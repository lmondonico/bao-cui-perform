"""Export frozen baseline inputs without fitting models or changing membership."""

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from data.conversion_scripts.check_compatibility import components, GEOMIX_ELEMENTS
from data.conversion_scripts.prepare_data import GEOMIX_TEMPLATES
from core.study import load_protocols, local_path
from data.conversion_scripts.prepare_splits import split_filename

STUDY_PLAN = ROOT / "config/study.json"
SPLITS = ("train", "validation", "test")
MODELS = ("smited", "fgcn", "molsets", "geomix")
CONVERTERS = {
    "smited": "baselines/FGCN_SMITED-IC/SMITED-IC/SMITED_convert_splits.py",
    "fgcn": "baselines/FGCN_SMITED-IC/FGCN/FGCN_convert_splits.py",
    "molsets": "baselines/MolSets/prepare_splits_data.py",
}


def module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def check_components(actual, expected, rtol=1e-9):
    actual = sorted(actual)
    expected = sorted((s, amount) for s, amount, _ in expected)
    if [s for s, _ in actual] != [s for s, _ in expected] or not np.allclose(
            [amount for _, amount in actual], [amount for _, amount in expected], rtol=rtol, atol=1e-12):
        raise ValueError("Converted component identities or amounts changed")


def native_csv(frame, model, target):
    converter = module(CONVERTERS[model], f"prepare_{model}")
    converter.TARGET = target
    function = converter.process_row_to_smited_format if model == "smited" else converter.process_row
    output = []
    for _, row in frame.iterrows():
        expected = components(row)
        if sum(role == "Salt" for _, _, role in expected) > 1:
            raise ValueError("single_salt_restriction")
        if model == "fgcn" and len(expected) > 6:
            raise ValueError("exceeds_six_components")
        converted = function(row)
        if converted is None:
            raise ValueError(f"Converter rejected Row_ID={row.Row_ID}")
        if model == "smited":
            fields = converted["smiles"].split("<sep>")
            actual = [(fields[i], float(fields[i + 1]) / 100) for i in range(0, len(fields) - 1, 2)]
            assert fields[-1] == f"{row.Temperature:.4g}"
            assert converted["temperature"] == row.Temperature and converted["ionic"] == row[target]
        else:
            actual = [(converted[f"smiles{i}"], converted[f"conc{i}"] / 100)
                      for i in range(1, 7) if converted[f"conc{i}"] > 0]
            assert converted["temperature ( C )"] == row.Temperature
            assert converted["log(cond) (uS/cm)"] == row[target]
        total = sum(x for _, x, _ in expected)
        check_components(actual, [(s, x / total, r) for s, x, r in expected], 1e-3 if model == "smited" else 1e-9)
        output.append(converted)
    return pd.DataFrame(output)


def prepare_fgcn(frames, native, output, target):
    path = "baselines/FGCN_SMITED-IC/FGCN/FGCN_preprocess_data.py"
    processor = module(path, "prepare_fgcn_features")
    processor.process_single_smiles = lru_cache(None)(processor.process_single_smiles)
    original_pad = processor.pad_graph
    padded = {}

    def pad(node, adjacency, maximum):
        key = id(node), maximum
        if key not in padded:
            padded[key] = original_pad(node, adjacency, maximum)
        return padded[key]

    processor.pad_graph = pad
    names = [f"smiles{i}" for i in range(1, 7)]
    all_smiles = {s for frame in native.values() for name in names for s in frame[name]}
    maximum = max(70, max(Chem.AddHs(Chem.MolFromSmiles(s)).GetNumAtoms() for s in all_smiles))
    for split, frame in native.items():
        rows = []
        for position, row in frame.iterrows():
            row_id = int(frames[split].iloc[position].Row_ID)
            value = processor.process_formulation_row((row_id, row), names, [f"conc{i}" for i in range(1, 7)],
                ["temperature ( C )"], ["log(cond) (uS/cm)"], max_atoms=maximum)
            if value is None or not all(np.isfinite(a).all() for graph in value["graphs"] for a in graph):
                raise ValueError(f"Invalid F-GCN features for Row_ID={row_id}")
            assert value["idx"] == row_id and value["properties"][0] == frames[split].iloc[position][target]
            rows.append(value)
        with (output / f"{split}_data_preprocessed.pkl").open("wb") as stream:
            pickle.dump(rows, stream, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_molsets(frames, output, target):
    converter = module(CONVERTERS["molsets"], "prepare_molsets_features")
    converter.smiles_to_graph = lru_cache(None)(converter.smiles_to_graph)
    for split, frame in frames.items():
        rows = []
        for _, row in frame.iterrows():
            copy = row.copy()
            copy["LIC"] = row[target]
            value = converter.process_row(copy, int(row.Row_ID))
            comp = components(row)
            solvent = [(s, x) for s, x, role in comp if role == "Solvent"]
            salt = [(s, x) for s, x, role in comp if role == "Salt"]
            masses = [x * Descriptors.MolWt(Chem.MolFromSmiles(s)) for s, x in solvent]
            assert value[0] == row.Row_ID and value[6] == row[target] and value[7] == row.Temperature + 273.15
            assert len(value[1]) == len(solvent)
            np.testing.assert_allclose(value[3], np.array(masses) / sum(masses), rtol=1e-9)
            np.testing.assert_allclose(value[4], salt[0][1] * 1000 / sum(masses), rtol=1e-9)
            if any(not np.isfinite(graph.x.numpy()).all() for graph in [*value[1], value[5]]):
                raise ValueError(f"Nonfinite MolSets features for Row_ID={row.Row_ID}")
            rows.append(value)
        name = "val" if split == "validation" else split
        with (output / f"data_{name}.pkl").open("wb") as stream:
            pickle.dump(rows, stream, protocol=pickle.HIGHEST_PROTOCOL)


@lru_cache(None)
def geomix_template(name):
    tree = ast.parse((ROOT / "baselines/GeoMix/dataset/CALiSol/preprocess/get_3d_salt_data.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == name)
    return np.array(ast.literal_eval(node.value.args[0]))


def ion_geometry(fragment, seed):
    smiles = Chem.MolToSmiles(fragment)
    if smiles in GEOMIX_TEMPLATES:
        name, center = GEOMIX_TEMPLATES[smiles]
        return geomix_template(name), np.array([center] + [9] * 6)
    mol = Chem.AddHs(fragment)
    if mol.GetNumAtoms() == 1:
        return np.zeros((1, 3)), np.array([mol.GetAtomWithIdx(0).GetAtomicNum()])
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol)
    elif AllChem.UFFHasAllMoleculeParams(mol):
        AllChem.UFFOptimizeMolecule(mol)
    return mol.GetConformer().GetPositions(), np.array([a.GetAtomicNum() for a in mol.GetAtoms()])


def geometry(smiles, seed):
    ions = [ion_geometry(f, seed) for f in Chem.GetMolFrags(Chem.MolFromSmiles(smiles), asMols=True)]
    if any(ion is None for ion in ions):
        return None
    placed = []
    for xyz, _ in ions:
        xyz = xyz - xyz.mean(axis=0)
        if placed:
            xyz[:, 0] += np.vstack(placed)[:, 0].max() - xyz[:, 0].min() + 4.0
        placed.append(xyz)
    return np.vstack(placed), np.concatenate([numbers for _, numbers in ions])


def prepare_geomix(frames, output, target, seeds, dataset):
    roles = {role: sorted({s for frame in frames.values() for _, row in frame.iterrows()
                          for s, _, r in components(row) if role == r}) for role in ("Solvent", "Salt")}
    for split, frame in frames.items():
        converted = []
        for _, row in frame.iterrows():
            comp = components(row)
            solvent = [(s, x) for s, x, role in comp if role == "Solvent"]
            salt = [(s, x) for s, x, role in comp if role == "Salt"]
            mass = sum(x * Descriptors.MolWt(Chem.MolFromSmiles(s)) for s, x in solvent)
            as_is = dataset == "calisol"
            value = {"k": row[target], "T": row.Temperature + 273.15, "c": salt[0][1] if as_is else salt[0][1] * 1000 / mass,
                     "salt": salt[0][0], "c units": "mol/kg", "solvent ratio type": "mol"}
            value.update({s: dict(solvent).get(s, 0.) for s in roles["Solvent"]})
            recovered_salt = value["c"] if as_is else value["c"] * mass / 1000
            check_components([(s, value[s]) for s in roles["Solvent"] if value[s] > 0]
                             + [(value["salt"], recovered_salt)], comp)
            converted.append(value)
        pd.DataFrame(converted).to_csv(output / split_filename(dataset, split), index=False)
    unsupported = sorted(s for smiles in roles.values() for s in smiles if any(
        a.GetAtomicNum() not in GEOMIX_ELEMENTS for a in Chem.AddHs(Chem.MolFromSmiles(s)).GetAtoms()))
    reasons = [f"unsupported GeoMix elements: {unsupported}"] if unsupported else []
    if not unsupported:
        for seed in seeds:
            arrays = {}
            failures = []
            for role, smiles_list in roles.items():
                records = {key: [] for key in ("name_list", "coordinates", "atomic_numbers", "edge_index", "edge_attr")}
                for smiles in smiles_list:
                    built = geometry(smiles, seed)
                    if built is None:
                        failures.append(smiles)
                        continue
                    xyz, numbers = built
                    distances = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
                    edges = np.argwhere((distances < 6.) & (distances > 0)).T
                    if not np.isfinite(xyz).all() or not edges.shape[1]:
                        failures.append(smiles)
                        continue
                    for key, value in {"name_list": smiles, "coordinates": xyz,
                        "atomic_numbers": numbers,
                        "edge_index": edges, "edge_attr": distances[edges[0], edges[1]]}.items():
                        records[key].append(value)
                arrays[role] = {}
                for key, values in records.items():
                    array = np.empty(len(values), dtype=object)
                    array[:] = values
                    arrays[role][key] = array
            if failures:
                reasons.append(f"no conformer for seed {seed}: {sorted(set(failures))}")
            else:
                for role, values in arrays.items():
                    np.savez(output / f"{role}_seed{seed}.npz", **values)
    return reasons


def protocol_for(split_dir):
    matches = [p for p in load_protocols().values() if (ROOT / p["split_dir"]).resolve() == Path(split_dir).resolve()]
    if len(matches) != 1:
        raise ValueError("Split directory must identify one protocol")
    return matches[0]


def verify_prepared(split_dir, model, target):
    split_dir = local_path(split_dir, ROOT / "data")
    protocol = protocol_for(split_dir)
    output = split_dir / "formats" / model
    if protocol["target"] != target:
        raise ValueError(f"Prepared target mismatch: {model}")
    if not output.is_dir() or not any(output.iterdir()):
        raise ValueError(f"Unavailable prepared {model} inputs: {split_dir}")
    details = {}
    if model == "fgcn":
        with (output / "train_data_preprocessed.pkl").open("rb") as stream:
            details["max_atoms"] = pickle.load(stream)[0]["graphs"][0][0].shape[0]
    if model == "geomix":
        details["salts"] = sorted(set().union(*(pd.read_csv(output / split_filename(protocol["dataset"], s)).salt for s in SPLITS)))
        details["geometry_seeds"] = sorted(int(path.stem.split("seed")[1]) for path in output.glob("Salt_seed*.npz"))
    return output, {"details": details}


def prepare(split_dir, model, target, seeds):
    split_dir = local_path(split_dir, ROOT / "data")
    protocol = protocol_for(split_dir)
    final = split_dir / "formats" / model
    output = split_dir / "formats" / f".{model}.tmp"
    shutil.rmtree(final, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    if protocol["target"] != target:
        raise ValueError("Prepared target must match the protocol")
    frames = {s: pd.read_csv(split_dir / split_filename(protocol["dataset"], s), low_memory=False) for s in SPLITS}
    if pd.concat(frames.values()).Row_ID.duplicated().any():
        raise ValueError("Observation membership overlaps between partitions")
    reasons = []
    for split, frame in frames.items():
        if not np.isfinite(frame[[target, "Temperature"]]).all().all():
            raise ValueError(f"Invalid source partition: {split}")
        for _, row in frame.iterrows():
            components(row)
        bad = frame.Salt_count.gt(1) if model in ("smited", "fgcn") else (
            frame.Salt_count.ne(1) | frame.Component_count.le(frame.Salt_count))
        if bad.any():
            reasons.append(f"{split}: {int(bad.sum())} rows outside the model's salt restriction")
    if not reasons:
        output.mkdir(parents=True)
        try:
            if model in ("smited", "fgcn"):
                native = {s: native_csv(frame, model, target) for s, frame in frames.items()}
                for split, frame in native.items():
                    frame.to_csv(output / split_filename(protocol["dataset"], split), index=False)
                if model == "smited":
                    path = ROOT / "baselines/FGCN_SMITED-IC/SMITED-IC/utils/SMITED_load.py"
                    pattern = re.compile(ast.literal_eval(ast.parse(path.read_text()).body[0].value))
                    vocabulary = set((path.parents[1] / "bert_vocab_curated.txt").read_text().splitlines())
                    for split, frame in native.items():
                        if frame.smiles.map(lambda s: len(pattern.findall(s)) + 2).gt(202).any():
                            reasons.append(f"{split}: formulations exceed 202 tokens")
                        molecules = {s for _, row in frames[split].iterrows() for s, _, _ in components(row)}
                        if {token for s in molecules for token in pattern.findall(s)} - vocabulary:
                            reasons.append(f"{split}: unknown SMI-TED tokens")
                else:
                    prepare_fgcn(frames, native, output, target)
            elif model == "molsets":
                prepare_molsets(frames, output, target)
            else:
                reasons.extend(prepare_geomix(frames, output, target, seeds, protocol["dataset"]))
        except Exception as error:
            reasons.append(f"preparation failed: {type(error).__name__}: {error}")
    if reasons:
        shutil.rmtree(output, ignore_errors=True)
    else:
        os.replace(output, final)
    if reasons:
        print(f"{model} unavailable for {split_dir.relative_to(ROOT)}: {'; '.join(reasons)}", flush=True)
    return not reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--split-dir", type=Path)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--target")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--geometry-seeds", type=int, nargs="+")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = json.loads(STUDY_PLAN.read_text())
    if args.all:
        required = {job["protocol"] for job in plan["jobs"] if job["model"] in args.models}
        protocols = [p for key, p in plan["protocols"].items() if key in required]
    else:
        protocols = [protocol_for(args.split_dir)]
    if not args.worker and not args.verify:
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        workers = []
        for model in args.models:
            environment = {"smited": "baselines/FGCN_SMITED-IC/.venv_smited",
                           "fgcn": "baselines/FGCN_SMITED-IC/.venv_smited",
                           "geomix": "baselines/GeoMix/.venv_geomix"}.get(model, ".venv-perform")
            command = [str(ROOT / environment / "bin/python"), "-s", "-B", str(Path(__file__)), "--models", model, "--worker"]
            command += ["--all"] if args.all else ["--split-dir", str(args.split_dir)]
            if args.target:
                command += ["--target", args.target]
            if args.geometry_seeds:
                command += ["--geometry-seeds", *map(str, args.geometry_seeds)]
            workers.append(subprocess.Popen(command, env=env, cwd=ROOT))
        for worker in workers:
            if worker.wait():
                raise subprocess.CalledProcessError(worker.returncode, worker.args)
        return
    jobs = [(ROOT / p["split_dir"], model, args.target or p["target"], args.geometry_seeds or [p["seed"]])
            for p in protocols for model in args.models]
    if args.verify:
        ready = sum((directory / "formats" / model).is_dir() for directory, model, _, _ in jobs)
    else:
        with ProcessPoolExecutor(max(1, os.cpu_count() // len(MODELS))) as pool:
            ready = sum(pool.map(prepare, *zip(*jobs)))
    print(f"{'Verified' if args.verify else 'Prepared'} {len(jobs)} input sets ({', '.join(args.models)}): {ready} ready, {len(jobs) - ready} unavailable.", flush=True)


if __name__ == "__main__":
    main()
