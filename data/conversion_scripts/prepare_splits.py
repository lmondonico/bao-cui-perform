"""Write the seed-42 five-fold splits shared by every model, including subset and strict CALiSol splits."""

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from sklearn.model_selection import KFold

if __package__:
    from .prepare_data import in_subset, plan_path, write_json
else:
    from prepare_data import in_subset, plan_path, write_json

DATASETS = ("calisol", "ilthermo", "des", "chew_enthalpy", "chew_density")
SEED = 42
SPLITS = ("train", "validation", "test")


def split_filename(dataset, partition):
    if partition not in SPLITS:
        raise ValueError(f"Unknown partition: {partition}")
    return f"{dataset}_{partition}.csv"


def canonical_smiles(value):
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {value}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def panel_exclusions(frame):
    fsi = canonical_smiles("[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F")
    tfsi = canonical_smiles("[Li+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F")
    bf4 = canonical_smiles("[Li+].[B-](F)(F)(F)F")
    pf6 = canonical_smiles("[Li+].[P-](F)(F)(F)(F)(F)F")
    pc = canonical_smiles("CC1COC(=O)O1")
    dec = canonical_smiles("CCOC(=O)OCC")
    dme = canonical_smiles("COCCOC")
    records = []
    for _, row in frame.iterrows():
        solvents, salts = set(), set()
        for prefix, count, target in (("Solvent", 5, solvents), ("Salt", 3, salts)):
            for index in range(1, count + 1):
                value = row.get(f"{prefix}_{index}")
                if pd.isna(value) or not str(value).strip():
                    continue
                smiles, amount = str(value).rsplit(":", 1)
                if float(amount) > 0:
                    target.add(canonical_smiles(smiles))
        reason = None
        if fsi in salts and len(solvents) in (1, 5):
            reason = "LiFSI_single_or_five_solvent_systems"
        elif tfsi in salts and solvents == {dme}:
            reason = "LiTFSI_DME"
        elif bf4 in salts and solvents in ({pc}, {pc, dec}):
            reason = "LiBF4_PC_or_PC_DEC"
        elif pf6 in salts and solvents == {pc}:
            reason = "LiPF6_PC"
        if reason:
            records.append({"Row_ID": int(row.Row_ID), "Composition_ID": row.Composition_ID,
                            "System_ID": row.System_ID, "reason": reason})
    return pd.DataFrame(records, columns=("Row_ID", "Composition_ID", "System_ID", "reason"))


def fold_partitions(frame, seed, grouped=False):
    values = pd.Series(sorted(frame.Composition_ID.unique())) if grouped else frame
    shuffled = values.sample(frac=1, random_state=seed).reset_index(drop=True)
    pool_end = int(0.9 * len(shuffled))
    pool, test = shuffled.iloc[:pool_end], shuffled.iloc[pool_end:]
    for fold, (train, validation) in enumerate(KFold(5, shuffle=True, random_state=42).split(pool), 1):
        parts = {"train": pool.iloc[train], "validation": pool.iloc[validation], "test": test}
        if grouped:
            parts = {name: frame[frame.Composition_ID.isin(groups)] for name, groups in parts.items()}
        yield fold, parts


def write_partition(output, relative, partitions, dataset, target, grouped=False, parent=None,
                    seed=SEED, fold=None, regime="cv"):
    ids = [set(part.Row_ID) for part in partitions.values()]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError(f"Overlapping observation membership: {relative}")
    if grouped:
        groups = [set(part.Composition_ID) for part in partitions.values()]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError(f"Overlapping composition membership: {relative}")
    files = {}
    for name, part in partitions.items():
        if part.empty:
            raise ValueError(f"Empty {name} partition: {relative}")
        filename = split_filename(dataset, name)
        path = output / relative / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(path, index=False, lineterminator="\n")
        files[filename] = {"path": str(path.relative_to(output)), "rows": len(part),
                           "compositions": int(part.Composition_ID.nunique()),
                           "chemical_systems": int(part.System_ID.nunique())}
    first = partitions["train"].iloc[0]
    transform = "identity" if target == "IC" else str(first.Target_transform)
    return {"dataset": dataset, "files": files, "split_dir": str(relative), "composition_disjoint": grouped,
            "parent": parent, "seed": seed, "fold": fold, "target": target, "regime": regime,
            "target_transform": transform, "transformation": transform,
            "target_unit": str(first.Physical_unit if target == "IC" else first.Target_unit)}


def freeze(data_root):
    check_root(data_root)
    path = plan_path(data_root)
    plan = json.loads(path.read_text())
    sources = plan["data"]
    output = data_root.parent
    protocols = {}
    for dataset in DATASETS:
        spec = sources["datasets"][dataset]
        frame = pd.read_csv(data_root / spec["path"], low_memory=False, float_precision="round_trip")
        if not frame.Row_ID.is_unique:
            raise ValueError(f"Duplicate Row_ID: {spec['path']}")
        relative = f"data/splits-{dataset.replace('_', '-')}"
        if dataset == "calisol":
            excluded = panel_exclusions(frame)
            excluded.to_csv(output / relative / "calisol_panel_exclusions.csv", index=False, lineterminator="\n")
            frame = frame[~frame.Row_ID.isin(excluded.Row_ID)].reset_index(drop=True)
            frame = pd.read_csv(io.StringIO(frame.to_csv(index=False, lineterminator="\n")), low_memory=False, float_precision="round_trip")
            frame.to_csv(output / relative / "calisol.csv", index=False, lineterminator="\n")
        children = [name for name, child in sources["datasets"].items() if child.get("parent") == dataset]
        members = {}
        for name in children:
            supplied = pd.read_csv(data_root / sources["datasets"][name]["path"], low_memory=False, float_precision="round_trip")
            expected = frame[[in_subset(name, r.Salt_count, r.Component_count, r.Components_JSON) for r in frame.itertuples()]]
            if set(supplied.Row_ID) != set(expected.Row_ID) or not 0 < len(supplied) < len(frame):
                raise ValueError(f"Subset rows differ from its parent: {name}")
            members[name] = set(supplied.Row_ID)
        target = "LIC" if dataset == "calisol" else "Target"
        for fold, parts in fold_partitions(frame, SEED):
            key = f"cv_{dataset}_fold{fold}"
            protocols[key] = write_partition(output, f"{relative}/fold_{fold}", parts, dataset, target, fold=fold)
            for name in children:
                sub_parts = {part_name: part[part.Row_ID.isin(members[name])] for part_name, part in parts.items()}
                protocols[f"cv_{name}_fold{fold}"] = write_partition(
                    output, f"data/splits-{name.replace('_', '-')}/fold_{fold}", sub_parts, name, target,
                    parent=key, fold=fold, regime="common_subset")
        if dataset == "calisol":
            for fold, parts in fold_partitions(frame, SEED, grouped=True):
                protocols[f"cv_calisol_lmo_fold{fold}"] = write_partition(
                    output, f"{relative}/strict_lmo/fold_{fold}", parts, dataset, target,
                    grouped=True, fold=fold, regime="strict_lmo")
    planned = {job["protocol"] for job in plan.get("jobs", [])}
    if planned and planned != set(protocols):
        raise RuntimeError("Prepared protocols differ from the experiment plan")
    plan["protocols"] = protocols
    write_json(path, plan)
    verify(data_root)


def verify(data_root):
    check_root(data_root)
    plan = json.loads(plan_path(data_root).read_text())
    for name, protocol in plan["protocols"].items():
        ids = []
        for filename, item in protocol["files"].items():
            frame = pd.read_csv(data_root.parent / item["path"], usecols=["Row_ID"])
            if len(frame) != item["rows"]:
                raise RuntimeError(f"Split row count changed: {item['path']}")
            ids.append(set(frame.Row_ID))
        if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
            raise RuntimeError(f"Overlapping split membership: {name}")
    print(f"Verified {len(plan['protocols'])} split protocols")


def check_root(data_root):
    if data_root.name != "data" or "_archive" in data_root.parts:
        raise ValueError("Only the repository data root is supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "verify"))
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    arguments = parser.parse_args()
    (freeze if arguments.command == "freeze" else verify)(arguments.data_root.resolve())
