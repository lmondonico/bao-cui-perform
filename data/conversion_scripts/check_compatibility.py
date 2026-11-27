"""Audit molecular coverage and model compatibility for every public record."""

import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(os.environ.get("PERFORM_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem

from modules.tokenizer.tokenizer_perform import Tokenizer
from modules.moiety_features import map_atom_to_moiety, build_moiety_edges
from data.conversion_scripts.prepare_data import GEOMIX_ELEMENTS, in_subset

MODELS = ("perform", "xgboost", "molformer", "attentivefp", "smited", "molsets", "fgcn", "geomix")
MOLFORMER_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


def components(row):
    result = []
    for role, count in (("Solvent", 5), ("Salt", 3)):
        for i in range(1, count + 1):
            value = row.get(f"{role}_{i}")
            if pd.isna(value) or value == "":
                continue
            smiles, amount = str(value).rsplit(":", 1)
            amount = float(amount)
            if not np.isfinite(amount) or amount <= 0:
                raise ValueError(f"Nonpositive/nonfinite component: {value}")
            result.append((smiles, amount, role))
    if len(result) != int(row["Component_count"]):
        raise ValueError(f"Component count changed for Row_ID={row['Row_ID']}")
    if sum(role == "Salt" for _, _, role in result) != int(row["Salt_count"]):
        raise ValueError(f"Salt count changed for Row_ID={row['Row_ID']}")
    return result


def molecular_audit(smiles_list):
    import torch
    tokenizer = Tokenizer()
    vocabulary = torch.load(ROOT / "baselines/Baseline_Utils/Molformer/training/pubchem_canon_zinc_final_vocab_sorted.pth", map_location="cpu", weights_only=False)
    token_pattern = re.compile(MOLFORMER_PATTERN)
    audit, graphs, hash_groups = {}, {}, {}
    for smiles in sorted(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            audit[smiles] = {"valid_smiles": False, "perform_issues": ["invalid_smiles"]}
            continue
        cliques, atom_edges, ids, matches = tokenizer.get_moieties(smiles)
        covered = set().union(*map(set, cliques)) if cliques else set()
        missing = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1 and a.GetIdx() not in covered]
        fragments = Chem.GetMolFrags(mol)
        missing_fragments = [list(fragment) for fragment in fragments if not (set(fragment) & covered)]
        graph = nx.Graph()
        graph.add_nodes_from((i, {"token": name, "detail": name if name.startswith("ring_") else
                                 Chem.MolFragmentToSmiles(mol, atomsToUse=cliques[i], isomericSmiles=True)})
                             for i, (name, _) in enumerate(matches))
        graph.add_edges_from(build_moiety_edges(atom_edges, map_atom_to_moiety(cliques)))
        signature = nx.weisfeiler_lehman_graph_hash(graph, node_attr="token") if len(graph) else "empty"
        issues = []
        if not ids:
            issues.append("empty_representation")
        if missing:
            issues.append("unrepresented_heavy_atoms")
        if missing_fragments:
            issues.append("unrepresented_component_fragment")
        tokens = token_pattern.findall(smiles)
        audit[smiles] = {
            "valid_smiles": True, "heavy_atoms": mol.GetNumHeavyAtoms(),
            "nonisomeric_smiles": Chem.MolToSmiles(mol, isomericSmiles=False),
            "atoms_with_hydrogen": Chem.AddHs(mol).GetNumAtoms(),
            "elements": sorted({a.GetSymbol() for a in mol.GetAtoms()}),
            "tokens": [name for name, _ in matches], "missing_atom_indices": missing,
            "missing_atoms": [mol.GetAtomWithIdx(i).GetSymbol() + f"({mol.GetAtomWithIdx(i).GetFormalCharge():+d})" for i in missing],
            "missing_fragments": missing_fragments, "perform_issues": issues,
            "graph_hash": signature,
            "molformer_token_count": len(tokens) + 2,
            "molformer_unknown_tokens": sorted(set(tokens) - set(vocabulary)),
            "molformer_unparsed": "".join(tokens) != smiles,
            "geomix_unsupported_elements": sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() not in GEOMIX_ELEMENTS}),
        }
        graphs[smiles] = graph
        hash_groups.setdefault(signature, []).append(smiles)
    collisions = []
    for group in hash_groups.values():
        remaining = list(group)
        while remaining:
            first = remaining.pop(0)
            same = [other for other in remaining if nx.is_isomorphic(graphs[first], graphs[other], node_match=lambda a, b: a["token"] == b["token"])]
            if same:
                collision = [first] + same
                collisions.append(collision)
                for other in collision:
                    kinds = {}
                    for candidate in collision:
                        if candidate == other:
                            continue
                        if audit[other]["nonisomeric_smiles"] == audit[candidate]["nonisomeric_smiles"]:
                            kind = "stereochemistry"
                        elif nx.is_isomorphic(graphs[other], graphs[candidate], node_match=lambda a, b: a == b):
                            if any(name.startswith("ring_") for name in audit[other]["tokens"]):
                                kind = "ring_fallback"
                            elif set(audit[other]["tokens"]) & {"benzene", "naphthalene"}:
                                kind = "ring_substitution"
                            else:
                                kind = "unexplained"
                        else:
                            kind = "unexplained"
                        kinds[candidate] = kind
                    audit[other]["collision_with"] = kinds
                    if "unexplained" in kinds.values():
                        audit[other]["perform_issues"].append("representation_collision")
                remaining = [other for other in remaining if other not in same]
    return audit, collisions


def row_issues(row, molecules, smited_pattern, smited_vocab, dataset_smiles):
    comp = components(row)
    issues = {model: [] for model in MODELS}
    salts = sum(role == "Salt" for _, _, role in comp)
    neutral = len(comp) - salts
    for smiles, _, _ in comp:
        info = molecules[smiles]
        if not info["valid_smiles"]:
            for model in MODELS:
                issues[model].append("invalid_smiles")
            continue
        issues["perform"].extend(reason for reason in info["perform_issues"] if reason != "representation_collision")
        if any(other in dataset_smiles and kind == "unexplained"
               for other, kind in info.get("collision_with", {}).items()):
            issues["perform"].append("representation_collision")
        if info["molformer_token_count"] > 202:
            issues["molformer"].append("molecule_exceeds_202_tokens")
        if info["molformer_unknown_tokens"] or info["molformer_unparsed"]:
            issues["molformer"].append("unknown_or_unparsed_molecular_tokens")
        if info["geomix_unsupported_elements"]:
            issues["geomix"].append("unsupported_element_features")
        if set(smited_pattern.findall(smiles)) - smited_vocab:
            issues["smited"].append("unknown_molecular_tokens_encoded_as_padding")
    for model in ("smited", "fgcn", "molsets", "geomix"):
        if salts > 1:
            issues[model].append("single_salt_restriction")
    for model in ("molsets", "geomix"):
        if salts == 0:
            issues[model].append("requires_salt_branch")
        if neutral == 0:
            issues[model].append("requires_neutral_branch")
    if len(comp) > 6:
        issues["fgcn"].append("exceeds_six_components")
    serialized = []
    for smiles, amount, _ in sorted(comp, key=lambda x: x[2] != "Salt"):
        serialized.extend((smiles, f"{amount / sum(c[1] for c in comp) * 100:.4g}"))
    serialized.append(f"{row['Temperature']:.4g}")
    tokens = smited_pattern.findall("<sep>".join(serialized))
    if len(tokens) + 2 > 202:
        issues["smited"].append("formulation_exceeds_202_tokens")
    return {model: sorted(set(reasons)) for model, reasons in issues.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from core.study import DATA, OUTPUT, load_data, local_path
    args.data_root = local_path(args.data_root, DATA)
    args.output = local_path(args.output, OUTPUT)
    manifest = load_data()
    paths = {name: args.data_root / item.get('benchmark_path', item['path']) for name, item in manifest['datasets'].items()
             if not item.get('parent')}
    if not paths:
        raise ValueError("No canonical public datasets found")
    frames = {name: pd.read_csv(path, low_memory=False) for name, path in paths.items()}
    all_smiles = {smiles for frame in frames.values() for _, row in frame.iterrows() for smiles, _, _ in components(row)}
    molecules, _ = molecular_audit(all_smiles)
    smited_source = ROOT / "baselines/FGCN_SMITED-IC/SMITED-IC/utils/SMITED_load.py"
    smited_pattern = re.compile(ast.literal_eval(ast.parse(smited_source.read_text()).body[0].value))
    smited_vocab = set((smited_source.parents[1] / "bert_vocab_curated.txt").read_text().splitlines())
    args.output.mkdir(parents=True, exist_ok=True)
    matrix = []
    for name, frame in frames.items():
        dataset_smiles = {s for _, row in frame.iterrows() for s, _, _ in components(row)}
        reasons = [row_issues(row, molecules, smited_pattern, smited_vocab, dataset_smiles) for _, row in frame.iterrows()]
        subsets = {name: frame.index}
        for subset, item in manifest['datasets'].items():
            if item.get('parent') == name:
                supplied = pd.read_csv(args.data_root / item['path'], low_memory=False)
                mask = [in_subset(subset, r.Salt_count, r.Component_count, r.Components_JSON) for r in frame.itertuples()]
                pd.testing.assert_frame_equal(supplied, frame.loc[mask].reset_index(drop=True), check_dtype=False)
                subsets[subset] = frame.index[mask]
        for label, indexes in subsets.items():
            for model in MODELS:
                failures = [i for i in indexes if reasons[i][model]]
                counts = {}
                for i in indexes:
                    for reason in reasons[i][model]:
                        counts[reason] = counts.get(reason, 0) + 1
                matrix.append({"dataset": label, "model": model, "status": "blocked" if failures else "pending_forward_train_predict_check",
                               "observations": len(indexes), "affected_observations": len(failures),
                               "reasons": json.dumps(counts, sort_keys=True)})
    pd.DataFrame(matrix).to_csv(args.output / "compatibility_matrix.csv", index=False)
    blocked = sum(row["status"] == "blocked" for row in matrix)
    print(f"Compatibility: {len(matrix)} model–dataset pairs checked, {blocked} blocked.")


if __name__ == "__main__":
    main()
