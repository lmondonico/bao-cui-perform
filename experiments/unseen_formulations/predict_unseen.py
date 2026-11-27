"""Predict the 12 in-house measured formulations with the five CALiSol LIC PerForm benchmark checkpoints."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from core.study import cache_path, load_protocols
from core.utils import get_tokenizer, load_preprocessed_data
from experiments.analyses import load_checkpoint
from modules.dataset_processing import CollateFn, FeatureEngineering
from modules.moiety_features import build_moiety_edges, get_adj_matrix, get_dist_matrix, map_atom_to_moiety

COLUMNS = [f"Solvent_{i}" for i in range(1, 6)] + [f"Salt_{i}" for i in range(1, 4)]


def canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def represent(smiles, tokenizer, vocab_size):
    atoms, edges, ids = tokenizer.get_moieties(smiles)[:3]
    covered = {atom for group in atoms for atom in group}
    mol = Chem.MolFromSmiles(smiles)
    if not ids or min(ids) < 1 or max(ids) >= vocab_size or any(a.GetAtomicNum() > 1 and a.GetIdx() not in covered for a in mol.GetAtoms()):
        raise ValueError(f"Not representable with the CALiSol vocabulary: {smiles}")
    nodes, links = list(range(len(atoms))), list(build_moiety_edges(edges, map_atom_to_moiety(atoms)))
    return {"moiety_ids": ids, "clique_atom_ids": atoms, "edges": edges,
            "adj_matrix": get_adj_matrix(nodes, links), "dist_matrix": get_dist_matrix(nodes, links)}


def inputs():
    frame = pd.read_csv(HERE / "unseen_formulations.csv")
    mapping, vocab_size, key_map, _ = load_preprocessed_data(custom_path=str(cache_path("calisol")))
    tokenizer = get_tokenizer()
    tokenizer.moiety_key_map = dict(key_map)
    tokenizer.key_moiety_map = {v: k for k, v in key_map.items()}
    tokenizer.next_key = vocab_size
    known = {canonical(s): s for s in mapping}
    new = []
    for column in COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        for index, value in frame[column].dropna().items():
            smiles, amount = value.rsplit(":", 1)
            key = known.get(canonical(smiles), smiles)
            if key not in mapping:
                mapping[key] = represent(key, tokenizer, vocab_size)
                new.append(key)
            frame.at[index, column] = f"{key}:{amount}"
    frame["Row_ID"], frame["LIC"] = frame.ID, 0.0
    return frame, mapping, vocab_size, key_map, sorted(set(new))


def main():
    frame, mapping, vocab_size, key_map, new = inputs()
    print("Molecules outside the CALiSol cache:", new)
    protocols = load_protocols()
    for fold in range(1, 6):
        protocol = f"cv_calisol_fold{fold}"
        checkpoint = ROOT / f"experiments/results/runs/calisol/calisol_benchmark_perform_LIC_fold{fold}/best_model.pth"
        model, state, identity = load_checkpoint(Namespace(checkpoint=checkpoint, protocol=protocol, device="cpu"),
                                                 protocols[protocol], required_target="LIC")
        if state["vocab_size"] != vocab_size or state["moiety_key_map"] != key_map:
            raise ValueError("Checkpoint vocabulary differs from the CALiSol cache")
        dataset = FeatureEngineering(frame, None, mapping, "LIC", "conductivity", 0.0, training=False)
        batch = next(iter(DataLoader(dataset, batch_size=len(frame), collate_fn=CollateFn(max_num_mols=8))))
        with torch.inference_mode():
            frame[f"fold{fold}_mS_cm"] = np.power(10.0, model(batch).reshape(-1).numpy())
    folds = frame.filter(like="fold")
    frame["Model_mS_cm"], frame["Model_SD_mS_cm"] = folds.mean(axis=1), folds.std(axis=1, ddof=1)
    frame["AE_mS_cm"] = (frame.Model_mS_cm - frame.Measured_mS_cm).abs()
    keep = ["ID", "Formulation", "Family", "Source", "Literature_T_C", "Literature_mS_cm", "Measured_mS_cm",
            "Model_mS_cm", "Model_SD_mS_cm", "AE_mS_cm", *folds.columns]
    frame[keep].to_csv(HERE / "unseen_predictions.csv", index=False, float_format="%.4f")
    print(frame[keep[:-5]].round(2).to_string(index=False))
    for name, part in [("all", frame), *frame.groupby("Family")]:
        print(f"{name}: n {len(part)}, MAE {part.AE_mS_cm.mean():.2f} mS/cm")


if __name__ == "__main__":
    main()
