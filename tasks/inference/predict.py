"""Predict one benchmark property for new mixtures with the five PerForm fold checkpoints of that task."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.study import cache_path
from core.utils import get_tokenizer, load_preprocessed_data
from modules.dataset_processing import CollateFn, FeatureEngineering
from modules.moiety_features import build_moiety_edges, get_adj_matrix, get_dist_matrix, map_atom_to_moiety
from modules.perform import PerForm, normalize_motif_state_dict
from modules.perform.perform_parm import ARCHITECTURE

TASKS = {"calisol": "LIC", "ilthermo": "Target", "des": "Target", "chew_density": "Target", "chew_enthalpy": "Target"}
COLUMNS = [f"Solvent_{i}" for i in range(1, 6)] + [f"Salt_{i}" for i in range(1, 4)]
INVERSE = {"log10": lambda x: np.power(10.0, x), "ln": np.exp, "identity": lambda x: x}


def checkpoint_path(task, fold):
    return ROOT / f"experiments/results/runs/{task}/{task}_benchmark_perform_{TASKS[task]}_fold{fold}/best_model.pth"


def load_model(task, fold, device):
    checkpoint = torch.load(checkpoint_path(task, fold), map_location="cpu", weights_only=False)
    model = PerForm(input_vocab_size=checkpoint["vocab_size"], max_num_molecules=checkpoint.get("max_num_molecules", 8),
                    target_label=TASKS[task], **checkpoint.get("architecture", ARCHITECTURE["PerForm"]))
    model.load_state_dict(normalize_motif_state_dict(checkpoint["model_state_dict"]), strict=True)
    return model.to(device).eval(), checkpoint


def represent(smiles, tokenizer, vocab_size):
    atoms, edges, ids = tokenizer.get_moieties(smiles)[:3]
    covered = {atom for group in atoms for atom in group}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not ids or min(ids) < 1 or max(ids) >= vocab_size or any(
            a.GetAtomicNum() > 1 and a.GetIdx() not in covered for a in mol.GetAtoms()):
        raise ValueError(f"Not representable with this task's moiety vocabulary: {smiles}")
    nodes, links = list(range(len(atoms))), list(build_moiety_edges(edges, map_atom_to_moiety(atoms)))
    return {"moiety_ids": ids, "clique_atom_ids": atoms, "edges": edges,
            "adj_matrix": get_adj_matrix(nodes, links), "dist_matrix": get_dist_matrix(nodes, links)}


def inputs(frame, task):
    mapping, vocab_size, key_map, _ = load_preprocessed_data(custom_path=str(cache_path(task)))
    tokenizer = get_tokenizer()
    tokenizer.moiety_key_map, tokenizer.key_moiety_map = dict(key_map), {v: k for k, v in key_map.items()}
    tokenizer.next_key = vocab_size
    known = {Chem.MolToSmiles(Chem.MolFromSmiles(s)): s for s in mapping}
    for column in COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        for index, value in frame[column].dropna().items():
            smiles, amount = str(value).rsplit(":", 1)
            mol = Chem.MolFromSmiles(smiles)
            key = known.get(Chem.MolToSmiles(mol), smiles) if mol is not None else smiles
            if key not in mapping:
                mapping[key] = represent(key, tokenizer, vocab_size)
            frame.at[index, column] = f"{key}:{amount}"
    return frame, mapping, vocab_size, key_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV with Solvent_1..5 and Salt_1..3 as SMILES:amount and Temperature in °C")
    parser.add_argument("--task", choices=TASKS, default="calisol")
    parser.add_argument("--output", type=Path, help="default: <input>_predictions.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    target = json.loads((ROOT / "config/study.json").read_text())["data"]["datasets"][args.task]["target"]
    original = pd.read_csv(args.input)
    frame, mapping, vocab_size, key_map = inputs(original.copy(), args.task)
    data = frame.assign(Row_ID=range(len(frame)), **{TASKS[args.task]: 0.0})
    dataset = FeatureEngineering(data, None, mapping, TASKS[args.task], "conductivity", 0.0, training=False)
    loader = DataLoader(dataset, batch_size=512, collate_fn=CollateFn(max_num_mols=8))
    folds = {}
    for fold in range(1, 6):
        model, checkpoint = load_model(args.task, fold, args.device)
        if checkpoint["vocab_size"] != vocab_size or checkpoint["moiety_key_map"] != key_map:
            raise ValueError("Checkpoint vocabulary differs from the task cache")
        with torch.inference_mode():
            values = [model({k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}).reshape(-1).cpu()
                      for batch in loader]
        folds[f"fold{fold}"] = INVERSE[target["transformation"]](torch.cat(values).double().numpy())
    folds = pd.DataFrame(folds, index=frame.index)
    output = original.join(folds).assign(prediction=folds.mean(axis=1), prediction_sd=folds.std(axis=1, ddof=1),
                                      unit=target["physical_unit"])
    path = args.output or args.input.with_name(f"{args.input.stem}_predictions.csv")
    output.to_csv(path, index=False)
    print(f"{len(output)} predictions ({target['physical_unit']}) -> {path}")


if __name__ == "__main__":
    main()
