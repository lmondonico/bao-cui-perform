"""Preprocess and cache molecular representations."""

import os
import sys
import numpy as np
import pandas as pd
import tqdm


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.moiety_features import (
    map_atom_to_moiety,
    build_moiety_edges,
    get_adj_matrix,
    get_dist_matrix,
)
from core.utils import get_tokenizer
from config.datasets import get_dataset_config

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def smiles_dict_preprocessing(data_path, dataset_type, process_id=None, output_path=None):
    print("Preprocessing molecules SMILES strings...")
    print(f"Dataset type: {dataset_type}")
    if process_id:
        print(f"Process ID: {process_id}")

    config = get_dataset_config(dataset_type)
    if not config:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    max_num_solvent_molecules = config["max_solvents"]
    max_num_salt_molecules = config["max_salts"]

    dtypes = {}
    for i in range(1, max_num_solvent_molecules + 1):
        dtypes[f"Solvent_{i}"] = str
    for i in range(1, max_num_salt_molecules + 1):
        dtypes[f"Salt_{i}"] = str
    dtypes["Row_ID"] = int

    if dataset_type == "conductivity":
        dtypes["IC"] = float
        dtypes["Temperature"] = float

    import csv

    dataset = pd.read_csv(
        data_path,
        header=0,
        dtype=dtypes,
        low_memory=False,
        quoting=csv.QUOTE_ALL,
        skipinitialspace=True,
        on_bad_lines="skip",
    )
    solvent_columns = [f"Solvent_{i}" for i in range(1, max_num_solvent_molecules + 1)]
    salt_columns = [f"Salt_{i}" for i in range(1, max_num_salt_molecules + 1)]

    unique_SMILES_in_dataset = []

    for mol_col in solvent_columns + salt_columns:
        if mol_col in dataset.columns:
            for value in dataset[mol_col].dropna():
                if pd.notna(value) and value != "":
                    try:
                        parts = str(value).split(":")
                        if len(parts) == 2:
                            smiles = parts[0].strip()
                            if smiles:
                                unique_SMILES_in_dataset.append(smiles)
                    except (ValueError, AttributeError):
                        pass

    unique_SMILES_in_toy_dataset = sorted(set(unique_SMILES_in_dataset))
    print(f"Total unique SMILES to preprocess: {len(unique_SMILES_in_toy_dataset)}")

    tokenizer = get_tokenizer()

    all_mols_dict = {
        smi: {
            "adj_matrix": None,
            "dist_matrix": None,
            "clique_atom_ids": None,
            "edges": None,
            "moiety_ids": None,
        }
        for smi in unique_SMILES_in_toy_dataset
    }

    for smiles in tqdm.tqdm(unique_SMILES_in_toy_dataset, desc="Preprocessing"):
        try:
            result = tokenizer.get_moieties(smiles)
            clique_atom_ids, edges, moiety_ids = result[0], result[1], result[2]

            atom_to_moiety = map_atom_to_moiety(clique_atom_ids)
            moiety_edges = build_moiety_edges(edges, atom_to_moiety)

            node_list = list(range(len(clique_atom_ids)))
            moiety_edges = list(moiety_edges)

            dist_matrix = get_dist_matrix(node_list, moiety_edges)
            adj_matrix = get_adj_matrix(node_list, moiety_edges)

            all_mols_dict[smiles]["adj_matrix"] = adj_matrix
            all_mols_dict[smiles]["dist_matrix"] = dist_matrix
            all_mols_dict[smiles]["clique_atom_ids"] = clique_atom_ids
            all_mols_dict[smiles]["edges"] = edges
            all_mols_dict[smiles]["moiety_ids"] = moiety_ids

        except Exception as e:
            print(f"Error processing mol {smiles}: {e}")
            all_mols_dict[smiles]["adj_matrix"] = np.array([[]], dtype=np.float32)
            all_mols_dict[smiles]["dist_matrix"] = np.array([[]], dtype=np.float32)
            all_mols_dict[smiles]["clique_atom_ids"] = []
            all_mols_dict[smiles]["edges"] = []
            all_mols_dict[smiles]["moiety_ids"] = []

    preprocessed_data = {
        "molecules": all_mols_dict,
        "moiety_key_map": tokenizer.moiety_key_map,
        "key_moiety_map": tokenizer.key_moiety_map,
        "vocab_size": max(tokenizer.moiety_key_map.values()) + 1
        if tokenizer.moiety_key_map
        else 1,
    }

    from config.paths import DATA_DIR

    if output_path is None:
        filename = f"preprocessed_mol_info_{process_id}.npy" if process_id else "preprocessed_mol_info.npy"
        output_path = os.path.join(DATA_DIR, filename)
    np.save(output_path, preprocessed_data)
    print("+" * 60)
    print("Molecules preprocessing COMPLETE!")
    print("+" * 60, "\n")

    return output_path


if __name__ == "__main__":
    from config.datasets import interactive_dataset_choice

    data_path, dataset_type = interactive_dataset_choice()
    if not data_path or not dataset_type:
        print("No dataset selected. Exiting.")
        sys.exit(1)

    smiles_dict_preprocessing(data_path, dataset_type)
