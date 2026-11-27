"""Preprocess molecular graphs for F-GCN training."""

DATASET = "smited"

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import os
from tqdm import tqdm
import pickle

import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

warnings.filterwarnings("ignore")


def electronegativity(element_label, return_dict=False, nan_swap=np.nan):
    """
    Pauling electronegativity from Inorganic Chemistry Principles of Structure and Reactivity second Ed. James E. Huheey
    """
    electneg = {}
    electneg["H"] = 2.2
    electneg["He"] = np.nan
    electneg["Li"] = 0.98
    electneg["Be"] = 1.57
    electneg["B"] = 2.04
    electneg["C"] = 2.55
    electneg["N"] = 3.04
    electneg["O"] = 3.44
    electneg["F"] = 3.98
    electneg["Ne"] = np.nan
    electneg["Na"] = 0.93
    electneg["Mg"] = 1.31
    electneg["Al"] = 1.61
    electneg["Si"] = 1.9
    electneg["P"] = 2.19
    electneg["S"] = 2.58
    electneg["Cl"] = 3.16
    electneg["Ar"] = np.nan
    electneg["K"] = 0.82
    electneg["Ca"] = 1
    electneg["Sc"] = 1.36
    electneg["Ti"] = 1.54
    electneg["Ti(II)"] = 1.54
    electneg["V"] = 1.63
    electneg["V(II)"] = 1.563
    electneg["Cr"] = 1.66
    electneg["Cr(II)"] = 1.66
    electneg["Mn"] = 1.55
    electneg["Mn(II)"] = 1.55
    electneg["Fe"] = 1.83
    electneg["Fe(II)"] = 1.83
    electneg["Fe(III)"] = 1.96
    electneg["Co"] = 1.88
    electneg["Co(II)"] = 1.88
    electneg["Ni"] = 1.91
    electneg["Ni(II)"] = 1.91
    electneg["Cu"] = 1.9
    electneg["Cu(I)"] = 1.9
    electneg["Cu(II)"] = 2
    electneg["Zn"] = 1.65
    electneg["Zn(II)"] = 1.65
    electneg["Ga"] = 1.81
    electneg["Ga(III)"] = 1.81
    electneg["Ge"] = 2.01
    electneg["Ge(IV)"] = 2.01
    electneg["As"] = 2.18
    electneg["As(III)"] = 2.18
    electneg["Se"] = 2.55
    electneg["Br"] = 2.96
    electneg["Kr"] = 2.9
    electneg["Rb"] = 0.82
    electneg["Sr"] = 0.95
    electneg["Y"] = 1.22
    electneg["Zr"] = 1.33
    electneg["Zr(II)"] = 1.33
    electneg["Nb"] = 1.6
    electneg["Mo"] = 2.16
    electneg["Mo(II)"] = 2.16
    electneg["Mo(III)"] = 2.19
    electneg["Mo(IV)"] = 2.24
    electneg["Mo(V)"] = 2.27
    electneg["Mo(VI)"] = 2.35
    electneg["Tc"] = 1.9
    electneg["Ru"] = 2.2
    electneg["Rh"] = 2.28
    electneg["Pd"] = 2.2
    electneg["Ag"] = 1.93
    electneg["Cd"] = 1.69
    electneg["In"] = 1.78
    electneg["Sn"] = 1.96
    electneg["Sn(II)"] = 1.8
    electneg["Sn(IV)"] = 1.96
    electneg["Sb"] = 2.05
    electneg["Te"] = 2.1
    electneg["I"] = 2.66
    electneg["Xe"] = 2.6
    electneg["Cs"] = 0.79
    electneg["Ba"] = 0.89
    electneg["La"] = 1.1
    electneg["Ce"] = 1.12
    electneg["Pr"] = 1.13
    electneg["Nd"] = 1.14
    electneg["Pm"] = np.nan
    electneg["Sm"] = 1.17
    electneg["Eu"] = np.nan
    electneg["Gd"] = 1.2
    electneg["Tb"] = np.nan
    electneg["Dy"] = 1.22
    electneg["Ho"] = 1.23
    electneg["Er"] = 1.24
    electneg["Tm"] = 1.25
    electneg["Yb"] = np.nan
    electneg["Lu"] = 1.27
    electneg["Hf"] = 1.3
    electneg["Ta"] = 1.5
    electneg["W"] = 2.36
    electneg["Re"] = 1.9
    electneg["Os"] = 2.2
    electneg["Ir"] = 2.2
    electneg["Pt"] = 2.28
    electneg["Au"] = 2.54
    electneg["Hg"] = 2
    electneg["Tl"] = 1.62
    electneg["Tl(I)"] = 1.62
    electneg["Tl(III)"] = 2.04
    electneg["Pb"] = 2.33
    electneg["Pb(II)"] = 1.87
    electneg["Pb(IV)"] = 2.33
    electneg["Bi"] = 2.02
    electneg["Po"] = 2
    electneg["At"] = 2.2
    electneg["Rn"] = np.nan
    electneg["Fr"] = 0.7
    electneg["Ra"] = 0.9
    electneg["Ac"] = 1.1
    electneg["Th"] = 1.3
    electneg["Pa"] = 1.5
    electneg["U"] = 1.7
    electneg["Np"] = 1.3
    electneg["Pu"] = 1.3
    electneg["Am"] = 1.3
    electneg["Cm"] = 1.3
    electneg["Bk"] = 1.3
    electneg["Cf"] = 1.3
    electneg["Es"] = 1.3
    electneg["Fm"] = 1.3
    electneg["Md"] = 1.3
    electneg["No"] = 1.3
    electneg["Lr"] = np.nan
    electneg["Rf"] = np.nan
    electneg["Db"] = np.nan
    electneg["Sg"] = np.nan
    electneg["Bh"] = np.nan
    electneg["Hs"] = np.nan
    electneg["Mt"] = np.nan
    electneg["Ds"] = np.nan
    electneg["Rg"] = np.nan
    electneg["Cn"] = np.nan
    electneg["Nh"] = np.nan
    electneg["Fl"] = np.nan
    electneg["Mc"] = np.nan
    electneg["Lv"] = np.nan
    electneg["Ts"] = np.nan
    electneg["Og"] = np.nan

    electneg = {k.lower(): v for k, v in electneg.items()}

    if not np.isnan(nan_swap):
        electneg = {
            (k if not np.isnan(v) else k): (v if not np.isnan(v) else nan_swap)
            for k, v in electneg.items()
        }

    if return_dict is True:
        return electneg
    elif element_label.lower() in electneg.keys():
        return electneg[element_label.lower()]
    else:
        raise RuntimeError(
            f"Error no electronegativity for element label {element_label}"
        )


def get_adjacency_matrix(mol, use_bond_order=False, self_connection=False):
    """Function to get a molecule adjacency matrix"""
    mat = GetAdjacencyMatrix(mol, useBO=use_bond_order)
    if self_connection is True:
        mat = mat + np.eye(mol.GetNumAtoms(), mol.GetNumAtoms())
    return mat


def process_single_smiles(smile, n_node_features=100):
    """Process a single SMILES string to node and adjacency matrix"""
    try:
        mol = Chem.MolFromSmiles(smile, sanitize=False)
        if mol is None:
            return None, None
        mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)

        n_atoms = mol.GetNumAtoms()

        node = np.zeros((n_atoms, n_node_features))
        for atom in mol.GetAtoms():
            node[atom.GetIdx(), atom.GetAtomicNum() - 1] = electronegativity(
                atom.GetSymbol(), nan_swap=0.0
            )

        adjacency_matrix = get_adjacency_matrix(
            mol, use_bond_order=True, self_connection=True
        )

        return node, adjacency_matrix
    except Exception as e:
        print(f"Error processing SMILES '{smile}': {e}")
        return None, None


def pad_graph(node, adj, max_atoms):
    """Pad node features and adjacency matrix to max_atoms size"""
    n_atoms = node.shape[0]

    if n_atoms >= max_atoms:
        return node[:max_atoms], adj[:max_atoms, :max_atoms]

    node_padded = np.zeros((max_atoms, node.shape[1]), dtype=np.float32)
    node_padded[:n_atoms] = node

    adj_padded = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    adj_padded[:n_atoms, :n_atoms] = adj

    return node_padded, adj_padded


def process_formulation_row(
    row_data,
    smiles_cols,
    conc_cols,
    loading_cols,
    property_cols,
    n_node_features=100,
    max_atoms=200,
):
    """Process a single formulation row (all 6 molecules) with padding"""
    idx, row = row_data

    smiles = [str(row[col]).strip() for col in smiles_cols]

    graphs = []
    for smile in smiles:
        node, adj = process_single_smiles(smile, n_node_features)
        if node is None:
            return None

        node_padded, adj_padded = pad_graph(node, adj, max_atoms)
        graphs.append((node_padded, adj_padded))

    concentrations = row[conc_cols].astype(float).to_numpy() / 100.0
    loading = row[loading_cols].astype(float).to_numpy()
    properties = row[property_cols].astype(float).to_numpy()

    return {
        "idx": idx,
        "graphs": graphs,
        "concentrations": concentrations,
        "loading": loading,
        "properties": properties,
    }


def preprocess_dataset(csv_file, output_file, n_processes=None):
    """Preprocess an entire dataset using multiprocessing"""
    print(f"\n{'=' * 60}")
    print(f"Preprocessing: {csv_file}")
    print(f"{'=' * 60}")

    data = pd.read_csv(csv_file)
    data.columns = [col.strip() for col in data.columns]

    smiles_cols = ["smiles1", "smiles2", "smiles3", "smiles4", "smiles5", "smiles6"]
    conc_cols = ["conc1", "conc2", "conc3", "conc4", "conc5", "conc6"]
    loading_cols = ["temperature ( C )"]
    property_cols = ["log(cond) (uS/cm)"]

    print(f"Total formulations: {len(data)}")

    max_atoms = 70
    print(f"Padding all molecules to {max_atoms} atoms")

    process_func = partial(
        process_formulation_row,
        smiles_cols=smiles_cols,
        conc_cols=conc_cols,
        loading_cols=loading_cols,
        property_cols=property_cols,
        n_node_features=100,
        max_atoms=max_atoms,
    )

    if n_processes is None:
        n_processes = cpu_count()

    print(f"Using {n_processes} CPU cores")

    with Pool(n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_func, data.iterrows()),
                total=len(data),
                desc="Processing formulations",
            )
        )

    results = [r for r in results if r is not None]
    print(f"Successfully processed: {len(results)}/{len(data)} formulations")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(results, f, protocol=4)

    print(f"Saved to: {output_file}")

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    return results


def main():
    script_dir = Path(__file__).parent
    splits_dir = script_dir / f"splits-{DATASET}"

    preprocessed_dir = script_dir / f"preprocessed-{DATASET}"
    train_csv = splits_dir / "train_data.csv"
    val_csv = splits_dir / "validation_data.csv"
    test_csv = splits_dir / "test_data.csv"

    train_output = preprocessed_dir / "train_data_preprocessed.pkl"
    val_output = preprocessed_dir / "validation_data_preprocessed.pkl"
    test_output = preprocessed_dir / "test_data_preprocessed.pkl"

    train_results = preprocess_dataset(train_csv, train_output)

    val_results = preprocess_dataset(val_csv, val_output)

    test_results = preprocess_dataset(test_csv, test_output)

    print(f"\n{'=' * 60}")
    print("Preprocessing complete!")
    print(f"Training samples: {len(train_results)}")
    print(f"Validation samples: {len(val_results)}")
    print(f"Test samples: {len(test_results)}")
    print(f"Preprocessed data saved to: {preprocessed_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
