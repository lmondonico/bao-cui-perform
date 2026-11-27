"""Convert supplied split CSVs to MolSets inputs."""

from rdkit import Chem
from rdkit.Chem import Descriptors
from pymatgen.core.periodic_table import Element
from torch_geometric.data import Data
import torch
import pandas as pd
import pickle


def smiles_to_graph(smiles, encoding='other'):
    """
    Convert SMILES string to PyTorch Geometric Data object.
    Adapted from data/prepare_data.py

    Args:
        smiles: SMILES string
        encoding: Feature encoding type ('onehot', 'atomic', or 'other')

    Returns:
        PyG Data object with molecular graph
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    num_nodes = mol.GetNumAtoms()

    node_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in ['Au', 'Cu']:
            atom.SetAtomicNum(6)
            symbol = 'C'
        element = Element(symbol)

        if encoding == 'onehot':
            node_feature = [int(symbol == x) for x in ['C', 'N', 'O', 'S', 'F', 'Cl']]
            node_feature += [atom.GetTotalNumHs(), atom.GetFormalCharge(), atom.GetMass()]
            node_features.append(node_feature)
        elif encoding == 'atomic':
            node_feature = [atom.GetAtomicNum(), atom.GetMass(), atom.GetFormalCharge(), atom.GetTotalNumHs()]
            node_feature += [element.X, element.van_der_waals_radius]
            node_features.append(node_feature)
        else:
            node_feature = [int(symbol == x) for x in ['C', 'N', 'O', 'B', 'S', 'F', 'Cl']]
            node_feature += [atom.GetAtomicNum(), atom.GetMass(), atom.GetFormalCharge(), atom.GetTotalNumHs()]
            node_feature += [element.X, element.van_der_waals_radius]
            node_features.append(node_feature)

    node_features = torch.tensor(node_features, dtype=torch.float)

    edge_index = []
    edge_weight = []
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        edge_index.append([start_idx, end_idx])
        edge_index.append([end_idx, start_idx])
        edge_weight.append(bond.GetBondTypeAsDouble())
        edge_weight.append(bond.GetBondTypeAsDouble())

    edge_index = torch.tensor(edge_index, dtype=torch.long).reshape(-1, 2).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    edge_index_loop = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
    edge_weight_loop = torch.ones(num_nodes, dtype=torch.float)
    edge_index = torch.cat([edge_index, edge_index_loop], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight_loop])

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)

    return data


def parse_component(component_str):
    """
    Parse component string in format 'SMILES:concentration'.

    Args:
        component_str: String like 'CCOC(=O)OC:5.763' or empty

    Returns:
        Tuple of (smiles, concentration) or None if empty
    """
    if pd.isna(component_str) or component_str == '':
        return None

    parts = component_str.split(':')
    if len(parts) != 2:
        return None

    smiles = parts[0].strip()
    try:
        concentration = float(parts[1].strip())
    except ValueError:
        return None

    return (smiles, concentration)


def process_row(row, index):
    """
    Process a single CSV row and convert to MolSets format.

    Args:
        row: Pandas DataFrame row
        index: Row index

    Returns:
        Tuple: (index, graph_list, mw_list, fraction_list, salt_mol, salt_graph, target, temperature)
    """
    solvents = []
    solvent_cols = ['Solvent_1', 'Solvent_2', 'Solvent_3', 'Solvent_4', 'Solvent_5']
    for col in solvent_cols:
        comp = parse_component(row[col])
        if comp is not None:
            solvents.append(comp)

    salts = []
    salt_cols = ['Salt_1', 'Salt_2', 'Salt_3']
    for col in salt_cols:
        comp = parse_component(row[col])
        if comp is not None:
            salts.append(comp)

    if len(solvents) == 0:
        raise ValueError(f"Row {index}: No solvents found")
    if len(salts) == 0:
        raise ValueError(f"Row {index}: No salts found")

    salt_smiles, salt_concentration = salts[0]

    solvent_mws = []
    for smiles, _ in solvents:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid solvent SMILES: {smiles}")
        mw = Descriptors.MolWt(mol)
        solvent_mws.append(mw)

    salt_mol_obj = Chem.MolFromSmiles(salt_smiles)
    if salt_mol_obj is None:
        raise ValueError(f"Invalid salt SMILES: {salt_smiles}")
    salt_mw = Descriptors.MolWt(salt_mol_obj)

    solvent_masses = [conc * mw for (_, conc), mw in zip(solvents, solvent_mws)]
    salt_mass = salt_concentration * salt_mw

    total_solvent_mass = sum(solvent_masses)
    weight_fractions = [mass / total_solvent_mass for mass in solvent_masses]

    solvent_mass_kg = total_solvent_mass / 1000.0
    salt_molality = salt_concentration / solvent_mass_kg

    graph_list = []
    for smiles, _ in solvents:
        graph = smiles_to_graph(smiles)
        graph_list.append(graph)

    salt_graph = smiles_to_graph(salt_smiles)

    target = float(row['LIC'])

    temp_celsius = float(row['Temperature'])
    temp_kelvin = temp_celsius + 273.15

    return (index, graph_list, solvent_mws, weight_fractions, salt_molality, salt_graph, target, temp_kelvin)


def prepare_split_data(csv_file, output_file):
    """
    Process a single CSV split file and save as pickle.

    Args:
        csv_file: Path to input CSV file
        output_file: Path to output pickle file
    """
    print(f"Processing {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} samples")

    data_list = []
    failed = 0

    for idx, row in df.iterrows():
        try:
            sample = process_row(row, idx)
            data_list.append(sample)
        except Exception as e:
            print(f"  Warning: Failed to process row {idx}: {e}")
            failed += 1
            continue

    print(f"  Successfully processed {len(data_list)} samples ({failed} failed)")

    with open(output_file, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"  Saved to {output_file}")

    return data_list
