import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pickle

densities = {'EC': 1.3210,
       'PC' : 1.205, 'DMC' : 1.07, 'EMC' : 0.902, 'DEC' : 0.975,
        'DME' : 0.86, 'DMSO' : 1.1004, 'AN': 0.786, 'MOEMC': 1.5, 'TFP': 1.487, 'EA' : 0.902,
       'MA': 0.932, 'FEC': 1.454, 'DOL': 1.06, '2-MeTHF' : 0.854, 'DMM': 0.902, 'Freon 11': 1.49, 'Methylene chloride': 1.33,
       'THF': 0.888, 'Toluene' : 0.867, 'Sulfolane' : 1.26, '2-Glyme': 0.937, '3-Glyme': 0.986, '4-Glyme': 1.009,
       '3-Me-2-Oxazolidinone': 1.17, '3-MeSulfolane': 1.20, 'Ethyldiglyme' : 0.937, 'DMF': 0.944,
       'Ethylbenzene': 0.866, 'Ethylmonoglyme': 0.965, 'Benzene': 0.876, 'g-Butyrolactone' : 1.13,
       'Cumene': 0.862, 'Propylsulfone' : 1.109, 'Pseudocumeme': 0.876, 'TEOS': 0.940, 'm-Xylene' : 0.860,
       'o-Xylene': 0.87596}

element_map = {
    1: 0,
    3: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    14: 7,
    15: 8,
    16: 9,
    17: 10,
    33: 11,
}

salt_list = [
    'LiPF6', 
    'LiBF4', 
    'LiFSI', 
    'LiTDI', 
    'LiPDI', 
    'LiTFSI', 
    'LiClO4', 
    'LiAsF6', 
    'LiBOB', 
    'LiCF3SO3', 
    'LiBPFPB', 
    'LiBMB', 
    'LiN(CF3SO2)2'
]

class ConductivityDataset(Dataset):
    """
    ConductivityDataset: A PyG dataset for loading and processing conductivity data from CALiSol23.
    """
    def __init__(self, dataset_path, npz_path, salt_npz_path, partition='all', max_samples=1e8, device='cpu'):
        super(ConductivityDataset, self).__init__()
        self.dataset_path = dataset_path
        self.npz_path = npz_path
        self.salt_npz_path = salt_npz_path
        self.partition = partition
        self.max_samples = int(max_samples)
        self.device = device
        self.dataset = self.load_dataset()
        self.solvent_data, self.salt_data = self.load_npz()

        print("Processing data...")
        dataset = dataset_path.replace("/", "_").replace(".", "_")
        os.makedirs("./data_processed/", exist_ok=True)
        dataset_process_path = "./data_processed/" + dataset + ".pkl"
        if os.path.exists(dataset_process_path):
            with open(dataset_process_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = self.process(self.dataset, self.solvent_data, self.salt_data)
            with open(dataset_process_path, 'wb') as f:
                pickle.dump(self.data, f)
        print(f"{partition} dataset total length: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_dataset(self):
        return pd.read_csv(self.dataset_path).head(self.max_samples)

    def load_npz(self):
        return np.load(self.npz_path, allow_pickle=True), np.load(self.salt_npz_path, allow_pickle=True)

    def _density_process(self, row):
        if row['c units'] == 'mol/kg':
            row['c'] = row['c']
        return row

    def process(self, dataset, solvent_data, salt_data):
        data = []

        unique_salts = dataset['salt'].unique()
        salt_to_idx = {salt: idx for idx, salt in enumerate(unique_salts)}

        for i in tqdm(range(min(len(self.dataset), self.max_samples))):
            row = dataset.iloc[i].copy()
            row = self._density_process(row)

            if pd.isna(row['k']):
                continue

            salt_one_hot = [0] * len(unique_salts)
            salt_one_hot[salt_to_idx[row['salt']]] = 1

            graph_data, salt_graph_data, ratio_data = self.get_graph(row, solvent_data, salt_data)
            graphs = [graph.to(self.device) for graph in graph_data]
            salt_graphs = [graph.to(self.device) for graph in salt_graph_data]
            ratios = torch.tensor(ratio_data, dtype=torch.float32)

            processed_item = {
                'k': row['k'], 
                'T': row['T'],
                'c': row['c'], 
                'salt': row['salt'], 
                'salt_graphs': salt_graphs, 
                'graphs': graphs,
                'ratios': ratios
            }
            data.append(processed_item)
        return data

    def convert_weight_ratio(self, solvent_names, solvent_ratios):
        densities_tensor = torch.tensor(list(densities.values()))
        sol_tensor = torch.tensor(solvent_ratios)

        sum_weight = torch.sum(torch.tensor(solvent_ratios) * densities_tensor)
        if sum_weight == 0.0:
            return sol_tensor
        return sol_tensor / sum_weight

    def convert_c(self, c, solvent_ratios):
        denom = sum(torch.tensor(solvent_ratios) / torch.tensor(list(densities.values())))
        if denom == 0.0:
            return c
        return c / denom

    def get_graph(self, row, solvent_data, salt_data):
        """
        Create a PyG Data object list based on a single row of data.
        """
        solvent_ratio_cols = self.dataset.columns[self.dataset.columns.get_loc('solvent ratio type') + 1:]
        solvent_ratios = row[solvent_ratio_cols].values.astype(float)
        solvent_names = solvent_data['name_list']
        salt_names = salt_data['name_list']

        graph_list = []
        ratio_list = []

        if row['solvent ratio type'] == 'v':
            solvent_ratios = self.convert_weight_ratio(solvent_names, solvent_ratios)

        if row['c units'] == 'mol/kg':
            row['c'] = self.convert_c(row['c'], solvent_ratios)

        for solvent_name, ratio in zip(solvent_names, solvent_ratios):
            if ratio > 0:
                idx = list(solvent_data['name_list']).index(solvent_name)

                coords = solvent_data['coordinates'][idx]
                atom_numbers = solvent_data['atomic_numbers'][idx]
                edge_index = solvent_data['edge_index'][idx]
                edge_attr = solvent_data['edge_attr'][idx]

                node_features = []
                for atom in atom_numbers:
                    atom_one_hot = [0] * len(element_map)
                    atom_one_hot[element_map[atom.item()]] = 1
                    node_features.append([atom.item() / 10, ratio] + atom_one_hot)

                node_features = torch.tensor(node_features, dtype=torch.float32).to(self.device)
                positions = torch.tensor(coords, dtype=torch.float32).to(self.device)
                edge_index = [edge_index[0], edge_index[1]]
                edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(self.device)
                
                graph = Data(
                    x=node_features,
                    pos=positions,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    name=solvent_name
                )
                graph_list.append(graph)
                ratio_list.append(ratio)

        salt_graph_list = []
        for salt_name in salt_names:
            if salt_name == row['salt']:
                idx = list(salt_data['name_list']).index(salt_name)

                coords = salt_data['coordinates'][idx]
                atom_numbers = salt_data['atomic_numbers'][idx]
                edge_index = salt_data['edge_index'][idx]
                edge_attr = salt_data['edge_attr'][idx]
                
                node_features = []
                for atom in atom_numbers:
                    atom_one_hot = [0] * len(element_map)
                    atom_one_hot[element_map[atom.item()]] = 1
                    node_features.append([atom.item()/10, row['c']] + atom_one_hot)

                node_features = torch.tensor(node_features, dtype=torch.float32).to(self.device)
                positions = torch.tensor(coords, dtype=torch.float32).to(self.device)
                edge_index = [edge_index[0], edge_index[1]]
                edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(self.device)

                graph = Data(
                    x=node_features,
                    pos=positions,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    name=salt_name
                )

                salt_graph_list.append(graph)

        return graph_list, salt_graph_list, ratio_list


def custom_collate_fn(batch, device):
    k_values = torch.tensor([item['k'] for item in batch], dtype=torch.float32, device=device)
    T_values = torch.tensor([item['T'] for item in batch], dtype=torch.float32, device=device)
    c_values = torch.tensor([item['c'] for item in batch], dtype=torch.float32, device=device)

    salt_one_hot = torch.zeros(len(batch), len(salt_list), dtype=torch.float32, device=device)
    for idx, item in enumerate(batch):
        salt_idx = salt_list.index(item['salt'])
        salt_one_hot[idx, salt_idx] = 1

    all_graphs = []
    all_salt_graphs = []
    g2b_indices_solv = []
    g2b_indices_salt = []
    batch_ratios = torch.empty((0,))

    for sample_idx, item in enumerate(batch):
        for graph in item['graphs']:
            all_graphs.append(graph)
            g2b_indices_solv.append(sample_idx)
        batch_ratios = torch.cat([batch_ratios, item['ratios']], dim=-1)
        for graph in item['salt_graphs']:
            all_salt_graphs.append(graph)
            g2b_indices_salt.append(sample_idx)

    batch_graphs = Batch.from_data_list(all_graphs).to(device)
    batch_salt_graphs = Batch.from_data_list(all_salt_graphs).to(device)

    n2g_indices_solv = torch.arange(len(batch_graphs.ptr) - 1, device=device).repeat_interleave((batch_graphs.ptr[1:] - batch_graphs.ptr[:-1]))
    g2b_indices_solv = torch.tensor(g2b_indices_solv, dtype=torch.long, device=device).clone().detach()
    n2b_indices_solv = g2b_indices_solv[n2g_indices_solv]
    n2g_indices_salt = torch.arange(len(batch_salt_graphs.ptr) - 1, device=device).repeat_interleave((batch_salt_graphs.ptr[1:] - batch_salt_graphs.ptr[:-1]))
    g2b_indices_salt = torch.tensor(g2b_indices_salt, dtype=torch.long, device=device).clone().detach()
    n2b_indices_salt = g2b_indices_salt[n2g_indices_salt]

    batch_graphs.n2g_indices = n2g_indices_solv
    batch_graphs.n2b_indices = n2b_indices_solv
    batch_graphs.g2b_indices = g2b_indices_solv.clone().detach()
    batch_salt_graphs.n2g_indices = n2b_indices_salt
    batch_salt_graphs.n2b_indices = n2b_indices_salt
    batch_salt_graphs.g2b_indices = g2b_indices_salt.clone().detach()

    n2g_indices = torch.cat([n2g_indices_solv, n2g_indices_salt+n2g_indices_solv.max().item()+1], dim=0)
    n2b_indices = torch.cat([n2b_indices_solv, n2b_indices_salt], dim=0)
    g2b_indices = torch.cat([g2b_indices_solv, g2b_indices_salt], dim=0)

    return {
        'k': k_values,
        'T': T_values,
        'c': c_values,
        'salt_one_hot': salt_one_hot,
        'graphs': batch_graphs,
        'salt_graphs': batch_salt_graphs,
        'ratios': batch_ratios,
        'n2g': n2g_indices,
        'n2b': n2b_indices,
        'g2b': g2b_indices
    }


if __name__ == '__main__':
    dataset_pth = './data/CALiSol23_test.csv'
    npz_pth = './data/CoordData/coords-solvent-pubchem.npz'
    salt_npz_path = './data/CoordData/coords-salts.npz'
    dataset = ConductivityDataset(dataset_pth, npz_pth, salt_npz_path, max_samples=200, device="cpu")

    dataloader = DataLoader(dataset, batch_size=30, shuffle=False, collate_fn=custom_collate_fn)
    for batch_idx, batch in enumerate(dataloader):  
        graphs = batch['graphs']
        if graphs.edge_index.shape[1] != graphs.edge_attr.shape[0]:
            print(graphs.edge_index.shape[1])
            print(graphs.edge_attr.shape)