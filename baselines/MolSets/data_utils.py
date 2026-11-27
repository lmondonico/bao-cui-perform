"""Prepare MolSets data with temperature inputs."""

import pickle
from torch_geometric.data import Dataset, Batch
import numpy as np


class GraphSetDataset(Dataset):
    """
    Dataset for graph sets with temperature support.
    Handles pre-split pickle files with temperature field.
    """
    def __init__(self, data_path, train_stats=None):
        """
        Initialize dataset from pickle file.

        Args:
            data_path: Path to pickle file
            train_stats: Optional dict with {'target_mean', 'target_std'} from training data.
                         If None, computes stats from this file (use for training set).
        """
        super().__init__()
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)

        targets = np.array([self.data_list[i][6] for i in range(len(self.data_list))])

        if train_stats is None:
            self.target_mean = np.mean(targets)
            self.target_std = np.std(targets)
        else:
            self.target_mean = train_stats['target_mean']
            self.target_std = train_stats['target_std']

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            Tuple: (index, graph_list, mw, fraction, salt_mol, salt_graph, normalized_target, temperature)
        """
        sample = self.data_list[idx]
        index, graph_list, mw, fraction, salt_mol, salt_graph, target, temperature = sample

        normalized_target = float((target - self.target_mean) / self.target_std)

        return (index, graph_list, mw, fraction, salt_mol, salt_graph, normalized_target, temperature)

    def get_orig(self, target):
        """
        Denormalize target back to original scale.

        Args:
            target: Normalized target value(s)

        Returns:
            Original target value(s)
        """
        return target * self.target_std + self.target_mean

    def get_stats(self):
        """
        Get normalization statistics.

        Returns:
            Dict with target_mean and target_std
        """
        return {
            'target_mean': self.target_mean,
            'target_std': self.target_std
        }


def graph_set_collate(batch):
    """
    Collate function for batching graph sets with temperature.

    Args:
        batch: List of samples from GraphSetDataset

    Returns:
        Tuple of batched data with temperature
    """
    indices, graph_lists, mws, weight_lists, mol_lists, salt_graphs, ys, temps = zip(*batch)

    batched_graph_sets = [Batch.from_data_list(g_list) for g_list in graph_lists]

    return batched_graph_sets, mws, weight_lists, mol_lists, salt_graphs, ys, temps
