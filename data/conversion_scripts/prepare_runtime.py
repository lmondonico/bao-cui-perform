"""Build and check the PerForm molecular caches for every dataset PerForm can use."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.study import DATA, OUTPUT, cache_path, load_data


def verify_molecules(source, output):
    cache = np.load(output, allow_pickle=True).item()
    frame = pd.read_csv(source, low_memory=False)
    columns = [f'{role}_{i}' for role, count in (('Solvent', 5), ('Salt', 3)) for i in range(1, count + 1)]
    smiles = {str(value).rsplit(':', 1)[0] for column in columns for value in frame[column].dropna()}
    if set(cache['molecules']) != smiles:
        raise RuntimeError(f'Molecular cache membership differs: {source}')
    for smiles, item in cache['molecules'].items():
        mol = Chem.MolFromSmiles(smiles)
        ids = item['moiety_ids']
        covered = {atom for group in item['clique_atom_ids'] for atom in group}
        if mol is None or not ids or any(a.GetAtomicNum() > 1 and a.GetIdx() not in covered for a in mol.GetAtoms()):
            raise RuntimeError(f'Incomplete molecular representation: {smiles}')
        if min(ids) < 1 or max(ids) >= cache['vocab_size']:
            raise RuntimeError(f'Invalid molecular token IDs: {smiles}')
        for key in ('adj_matrix', 'dist_matrix'):
            values = np.asarray(item[key])
            if values.shape != (len(ids), len(ids)) or not np.isfinite(values).all():
                raise RuntimeError(f'Invalid {key}: {smiles}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Check existing caches without rebuilding them')
    args = parser.parse_args()
    sources = load_data()['datasets']
    with (OUTPUT / 'compatibility/compatibility_matrix.csv').open() as stream:
        names = [row['dataset'] for row in csv.DictReader(stream) if row['model'] == 'perform' and row['status'] != 'blocked']
    for name in names:
        source, output = DATA / sources[name]['path'], cache_path(name)
        if not args.verify:
            from core.smiles_preprocessing import smiles_dict_preprocessing
            smiles_dict_preprocessing(str(source), 'conductivity', output_path=str(output))
        verify_molecules(source, output)
    print(f'Verified {len(names)} PerForm molecular caches')


if __name__ == '__main__':
    main()
