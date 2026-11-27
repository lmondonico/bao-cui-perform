"""Figure S13: TEOS, F5DEE and LiTFSI above their local and global attention maps (final MOTIF layer, head mean)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import ImageOps
from rdkit import Chem
from rdkit.Chem import Draw

from common import DATA, FONT, plt, save

ATTENTION = {'local': ['#FEFBD8', '#F7F7B8DB', '#F7D98E', '#F2C66D', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#B10026'],
             'global': ['#FFFFFFD8', '#C0D3E8', '#9FBAD6', '#6F9EC5', '#6BAED6', '#4292C6', '#2171B5', '#08519C', '#08306B']}


def attention():
    molecules = json.loads((DATA / 'attention.json').read_text())
    fig = plt.figure(figsize=(19, 31), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[.8, 4.6])
    maps = bottom.subplots(len(molecules), 2)
    for ax, letter, mol in zip(top.subplots(1, len(molecules)), 'abc', molecules):
        image = Draw.MolToImage(Chem.MolFromSmiles(mol['smiles']), size=(900, 900), kekulize=True)
        ax.imshow(image.crop(ImageOps.invert(image.convert('L')).getbbox()))
        ax.set_adjustable('datalim')
        ax.axis('off')
        ax.set_title(f'({letter}) {mol["name"]}', fontsize=FONT + 6, fontweight='bold')
    for column, (key, colors) in zip(maps.T, ATTENTION.items()):
        cmap = LinearSegmentedColormap.from_list(key, colors)
        vmax = max(np.max(mol[f'{key}_attention']) for mol in molecules)
        for ax, mol in zip(column, molecules):
            labels = range(len(mol['moiety_names']))
            ax.set_xticks(labels, mol['moiety_names'], rotation=45, ha='right', rotation_mode='anchor', fontsize=FONT - 4)
            ax.set_yticks(labels, mol['moiety_names'], fontsize=FONT - 4)
            ax.minorticks_off()
            ax.tick_params(length=0)
            image = ax.imshow(np.ma.masked_equal(mol[f'{key}_attention'], 0), cmap=cmap, vmin=0, vmax=vmax)
        column[0].set_title(f'{key.capitalize()} attention', fontsize=FONT + 4, fontweight='bold', pad=14)
        bar = bottom.colorbar(image, ax=column, location='bottom', shrink=.8, pad=.02)
        bar.set_label('Attention weight', fontsize=FONT)
        bar.ax.tick_params(labelsize=FONT - 4)
    for ax, letter, mol in zip(maps[:, 0], 'abc', molecules):
        ax.set_ylabel(f'({letter}) {mol["name"]}', fontsize=FONT + 4, fontweight='bold', labelpad=34)
    bottom.align_ylabels(maps[:, 0])
    save(fig, 'attention-study/moiety_attention', layout=False)


if __name__ == '__main__':
    attention()
