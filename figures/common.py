"""Data, style and plotting helpers shared by the figure scripts; data come from figures/data/."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'Arial', 'axes.linewidth': 1.5, 'legend.frameon': False,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'xtick.major.size': 7, 'ytick.major.size': 7, 'xtick.minor.size': 4, 'ytick.minor.size': 4,
                     'xtick.major.width': 1.5, 'ytick.major.width': 1.5, 'xtick.minor.width': 1, 'ytick.minor.width': 1})

HERE = Path(__file__).parent
DATA = HERE / 'data'
FONT = 22
SPLITS = {'Training': 'orange', 'Validation': 'blue', 'Test': 'green'}
DATASETS = {'calisol': ('CALiSol (LIC target)', '#F18F01'),
            'ilthermo': ('ILThermo (full dataset)', '#2E86AB'),
            'des': ('DES (full dataset)', '#A23B72'),
            'chew_density': ('CheW density', '#3B7A57'),
            'chew_enthalpy': ('CheW enthalpy of mixing', '#6A4C93'),
            'ilthermo_common': ('ILThermo (common subset)', '#2E86AB'),
            'des_common': ('DES (common subset)', '#A23B72')}
KEY = ['dataset', 'model', 'target', 'fold']
PREDICTIONS = dict(tuple(pd.read_csv(DATA / 'predictions.csv.gz').groupby(KEY)))
LOSSES = dict(tuple(pd.read_csv(DATA / 'losses.csv').groupby(KEY)))
METRICS = pd.read_csv(DATA / 'metrics.csv').set_index(KEY).sort_index()


def summary(dataset, model, target):
    values = METRICS.loc[(dataset, model, target)]
    return [f'{values[c].mean():.3f} ± {values[c].std():.3f}' for c in ('rmse', 'r2')]


def style(axes, labels=False):
    for ax in np.ravel(axes):
        if labels:
            ax.set(xlabel='Actual', ylabel='Predicted')
        ax.tick_params(labelsize=FONT - 3)
        ax.xaxis.label.set_fontsize(FONT)
        ax.yaxis.label.set_fontsize(FONT)
        ax.title.set_fontsize(FONT + 2)
        ax.title.set_fontweight('bold')


def corner(ax, text):
    ax.text(.97, .03, text, transform=ax.transAxes, fontsize=FONT - 2, fontweight='bold', ha='right', va='bottom')


def legend(fig):
    handles = [Line2D([], [], color=c, marker='o', ls='', markersize=FONT * .7, alpha=.6, label=n) for n, c in SPLITS.items()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 0), ncol=3, frameon=False, fontsize=FONT)


def save(fig, name, layout=True):
    if layout:
        fig.tight_layout(h_pad=1.8, w_pad=1.8)
        fig.tight_layout(h_pad=1.8, w_pad=1.8)
    path = HERE / f'{name}.png'
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def parity(ax, frame, title=''):
    for split, color in SPLITS.items():
        part = frame[frame.split == split]
        ax.scatter(part.truth, part.prediction, color=color, alpha=.6, s=5)
    low, high = frame[['truth', 'prediction']].min().min(), frame[['truth', 'prediction']].max().max()
    ax.plot([low, high], [low, high], 'k--', lw=1.5, alpha=.8)
    ax.set_title(title)


def smooth(values, window=9):
    values, half = np.asarray(values), window // 2
    return [values[i - k:i + k + 1].mean() for i, k in enumerate(min(half, i, len(values) - 1 - i) for i in range(len(values)))]


def loss(ax, history, title):
    for column, (name, color) in zip(('train_loss', 'val_loss'), SPLITS.items()):
        ax.plot(history.epoch, history[column], color=color, alpha=.35, lw=1)
        ax.plot(history.epoch, smooth(history[column]), color=color, lw=2, label=name)
    ax.set(xlabel='Epoch', ylabel='Loss', title=title)
    ax.legend(frameon=False, fontsize=FONT - 2)
