"""Figures S4-S10: parity of every trained model, one figure per dataset in DATASETS order."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATASETS, FONT, PREDICTIONS, corner, legend, parity, plt, save, style, summary

MODELS = {'perform': 'PerForm', 'xgboost': 'XGBoost', 'molformer': 'MM-MoLFormer', 'attentivefp': 'AttentiveFP',
          'smited': 'SMI-TED-IC', 'molsets': 'MolSets-SAGE', 'fgcn': 'F-GCN', 'geomix': 'GeoMix-TFN'}


def baseline_comparison(dataset):
    target = 'LIC' if dataset == 'calisol' else 'Target'
    models = [m for m in MODELS if (dataset, m, target, 1) in PREDICTIONS]
    rows = -(-len(models) // 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 6.2 * rows), squeeze=False)
    for ax in axes.flat[len(models):]:
        ax.set_axis_off()
    for ax, model in zip(axes.flat, models):
        parity(ax, PREDICTIONS[(dataset, model, target, 1)])
        ax.text(.03, .97, MODELS[model], transform=ax.transAxes, fontsize=FONT + 1, fontweight='bold', va='top')
        corner(ax, 'RMSE: {}\nR²: {}'.format(*summary(dataset, model, target)))
    style(axes, labels=True)
    for ax in axes.flat:
        ax.tick_params(axis='x', pad=9)
    fig.suptitle(DATASETS[dataset][0], fontsize=FONT + 8, fontweight='bold', y=1.005)
    legend(fig)
    save(fig, f'baselines-plots/baseline_comparison_{dataset}')


if __name__ == '__main__':
    for dataset in DATASETS:
        baseline_comparison(dataset)
