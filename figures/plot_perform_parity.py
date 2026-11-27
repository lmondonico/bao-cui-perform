"""Figure 2: PerForm test parity with KDE marginals; also a 2 x 4 variants figure that is not in the paper."""
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from common import DATASETS, FONT, PREDICTIONS, plt, save, style, summary

BENCHMARKS = ['calisol', 'ilthermo', 'des', 'chew_density', 'chew_enthalpy']
VARIANTS = [('calisol', 'LIC', 'CALiSol\n(LIC target)'), ('calisol', 'IC', 'CALiSol\n(IC target)'),
            ('ilthermo', 'Target', 'ILThermo\n(full dataset)'), ('ilthermo_common', 'Target', 'ILThermo\n(common subset)'),
            ('des', 'Target', 'DES\n(full dataset)'), ('des_common', 'Target', 'DES\n(common subset)'),
            ('chew_density', 'Target', 'CheW\ndensity'), ('chew_enthalpy', 'Target', 'CheW enthalpy\nof mixing')]


def kde_panel(fig, cell, dataset, target, label):
    test = PREDICTIONS[(dataset, 'perform', target, 1)].query("split == 'Test'")
    truth, pred = test.truth.to_numpy(), test.prediction.to_numpy()
    color = DATASETS[dataset][1]
    inner = cell.subgridspec(2, 2, width_ratios=[6, 1], height_ratios=[1, 6], wspace=.04, hspace=.04)
    ax = fig.add_subplot(inner[1, 0])
    top, right = fig.add_subplot(inner[0, 0], sharex=ax), fig.add_subplot(inner[1, 1], sharey=ax)
    ax.scatter(truth, pred, s=8, c=color, alpha=.35, edgecolors='none', rasterized=True)
    low, high = min(truth.min(), pred.min()), max(truth.max(), pred.max())
    lims = (low - .05 * (high - low), high + .05 * (high - low))
    ax.plot(lims, lims, 'k--', lw=1.1, alpha=.7)
    ax.set(xlim=lims, ylim=lims)
    style(ax, labels=True)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 1.5, 2, 2.5, 3, 4, 5, 10]))
    x, y = (truth - lims[0]) / (lims[1] - lims[0]), (pred - lims[0]) / (lims[1] - lims[0])
    corners = [(.04, .96, 'left', 'top'), (.96, .04, 'right', 'bottom')]
    if np.sum((x < .75) & (y > .75)) > np.sum((x > .25) & (y < .25)):
        corners.reverse()
    metrics = 'RMSE = {}\nR² = {}'.format(*summary(dataset, 'perform', target))
    for (px, py, ha, va), text, size, tint in zip(corners, (metrics, label), (FONT - 4, FONT), ('black', color)):
        ax.text(px, py, text, transform=ax.transAxes, fontsize=size, fontweight='bold', color=tint, ha=ha, va=va)
    for side, values in ((top, truth), (right, pred)):
        grid = np.linspace(*lims, 400)
        density = gaussian_kde(values)(grid)
        fill, curve = (side.fill_between, (grid, density)) if side is top else (side.fill_betweenx, (density, grid))
        fill(grid, density, color=color, alpha=.55, lw=0)
        side.plot(*curve, color=color, lw=1.4)
        side.axis('off')


def perform_parity():
    fig = plt.figure(figsize=(22, 14))
    grid = fig.add_gridspec(2, 6, left=.06, right=.99, top=.98, bottom=.07, wspace=.8, hspace=.17)
    panels = [variant for variant in VARIANTS if variant[0] in BENCHMARKS and variant[1] != 'IC']
    for cell, variant in zip([grid[0, 0:2], grid[0, 2:4], grid[0, 4:6], grid[1, 1:3], grid[1, 3:5]], panels):
        kde_panel(fig, cell, *variant)
    save(fig, 'parity_plot_perform_mosaic', layout=False)


def perform_variants():
    fig = plt.figure(figsize=(28, 14))
    grid = fig.add_gridspec(2, 4, left=.05, right=.99, top=.98, bottom=.07, wspace=.31, hspace=.17)
    for cell, (dataset, target, label) in zip(grid, VARIANTS):
        kde_panel(fig, cell, dataset, target, label)
    save(fig, 'parity_plot_perform_variants', layout=False)


if __name__ == '__main__':
    perform_parity()
    perform_variants()
