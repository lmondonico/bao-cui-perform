"""Figure S1: CALiSol parity for the models trained on LIC and on IC."""
from common import FONT, PREDICTIONS, corner, legend, parity, plt, save, style, summary


def ic_lic_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.4))
    for ax, letter, target in zip(axes, 'ab', ('LIC', 'IC')):
        parity(ax, PREDICTIONS[('calisol', 'perform', target, 1)])
        ax.set_title(f'Trained on {target}', y=.93)
        ax.text(-.16, 1.03, f'({letter})', transform=ax.transAxes, fontsize=FONT + 4, fontweight='bold', va='bottom')
        corner(ax, 'RMSE: {}\nR²: {}'.format(*summary('calisol', 'perform', target)))
        style(ax, labels=True)
    legend(fig)
    save(fig, 'parity_comparison_LIC_IC')


if __name__ == '__main__':
    ic_lic_comparison()
