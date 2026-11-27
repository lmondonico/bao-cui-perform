"""Figures S11 and S12: CALiSol parity and loss curves for the full model and the component ablations."""
from common import LOSSES, PREDICTIONS, legend, loss, parity, plt, save, style

ABLATIONS = {'perform': 'Full Model', 'no_motif': 'No Moiety Encoder', 'no_molar_ratio': 'No Molar Ratio',
             'no_attention': 'No Formulation Attention', 'no_temperature': 'No Temperature', 'minimal': 'Minimal Model'}


def component_ablations():
    parity_fig, parity_axes = plt.subplots(3, 2, figsize=(13, 19.5))
    loss_fig, loss_axes = plt.subplots(3, 2, figsize=(13, 19.5))
    for (model, title), pa, la in zip(ABLATIONS.items(), parity_axes.flat, loss_axes.flat):
        parity(pa, PREDICTIONS[('calisol', model, 'LIC', 1)], title)
        loss(la, LOSSES[('calisol', model, 'LIC', 1)], title)
    style(parity_axes, labels=True)
    style(loss_axes)
    legend(parity_fig)
    save(parity_fig, 'ablation_parity_mosaic')
    save(loss_fig, 'ablation_loss_mosaic')


if __name__ == '__main__':
    component_ablations()
