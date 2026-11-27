"""Figures S2 (LIC) and S3 (IC): CALiSol PerForm per fold, parity (left) and loss (right)."""
from common import LOSSES, METRICS, PREDICTIONS, corner, legend, loss, parity, plt, save, style


def cv_folds(target):
    fig, axes = plt.subplots(5, 2, figsize=(13, 28))
    for fold, (pa, la) in enumerate(axes, 1):
        key = ('calisol', 'perform', target, fold)
        parity(pa, PREDICTIONS[key], f'Fold {fold}: parity')
        corner(pa, 'Test RMSE: {:.3f}\nTest R²: {:.3f}'.format(*METRICS.loc[key, ['rmse', 'r2']]))
        style(pa, labels=True)
        loss(la, LOSSES[key], f'Fold {fold}: loss')
        style(la)
    legend(fig)
    save(fig, f'cv_folds_{target}')


if __name__ == '__main__':
    cv_folds('LIC')
    cv_folds('IC')
