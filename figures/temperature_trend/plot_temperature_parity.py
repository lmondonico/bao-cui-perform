"""Figure 3: CALiSol parity colored by temperature, for the models trained on LIC and on IC."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import FONT, PREDICTIONS, plt, save, style


def temperature_parity():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    for ax, target in zip(axes, ('LIC', 'IC')):
        frame = PREDICTIONS[('calisol', 'perform', target, 1)]
        scatter = ax.scatter(frame.truth, frame.prediction, c=frame.temperature, cmap='coolwarm', vmin=-40, vmax=110,
                             alpha=.5, s=16, edgecolors='none')
        low, high = frame[['truth', 'prediction']].min().min(), frame[['truth', 'prediction']].max().max()
        ax.plot([low, high], [low, high], 'k--', lw=1.5, alpha=.8)
        ax.set_title(f'Trained on {target}')
        style(ax, labels=True)
    fig.tight_layout()
    position = axes[1].get_position()
    bar = fig.colorbar(scatter, cax=fig.add_axes([1, position.y0, .015, position.height]), extend='both')
    bar.set_label('Temperature (°C)', fontsize=FONT, fontweight='bold')
    bar.ax.tick_params(labelsize=FONT - 3)
    save(fig, 'temperature_trend/temperature_parity_comparison', layout=False)


if __name__ == '__main__':
    temperature_parity()
