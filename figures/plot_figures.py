"""Run every figure script in figures/ (Figure 1 is figures/framework/perform_framework.tex)."""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SCRIPTS = ['plot_perform_parity.py', 'plot_cv_folds.py', 'plot_ic_lic_comparison.py', 'plot_ablation_comparison.py',
           'temperature_trend/plot_temperature_parity.py', 'attention-study/plot_moiety_attention.py',
           'baselines-plots/plot_baseline_comparison.py']

if __name__ == '__main__':
    for script in SCRIPTS:
        runpy.run_path(str(HERE / script), run_name='__main__')
