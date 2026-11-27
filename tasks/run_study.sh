#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/experiments/results/cache/matplotlib"
python_bin="$PWD/.venv-perform/bin/python"
mode="${1:-run}"
if [[ "$mode" == "--status" ]]; then
    exec "$python_bin" -s -B - <<'PY'
import collections
import json
from pathlib import Path
output = Path('experiments/results')
plan = json.loads(Path('config/study.json').read_text())
print('Training:', 'enabled' if plan.get('training_enabled') is True else 'stopped')
print('Planned fits:', len(plan['jobs']))
ledger = output / 'job_ledger.json'
state = json.loads(ledger.read_text()) if ledger.exists() else {'jobs': {}}
print('Official study jobs:', dict(collections.Counter(j['status'] for j in state['jobs'].values())))
print('Official outputs:', output)
PY
fi
if [[ "$mode" != "--prepare" && "$mode" != "--check" ]]; then
    "$python_bin" -s -B -c 'from core.study import require_training_enabled; require_training_enabled()'
fi
if [[ "$mode" == "--prepare" ]]; then
    "$python_bin" -s -B data/conversion_scripts/prepare_data.py
    "$python_bin" -s -B data/conversion_scripts/validate_data.py
    "$python_bin" -s -B data/conversion_scripts/prepare_splits.py freeze
    "$python_bin" -s -B data/conversion_scripts/check_compatibility.py --data-root data --output experiments/results/compatibility
    "$python_bin" -s -B data/conversion_scripts/prepare_runtime.py
    "$python_bin" -s -B data/conversion_scripts/prepare_inputs.py --all
fi
if [[ -d data/raw ]]; then
    "$python_bin" -s -B data/conversion_scripts/validate_data.py
fi
"$python_bin" -s -B data/conversion_scripts/prepare_splits.py verify
"$python_bin" -s -B data/conversion_scripts/prepare_inputs.py --all --verify
if [[ "$mode" != "--prepare" ]]; then
    "$python_bin" -s -B data/conversion_scripts/check_compatibility.py --data-root data --output experiments/results/compatibility
fi
"$python_bin" -s -B data/conversion_scripts/prepare_runtime.py --verify
"$python_bin" -s -B experiments/metrics.py --prepared-inputs
if [[ "$mode" == "--check" || "$mode" == "--prepare" ]]; then
    exit 0
fi
export PERFORM_ROOT="$PWD"
export XDG_CACHE_HOME="$PWD/experiments/results/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
exec "$python_bin" -s -u tasks/training/train_kfold.py "$@"
