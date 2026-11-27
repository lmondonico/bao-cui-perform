"""Paths, plan access and run guards for the public study."""
import fcntl
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUTPUT = ROOT / 'experiments/results'
PLAN = ROOT / 'config/study.json'


def load_data():
    return json.loads(PLAN.read_text())['data']


def load_protocols():
    return json.loads(PLAN.read_text())['protocols']



def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def local_path(path, parent=ROOT):
    path = Path(path).resolve()
    if not path.is_relative_to(parent.resolve()) or '_archive' in path.parts:
        raise ValueError(f'Path outside active study: {path}')
    return path


def cache_path(dataset):
    return DATA / f"splits-{dataset.replace('_', '-')}" / 'preprocessed_mol_info.npy'


def run_dir(job, smoke=False):
    return OUTPUT / ('feasibility' if smoke else 'runs') / job['dataset'] / job['id']


def read_run(directory):
    path = Path(directory) / 'run.json'
    return json.loads(path.read_text()) if path.exists() else {}


def update_run(directory, **fields):
    atomic_json(Path(directory) / 'run.json', {**read_run(directory), **fields})


def verify_identity(directory, identity):
    directory = local_path(directory, OUTPUT)
    directory.mkdir(parents=True, exist_ok=True)
    run = read_run(directory)
    if run:
        if run.get('identity') != identity:
            raise RuntimeError(f'Resume identity mismatch: {directory}')
    elif any(directory.iterdir()):
        raise RuntimeError(f'Output without identity: {directory}')
    else:
        update_run(directory, identity=identity)
    return directory


def verify_completion(directory, identity):
    run = read_run(local_path(directory, OUTPUT))
    complete = run.get('completion')
    if not complete or run.get('identity') != identity:
        raise RuntimeError(f'Completed run does not match its job: {directory}')
    missing = [name for name in complete['outputs'] if not (Path(directory) / name).is_file()]
    if missing:
        raise RuntimeError(f'Missing outputs in {directory}: {missing}')
    return complete


def require_training_enabled():
    if json.loads(PLAN.read_text()).get('training_enabled') is not True:
        raise RuntimeError('Training is stopped in config/study.json; preparation and checks only.')


def acquire_gpu_slot(gpu, slot):
    if not str(gpu).isdigit() or slot not in (0, 1):
        raise ValueError('Expected one GPU index and slot 0 or 1')
    locks = []
    try:
        for path, mode in ((OUTPUT / f'gpu_{gpu}.lock', fcntl.LOCK_SH),
                           (OUTPUT / f'gpu_{gpu}_slot_{slot}.lock', fcntl.LOCK_EX)):
            stream = path.open('a')
            locks.append(stream)
            fcntl.flock(stream, mode | fcntl.LOCK_NB)
    except Exception:
        for stream in locks:
            stream.close()
        raise
    return locks
