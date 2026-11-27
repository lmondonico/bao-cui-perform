"""Train one model on a frozen train/validation/test split."""
import argparse
import csv
import importlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.study import OUTPUT, acquire_gpu_slot, cache_path, run_dir, load_data, load_protocols, local_path, read_run, update_run, verify_identity
from core.utils import evaluate_model, calculate_metrics, save_predictions_to_csv
from data.conversion_scripts.prepare_splits import split_filename
from modules.dataset_processing import CollateFn
from tasks.training.train_parm import loader_batch_size, num_epochs as DEFAULT_EPOCHS, setup_criterion, setup_optimizer, setup_scheduler

NW = int(os.environ.get("PERFORM_WORKERS", "3"))
MAX_MOLS = 8


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one(
    model_factory,
    data,
    out_dir,
    seed,
    device,
    proto,
    label,
    epochs=None,
    lr=7e-5,
    patience=None,
    meta=None,
    identity=None,
):
    """Train one seed, or return an existing valid result."""
    if identity is not None:
        from core.study import verify_identity
        verify_identity(out_dir, identity)
    os.makedirs(out_dir, exist_ok=True)
    prev = read_run(out_dir).get("training", {})
    if "test_rmse" in prev and all(os.path.isfile(os.path.join(out_dir, name)) for name in
                                   ("loss_history.csv", "regr_predictions.csv", "best_model.pth")):
        return prev

    epochs = epochs or DEFAULT_EPOCHS
    patience = patience or epochs
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model = model_factory(data["vocab"]).to(device)
    nparam = count_params(model)

    collate = CollateFn(max_num_mols=MAX_MOLS)
    train_loader = DataLoader(
        data["train"], batch_size=loader_batch_size, shuffle=True,
        collate_fn=collate, num_workers=NW, pin_memory=True, drop_last=True,
        persistent_workers=identity is None and NW > 0, generator=generator,
    )
    val_loader = DataLoader(
        data["val"], batch_size=loader_batch_size, shuffle=False,
        collate_fn=collate, num_workers=max(1,NW-1), pin_memory=True, drop_last=False,
        persistent_workers=True, generator=torch.Generator().manual_seed(seed + 1000000),
    )

    criterion = setup_criterion()
    optimizer = setup_optimizer(model, learning_rate=lr)
    scheduler = setup_scheduler(optimizer, train_loader, epochs, max_lr=lr)

    weights_path = os.path.join(out_dir, "best_model.pth")
    best_val = float("inf")
    pat = 0
    t0 = time.time()
    history = []
    resume_path = os.path.join(out_dir, "resume.pth")
    start_epoch = 0
    elapsed_before = 0.0
    peak_allocated = 0
    if identity is not None and os.path.isfile(resume_path):
        import random
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved["identity"] != identity:
            raise RuntimeError("Checkpoint identity mismatch")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = saved["epoch"] + 1
        best_val, pat, history = saved["best_val"], saved["patience_counter"], saved["history"]
        elapsed_before = saved["elapsed_seconds"]
        peak_allocated = saved.get('training_peak_gpu_allocated_bytes', 0)
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, epochs if pat < patience else start_epoch):
        if identity is not None:
            generator.manual_seed(seed + epoch)
        model.train()
        tl, n = 0.0, 0
        for batch in train_loader:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            y = batch["y"].reshape(-1)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
            tl += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = tl / max(n, 1)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                y = batch["y"].reshape(-1)
                out = model(batch)
                vl += criterion(out, y).item() * y.size(0)
                vn += y.size(0)
        val_loss = vl / max(vn, 1)
        if device.type == 'cuda':
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]})

        if val_loss < best_val:
            best_val = val_loss
            pat = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": data["vocab"],
                "moiety_key_map": data["mkm"],
                "key_moiety_map": data["kmm"],
                "label": label, "seed": seed, "identity": identity,
            }, weights_path + ".tmp")
            os.replace(weights_path + ".tmp", weights_path)
        else:
            pat += 1
        if identity is not None:
            import random
            elapsed_now = elapsed_before + time.time() - t0
            torch.save({
                "identity": identity, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val": best_val, "patience_counter": pat, "history": history,
                "elapsed_seconds": elapsed_now,
                "training_peak_gpu_allocated_bytes": peak_allocated,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }, resume_path + ".tmp")
            os.replace(resume_path + ".tmp", resume_path)
            pd.DataFrame(history).to_csv(os.path.join(out_dir, "loss_history.csv"), index=False)
        if pat >= patience:
            break

    elapsed = elapsed_before + time.time() - t0

    eval_model = model_factory(data["vocab"]).to(device)
    loaders = {}
    for name, ds in (("Training", data.get("train_eval", data["train"])), ("Validation", data["val"]), ("Test", data["test"])):
        loaders[name] = DataLoader(ds, batch_size=loader_batch_size, shuffle=False,
                                   collate_fn=collate, num_workers=max(1,NW-1), drop_last=False,
                                   generator=torch.Generator().manual_seed(seed + 2000000))
    tr_r = evaluate_model(eval_model, loaders["Training"], device, setup_criterion, task="Training", weights_path=weights_path)
    va_r = evaluate_model(eval_model, loaders["Validation"], device, setup_criterion, task="Validation", weights_path=weights_path)
    te_r = evaluate_model(eval_model, loaders["Test"], device, setup_criterion, task="Test", weights_path=weights_path)

    metrics = calculate_metrics(tr_r, va_r, te_r, show_results=False)
    save_predictions_to_csv(tr_r, va_r, te_r, save_path=out_dir)

    result = {
        "label": label, "seed": seed, "proto": proto,
        "params": nparam,
        "epochs_run": len(history),
        "train_hours": elapsed / 3600.0,
        "training_peak_gpu_allocated_bytes": peak_allocated if device.type == 'cuda' else None,
        "test_rmse": float(metrics["Test"]["rmse"]),
        "test_r2": float(metrics["Test"]["r2"]),
        "val_rmse": float(metrics["Validation"]["rmse"]),
        "val_r2": float(metrics["Validation"]["r2"]),
        "meta": meta or {},
    }
    update_run(out_dir, training=result)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "loss_history.csv"), index=False)
    del eval_model, model
    torch.cuda.empty_cache()
    return result


VARIANTS = {
    'no_motif': ('perform_no_motif', 'PerFormNoMOTIF'),
    'no_molar_ratio': ('perform_no_molar_ratio', 'PerFormNoMolarRatio'),
    'no_attention': ('perform_no_attention', 'PerFormNoAttention'),
    'no_temperature': ('perform_no_temperature', 'PerFormNoTemperature'),
    'minimal': ('perform_minimal', 'PerFormMinimal'),
    'no_relu': ('perform_no_relu_attention', 'PerFormNoReLUAttention'),
    'vanilla': ('perform_vanilla_attention', 'PerFormVanillaAttention'),
    'hadamard': ('perform_hadamard_attention', 'PerFormHadamardAttention'),
    'additive_distance': ('perform_additive_distance', 'PerFormAdditiveDistance'),
    'role_assigned': ('perform_role_assigned', 'PerFormRoleAssigned'),
    'temp_concat': ('perform_temp_concat', 'PerFormTemperatureConcat'),
    'mr_concat': ('perform_mr_concat', 'PerFormMolarRatioConcat'),
    'deep_sets': ('perform_deep_sets', 'PerFormDeepSets'),
}


def load_protocol(key):
    proto = load_protocols()[key]
    return proto, local_path(ROOT / proto['split_dir'], ROOT / 'data')


def model_factory(job, vocab):
    from modules.perform.perform_parm import ARCHITECTURE
    kwargs = dict(ARCHITECTURE['PerForm'], input_vocab_size=vocab, max_num_molecules=8)
    if job['model'] in ('perform', 'full'):
        from modules.perform import PerForm
        return PerForm(**kwargs, target_label=job['target'])
    module, name = VARIANTS[job['model']]
    cls = getattr(importlib.import_module('tasks.ablation.ablation_models.' + module), name)
    if 'target_label' in inspect.signature(cls).parameters:
        kwargs['target_label'] = job['target']
    return cls(**kwargs)


def prepare_perform(job, split_dir, cache_path):
    import pandas as pd
    from core.utils import get_tokenizer, load_preprocessed_data
    from modules.dataset_processing import FeatureEngineering
    mapping, vocab, forward, reverse = load_preprocessed_data(custom_path=str(cache_path))
    data = {'vocab':vocab, 'mkm':forward, 'kmm':reverse}
    for part, source_part in [('train', 'train'), ('val', 'validation'), ('test', 'test'), ('train_eval', 'train')]:
        frame = pd.read_csv(split_dir / split_filename(job['dataset'], source_part))
        for col in [f'Solvent_{i}' for i in range(1,6)] + [f'Salt_{i}' for i in range(1,4)]:
            for value in frame[col].dropna().unique():
                smiles, amount = str(value).rsplit(':', 1)
                if float(amount) <= 0 or smiles not in mapping or not mapping[smiles]['moiety_ids']:
                    raise ValueError(f'Invalid component in frozen split: {value}')
        data[part] = FeatureEngineering(frame, get_tokenizer(), mapping, job['target'], 'conductivity',
                                       .03 if part == 'train' else 0., training=part == 'train')
        if len(data[part]) != len(frame):
            raise RuntimeError('Adapter changed record count')
    return data


def expected_identity(job, proto, smoke=False):
    transformation = 'identity' if job['target'] == 'IC' else load_data()['datasets'][job['dataset']]['target']['transformation']
    return {'dataset': job['dataset'], 'split': job['protocol'], 'split_dir': proto['split_dir'],
            'target': job['target'], 'transformation': transformation, 'model': job['model'],
            'seed': job['seed'], 'run': job.get('run'), 'fold': job.get('fold'),
            'configuration': {k: job.get(k) for k in ('epochs', 'patience', 'regime')}, 'smoke': smoke,
            'split_rows': {name: item['rows'] for name, item in proto['files'].items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--plan', default=str(ROOT / 'config/study.json'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--gpu-slot', type=int, choices=(0, 1), default=0)
    parser.add_argument('--cpu-threads', type=int, default=12)
    args = parser.parse_args()
    from core.study import require_training_enabled
    require_training_enabled()
    job = next(j for j in json.loads(Path(args.plan).read_text())['jobs'] if j['id'] == args.job)
    proto, split_dir = load_protocol(job['protocol'])
    with (OUTPUT/'compatibility/compatibility_matrix.csv').open() as stream:
        compatibility=list(csv.DictReader(stream))
    model='perform' if job['model'] in ('perform','full', *VARIANTS) else job['model']
    pairing=next((r for r in compatibility if r['dataset']==job['dataset'] and r['model']==model),None)
    if not pairing or pairing['status']=='blocked':
        raise RuntimeError(f'Unsupported comparison: {pairing}')
    if pairing.get('prepared_inputs_ready', '').lower() == 'false':
        raise RuntimeError(f'Unavailable prepared inputs: {pairing["prepared_inputs_reasons"]}')
    out = run_dir(job, args.smoke)
    identity = expected_identity(job, proto, args.smoke)
    from core.study import verify_completion
    if 'completion' in read_run(out):
        verify_completion(out, identity)
        return
    verify_identity(out, identity)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    gpu_locks=acquire_gpu_slot(gpu,args.gpu_slot)
    update_run(out,execution={'gpu':gpu,'slot':args.gpu_slot,'baseline_cpu_threads':args.cpu_threads,
                'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PERFORM_WORKERS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')}})
    started = time.time()
    if job['model'] in ('perform','full', *VARIANTS):
        import torch
        from core.runtime_utils import count_parameters, count_parameters_by_component
        data = prepare_perform(job, split_dir, cache_path(job['dataset']))
        if args.smoke:
            from torch.utils.data import Subset
            for name in ('train','val','test','train_eval'):
                data[name] = Subset(data[name], range(min(128, len(data[name]))))
        model = model_factory(job, data['vocab'])
        update_run(out, parameters={'total':count_parameters(model), 'components':count_parameters_by_component(model)})
        del model
        train_one(lambda vocab:model_factory(job,vocab), data, str(out), job['seed'], torch.device('cuda:0'),
                  job['protocol'],job['id'],epochs=2 if args.smoke else job['epochs'],
                  patience=job.get('patience',job['epochs']),identity=identity)
    else:
        baseline_split_dir=split_dir
        if args.smoke and job['model'] in ('xgboost', 'attentivefp', 'molformer'):
            import pandas as pd
            baseline_split_dir=out/'diagnostic_splits'
            baseline_split_dir.mkdir(exist_ok=True)
            for part in ('train', 'validation', 'test'):
                name = split_filename(job['dataset'], part)
                pd.read_csv(split_dir/name).iloc[:128].to_csv(baseline_split_dir/name,index=False)
        cmd=[sys.executable,'-s',str(ROOT/'baselines/train_baseline.py'),
             '--model',job['model'],'--dataset',job['dataset'],'--split-dir',str(baseline_split_dir),'--target',job['target'],
             '--output',str(out),'--seed',str(job['seed']),'--cpu-threads',str(args.cpu_threads)]
        if args.smoke: cmd.append('--smoke')
        subprocess.run(cmd,check=True,cwd=ROOT,pass_fds=tuple(stream.fileno() for stream in gpu_locks))
    from experiments.metrics import verify_run
    limited_smoke = args.smoke and job['model'] not in ('fgcn', 'smited', 'molsets', 'geomix')
    verify_run(out, split_dir, job, smoke=limited_smoke)
    loss_outputs={'xgboost':('loss_history.csv','loss_history.png'),
                  'attentivefp':('loss_history.csv','attentivefp_training_loss.png'),
                  'molformer':('loss_history.csv','molformer_training_loss.png'),
                  'molsets':('loss_history.csv','loss_history.png'),
                  'fgcn':('FGCN_training_history.csv','FGCN_loss.png'),
                  'smited':('public_training_history.csv','public_loss.png'),
                  'geomix':(f'{out.name}_training_history.csv',f'{out.name}_loss.png')}
    is_perform=job['model'] in ('perform','full', *VARIANTS)
    required=['regr_predictions.csv','predictions_with_provenance.csv','regr_parity_plot.png','test_strata.csv']
    required += ['loss_history.csv','loss_history.png'] if is_perform else []
    required.extend(loss_outputs.get(job['model'], ()))
    missing=[name for name in required if not (out/name).is_file()]
    if missing:raise RuntimeError(f'Missing required artifacts: {missing}')
    checkpoints=[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.suffix in ('.pth','.pt','.ubj','.h5','.keras')
                 and p.name != 'resume.pth' and not any(x in p.relative_to(out).parts for x in ('cache','inputs','diagnostic_splits'))]
    if not checkpoints:
        raise RuntimeError('No trained checkpoint saved')
    update_run(out,completion={'elapsed_seconds':time.time()-started,'checkpoints':checkpoints,'outputs':required+checkpoints})



if __name__ == '__main__':
    main()
