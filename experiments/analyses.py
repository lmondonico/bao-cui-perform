"""Run CALiSol analyses, metrics and figures."""

import argparse
import importlib.util
import json
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.study import cache_path, read_run
from data.conversion_scripts.prepare_splits import split_filename


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def public_path(path):
    path = path.resolve()
    allowed = (ROOT / "data", ROOT / "experiments/results")
    if not any(path.is_relative_to(parent) for parent in allowed):
        raise ValueError(f"Not an explicit repository-local public-study path: {path}")
    if {"_archive", ".local_archive"}.intersection(path.parts):
        raise ValueError("Archived artifacts cannot enter a new analysis")
    return path


def inputs(args):
    data_root = public_path(args.data_root)
    from data.conversion_scripts import prepare_splits as prepare
    prepare.verify(data_root)
    from core.study import load_protocols
    protocol = load_protocols()[args.protocol]
    if protocol["dataset"] != "calisol":
        raise ValueError("A frozen, panel-protected CALiSol protocol is required")
    split_dir = public_path(ROOT / protocol["split_dir"])
    frames = [pd.read_csv(split_dir / split_filename(protocol['dataset'], part)) for part in ('train', 'validation', 'test')]
    frame = pd.concat(frames, ignore_index=True)
    if frame.Row_ID.duplicated().any():
        raise ValueError("Overlapping observation membership")
    return protocol, split_dir, frame


def load_checkpoint(args, protocol, required_target=None):
    import torch
    from modules.perform import PerForm, normalize_motif_state_dict
    from modules.perform.perform_parm import ARCHITECTURE

    checkpoint_path = public_path(args.checkpoint)
    if checkpoint_path.name != 'best_model.pth':
        raise ValueError('Use the validation-selected best_model.pth checkpoint')
    run = read_run(checkpoint_path.parent)
    identity = run["identity"]
    if "completion" not in run or identity.get("smoke"):
        raise ValueError("A completed full-training checkpoint identity is required")
    if identity["dataset"] != "calisol" or identity["model"] != "perform":
        raise ValueError("A newly trained full PerForm CALiSol checkpoint is required")
    if required_target is not None and identity["target"] != required_target:
        raise ValueError(f"Analysis requires target {required_target}")
    if identity["split"] != args.protocol:
        raise ValueError("Checkpoint split differs from analysis inputs")
    from core.study import verify_completion
    from tasks.training.train_holdout import expected_identity
    jobs = json.loads((ROOT / 'config/study.json').read_text())['jobs']
    job = next((job for job in jobs if job['id'] == checkpoint_path.parent.name), None)
    if job is None:
        raise ValueError('Checkpoint does not belong to the current study plan')
    verify_completion(checkpoint_path.parent, expected_identity(job, protocol))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.setdefault("architecture", ARCHITECTURE["PerForm"])
    checkpoint.setdefault("max_num_molecules", 8)
    model = PerForm(input_vocab_size=checkpoint["vocab_size"], max_num_molecules=checkpoint["max_num_molecules"],
                    target_label=identity["target"], **checkpoint["architecture"]).to(args.device).eval()
    model.load_state_dict(normalize_motif_state_dict(checkpoint["model_state_dict"]), strict=True)
    return model, checkpoint, identity


EXAMPLES = {"TEOS": "CCO[Si](OCC)(OCC)OCC", "F5DEE": "FC(F)COCCOCC(F)(F)F",
            "LiTFSI": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Li+]"}


def attention(args, protocol, frame, output):
    import torch

    model, checkpoint, identity = load_checkpoint(args, protocol, required_target="LIC")
    method = load_module("molecular_attention", ROOT / "experiments/attention.py")
    counts = Counter()
    for column in method.COMPONENT_COLUMNS:
        for value in frame[column].dropna():
            smiles, amount = str(value).rsplit(":", 1)
            if float(amount) > 0:
                counts[smiles] += 1
    tokenizer = method.configure_tokenizer(checkpoint)
    molecules, representation = method.process_molecules(list(EXAMPLES.values()), tokenizer, checkpoint)
    batch = method.make_batch(molecules, torch.device(args.device))
    with torch.inference_mode():
        _, weights = model(method.full_model_batch(batch, model.max_num_molecules), return_attention=True)
    local = weights[0]["local_attention"].cpu().numpy().astype(np.float64)
    global_ = weights[0]["global_attention"].cpu().numpy().astype(np.float64)
    pairs, ratios, audit = method.extract_attention_values(molecules, local, global_)
    pd.DataFrame(pairs).to_csv(output / "attention_pairs.csv", index=False)
    pd.DataFrame(ratios).to_csv(output / "attention_ratios.csv", index=False)
    examples = []
    for index, (name, molecule) in enumerate(zip(EXAMPLES, molecules)):
        count = molecule["n_moieties"]
        examples.append({"name": name, "smiles": molecule["smiles"], "moiety_names": molecule["moiety_names"],
                         "local_attention": local[index, :, :count, :count].mean(axis=0),
                         "global_attention": global_[index, :, :count, :count].mean(axis=0)})
    with (output / "attention_data.pkl").open("wb") as handle:
        pickle.dump(examples, handle)
    save_json(output / "attention_audit.json", {"checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)),
        "protocol": args.protocol, "target": identity["target"], "representation": representation, "ratios": audit,
        "selection": "Fixed example molecules of the original manuscript; no target/prediction selection.",
        "examples": [{"name": value["name"], "smiles": value["smiles"], "occurrences": counts[value["smiles"]]} for value in examples]})


def benchmark_inference(model, dataset, collate, device, warmup=10, repeats=100, batch_size=1024):
    import torch
    from torch.utils.data import DataLoader
    from core.runtime_utils import RuntimeTracker, measure_memory

    def move(batch):
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    sample = move(next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate))))
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)
    gpu = torch.device(device).type == "cuda"
    latency = []
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        for _ in range(repeats):
            with RuntimeTracker() as timer:
                result = model(sample)
            latency.append(timer.get_elapsed_seconds())
        sample_size = int(result.shape[0])
        model(move(next(iter(loader))))
        observations = 0
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        with RuntimeTracker() as timer:
            for batch in loader:
                result = model(move(batch))
                observations += int(result.shape[0])
        memory = measure_memory(reset=False) if gpu else {"peak_memory_gb": None, "allocated_gb": None}
    if observations != len(dataset):
        raise ValueError("Inference timing omitted or duplicated observations")
    return {"latency_batch_size": sample_size, "latency_warmup_forwards": warmup,
            "latency_timed_forwards": repeats, "latency_mean_ms": float(np.mean(latency) * 1000),
            "throughput_batch_size": batch_size, "throughput_warmup_batches": 1,
            "throughput_observations": observations, "throughput_seconds": timer.get_elapsed_seconds(),
            "throughput_pred_per_sec": observations / timer.get_elapsed_seconds(), **memory}, latency, sample


def computational_runtime(args, protocol, frame, output):
    import torch
    from core.runtime_utils import count_parameters, count_parameters_by_component, measure_flops, save_runtime_stats
    from core.utils import load_preprocessed_data
    from modules.dataset_processing import FeatureEngineering, CollateFn

    ledger_path = ROOT / "experiments/results/job_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if any(job.get("status") == "running" for job in ledger["jobs"].values()):
        raise RuntimeError("Computational profiling waits until all training workers are idle")
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(args.device)
    model, checkpoint, identity = load_checkpoint(args, protocol, required_target="LIC")
    from core.study import load_data
    source = load_data()['datasets']['calisol']
    cache = cache_path('calisol')
    mapping, vocabulary, moiety_map, _ = load_preprocessed_data(custom_path=str(cache))
    if vocabulary != checkpoint["vocab_size"] or moiety_map != checkpoint["moiety_key_map"]:
        raise ValueError("Runtime representation differs from checkpoint")
    dataset = FeatureEngineering(frame, None, mapping, "LIC", "conductivity", 0.0, training=False)
    inference, latency, sample = benchmark_inference(model, dataset, CollateFn(max_num_mols=8), args.device)
    pd.DataFrame({"forward": range(1, len(latency) + 1), "batch_size": inference["latency_batch_size"],
                  "elapsed_seconds": latency}).to_csv(output / "latency_samples.csv", index=False)
    parameters = count_parameters(model)
    breakdown = count_parameters_by_component(model)
    assigned = sum(value["params"] for value in breakdown.values())
    if assigned != parameters["trainable"]:
        remainder = parameters["trainable"] - assigned
        breakdown["unassigned_by_existing_helper"] = {"params": remainder, "percent": 100 * remainder / parameters["trainable"]}
    pd.DataFrame([{"component": key, **value} for key, value in breakdown.items()]).to_csv(output / "parameter_breakdown.csv", index=False)
    flops, flops_reason = None, ""
    try:
        cpu_sample = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
        flops = measure_flops(model, (cpu_sample,))
        if flops is None:
            flops_reason = "The existing optional fvcore profiler is not installed."
    except Exception as error:
        flops_reason = f"Existing profiler unavailable: {type(error).__name__}: {error}"
    finally:
        model.to(args.device)
    metrics = read_run(args.checkpoint.parent)["training"]
    hours, epochs = metrics.get("train_hours"), metrics.get("epochs_run")
    training_peak = metrics.get('training_peak_gpu_allocated_bytes')
    stats = {"checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)), "protocol": args.protocol, "target": identity["target"],
             "device": str(args.device), "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
             "gpu_name": torch.cuda.get_device_name() if torch.device(args.device).type == "cuda" else None,
             "model": {**parameters, "flops": flops, "flops_batch_size": inference["latency_batch_size"], "flops_limitation": flops_reason},
             "inference": inference, "training": {"total_hours": hours, "epochs_run": epochs,
                 "mean_epoch_seconds": hours * 3600 / epochs if hours is not None and epochs else None,
                 "peak_memory_gb": training_peak / 1e9 if training_peak is not None else None,
                 "peak_memory_definition": "Per-process PyTorch allocation during fitting, including validation; maximum retained across resumed sessions.",
                 "peak_memory_limitation": None if training_peak is not None else "No GPU allocation measurement was recorded for this run."},
             "method": "Batch two, ten warmup and 100 timed forwards; batch 1024 with one warmup then the complete retained CALiSol corpus. Throughput includes collation and device transfer. Peak inference allocation is measured during that full pass. Latency is per batch of two, not per single prediction.",
             "corpus": {"observations": len(frame), "compositions": int(frame.Composition_ID.nunique()),
                        "membership": "All three partitions of the frozen panel-protected protocol; labels are not used for timing."},
             "sources": ["experiments/analyses.py", "core/runtime_utils.py"]}
    save_runtime_stats(stats, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", choices=("all", "metrics", "figures", "controls", "attention", "runtime"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", default="cv_calisol_fold1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.analysis in ("metrics", "figures"):
        if args.plan_only:
            parser.error("--plan-only applies only to all")
        if args.analysis == "metrics":
            from experiments.metrics import aggregate
            aggregate()
        else:
            import runpy
            runpy.run_path(str(ROOT / "figures/plot_figures.py"), run_name="__main__")
        return
    if args.output is None:
        if args.analysis != "all":
            parser.error("--output is required for an individual analysis")
        args.output = ROOT / "experiments/results/analyses"
    sys.path.insert(0, str(ROOT))
    if args.analysis == "all":
        from experiments.analysis_queue import run_all
        run_all(args)
        return
    if args.plan_only:
        raise ValueError("--plan-only applies only to the all command")
    protocol, split_dir, frame = inputs(args)
    output = public_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing analysis output: {output}")
    if args.analysis == "controls":
        from core.study import require_training_enabled
        require_training_enabled()
        script = ROOT / "baselines/temperature_only/run_temperature_baselines.py"
        subprocess.run([sys.executable, "-s", "-B", str(script), "--split-dir", str(split_dir),
                        "--output", str(output), "--device", args.device, "--seed", str(args.seed)], check=True)
    else:
        output.mkdir(parents=True)
        if args.analysis in ("attention", "runtime"):
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required for learned-model analyses")
            if args.analysis == "attention":
                attention(args, protocol, frame, output)
            else:
                computational_runtime(args, protocol, frame, output)
    methods = {"controls": ["baselines/temperature_only/run_temperature_baselines.py"],
               "attention": ["experiments/attention.py"],
               "runtime": ["experiments/analyses.py", "core/runtime_utils.py"]}[args.analysis]
    save_json(output / "completed.json", {"analysis": args.analysis, "protocol": args.protocol, "methods": methods,
              "files": sorted(p.name for p in output.iterdir() if p.is_file())})


if __name__ == "__main__":
    main()
