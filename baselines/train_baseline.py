"""Run an existing baseline with frozen public splits and audited adapters."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(os.environ.get("PERFORM_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))
from data.conversion_scripts.prepare_inputs import verify_prepared
from core.study import local_path, update_run
from data.conversion_scripts.prepare_splits import split_filename
SPLITS = {"train": "Training", "validation": "Validation", "test": "Test"}
BASELINES = {
    "xgboost": "baselines/XGBoost_baseline.py",
    "attentivefp": "baselines/AttentiveFP_baseline.py",
    "molformer": "baselines/Molformer_baseline.py",
    "molsets": "baselines/MolSets/train_benchmark.py",
    "smited": "baselines/FGCN_SMITED-IC/SMITED-IC/SMITED_finetune.py",
    "fgcn": "baselines/FGCN_SMITED-IC/FGCN/FGCN_finetune.py",
    "geomix": "baselines/GeoMix/main_splits.py",
}


def record_model(model, output):
    if hasattr(model, "parameters"):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        components = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
    else:
        import numpy as np
        total = model.count_params()
        trainable = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
        components = {layer.name: layer.count_params() for layer in model.layers}
    update_run(output, parameters={"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable), "components": components})


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one source occurrence: {old!r}")
    return text.replace(old, new)


def setting(text, name, value):
    expression = f"Path({str(value)!r})" if isinstance(value, Path) else repr(value)
    text, count = re.subn(r"^" + re.escape(name) + r"\s*=.*$", lambda _: name + " = " + expression, text, count=1, flags=re.M)
    if count != 1:
        raise ValueError(f"Missing source setting {name}")
    return text


def execute(source, original, output):
    (output / "source_executed.py").write_text(source)
    namespace = {"__name__": "__main__", "__file__": str(original), "_public_record_model": lambda model: record_model(model, output)}
    sys.path.insert(0, str(original.parent))
    exec(compile(source, str(original), "exec"), namespace)
    return namespace


def canonical_predictions(frames, prediction_file, output, target):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    predictions = pd.read_csv(prediction_file)
    predictions = predictions.rename(columns={"split": "Task", "actual": "Truth", "predicted": "Prediction", "actual_lic": "Truth", "predicted_lic": "Prediction"})
    names = {"train": "Training", "training": "Training", "val": "Validation", "validation": "Validation", "test": "Test"}
    predictions["Task"] = predictions.Task.str.lower().map(names)
    all_rows, metrics = [], {}
    for split, task in SPLITS.items():
        expected = frames[split]
        actual = predictions.loc[predictions.Task == task].reset_index(drop=True)
        if len(actual) != len(expected):
            raise ValueError(f"{task}: adapter changed row count {len(expected)} -> {len(actual)}")
        if "Row_ID" in actual:
            if set(actual.Row_ID) != set(expected.Row_ID) or actual.Row_ID.duplicated().any():
                raise ValueError(f"{task}: prediction IDs do not match frozen split")
            actual = actual.set_index("Row_ID").loc[expected.Row_ID].reset_index()
        else:
            actual["Row_ID"] = expected.Row_ID.to_numpy()
        if not np.allclose(actual.Truth, expected[target], rtol=2e-5, atol=2e-5):
            raise ValueError(f"{task}: prediction order or target values changed")
        if not np.isfinite(actual[["Truth", "Prediction"]].to_numpy()).all():
            raise ValueError(f"{task}: nonfinite prediction")
        actual["Truth"] = expected[target].to_numpy()
        for column in ("Source_ID", "Composition_ID", "System_ID", "Temperature", "Salt_count", "Component_count"):
            actual[column] = expected[column].to_numpy()
        all_rows.append(actual)
        metrics[task] = {"rmse": float(np.sqrt(mean_squared_error(actual.Truth, actual.Prediction))), "mae": float(mean_absolute_error(actual.Truth, actual.Prediction)), "r2": float(r2_score(actual.Truth, actual.Prediction)), "n": len(actual), "unique_compositions": int(actual.Composition_ID.nunique())}
    joined = pd.concat(all_rows, ignore_index=True)
    joined.to_csv(output / "regr_predictions.csv", index=False)
    payload = {"target": target, "metrics": metrics, "test_rmse": metrics["Test"]["rmse"], "test_r2": metrics["Test"]["r2"], "test_mae": metrics["Test"]["mae"], "n_test": metrics["Test"]["n"]}
    update_run(output, training=payload)
    return payload


def run_simple(args, frames, source, original):
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, str(ROOT / "baselines"))
    from Baseline_Utils import GNN_regr_utils as gnn
    parsed = [gnn.parse_smiles_ratio_format(frames[split]) for split in SPLITS]
    gnn.load_split_datasets = lambda *a, **k: tuple(frame.copy() for frame in parsed)
    for name, value in (("DATASET", "public"), ("TARGET", args.target), ("RANDOM_SEED", args.seed), ("RESULTS_DIR", str(args.output))):
        source = setting(source, name, value)
    if args.model == "xgboost":
        source = source.replace('"n_jobs": -1', '"n_jobs": ' + str(args.cpu_threads))
        source = replace(source, "model.fit(X_train, y_train)", "model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)")
        namespace = execute(source, original, args.output)
        history = namespace["model"].evals_result()
        pd.DataFrame({"Round": range(1, len(history["validation_0"]["rmse"]) + 1),
                      "Training_RMSE": history["validation_0"]["rmse"],
                      "Validation_RMSE": history["validation_1"]["rmse"]}).to_csv(args.output / "loss_history.csv", index=False)
        namespace["model"].save_model(args.output / "model.ubj")
        trees = namespace["model"].get_booster().trees_to_dataframe()
        update_run(args.output, parameters={"parameter_definition": "trees", "trees": int(trees.Tree.nunique()), "nodes": len(trees), "leaves": int((trees.Feature == "Leaf").sum())})
    else:
        source = setting(source, "GPU_ID", 0)
        source = replace(source, {"molformer": "patience=50,", "attentivefp": "patience=15,"}[args.model], "patience=20,")
        if args.smoke:
            source = setting(source, "num_epochs", 2)
            source = setting(source, "batch_size", 8)
        train_original = gnn.train_gnn_model
        def train_with_final_evaluation(model, train_loader, val_loader, *a, **kw):
            train_original(model, train_loader, val_loader, *a, **kw)
            device, weights = a[1], a[2]
            train_loader.dataset.concentration_noise = 0.0
            return (gnn.evaluate_gnn_model(model, train_loader, device, task="Training", weights_path=weights), gnn.evaluate_gnn_model(model, val_loader, device, task="Validation", weights_path=weights))
        gnn.train_gnn_model = train_with_final_evaluation
        datasets_original = gnn.create_gnn_datasets
        initialize_original = gnn.initialize_gnn_model
        def initialize_with_count(*a, **kw):
            model = initialize_original(*a, **kw)
            record_model(model, args.output)
            return model
        gnn.initialize_gnn_model = initialize_with_count
        def deterministic_evaluation(*a, **kw):
            datasets = datasets_original(*a, **kw)
            for dataset in datasets[1:]:
                dataset.concentration_noise = 0.0
            return datasets
        gnn.create_gnn_datasets = deterministic_evaluation
        def save_with_ids(*results, save_path, **kw):
            pd.concat([pd.DataFrame({"Task": r["task"], "Row_ID": r["elf_codes"], "Truth": r["truths"], "Prediction": r["predictions"]}) for r in results]).to_csv(Path(save_path) / "regr_predictions.csv", index=False)
        gnn.save_gnn_predictions_simple_csv = save_with_ids
        execute(source, original, args.output)
    return args.output / "regr_predictions.csv"


def run_molsets(args, frames, source, original):
    inputs, _ = verify_prepared(args.split_dir, "molsets", args.target)
    if args.dataset.startswith("ilthermo"):
        import pickle
        from data.conversion_scripts.check_compatibility import components
        fraction = {int(row.Row_ID): next(x for _, x, role in components(row) if role == "Salt") for frame in frames.values() for _, row in frame.iterrows()}
        scaled = args.output / "scaled_inputs"
        scaled.mkdir(exist_ok=True)
        for name in ("train", "val", "test"):
            rows = pickle.loads((inputs / f"data_{name}.pkl").read_bytes())
            (scaled / f"data_{name}.pkl").write_bytes(pickle.dumps([(*row[:4], fraction[int(row[0])], *row[5:]) for row in rows], protocol=pickle.HIGHEST_PROTOCOL))
        inputs = scaled
    for name in ("train", "val", "test"):
        source = replace(source, f'f"splits-{{DATASET}}/data_{name}.pkl"', repr(str(inputs / f"data_{name}.pkl")))
    source = setting(source, "DATASET", "public")
    source = setting(source, "GPU_ID", 0)
    source = setting(source, "SEED", args.seed)
    source = replace(source, "best_model = model.state_dict()", "best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}")
    source = replace(source, "best_val_loss = np.inf", "loss_history = []\nbest_val_loss = np.inf")
    source = replace(source, "    scheduler.step(val_loss)", "    scheduler.step(val_loss)\n    loss_history.append({'Epoch': epoch + 1, 'Training': train_loss, 'Validation': val_loss})\n    pd.DataFrame(loss_history).to_csv('loss_history.csv', index=False)")
    source = replace(source, "results = {}", "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)\ntrain_loader.collate_fn = graph_set_collate\nresults = {}")
    source = replace(source, '"max_ep": 10000', f'"max_ep": {2 if args.smoke else 500}')
    if args.smoke:
        source = replace(source, '"es_patience": 10', '"es_patience": -1')
    namespace = execute(source, original, args.output)
    record_model(namespace["model"], args.output)
    return args.output / "regr_predictions_public.csv"


def run_smited(args, frames, source, original):
    inputs, _ = verify_prepared(args.split_dir, "smited", args.target)
    for name, value in (("DATASET", "public"), ("GPU_ID", 0), ("DATA_ROOT", str(inputs)), ("MODEL_PATH", str(original.parent)), ("CHECKPOINTS_FOLDER", str(args.output / "checkpoints")), ("RESULTS_FOLDER", str(args.output)), ("START_SEED", args.seed)):
        source = setting(source, name, value)
    for part in SPLITS:
        source = replace(source, f'f"{{config.data_root}}/{part}_data.csv"', repr(str(inputs / split_filename(args.dataset, part))))
    source = replace(source, "    # Reinitialize prediction head", "    model._set_seed(config.start_seed)\n    # Reinitialize prediction head")
    source = replace(source, "    trainer.fit(max_epochs=config.max_epochs)", "    trainer.fit(max_epochs=config.max_epochs)\n    trainer.train_loader = torch.utils.data.DataLoader(trainer.train_loader.dataset, batch_size=config.n_batch, shuffle=False)")
    source = replace(source, "    # Train\n    trainer.fit", "    for expected, frame in zip((df_train, df_valid, df_test), (trainer.df_train, trainer.df_valid, trainer.df_test)):\n        columns = ['smiles', config.measure_name]\n        if not frame[columns].reset_index(drop=True).equals(expected[columns].reset_index(drop=True)):\n            raise ValueError('SMI-TED changed ordered records or targets')\n        if not frame['canon_smiles'].equals(frame['smiles']):\n            raise ValueError('SMI-TED changed formulation strings')\n        for value in frame['canon_smiles']:\n            if len(model.tokenizer.tokenize(value)) + 2 > model.max_len:\n                raise ValueError('SMI-TED formulation would be truncated')\n    # Train\n    trainer.fit")
    source = replace(source, "    trainer.compile(", "    _public_record_model(model)\n    trainer.compile(")
    source = source.replace("predictions.cpu().numpy()", "predictions.reshape(-1).cpu().numpy()")
    source = replace(source, "    trainer.fit(max_epochs=config.max_epochs)\n    trainer.train_loader",
                     "    best, wait = trainer.best_vloss, 0\n    for epoch in range(trainer.start_epoch, config.max_epochs + 1):\n"
                     "        trainer.start_epoch = epoch\n        trainer.fit(max_epochs=epoch)\n"
                     "        best, wait = (trainer.best_vloss, 0) if trainer.best_vloss < best else (best, wait + 1)\n"
                     "        if wait >= 20:\n            break\n    trainer.train_loader")
    if args.smoke:
        source = setting(source, "MAX_EPOCHS", 1)
        source = setting(source, "BATCH_SIZE", 8)
    execute(source, original, args.output)
    return args.output / "public_predictions.csv"


def run_fgcn(args, frames, source, original):
    import numpy as np
    import pandas as pd
    inputs, manifest = verify_prepared(args.split_dir, "fgcn", args.target)
    max_atoms = manifest["details"]["max_atoms"]
    source = setting(source, "DATASET", "public")
    source = setting(source, "GPU_ID", 0)
    source = setting(source, "results_dir", args.output)
    source = setting(source, "preprocessed_dir", inputs)
    source = setting(source, "max_atoms", max_atoms)
    source = replace(source, "    validation_data=val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),", "    validation_data=val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),\n    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(results_dir / 'best.weights.h5'), monitor='val_loss', save_best_only=True, save_weights_only=True), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)],")
    source = replace(source, "# Save model\nelectrolyte_model.save", "electrolyte_model.load_weights(str(results_dir / 'best.weights.h5'))\n# Save model\nelectrolyte_model.save")
    source = replace(source, 'warnings.filterwarnings("ignore")', f'warnings.filterwarnings("ignore")\ntf.keras.utils.set_random_seed({args.seed})')
    source = replace(source, "dataset = list_to_tf_dataset(train_data)\nval_dataset = list_to_tf_dataset(val_data)\ntest_dataset = list_to_tf_dataset(test_data)",
                     "with tf.device('/CPU:0'):\n    dataset = list_to_tf_dataset(train_data)\n    val_dataset = list_to_tf_dataset(val_data)\n    test_dataset = list_to_tf_dataset(test_data)")
    if args.smoke:
        source = setting(source, "epochs", 2)
    namespace = execute(source, original, args.output)
    record_model(namespace["electrolyte_model"], args.output)
    rows = []
    for split, prefix in (("train", "train"), ("validation", "val"), ("test", "test")):
        rows.append(pd.DataFrame({"Task": SPLITS[split], "Truth": np.asarray(namespace[prefix + "_actuals"]).reshape(-1), "Prediction": np.asarray(namespace[prefix + "_predictions"]).reshape(-1)}))
    path = args.output / "predictions_native.csv"
    pd.concat(rows).to_csv(path, index=False)
    return path


def run_geomix(args, frames, source, original):
    inputs, manifest = verify_prepared(args.split_dir, "geomix", args.target)
    sys.path.insert(0, str(original.parent))
    from dataset.CALiSol import ConductivityDataset as data_module
    data_module.salt_list = manifest["details"]["salts"]
    data_module.ConductivityDataset.convert_c = lambda self, concentration, ratios: concentration
    if args.seed not in manifest["details"]["geometry_seeds"]:
        raise ValueError(f"No frozen geometry for seed {args.seed}")
    source = replace(source, "            if valid_mask.any():", "            if not valid_mask.all():\n                raise ValueError('Nonfinite GeoMix predictions; refusing row loss')\n            if valid_mask.any():")
    source = replace(source, "    split_results = {}  # {split_name: (targets, preds)}", "    loader_train = DataLoader(dataset_train, batch_size=effective_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False, num_workers=0)\n    split_results = {}")
    source = replace(source, "    if args.optimizer == 'AdamW':", "    _public_record_model(model)\n    if args.optimizer == 'AdamW':")
    source = replace(source, "    history_df.to_csv(history_csv, index=False)", "    val_history['epoch'] = [epoch for epoch in range(len(train_losses)) if epoch >= warmup_epochs and epoch % args.test_interval == 0]\n    history_df.merge(val_history, on='epoch', how='left', validate='one_to_one').to_csv(history_csv, index=False)")
    source = replace(source, "    val_epoch_list = list(range(warmup_epochs, warmup_epochs + len(val_losses) * args.test_interval, args.test_interval))", "    val_epoch_list = val_history.epoch.tolist()")
    import pandas as pd
    scaled = args.output / "scaled_inputs"
    scaled.mkdir(exist_ok=True)
    mean, std = (lambda k: (k.mean(), k.std()))(pd.read_csv(inputs / split_filename(args.dataset, "train")).k)
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(inputs / split_filename(args.dataset, split)).assign(k=lambda f: (f.k - mean) / std)
        if args.dataset.startswith("ilthermo"):
            frame["c"] = 1 - frame[frame.columns[frame.columns.get_loc("solvent ratio type") + 1:]].sum(axis=1)
        frame.to_csv(scaled / split_filename(args.dataset, split), index=False)
    command = [str(original), "--dataset", "CALiSol", "--model", "GeoMix", "--encoder", "TFN", "--seed", str(args.seed), "--train_csv", str(scaled / split_filename(args.dataset, 'train')), "--val_csv", str(scaled / split_filename(args.dataset, 'validation')), "--test_csv", str(scaled / split_filename(args.dataset, 'test')), "--npz_dir", str(inputs / f"Solvent_seed{args.seed}.npz"), "--salt_npz_dir", str(inputs / f"Salt_seed{args.seed}.npz"), "--results_dir", str(args.output.parent), "--run_name", args.output.name, "--patience", "20"]
    if args.smoke:
        command += ["--epochs", "6", "--batch_size", "8"]
    sys.argv = command
    execute(source, original, args.output)
    predictions = pd.read_csv(args.output / f"{args.output.name}_predictions.csv")
    predictions[["actual_lic", "predicted_lic"]] = predictions[["actual_lic", "predicted_lic"]] * std + mean
    predictions.to_csv(args.output / "predictions_native.csv", index=False)
    return args.output / "predictions_native.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=BASELINES, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    from core.study import require_training_enabled
    require_training_enabled()
    args.output, args.split_dir = local_path(args.output), local_path(args.split_dir)
    environments = {"smited": "baselines/FGCN_SMITED-IC/.venv_smited", "fgcn": "baselines/FGCN_SMITED-IC/.venv_smited", "geomix": "baselines/GeoMix/.venv_geomix"}
    python = ROOT / environments.get(args.model, ".venv-perform") / "bin/python"
    if not args.worker:
        (args.output / "tmp").mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(args.output / "cache/matplotlib"), XDG_CACHE_HOME=str(args.output / "cache"), OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS="1", PERFORM_ROOT=str(ROOT))
        environment.update({"TMPDIR": str(args.output / "tmp"), "CUDA_CACHE_PATH": str(args.output / "cache/cuda"), "TORCH_EXTENSIONS_DIR": str(args.output / "cache/extensions"), "TORCHINDUCTOR_CACHE_DIR": str(args.output / "cache/inductor"), "HF_HOME": str(args.output / "cache/huggingface")})
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        return subprocess.call([str(python), "-s", "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], env=environment)
    import random
    import numpy as np
    import pandas as pd
    from data.conversion_scripts.check_compatibility import components
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.model != "fgcn":
        import torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(args.cpu_threads)
    frames = {split: pd.read_csv(args.split_dir / split_filename(args.dataset, split)) for split in SPLITS}
    for split, frame in frames.items():
        if frame.empty or frame.Row_ID.duplicated().any() or not np.isfinite(frame[args.target]).all():
            raise ValueError(f"Invalid {split} records")
        for _, row in frame.iterrows():
            components(row)
        if args.model in ("molsets", "geomix") and ((frame.Salt_count != 1).any() or (frame.Component_count <= frame.Salt_count).any()):
            raise ValueError(f"{args.model} requires exactly one salt and a neutral component")
        if args.model in ("smited", "fgcn") and (frame.Salt_count > 1).any():
            raise ValueError(f"{args.model} single-salt restriction")
    args.output.mkdir(parents=True, exist_ok=True)
    original = ROOT / BASELINES[args.model]
    source = original.read_text()
    if args.model in ("molsets", "smited", "fgcn", "geomix"):
        verify_prepared(args.split_dir, args.model, args.target)
    selection = "fixed original 100 boosting rounds; no checkpoint selection" if args.model == "xgboost" else "validation-selected checkpoints"
    provenance = {"model": args.model, "target": args.target, "seed": args.seed, "smoke": args.smoke, "python": str(python), "source": str(original.relative_to(ROOT)), "adjustments": ["exact split inputs", "no ingredient or observation loss", "deterministic evaluation", selection]}
    if args.model == "geomix":
        provenance["adjustments"].append("target standardized with training mean and SD; predictions mapped back")
    if args.model in ("geomix", "molsets") and args.dataset.startswith("ilthermo"):
        provenance["adjustments"].append("c given as salt mole fraction instead of molality")
    update_run(args.output, provenance=provenance)
    os.chdir(args.output)
    runner = {"molsets": run_molsets, "smited": run_smited, "fgcn": run_fgcn, "geomix": run_geomix}.get(args.model, run_simple)
    prediction_path = runner(args, frames, source, original)
    canonical_predictions(frames, prediction_path, args.output, args.target)
    print(f"{args.model}: saved predictions and metrics to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
