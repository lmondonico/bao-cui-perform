"""Repository paths for models, data and results."""

import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())

CORE_DIR = os.path.join(PROJECT_ROOT, "core")
MODULES_DIR = os.path.join(PROJECT_ROOT, "modules")
PERFORM_DIR = os.path.join(MODULES_DIR, "perform")
TOKENIZER_DIR = os.path.join(MODULES_DIR, "tokenizer")
TASKS_DIR = os.path.join(PROJECT_ROOT, "tasks")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
EXPERIMENT_RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
RESULTS_DIR = os.path.join(EXPERIMENT_RESULTS_DIR, "runs")

ABLATION_DIR = os.path.join(TASKS_DIR, "ablation")
ABLATION_MODELS_DIR = os.path.join(ABLATION_DIR, "ablation_models")

INFERENCE_DIR = os.path.join(TASKS_DIR, "inference")

CORE_MODULES = {
    "imports": os.path.join(CORE_DIR, "imports.py"),
    "utils": os.path.join(CORE_DIR, "utils.py"),
    "plotting": os.path.join(CORE_DIR, "plotting.py"),
    "smiles_preprocessing": os.path.join(CORE_DIR, "smiles_preprocessing.py"),
}

MODULES = {
    "dataset_processing": os.path.join(MODULES_DIR, "dataset_processing.py"),
    "moiety_features": os.path.join(MODULES_DIR, "moiety_features.py"),
    "perform_model": os.path.join(PERFORM_DIR, "perform_model.py"),
    "perform_components": os.path.join(PERFORM_DIR, "perform_components.py"),
    "perform_parm": os.path.join(PERFORM_DIR, "perform_parm.py"),
    "perform_no_motif": os.path.join(
        ABLATION_MODELS_DIR, "perform_no_motif.py"
    ),
    "perform_no_molar_ratio": os.path.join(
        ABLATION_MODELS_DIR, "perform_no_molar_ratio.py"
    ),
    "perform_no_temperature": os.path.join(
        ABLATION_MODELS_DIR, "perform_no_temperature.py"
    ),
    "perform_no_attention": os.path.join(
        ABLATION_MODELS_DIR, "perform_no_attention.py"
    ),
    "perform_minimal": os.path.join(ABLATION_MODELS_DIR, "perform_minimal.py"),
}

MODEL_CONFIGS = {
    "perform": {
        "name": "PerForm",
        "description": "PerForm",
        "module_path": "modules.perform.perform_model",
        "class_name": "PerForm",
        "parm_module": "modules.perform.perform_parm",
    },
    "perform-no-motif": {
        "name": "PerForm-NO-MOTIF",
        "description": "Ablation variant without MOTIF (simple embedding)",
        "module_path": "tasks.ablation.ablation_models.perform_no_motif",
        "class_name": "PerFormNoMOTIF",
        "parm_module": "tasks.ablation.ablation_parm",
    },
    "perform-no-molar-ratio": {
        "name": "PerForm-NO-MOLAR-RATIO",
        "description": "Ablation variant without MolarRatioEncoder",
        "module_path": "tasks.ablation.ablation_models.perform_no_molar_ratio",
        "class_name": "PerFormNoMolarRatio",
        "parm_module": "tasks.ablation.ablation_parm",
    },
    "perform-no-temperature": {
        "name": "PerForm-NO-TEMPERATURE",
        "description": "Ablation variant without TemperatureEncoder",
        "module_path": "tasks.ablation.ablation_models.perform_no_temperature",
        "class_name": "PerFormNoTemperature",
        "parm_module": "tasks.ablation.ablation_parm",
    },
    "perform-no-attention": {
        "name": "PerForm-NO-ATTENTION",
        "description": "Ablation variant without FormulationAttention",
        "module_path": "tasks.ablation.ablation_models.perform_no_attention",
        "class_name": "PerFormNoAttention",
        "parm_module": "tasks.ablation.ablation_parm",
    },
    "perform-minimal": {
        "name": "PerForm-MINIMAL",
        "description": "Minimal ablation variant (MOTIF only)",
        "module_path": "tasks.ablation.ablation_models.perform_minimal",
        "class_name": "PerFormMinimal",
        "parm_module": "tasks.ablation.ablation_parm",
    },
}

TOKENIZER_MODULE = os.path.join(TOKENIZER_DIR, "tokenizer_perform.py")
MOIETY_DICT_MODULE = os.path.join(TOKENIZER_DIR, "tokenizer_moiety_dict.py")

TASKS = {
    "train_holdout": os.path.join(TASKS_DIR, "training", "train_holdout.py"),
    "train_kfold": os.path.join(TASKS_DIR, "training", "train_kfold.py"),
    "train_parm": os.path.join(TASKS_DIR, "training", "train_parm.py"),
    "train_loss_function": os.path.join(
        TASKS_DIR, "training", "train_loss_function.py"
    ),
    "ablation_parm": os.path.join(ABLATION_DIR, "ablation_parm.py"),
    "predict": os.path.join(INFERENCE_DIR, "predict.py"),
}

DATASETS = {
    "conductivity": {
        "CALISOL_23": os.path.join(DATA_DIR, "splits-calisol", "calisol.csv"),
    },
}


def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [RESULTS_DIR, DATA_DIR, ABLATION_DIR, ABLATION_MODELS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_venv_python():
    """Get the path to the virtual environment Python."""
    return os.path.join(PROJECT_ROOT, ".venv-perform", "bin", "python")


def get_model_config(model_key):
    """Get model configuration by key."""
    return MODEL_CONFIGS.get(model_key)
