# Predicting Multicomponent Mixture Properties with Hierarchical Representation Learning

**Results and trained models:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24574337.svg)](https://doi.org/10.5281/zenodo.24574337)

Code accompanying **“Predicting Multicomponent Mixture Properties with Hierarchical Representation Learning”** (2026).

PerForm predicts mixture properties using molecular moieties and hierarchical attention.

## Setup and training

Download `perform_results.zip` from Zenodo, then run from the repository root:

```bash
unzip perform_results.zip
mv perform_results/data/splits-* data/ && mv perform_results/experiments/results experiments/
micromamba create -p ./.venv-perform -f environment.yml
bash tasks/run_study.sh --gpus 0,1,2,3 --jobs-per-gpu 2
```

Training requires `"training_enabled": true` in `config/study.json`. Baselines use the environments in `baselines/` and these files from their original releases:

- MM-MoLFormer: `baselines/Baseline_Utils/Molformer/training/pubchem_canon_zinc_final_vocab_sorted.pth`
- SMI-TED-IC: `baselines/FGCN_SMITED-IC/SMITED-IC/smi-ted-Light_40.pt`
- F-GCN: `baselines/FGCN_SMITED-IC/FGCN/pretrained_weights/my_checkpoint.*`

## Inference

```bash
micromamba run -p ./.venv-perform python tasks/inference/predict.py my_mixtures.csv --task calisol
```

The CSV lists `Solvent_1..5` and `Salt_1..3` as `SMILES:amount` and `Temperature` in °C. Tasks: `calisol`, `ilthermo`, `des`, `chew_density`, `chew_enthalpy`. The output is the mean ± SD of the five fold checkpoints (`experiments/results/runs/`) in physical units; `load_model(task, fold, device)` in `predict.py` loads one.

Figure scripts and source data are in `figures/`.

## Data

CALiSol-23, ILThermo (CheMixHub), DES (Makarov and Kolker) and CheW (Chew et al.); see the SI.
