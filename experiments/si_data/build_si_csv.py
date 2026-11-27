"""Build the machine-readable SI files for the 12 in-house conductivity measurements and the attention examples."""

import json
import re
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "unseen_formulations"))
import predict_unseen as pu
from experiments.analyses import EXAMPLES

SOLVENTS = [f"Solvent_{i}" for i in range(1, 6)]
RATIO = {"w": "weight", "v": "volume", "mol": "mole"}
FAMILY = {"carbonate": "carbonate-based", "carbonate-fluoroether": "carbonate-based", "ether": "ether-based"}
NAMES = {
    "CC1COC(=O)O1": ("propylene carbonate", "PC"),
    "CCOC(=O)OCC": ("diethyl carbonate", "DEC"),
    "CCOC(=O)OC": ("ethyl methyl carbonate", "EMC"),
    "COC(=O)OC": ("dimethyl carbonate", "DMC"),
    "FC(F)C(F)(F)COC(F)(F)C(F)F": ("1,1,2,2-tetrafluoroethyl 2,2,3,3-tetrafluoropropyl ether", "TTE"),
    "FC(F)(F)COCC(F)(F)F": ("bis(2,2,2-trifluoroethyl) ether", "BTFE"),
    "C1CCOC1": ("tetrahydrofuran", "THF"),
    "CCCOCCOC": ("1-methoxy-2-propoxyethane", "MPE"),
    "CCOCCOCC": ("1,2-diethoxyethane", "DEE"),
    "FC(F)(F)COCCOCC(F)(F)F": ("1,2-bis(2,2,2-trifluoroethoxy)ethane", "F6DEE"),
    "COCOC": ("dimethoxymethane", "DMM"),
    "COCC(F)(F)C(F)(F)COC": ("2,2,3,3-tetrafluoro-1,4-dimethoxybutane", "FDMB"),
    "COCCOC": ("1,2-dimethoxyethane", "DME"),
    "F[B-](F)(F)F.[Li+]": ("lithium tetrafluoroborate", "LiBF4"),
    "F[P-](F)(F)(F)(F)F.[Li+]": ("lithium hexafluorophosphate", "LiPF6"),
    "O=S(=O)(F)[N-]S(=O)(=O)F.[Li+]": ("lithium bis(fluorosulfonyl)imide", "LiFSI"),
    "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Li+]": ("lithium bis(trifluoromethanesulfonyl)imide", "LiTFSI"),
    "CCO[Si](OCC)(OCC)OCC": ("tetraethyl orthosilicate", "TEOS"),
    "FC(F)COCCOCC(F)(F)F": ("2-[2-(2,2-difluoroethoxy)ethoxy]-1,1,1-trifluoroethane", "F5DEE"),
    "O=C1OCCO1": ("ethylene carbonate", "EC"),
}


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles}")
    return Chem.MolToSmiles(mol)


def split(value):
    smiles, amount = value.rsplit(":", 1)
    return smiles, float(amount)


def composition(pairs):
    return frozenset((canonical(s), round(a, 6)) for s, a in pairs)


def dois():
    text = (HERE.parent / "unseen_formulations/unseen_validation.bib").read_text()
    entries = re.findall(r"@\w+\{(\w+),(.*?)\n\}", text, re.S)
    return {key: re.search(r"doi\s*=\s*\{([^}]+)\}", body).group(1) for key, body in entries}


def calisol_units():
    frame = pd.read_csv(ROOT / "data/splits-calisol/calisol_eligible.csv", low_memory=False)
    units = {}
    for row in frame.itertuples():
        key = composition((c["smiles"], c["amount"]) for c in json.loads(row.Components_JSON))
        units.setdefault(key, set()).add((row.Source_ratio_type, row.Source_concentration_unit))
    return units


def units(row, solvents, salt, calisol):
    known = calisol.get(composition([*solvents, salt]), set())
    if len(known) > 1:
        raise ValueError(f"#{row.ID}: conflicting CALiSol units {known}")
    ratio, unit = next(iter(known), (None, None))
    if len(solvents) == 1:
        ratio = "single solvent"
    elif "v/v" in row.Formulation:
        ratio = "volume"
    else:
        ratio = RATIO.get(ratio)
    unit = "mol/L" if re.search(r"\d M ", row.Formulation) else unit
    if ratio is None or unit is None:
        raise ValueError(f"#{row.ID}: ratio type or salt unit unclear")
    return ratio, unit


def predict(frame, mapping, vocab_size, key_map):
    protocols, folds = pu.load_protocols(), {}
    for fold in range(1, 6):
        protocol = f"cv_calisol_fold{fold}"
        checkpoint = ROOT / f"experiments/results/runs/calisol/calisol_benchmark_perform_LIC_fold{fold}/best_model.pth"
        model, state, _ = pu.load_checkpoint(Namespace(checkpoint=checkpoint, protocol=protocol, device="cpu"),
                                             protocols[protocol], required_target="LIC")
        if state["vocab_size"] != vocab_size or state["moiety_key_map"] != key_map:
            raise ValueError("Checkpoint vocabulary differs from the CALiSol cache")
        dataset = pu.FeatureEngineering(frame, None, mapping, "LIC", "conductivity", 0.0, training=False)
        batch = next(iter(DataLoader(dataset, batch_size=len(frame), collate_fn=pu.CollateFn(max_num_mols=8))))
        with torch.inference_mode():
            folds[f"PerForm_fold{fold}_mS_cm"] = np.power(10.0, model(batch).reshape(-1).double().numpy())
    return pd.DataFrame(folds, index=frame.index)


def main():
    frame, mapping, vocab_size, key_map, _ = pu.inputs()
    folds = predict(frame, mapping, vocab_size, key_map)
    stored = pd.read_csv(HERE.parent / "unseen_formulations/unseen_predictions.csv")
    if not np.allclose(folds.to_numpy(), stored.filter(like="fold").to_numpy(), atol=1e-4):
        raise ValueError("Predictions differ from unseen_formulations/unseen_predictions.csv")
    doi, calisol = dois(), calisol_units()

    rows, molecules = [], {}
    for row, (_, fold) in zip(frame.itertuples(), folds.iterrows()):
        solvents = [split(getattr(row, c)) for c in SOLVENTS if pd.notna(getattr(row, c))]
        salt = split(row.Salt_1)
        ratio, unit = units(row, solvents, salt, calisol)
        out = {"ID": row.ID, "Formulation": row.Formulation.split(" (")[0], "Family": FAMILY[row.Family],
               "Reference_DOI": doi.get(row.Source, ""), "Literature_T_C": row.Literature_T_C, "Literature_conductivity_mS_cm": row.Literature_mS_cm,
               "Measured_T_C": row.Temperature, "Measured_conductivity_mS_cm": row.Measured_mS_cm}
        for i in range(5):
            out[f"Solvent_{i + 1}_SMILES"] = solvents[i][0] if i < len(solvents) else ""
        for i in range(5):
            out[f"Solvent_{i + 1}_fraction"] = solvents[i][1] if i < len(solvents) else np.nan
        out |= {"Solvent_ratio_type": ratio, "Salt_SMILES": salt[0], "Salt_amount": salt[1], "Salt_amount_unit": unit}
        out |= fold.to_dict()
        out["PerForm_mean_mS_cm"], out["PerForm_SD_mS_cm"] = fold.mean(), fold.std(ddof=1)
        out["Absolute_error_mS_cm"] = abs(out["PerForm_mean_mS_cm"] - row.Measured_mS_cm)
        rows.append(out)
        for smiles, _ in [*solvents, salt]:
            molecules.setdefault(smiles, None)
    for smiles in (EXAMPLES["TEOS"], EXAMPLES["F5DEE"], "O=C1OCCO1"):
        molecules.setdefault(smiles, None)

    pd.DataFrame(rows).to_csv(HERE / "perform_in_house_conductivity.csv", index=False)
    names, table = {canonical(s): v for s, v in NAMES.items()}, []
    for smiles in molecules:
        name, abbreviation = names[canonical(smiles)]
        table.append({"Name": name, "Abbreviation": abbreviation, "SMILES": smiles,
                      "InChIKey": Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))})
    pd.DataFrame(table).to_csv(HERE / "perform_smiles.csv", index=False)
    print(f"perform_in_house_conductivity.csv: {len(rows)} rows; perform_smiles.csv: {len(table)} rows")


if __name__ == "__main__":
    main()
