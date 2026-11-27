"""Convert supplied splits to F-GCN component tables."""

TARGET = "LIC"

import pandas as pd


def parse_smiles_ratio(cell_value):
    """
    Parse a cell containing 'SMILES:ratio' format.

    Args:
        cell_value: String in format 'SMILES:ratio' or empty/NaN

    Returns:
        tuple: (smiles, ratio) or (None, None) if empty
    """
    if pd.isna(cell_value) or cell_value == "":
        return None, None

    parts = cell_value.rsplit(":", 1)
    if len(parts) == 2:
        smiles = parts[0].strip()
        try:
            ratio = float(parts[1].strip())
            return smiles, ratio
        except ValueError:
            print(f"Warning: Could not parse ratio from '{cell_value}'")
            return None, None
    return None, None


def process_row(row):
    """
    Process a single row from the original dataset to FGCN format.

    Args:
        row: pandas Series with original dataset columns

    Returns:
        dict: Processed row data in FGCN format or None if row should be filtered
    """
    salt_columns = ["Salt_1", "Salt_2", "Salt_3"]
    salts = []
    for col in salt_columns:
        smiles, ratio = parse_smiles_ratio(row[col])
        if smiles is not None:
            salts.append((smiles, ratio))

    if len(salts) >= 2:
        return None

    solvent_columns = ["Solvent_1", "Solvent_2", "Solvent_3", "Solvent_4", "Solvent_5"]
    solvents = []
    for col in solvent_columns:
        smiles, ratio = parse_smiles_ratio(row[col])
        if smiles is not None:
            solvents.append((smiles, ratio))

    all_components = salts + solvents

    if len(all_components) == 0:
        return None

    total_ratio = sum(ratio for _, ratio in all_components)

    if total_ratio == 0:
        return None

    components_percent = [
        (smiles, (ratio / total_ratio) * 100.0) for smiles, ratio in all_components
    ]

    output = {}
    for i in range(6):
        if i < len(components_percent):
            smiles, conc = components_percent[i]
            output[f"smiles{i + 1}"] = smiles
            output[f"conc{i + 1}"] = conc
        else:
            output[f"smiles{i + 1}"] = "O"
            output[f"conc{i + 1}"] = 0.0

    output["temperature ( C )"] = row["Temperature"]
    output["log(cond) (uS/cm)"] = row[TARGET]

    return output
