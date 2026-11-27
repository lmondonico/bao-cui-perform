"""Verify prepared records against the raw source files."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

if __package__:
    from . import prepare_data as preparation
else:
    import prepare_data as preparation


def validate(root):
    preparation.check_root(root)
    manifest = json.loads(preparation.plan_path(root).read_text())["data"]
    datasets = {name: preparation.read_csv(root / item["path"]) for name, item in manifest["datasets"].items()}
    for name, rows in datasets.items():
        item = manifest["datasets"][name]
        assert not item.get("excluded", {}).get("unreviewed_connected_charge_separation"), name
        assert len(rows) == item["observations"] == len({row["Row_ID"] for row in rows}), name
        keys = {}
        for row in rows:
            assert keys.setdefault(row["Composition_ID"], row["Composition_key"]) == row["Composition_key"], name
            components = json.loads(row["Components_JSON"])
            assert len(components) == int(row["Component_count"])
            assert sum(component["salt"] for component in components) == int(row["Salt_count"])
            assert name == "calisol" or abs(sum(component["amount"] for component in components) - 1) < 1e-12
            assert all(component["amount"] > 0 for component in components)
            grouped = preparation.json_text([[component["smiles"], f"{component['amount']:.12f}"] for component in components])
            assert row["Composition_key"] == grouped
            exported = {}
            for field, value in row.items():
                if (field.startswith("Solvent_") or field.startswith("Salt_")) and field != "Salt_count" and value:
                    smiles, amount = value.rsplit(":", 1)
                    exported[smiles] = float(amount)
            assert exported == {component["smiles"]: component["amount"] for component in components}
            original = defaultdict(float)
            for smiles, amount in json.loads(row["Source_components_JSON"]):
                if abs(amount) > preparation.TOLERANCE:
                    original[preparation.molecule(smiles)[0]] += amount
            total = 1 if name == "calisol" else sum(original.values())
            assert original.keys() == exported.keys()
            assert all(abs(exported[smiles] - amount / total) < 1e-14 for smiles, amount in original.items())
            target = float(row["Target"])
            physical = math.exp(target) if row["Target_transform"] == "ln" else 10 ** target if row["Target_transform"] == "log10" else target
            assert math.isclose(float(row["Physical_value"]), physical, rel_tol=1e-12, abs_tol=1e-15)
            assert math.isclose(float(row["Temperature"]) + 273.15, float(row["Source_temperature_K"]), abs_tol=1e-10)
    for name, parent in preparation.SUBSETS.items():
        full = {row["Row_ID"]: row for row in datasets[parent]}
        subset = {row["Row_ID"]: row for row in datasets[name]}
        assert subset == {identifier: row for identifier, row in full.items()
                          if preparation.in_subset(name, row["Salt_count"], row["Component_count"], row["Components_JSON"])}
        assert 0 < len(subset) < len(full)
    raw = preparation.read_csv(root / "raw/calisol/calisol23.csv")
    for row in datasets["calisol"]:
        original = raw[int(row["Source_row"]) - 1]
        assert float(row["IC"]) == float(original["k"])
        assert float(row["LIC"]) == math.log10(float(original["k"]))
        amounts = json.loads(row["Source_components_JSON"])
        assert amounts[-1][1] == float(original["c"])
    raw = preparation.read_csv(root / "raw/ilthermo/processed_IlThermoData.csv")
    for row in datasets["ilthermo"]:
        original = raw[int(row["Source_row"]) - 1]
        assert original["property"] == "Viscosity" and original["unit"] == "Pa•s"
        assert float(row["Target"]) == float(original["value"])
    raw = preparation.read_csv(root / "raw/des/database_curated.csv")
    for row in datasets["des"]:
        assert int(row["Salt_count"]) <= 1
        assert float(row["Target"]) == float(raw[int(row["Source_row"]) - 1]["Viscosity {lg(mPa*s)}"])
    raw = preparation.read_csv(root / "raw/chew/original_publisher_mixtures.csv")
    for name, field in [("chew_density", "density_grams_per_cubic_meter"), ("chew_enthalpy", "enthalpy_of_mixing_kJ_per_mol")]:
        for row in datasets[name]:
            assert float(row["Target"]) == float(raw[int(row["Source_row"]) - 1][field])
            assert float(row["Source_temperature_K"]) == 300
            assert int(row["Salt_count"]) == 0
    assert min(float(row["Target"]) for row in datasets["chew_enthalpy"]) < 0
    assert [(r["Row_ID"], r["Composition_ID"]) for r in datasets["chew_density"]] == [(r["Row_ID"], r["Composition_ID"]) for r in datasets["chew_enthalpy"]]
    return sum(map(len, datasets.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    args = parser.parse_args()
    print(f"Verified {validate(args.data_root.resolve()):,} records against the raw source files.")


if __name__ == "__main__":
    main()
