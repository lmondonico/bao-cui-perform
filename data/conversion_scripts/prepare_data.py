"""Convert the cleaned public source files into the shared benchmark format."""

import argparse
import ast
import csv
import io
import json
import math
import sys
import urllib.request
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CHEMIX_VERSION = "356c16b99e38b56a91b5219d33db410fc372d538"
DES_VERSION = "95174c81f45448aa8ece122deb4622f3d7e5d667"
CALISOL_CODE_VERSION = "02334f51d581b45d6760fa02048ffa0d84c056d8"
CHEMIX = f"https://raw.githubusercontent.com/chemcognition-lab/chemixhub/{CHEMIX_VERSION}"
DES = f"https://raw.githubusercontent.com/saparicioubu/DES-viscosity/{DES_VERSION}"
CALISOL = f"https://raw.githubusercontent.com/Pele0599/CALiSol-23/{CALISOL_CODE_VERSION}"
CHEW_SI = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41524-025-01552-2/MediaObjects/41524_2025_1552_MOESM3_ESM.csv"
TOLERANCE = 1e-12
GROUP_DECIMALS = 12
FRACTION_SUM_TOLERANCE = 1e-4
NEUTRAL_SALT_IDENTITIES = {"CN(C)CCOC(c1ccccc1)c1ccccc1.Cl": 8980,
                           "Cc1ncc(CO)c(CO)c1O.Cl": 6019}
NEUTRAL_SALT_CANONICAL = {Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in NEUTRAL_SALT_IDENTITIES}
CONNECTED_CHARGE_IDENTITIES = {
    "C[N+](C)(C)CC(=O)[O-]": {"name": "betaine", "cid": 247, "salt": True},
    "C[N+](C)(C)C[C@H](O)CC(=O)[O-]": {"name": "L-carnitine", "cid": 10917, "salt": True},
    "C[N+](=O)[O-]": {"name": "nitromethane", "cid": 6375, "salt": False},
    "CCCCCCCCCCCC[N+](C)(C)[O-]": {"name": "lauramine oxide", "cid": 15433, "salt": False},
}
GEOMIX_ELEMENTS = {1, 3, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33}
GEOMIX_TEMPLATES = {"F[P-](F)(F)(F)(F)F": ("coord_LiPF6", 15), "F[As-](F)(F)(F)(F)F": ("coord_LiAsF6", 33)}
SUBSETS = {"ilthermo_common": "ilthermo", "des_common": "des"}


def json_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def plan_path(root):
    return root.parent / "config/study.json"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields or list(dict.fromkeys(key for row in rows for key in row)))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(handle.getvalue())


def read_csv(path, **kwargs):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, **kwargs))


def finite(value, label):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"missing_or_invalid_{label}") from None
    if not math.isfinite(result):
        raise ValueError(f"missing_or_invalid_{label}")
    return result


@lru_cache(maxsize=None)
def molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("invalid_source_smiles")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    fragments = Chem.GetMolFrags(mol, asMols=True)
    charges = [sum(atom.GetFormalCharge() for atom in fragment.GetAtoms()) for fragment in fragments]
    atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    salt = (any(q > 0 for q in charges) and any(q < 0 for q in charges)) or canonical in NEUTRAL_SALT_CANONICAL
    if len(fragments) == 1 and any(q > 0 for q in atom_charges) and any(q < 0 for q in atom_charges):
        if canonical not in CONNECTED_CHARGE_IDENTITIES:
            raise ValueError("unreviewed_connected_charge_separation")
        salt = CONNECTED_CHARGE_IDENTITIES[canonical]["salt"]
    if sum(charges) != 0:
        raise ValueError("incomplete_non_neutral_component_identity")
    if len(fragments) > 1 and not salt:
        raise ValueError("unresolved_disconnected_neutral_component_identity")
    return canonical, salt, Descriptors.MolWt(mol)


@lru_cache(maxsize=None)
def geomix_ready(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if any(atom.GetAtomicNum() not in GEOMIX_ELEMENTS for atom in mol.GetAtoms()):
        return False
    return all(Chem.MolToSmiles(f) in GEOMIX_TEMPLATES or max(a.GetDegree() for a in f.GetAtoms()) < 6
               for f in Chem.GetMolFrags(mol, asMols=True))


def in_subset(name, salt_count, component_count, components_json):
    if int(salt_count) != 1:
        return False
    smiles = [component["smiles"] for component in json.loads(components_json)]
    return int(component_count) > 1 and all(geomix_ready(s) for s in smiles)


class Preparation:
    def __init__(self, root):
        check_root(root)
        self.root = root
        self.datasets = {}
        self.exclusions = defaultdict(Counter)

    def download(self, name, url):
        path = self.root / "raw" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(url, headers={"User-Agent": "public-dataset-research/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                path.write_bytes(response.read())
        return path

    def reject(self, dataset, reason):
        self.exclusions[dataset][reason] += 1

    def verify_connected_charge_identities(self):
        for smiles, identity in CONNECTED_CHARGE_IDENTITIES.items():
            cid = identity["cid"]
            path = self.download(f"chemical_identities/pubchem_cid_{cid}.json", f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES/JSON")
            public_smiles = json.loads(path.read_text())["PropertyTable"]["Properties"][0]["SMILES"]
            if Chem.MolToSmiles(Chem.MolFromSmiles(public_smiles)) != smiles:
                raise RuntimeError(f"Independent connected-charge molecular identity mismatch: {cid}")

    def record(self, dataset, index, source_id, components, value, temperature, transform, unit, reference="", extra=None):
        temperature = finite(temperature, "temperature")
        value = finite(value, "target")
        if temperature <= 0:
            raise ValueError("nonpositive_kelvin_temperature")
        totals = defaultdict(float)
        salt_flags = {}
        corrections = 0
        for smiles, amount in components:
            amount = finite(amount, "component_amount")
            if amount < -TOLERANCE:
                raise ValueError("negative_component_amount")
            if abs(amount) <= TOLERANCE:
                corrections += amount != 0
                continue
            canonical, salt, _ = molecule(smiles.strip())
            totals[canonical] += amount
            salt_flags[canonical] = salt
        total = sum(totals.values())
        if not totals or total <= 0:
            raise ValueError("no_positive_component")
        scale = 1 if dataset == "calisol" else total
        normalized = [{"smiles": smi, "amount": amount / scale, "salt": salt_flags[smi],
                       "salt_kind": ("verified_neutral_hydrochloride_depiction" if smi in NEUTRAL_SALT_CANONICAL
                                     else "inner_salt" if salt_flags[smi] and "." not in smi
                                     else "disconnected_ions" if salt_flags[smi] else "not_salt")}
                      for smi, amount in sorted(totals.items())]
        grouped = [[c["smiles"], f"{c['amount']:.{GROUP_DECIMALS}f}"] for c in normalized]
        if any(float(fraction) == 0 for _, fraction in grouped):
            raise ValueError("positive_fraction_below_grouping_precision")
        physical = math.exp(value) if transform == "ln" else 10 ** value if transform == "log10" else value
        if not math.isfinite(physical):
            raise ValueError("nonfinite_inverse_target")
        salts = [c for c in normalized if c["salt"]]
        solvents = [c for c in normalized if not c["salt"]]
        if dataset == "des" and len(salts) > 1:
            raise ValueError("des_multi_salt_outside_single_salt_scope")
        record = {"Row_ID": index, "Source_ID": source_id, "Source_row": index, "Target": value,
                  "Physical_value": physical, "Target_transform": transform,
                  "Target_unit": f"{transform}({unit})" if transform != "identity" else unit,
                  "Physical_unit": unit, "Temperature": temperature - 273.15,
                  "Source_temperature_K": temperature,
                  "Composition_ID": None,
                  "Composition_key": json_text(grouped),
                  "System_ID": None,
                  "Salt_count": len(salts), "Component_count": len(normalized),
                  "Inner_salt_count": sum(c["salt_kind"] == "inner_salt" for c in normalized),
                  "Components_JSON": json_text(normalized),
                  "Source_components_JSON": json_text(components),
                  "Rounded_zero_components": corrections, "Source_reference": reference,
                  "Adapter_capacity_valid": len(solvents) <= 5 and len(salts) <= 3}
        for prefix, entries, capacity in [("Solvent", solvents, 5), ("Salt", salts, 3)]:
            for position in range(max(capacity, len(entries))):
                record[f"{prefix}_{position + 1}"] = (f"{entries[position]['smiles']}:{entries[position]['amount']:.17g}"
                                                          if position < len(entries) else "")
        if extra:
            record.update(extra)
        return record

    def calisol(self):
        self.download("calisol/article_metadata.json", "https://api.figshare.com/v2/articles/24559960")
        raw = self.download("calisol/calisol23.csv", "https://ndownloader.figshare.com/files/43151344")
        structures = self.download("calisol/calisolsmile.csv", CALISOL + "/calisolsmile.csv")
        self.download("calisol/CALiSol-23.ipynb", CALISOL + "/CALiSol-23.ipynb")
        self.download("calisol/article.html", "https://www.nature.com/articles/s41597-024-03575-8")
        with structures.open(encoding="utf-8-sig", newline="") as handle:
            mapping_rows = [[value.strip() for value in row] for row in csv.reader(handle, delimiter=";")]
        salts = {row[0]: row[2] for row in mapping_rows[:13]}
        solvents = {key: mapping_rows[13 + index][1] for index, key in enumerate(["EC", "PC", "DMC", "EMC"])}
        solvents.update({row[1]: row[2] for row in mapping_rows[17:]})
        rows = read_csv(raw)
        output = []
        for index, row in enumerate(rows, 1):
            try:
                conductivity = finite(row["k"], "conductivity")
                unknown = set(row) - {"", "doi", "k", "T", "c", "salt", "c units", "solvent ratio type"} - set(solvents)
                if unknown:
                    raise RuntimeError(f"Unmapped solvent columns: {unknown}")
                ratios = [(smiles, finite(row[name], "solvent_ratio")) for name, smiles in solvents.items()]
                components = [(smiles, amount) for smiles, amount in ratios if amount > 0]
                components.append((salts[row["salt"].strip()], finite(row["c"], "salt_concentration")))
                output.append(self.record("calisol", index, f"calisol23.v1:{row['']}", components,
                                          math.log10(conductivity), row["T"], "log10", "mS/cm", row["doi"],
                                          {"IC": conductivity, "LIC": math.log10(conductivity),
                                           "Source_ratio_type": row["solvent ratio type"],
                                           "Source_concentration_unit": row["c units"]}))
            except ValueError as exc:
                self.reject("calisol", str(exc))
        self.datasets["calisol"] = output

    def ilthermo(self):
        base = CHEMIX + "/datasets/ionic-liquids/processed_data/"
        raw = self.download("ilthermo/processed_IlThermoData.csv", base + "processed_IlThermoData.csv")
        compound_file = self.download("ilthermo/compounds.csv", base + "compounds.csv")
        self.download("ilthermo/data_processing.py", base + "data_processing.py")
        for smiles, cid in NEUTRAL_SALT_IDENTITIES.items():
            identity_file = self.download(f"ilthermo/pubchem_cid_{cid}.json", f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES/JSON")
            public_smiles = json.loads(identity_file.read_text())["PropertyTable"]["Properties"][0]["SMILES"]
            if Chem.MolToSmiles(Chem.MolFromSmiles(public_smiles)) != Chem.MolToSmiles(Chem.MolFromSmiles(smiles)):
                raise RuntimeError(f"Independent neutral hydrochloride identity mismatch: {cid}")
        compounds = {int(row["compound_id"]): row for row in read_csv(compound_file)}
        rows = read_csv(raw)
        output = []
        for index, row in enumerate(rows, 1):
            if row["property"] != "Viscosity":
                continue
            try:
                identifiers = ast.literal_eval(row["cmp_ids"])
                fractions = ast.literal_eval(row["cmp_mole_fractions"])
                if len(identifiers) != len(fractions):
                    raise ValueError("component_fraction_length_mismatch")
                components = [(compounds[int(identifier)]["smiles"], finite(fraction, "mole_fraction"))
                              for identifier, fraction in zip(identifiers, fractions)]
                if abs(sum(fractions) - 1) > FRACTION_SUM_TOLERANCE:
                    raise ValueError("mole_fractions_do_not_sum_to_one")
                output.append(self.record("ilthermo", index, f"chemixhub:{CHEMIX_VERSION}:ionic-liquids:{index}", components,
                                          row["value"], row["Temperature, K"], "ln", "Pa s", "NIST ILThermo via chemixhub",
                                          {"Pressure_kPa": row["Pressure, kPa"],
                                           "Source_pressure_assumed": row["Pressure_assumption"],
                                           "Source_compound_IDs": row["cmp_ids"]}))
            except ValueError as exc:
                self.reject("ilthermo", str(exc))
        self.datasets["ilthermo"] = output

    def des(self):
        raw = self.download("des/database_curated.csv", DES + "/data/database_curated.csv")
        self.download("des/DATA_PROVENANCE.md", DES + "/DATA_PROVENANCE.md")
        rows = read_csv(raw)
        output = []
        for index, row in enumerate(rows, 1):
            try:
                components = [(row["SMILES(HBA)"], finite(row["X(HBA)"], "mole_fraction")),
                              (row["SMILES(HBD)"], finite(row["X(HBD)"], "mole_fraction"))]
                if abs(sum(amount for _, amount in components) - 1) > FRACTION_SUM_TOLERANCE:
                    raise ValueError("mole_fractions_do_not_sum_to_one")
                output.append(self.record("des", index, f"DES-viscosity:{DES_VERSION}:{index}", components,
                                          row["Viscosity {lg(mPa*s)}"], row["Temperature {K}"], "log10", "mPa s", row["References"],
                                          {"Source_molecule_ID": row["MOLECULEID"], "Source_split": row["Set"], "Source_name": row["NAME"],
                                           "Source_fraction_sum": sum(amount for _, amount in components)}))
            except ValueError as exc:
                self.reject("des", str(exc))
        self.datasets["des"] = output

    def chew(self):
        raw = self.download("chew/original_publisher_mixtures.csv", CHEW_SI)
        self.download("chew/article.html", "https://www.nature.com/articles/s41524-025-01552-2")
        base = CHEMIX + "/datasets/miscible-solvent/processed_data/"
        self.download("chew/data_processing.py", base + "data_processing.py")
        rows = read_csv(raw)
        for dataset, field, unit in [("chew_density", "density_grams_per_cubic_meter", "g/cm^3"),
                                     ("chew_enthalpy", "enthalpy_of_mixing_kJ_per_mol", "kJ/mol")]:
            output = []
            for index, row in enumerate(rows, 1):
                try:
                    components = []
                    for position in range(5):
                        smiles, amount = row[f"SMILES_{position}"], row[f"comp_{position}"]
                        if not smiles and not amount:
                            continue
                        if not smiles or not amount:
                            raise ValueError("missing_component_or_amount")
                        components.append((smiles, finite(amount, "mole_percent") / 100))
                    if abs(sum(amount for _, amount in components) - 1) > FRACTION_SUM_TOLERANCE:
                        raise ValueError("mole_fractions_do_not_sum_to_one")
                    output.append(self.record(dataset, index, f"Chew2025:MOESM3:ID={row['ID']}", components,
                                              row[field], 300.0, "identity", unit, "10.1038/s41524-025-01552-2",
                                              {"Source_mixture_ID": row["ID"], "Source_name": row["label"],
                                               "Temperature_convention": "300 K final NVT production; fixed cell derived from 293 K NPT equilibration",
                                               "Source_target_column": field}))
                except ValueError as exc:
                    self.reject(dataset, str(exc))
            self.datasets[dataset] = output

    def save(self):
        plan = json.loads(plan_path(self.root).read_text())
        manifests = {}
        for dataset, rows in self.datasets.items():
            for column, key in (("Composition_ID", lambda r: r["Composition_key"]),
                                ("System_ID", lambda r: json_text(sorted(c["smiles"] for c in json.loads(r["Components_JSON"]))))):
                ids = {value: i for i, value in enumerate(sorted({key(r) for r in rows}), 1)}
                for row in rows:
                    row[column] = ids[key(row)]
            path = self.root / f"splits-{dataset.replace('_', '-')}" / ("calisol_eligible.csv" if dataset == "calisol" else f"{dataset}.csv")
            write_csv(path, rows)
            target = {"column": "Target", "transformation": rows[0]["Target_transform"],
                      "physical_unit": rows[0]["Physical_unit"], "unit": rows[0]["Target_unit"]}
            subsets = {dataset: (rows, path)}
            for name, parent in SUBSETS.items():
                if parent == dataset:
                    subset = [r for r in rows if in_subset(name, r["Salt_count"], r["Component_count"], r["Components_JSON"])]
                    subsets[name] = subset, self.root / f"splits-{name.replace('_', '-')}/{name}.csv"
                    write_csv(subsets[name][1], subset, list(rows[0]))
            for name, (subset, subset_path) in subsets.items():
                manifests[name] = {"path": str(subset_path.relative_to(self.root)), "target": target,
                                   "observations": len(subset), "compositions": len({r["Composition_ID"] for r in subset}),
                                   "systems": len({r["System_ID"] for r in subset})}
                if name != dataset:
                    manifests[name]["parent"] = dataset
            if self.exclusions[dataset]:
                manifests[dataset]["excluded"] = dict(self.exclusions[dataset])
            if dataset == "calisol":
                manifests[dataset]["benchmark_path"] = "splits-calisol/calisol.csv"
        plan["data"] = {"datasets": manifests}
        write_json(plan_path(self.root), plan)
        print("Prepared rows: " + ", ".join(f"{name}={item['observations']:,}" for name, item in manifests.items()))


def check_root(root):
    if root.name != "data":
        raise ValueError("Only the explicit repository data root is supported")
    if any(part in root.parts for part in ("_archive", ".local_archive")):
        raise ValueError("Archive inputs are forbidden")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    args = parser.parse_args()
    preparation = Preparation(args.data_root.resolve())
    preparation.verify_connected_charge_identities()
    preparation.calisol()
    preparation.ilthermo()
    preparation.des()
    preparation.chew()
    preparation.save()


if __name__ == "__main__":
    main()
