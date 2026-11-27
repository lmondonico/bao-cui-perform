"""
Functions to obtain the atomic ratios from PNAS paper
"""

import pandas as pd
from rdkit import Chem


class PNASFeaturizer:
    def __init__(self):
        self.inorganic_elements = set(["F", "O", "N", "S", "P", "B", "Si", "Cl", "Br", "I"])
        self.temp_min = -40.0
        self.temp_max = 80.0

    def count_atoms(self, smiles):
        """Count all atoms including implicit hydrogens in a molecule"""
        mol = Chem.MolFromSmiles(smiles)
        total_atoms = 0
        for atom in mol.GetAtoms():
            total_atoms += 1 + atom.GetTotalNumHs()
        return total_atoms

    def count_element(self, smiles, element):
        """Count occurrences of a specific element in a molecule"""
        mol = Chem.MolFromSmiles(smiles)
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == element)

    def count_organic_inorganic(self, smiles):
        """Count organic (C,H) vs inorganic atoms"""
        mol = Chem.MolFromSmiles(smiles)
        organic = 0
        inorganic = 0

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            num_h = atom.GetTotalNumHs()

            if symbol == "C":
                organic += 1 + num_h
            elif symbol in self.inorganic_elements:
                inorganic += 1

        return organic, inorganic

    def normalize_temperature(self, temp):
        """Normalize temperature to [0, 1] range"""
        return (temp - self.temp_min) / (self.temp_max - self.temp_min)

    def calculate_features(self, sample, include_temperature=True):
        """
        Calculate PNAS features for a sample (optionally including temperature)
        Sample should contain SMILES strings and concentrations for molecules 1-8

        Args:
            sample: DataFrame row with Solvent_i, Conc_Solvent_i, Salt_i, Conc_Salt_i columns
            include_temperature: If True, adds normalized temperature as 5th feature

        Returns:
            dict with features: sO, FO, InOr, aC (and Temp_norm if include_temperature=True)
        """
        features = {}

        solvent_O = 0
        solvent_total_atoms = 0
        total_F = 0
        total_O = 0
        total_organic = 0
        total_inorganic = 0
        salt_C = 0
        salt_total = 0

        for i in range(1, 6):
            mol_key = f"Solvent_{i}"
            conc_key = f"Conc_Solvent_{i}"

            if mol_key in sample and pd.notna(sample[mol_key]):
                smiles = sample[mol_key]
                conc = float(sample[conc_key])

                solvent_O += self.count_element(smiles, "O") * conc
                solvent_total_atoms += self.count_atoms(smiles) * conc

                total_F += self.count_element(smiles, "F") * conc
                total_O += self.count_element(smiles, "O") * conc
                org, inorg = self.count_organic_inorganic(smiles)
                total_organic += org * conc
                total_inorganic += inorg * conc

        for i in range(1, 4):
            mol_key = f"Salt_{i}"
            conc_key = f"Conc_Salt_{i}"

            if mol_key in sample and pd.notna(sample[mol_key]):
                smiles = sample[mol_key]
                conc = float(sample[conc_key])

                total_F += self.count_element(smiles, "F") * conc
                total_O += self.count_element(smiles, "O") * conc
                org, inorg = self.count_organic_inorganic(smiles)
                total_organic += org * conc
                total_inorganic += inorg * conc

                salt_C += self.count_element(smiles, "C") * conc
                salt_total += self.count_atoms(smiles) * conc

        features["sO"] = solvent_O / solvent_total_atoms if solvent_total_atoms > 0 else 0
        features["FO"] = total_F / total_O if total_O > 0 else 0
        features["InOr"] = total_inorganic / total_organic if total_organic > 0 else 0
        features["aC"] = salt_C / salt_total if salt_total > 0 else 0

        if include_temperature and "Temperature" in sample:
            features["Temp_norm"] = self.normalize_temperature(sample["Temperature"])

        return features
