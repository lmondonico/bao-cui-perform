"""
Module for tokenizing molecules into moieties and rings.
"""

import os
import sys
import colorsys
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from io import BytesIO


sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.tokenizer.tokenizer_moiety_dict import MOIETIES


class Tokenizer:
    def __init__(self):
        self.fg_dict = MOIETIES
        self.moiety_key_map = {
            name: idx + 1 for idx, name in enumerate(self.fg_dict.keys())
        }
        self.key_moiety_map = {idx: name for name, idx in self.moiety_key_map.items()}
        self.next_key = max(self.moiety_key_map.values()) + 1
        self.global_color_map = {}

    def detect_MOIETIES(self, mol, fg_dict):
        """
        Detect moieties in a molecule, excluding atoms corresponding to '[#6]' in the SMARTS patterns.
        Each functional group instance is listed separately.
        Overlapping moieties are handled by keeping only the largest ones, giving less priority to rings.
        """
        fg_matches = []

        for fg_name, smarts in fg_dict.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            pattern_atom_labels = [atom.GetSmarts() for atom in pattern.GetAtoms()]
            matches = mol.GetSubstructMatches(pattern, uniquify=False)
            for match in matches:
                included_atoms = []
                for pat_idx, mol_idx in enumerate(match):
                    label = pattern_atom_labels[pat_idx]
                    if label != "[#6]":
                        included_atoms.append(mol_idx)
                if included_atoms:
                    fg_matches.append((fg_name, set(included_atoms), "moiety"))

        ring_matches = self.detect_rings(mol)

        all_matches = fg_matches + ring_matches

        non_overlapping_matches = self.filter_overlapping_matches(all_matches)

        return non_overlapping_matches

    def detect_rings(self, mol):
        """
        Detect rings in the molecule.
        Each ring system is identified and labeled according to the specified format.
        """
        ring_matches = []
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        ring_systems = self.group_rings(rings)
        for ring_atoms_list in ring_systems:
            ring_sizes = [len(ring) for ring in ring_atoms_list]
            ring_sizes_str = "_&_".join(str(size) for size in sorted(ring_sizes))
            if len(ring_sizes) == 1:
                ring_name = f"ring_{ring_sizes_str}"
            else:
                ring_name = f"ring_{ring_sizes_str}"
            atom_indices = set()
            for ring in ring_atoms_list:
                atom_indices.update(ring)
            ring_matches.append((ring_name, atom_indices, "ring"))
            if ring_name not in self.moiety_key_map:
                self.moiety_key_map[ring_name] = self.next_key
                self.key_moiety_map[self.next_key] = ring_name
                self.global_color_map[self.next_key] = self.generate_single_color(
                    self.next_key
                )
                self.next_key += 1
        return ring_matches

    def group_rings(self, rings):
        """
        Group rings that share atoms into ring systems.
        Returns a list of lists of rings (each ring is a tuple of atom indices).
        """
        if not rings:
            return []

        ring_sets = [set(ring) for ring in rings]
        ring_systems = []

        while ring_sets:
            first, *rest = ring_sets
            first = set(first)
            merged = False
            rest2 = []
            for r in rest:
                if first & r:
                    first |= r
                    merged = True
                else:
                    rest2.append(r)
            if merged:
                ring_sets = [first] + rest2
            else:
                ring_systems.append([tuple(first)])
                ring_sets = rest2

        final_ring_systems = []
        for system in ring_systems:
            if not system:
                continue
            system_atoms = system.pop()
            individual_rings = []
            for ring in rings:
                if set(ring).issubset(system_atoms):
                    individual_rings.append(ring)
            final_ring_systems.append(individual_rings)

        return final_ring_systems

    def filter_overlapping_matches(self, matches):
        """
        Filter out overlapping moieties by keeping only the largest ones,
        giving priority to non-ring moieties over ring moieties.
        """
        matches.sort(key=lambda x: (x[2] == "ring", -len(x[1])))
        non_overlapping = []
        used_atoms = set()
        for fg_name, atom_indices, moiety_type in matches:
            if not atom_indices & used_atoms:
                non_overlapping.append((fg_name, atom_indices))
                used_atoms.update(atom_indices)
        return non_overlapping

    def generate_single_color(self, key):
        """
        Generate a unique color based on the moiety key using a combination of hue, saturation, and value.
        """
        hue = (key * 0.618033988749895) % 1
        saturation = 0.6 + 0.4 * (key % 2)
        value = 0.95
        return colorsys.hsv_to_rgb(hue, saturation, value)

    def highlight_and_display_molecule(self, mol, matches, save_path=None):
        """
        Assign colors to each functional group and display the molecule.
        """
        highlight_atom_colors = {}
        for fg_name, atom_indices in matches:
            key = self.moiety_key_map.get(fg_name, -1)
            color = (
                (0.0, 0.0, 0.0)
                if key == -1
                else self.global_color_map.get(key, self.generate_single_color(key))
            )
            for atom in atom_indices:
                highlight_atom_colors[atom] = color

        drawer = rdMolDraw2D.MolDraw2DCairo(1000, 800)
        options = drawer.drawOptions()
        options.addAtomIndices = True
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(highlight_atom_colors.keys()),
            highlightAtomColors=highlight_atom_colors,
        )
        drawer.FinishDrawing()
        img = drawer.GetDrawingText()

        image = Image.open(BytesIO(img))

        if save_path:
            image.save(save_path, dpi=(900, 900))

        else:
            image.show()

    def get_moieties(self, smiles):
        """
        Get moieties (moieties and rings) from a SMILES string.
        """

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Invalid SMILES string.")
            return

        matches = self.detect_MOIETIES(mol, self.fg_dict)

        clique_atom_ids = [sorted(list(atom_indices)) for _, atom_indices in matches]

        moiety_ids = [self.moiety_key_map.get(fg_name, -1) for fg_name, _ in matches]

        atom_to_moiety = {}
        for moiety_idx, (fg_name, atom_indices) in enumerate(matches):
            for atom in atom_indices:
                atom_to_moiety[atom] = moiety_idx

        edges = set()
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            moiety1 = atom_to_moiety.get(begin)
            moiety2 = atom_to_moiety.get(end)
            if moiety1 is not None and moiety2 is not None and moiety1 != moiety2:
                edges.add(tuple(sorted((begin, end))))
        edges = sorted(list(edges))

        return clique_atom_ids, edges, moiety_ids, matches

    def get_vocab_size(self):
        """Get the vocabulary size for this tokenizer."""
        if self.moiety_key_map:
            return max(self.moiety_key_map.values()) + 1
        else:
            return 1

    def encode(self, smiles):
        """Encode a SMILES string into token IDs."""
        try:
            clique_atom_ids, edges, moiety_ids, matches = self.get_moieties(smiles)
            return moiety_ids
        except:
            return [0]
