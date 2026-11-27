"""Tokenization and learned molecular attention for the paper examples."""

from collections import defaultdict
from typing import Any

import numpy as np
import torch


COMPONENT_COLUMNS = [*(f"Solvent_{i}" for i in range(1, 6)), *(f"Salt_{i}" for i in range(1, 4))]
TEMPERATURE_K = 298.15
DISCONNECTED_DISTANCE = 1e8


def configure_tokenizer(checkpoint: dict[str, Any]) -> Any:
    from core.utils import get_tokenizer

    tokenizer = get_tokenizer()
    tokenizer.moiety_key_map = dict(checkpoint["moiety_key_map"])
    tokenizer.key_moiety_map = dict(checkpoint["key_moiety_map"])
    tokenizer.next_key = max(tokenizer.moiety_key_map.values()) + 1
    return tokenizer


def process_molecules(
    smiles_values: list[str],
    tokenizer: Any,
    checkpoint: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from modules.moiety_features import (
        build_moiety_edges,
        get_adj_matrix,
        get_dist_matrix,
        map_atom_to_moiety,
    )

    starting_map = dict(tokenizer.moiety_key_map)
    molecules: list[dict[str, Any]] = []
    off_diagonal_adjacency_checks = 0
    off_diagonal_adjacency_failures = 0

    for smiles in smiles_values:
        result = tokenizer.get_moieties(smiles)
        if result is None:
            raise ValueError(f"Tokenizer rejected corpus SMILES: {smiles}")
        clique_atom_ids, edges, moiety_ids, matches = result
        if not moiety_ids:
            raise ValueError(f"Tokenizer returned no moieties: {smiles}")
        if min(moiety_ids) <= 0 or max(moiety_ids) >= checkpoint["vocab_size"]:
            raise ValueError(f"Token ID outside checkpoint vocabulary: {smiles}")

        atom_to_moiety = map_atom_to_moiety(clique_atom_ids)
        moiety_edges = sorted(build_moiety_edges(edges, atom_to_moiety))
        nodes = list(range(len(clique_atom_ids)))
        raw_adj = get_adj_matrix(nodes, moiety_edges).astype(np.float32)
        raw_dist = get_dist_matrix(nodes, moiety_edges).astype(np.float32)
        adjacency_mask = ((1.0 - raw_adj) * -1e9).astype(np.float32)

        if len(nodes) > 1:
            off_diagonal = ~np.eye(len(nodes), dtype=bool)
            adjacency_identity = raw_adj[off_diagonal] == (
                raw_dist[off_diagonal] == 1
            )
            off_diagonal_adjacency_checks += 1
            if not bool(np.all(adjacency_identity)):
                off_diagonal_adjacency_failures += 1

        names = [
            tokenizer.key_moiety_map.get(token_id, f"UNKNOWN_{token_id}")
            for token_id in moiety_ids
        ]
        molecules.append(
            {
                "smiles": smiles,
                "tokens": np.asarray(moiety_ids, dtype=np.int64),
                "moiety_names": names,
                "matches": matches,
                "raw_adj": raw_adj,
                "raw_dist": raw_dist,
                "adjacency_mask": adjacency_mask,
                "n_moieties": len(moiety_ids),
            }
        )

    if tokenizer.moiety_key_map != starting_map:
        added = sorted(set(tokenizer.moiety_key_map) - set(starting_map))
        raise ValueError(f"Corpus generated tokens absent from checkpoint map: {added}")

    audit = {
        "molecules_processed": len(molecules),
        "total_moiety_instances": int(sum(x["n_moieties"] for x in molecules)),
        "minimum_moieties": int(min(x["n_moieties"] for x in molecules)),
        "maximum_moieties": int(max(x["n_moieties"] for x in molecules)),
        "off_diagonal_adjacency_checks": off_diagonal_adjacency_checks,
        "off_diagonal_adjacency_failures": off_diagonal_adjacency_failures,
        "checkpoint_map_unchanged": True,
    }
    return molecules, audit


def make_batch(
    molecules: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_size = len(molecules)
    width = max(molecule["n_moieties"] for molecule in molecules)
    tokens = np.zeros((batch_size, width), dtype=np.int64)
    adjacency = np.full((batch_size, width, width), -1e9, dtype=np.float32)
    distance = np.full((batch_size, width, width), 1e9, dtype=np.float32)

    for row, molecule in enumerate(molecules):
        count = molecule["n_moieties"]
        tokens[row, :count] = molecule["tokens"]
        adjacency[row, :count, :count] = molecule["adjacency_mask"]
        distance[row, :count, :count] = molecule["raw_dist"]

    return {
        "x": torch.from_numpy(tokens).to(device),
        "adjacency": torch.from_numpy(adjacency).to(device),
        "distance": torch.from_numpy(distance).to(device),
    }


def full_model_batch(
    batch: dict[str, torch.Tensor],
    max_num_molecules: int,
) -> dict[str, torch.Tensor]:
    full: dict[str, torch.Tensor] = {}
    batch_size, width = batch["x"].shape
    for molecule_index in range(1, max_num_molecules + 1):
        if molecule_index == 1:
            full[f"x{molecule_index}"] = batch["x"]
            full[f"adjoin_matrix{molecule_index}"] = batch["adjacency"]
            full[f"dist_matrix{molecule_index}"] = batch["distance"]
            full[f"mr{molecule_index}"] = torch.ones(
                batch_size, 1, device=batch["x"].device
            )
        else:
            full[f"x{molecule_index}"] = torch.zeros(
                batch_size,
                width,
                dtype=torch.long,
                device=batch["x"].device,
            )
            full[f"adjoin_matrix{molecule_index}"] = torch.full(
                (batch_size, width, width),
                -1e9,
                device=batch["x"].device,
            )
            full[f"dist_matrix{molecule_index}"] = torch.full(
                (batch_size, width, width),
                1e9,
                device=batch["x"].device,
            )
            full[f"mr{molecule_index}"] = torch.zeros(
                batch_size, 1, device=batch["x"].device
            )
    full["temperature"] = torch.full(
        (batch_size,), TEMPERATURE_K, device=batch["x"].device
    )
    return full


def extract_attention_values(molecules, local_attention, global_attention):
    pairs, ratios = [], []
    zero_denominators = 0
    for molecule_index, molecule in enumerate(molecules):
        count = molecule["n_moieties"]
        local = local_attention[molecule_index, :, :count, :count].mean(axis=0)
        global_ = global_attention[molecule_index, :, :count, :count].mean(axis=0)
        groups = defaultdict(list)
        for query in range(count):
            for key in range(count):
                if query == key:
                    continue
                distance = float(molecule["raw_dist"][query, key])
                if molecule["raw_adj"][query, key] > 0.5 and distance == 1.0:
                    groups[("local_bonded", 1)].append((query, key, float(local[query, key])))
                if 0 < distance < DISCONNECTED_DISTANCE:
                    if not np.isclose(distance, round(distance)):
                        raise ValueError("Shortest-path distance was not integral")
                    groups[("global_fixed_distance", int(round(distance)))].append((query, key, float(global_[query, key])))
        for (branch, distance), entries in sorted(groups.items()):
            if len(entries) < 2:
                continue
            values = np.array([entry[2] for entry in entries])
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid attention weights")
            minimum, maximum = float(values.min()), float(values.max())
            ratio = maximum / minimum if minimum > 0 else None
            zero_denominators += ratio is None
            ratios.append({"smiles": molecule["smiles"], "branch": branch, "distance": distance,
                           "n_directed_pairs": len(entries), "minimum": minimum, "maximum": maximum, "ratio": ratio})
            for query, key, value in entries:
                pairs.append({"smiles": molecule["smiles"], "branch": branch, "distance": distance,
                              "query_index": query, "key_index": key, "query_moiety": molecule["moiety_names"][query],
                              "key_moiety": molecule["moiety_names"][key], "attention": value})
    return pairs, ratios, {
        "pair_weight_rows": len(pairs), "ratio_groups": len(ratios), "zero_denominator_groups": zero_denominators,
        "definition": "Maximum/minimum head-averaged final-layer attention within each molecule's bonded group or fixed finite distance group; zero minima remain undefined.",
        "excluded_pairs": "Diagonal and disconnected pairs; groups with fewer than two directed pairs.",
    }
