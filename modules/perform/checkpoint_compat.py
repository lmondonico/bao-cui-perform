"""Read-only compatibility for checkpoints saved before the MOTIF rename."""

LEGACY_ARCHITECTURE_KEYS = {
    "moiety_att_num_layers": "motif_num_layers",
    "moiety_att_num_heads": "motif_num_heads",
    "moiety_encoder_dropout": "motif_dropout",
}


def normalize_motif_architecture(architecture):
    """Return architecture settings using the published MOTIF names."""
    normalized = dict(architecture)
    for legacy_key, motif_key in LEGACY_ARCHITECTURE_KEYS.items():
        if legacy_key not in normalized:
            continue
        if motif_key in normalized and normalized[motif_key] != normalized[legacy_key]:
            raise ValueError(f"Conflicting architecture values for {motif_key}")
        normalized[motif_key] = normalized.pop(legacy_key)
    return normalized


def _rename_motif_path(path):
    return ".".join(
        "motif" if part == "encoder_moiety" else part for part in path.split(".")
    )


def normalize_motif_state_dict(state_dict):
    """Translate legacy MOTIF module paths without modifying the checkpoint."""
    normalized = state_dict.__class__()
    for key, value in state_dict.items():
        motif_key = _rename_motif_path(key)
        if motif_key in normalized:
            raise ValueError(f"Duplicate model state key after MOTIF rename: {motif_key}")
        normalized[motif_key] = value

    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        normalized._metadata = {
            _rename_motif_path(key): value for key, value in metadata.items()
        }
    return normalized
