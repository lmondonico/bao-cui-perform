"""Measure model parameters, computation, memory and runtime."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class RuntimeTracker:
    """Context manager for timing code sections."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_time = time.time() - self.start_time

    def get_elapsed_seconds(self) -> float:
        """Return elapsed time in seconds."""
        return self.elapsed_time if self.elapsed_time is not None else 0.0


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total, trainable, and non-trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params
    }


def count_parameters_by_component(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Count parameters by model component with percentage breakdown.

    Designed for PerForm model architecture.

    Args:
        model: PerForm PyTorch model

    Returns:
        Dictionary mapping component names to their parameter counts and percentages
    """
    def count_params(module):
        """Count trainable parameters in a module."""
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    breakdown = {}
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if hasattr(model, 'motif'):
        motif_params = count_params(model.motif)
        breakdown['motif'] = {
            'params': motif_params,
            'percent': round(100 * motif_params / total_params, 1)
        }

    if hasattr(model, 'transf_mlp') and hasattr(model, 'transf_norm'):
        molecule_layer_params = count_params(model.transf_mlp) + count_params(model.transf_norm)
        breakdown['molecule_layer'] = {
            'params': molecule_layer_params,
            'percent': round(100 * molecule_layer_params / total_params, 1)
        }

    if hasattr(model, 'molecular_attention'):
        formulation_params = count_params(model.molecular_attention)
        breakdown['formulation_attention'] = {
            'params': formulation_params,
            'percent': round(100 * formulation_params / total_params, 1)
        }

    if hasattr(model, 'molar_ratio_encoder'):
        mr_params = count_params(model.molar_ratio_encoder)
        breakdown['molar_ratio_encoder'] = {
            'params': mr_params,
            'percent': round(100 * mr_params / total_params, 1)
        }

    if hasattr(model, 'temperature_encoder'):
        temp_params = count_params(model.temperature_encoder)
        breakdown['temperature_encoder'] = {
            'params': temp_params,
            'percent': round(100 * temp_params / total_params, 1)
        }

    if hasattr(model, 'prediction_head'):
        head_params = count_params(model.prediction_head)
        breakdown['prediction_head'] = {
            'params': head_params,
            'percent': round(100 * head_params / total_params, 1)
        }

    if hasattr(model, 'global_pooling'):
        pooling_params = count_params(model.global_pooling)
        if pooling_params > 0:
            breakdown['global_pooling'] = {
                'params': pooling_params,
                'percent': round(100 * pooling_params / total_params, 1)
            }

    return breakdown


def measure_flops(model: nn.Module, sample_input: tuple) -> Optional[int]:
    """
    Measure FLOPs (floating point operations) for a single forward pass.

    Args:
        model: PyTorch model
        sample_input: Tuple of input tensors (same format as model forward pass)

    Returns:
        Total FLOPs, or None if fvcore is not available
    """
    try:
        from fvcore.nn import FlopCountAnalysis

        model_cpu = model.cpu()
        sample_input_cpu = tuple(x.cpu() if isinstance(x, torch.Tensor) else x
                                  for x in sample_input)

        flops = FlopCountAnalysis(model_cpu, sample_input_cpu)
        total_flops = flops.total()

        if torch.cuda.is_available():
            model.cuda()

        return int(total_flops)
    except ImportError:
        print("Warning: fvcore not installed. FLOPs measurement skipped.")
        print("Install with: pip install fvcore")
        return None


def measure_memory(reset: bool = True) -> Dict[str, float]:
    """
    Measure peak GPU memory usage.

    Args:
        reset: If True, reset peak memory statistics after measurement

    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {"peak_memory_gb": 0.0, "allocated_gb": 0.0}

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    current_memory = torch.cuda.memory_allocated() / 1024**3

    if reset:
        torch.cuda.reset_peak_memory_stats()

    return {
        "peak_memory_gb": round(peak_memory, 3),
        "allocated_gb": round(current_memory, 3)
    }


def save_runtime_stats(stats: Dict[str, Any], output_path: Path, filename: str = "runtime_stats.json"):
    """
    Save runtime statistics to JSON file.

    Args:
        stats: Dictionary of runtime statistics
        output_path: Directory to save the file
        filename: Name of the output file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nRuntime statistics saved to: {filepath}")


def format_runtime_summary(stats: Dict[str, Any]) -> str:
    """
    Format runtime statistics as a readable string.

    Args:
        stats: Dictionary of runtime statistics

    Returns:
        Formatted string summary
    """
    lines = ["\n" + "="*60]
    lines.append("RUNTIME STATISTICS SUMMARY")
    lines.append("="*60)

    if "model" in stats:
        lines.append("\nModel Complexity:")
        lines.append(f"  Parameters (total):     {stats['model'].get('parameters', 0):,}")
        lines.append(f"  Parameters (trainable): {stats['model'].get('trainable_parameters', 0):,}")
        if stats['model'].get('flops') is not None:
            flops_m = stats['model']['flops'] / 1e6
            flops_g = stats['model']['flops'] / 1e9
            lines.append(f"  FLOPs per forward pass: {flops_m:.2f}M ({flops_g:.3f}G)")

    if "training" in stats:
        lines.append("\nTraining Performance:")
        lines.append(f"  Time per epoch:   {stats['training'].get('time_per_epoch_sec', 0):.2f} sec")
        lines.append(f"  Total time:       {stats['training'].get('total_time_min', 0):.2f} min")
        lines.append(f"  Peak GPU memory:  {stats['training'].get('peak_memory_gb', 0):.2f} GB")

    if "inference" in stats:
        lines.append("\nInference Performance:")
        lines.append(f"  Throughput (batch=1024): {stats['inference'].get('throughput_pred_per_sec', 0):.1f} pred/sec")
        lines.append(f"  Latency (single pred):   {stats['inference'].get('latency_ms', 0):.2f} ms")
        lines.append(f"  Peak GPU memory:         {stats['inference'].get('peak_memory_gb', 0):.2f} GB")

    lines.append("="*60 + "\n")
    return "\n".join(lines)


def print_runtime_summary(stats: Dict[str, Any]):
    """Print formatted runtime statistics."""
    print(format_runtime_summary(stats))
