"""
Integrated Gradients explainability module for 1D ECG signals.

Computes path integral of input gradients along linear interpolation path from
a zero baseline to the input waveform.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


def compute_integrated_gradients(
    model: nn.Module,
    waveform_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
) -> Tuple[np.ndarray, float]:
    """
    Computes Integrated Gradients attribution and returns completeness error.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model in eval mode.
    waveform_tensor : torch.Tensor
        Input waveform tensor of shape (1, 216) or (1, 1, 216).
    target_class : int
        Target integer class index to explain.
    device : torch.device
        Compute device.
    baseline : Optional[torch.Tensor], default=None
        Baseline tensor of shape matching waveform_tensor. Defaults to zeros.
    n_steps : int, default=50
        Number of interpolation steps along the path.
        
    Returns
    -------
    Tuple[np.ndarray, float]
        (integrated_gradients_array, completeness_error)
    """
    model.eval()
    model.to(device)

    if waveform_tensor.dim() == 2:
        waveform_tensor = waveform_tensor.unsqueeze(0)
    elif waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0).unsqueeze(0)

    input_x = waveform_tensor.to(device).detach()

    if baseline is None:
        baseline_x = torch.zeros_like(input_x)
    else:
        baseline_x = baseline.to(device).detach()
        if baseline_x.dim() == 2:
            baseline_x = baseline_x.unsqueeze(0)

    # Calculate target class scores at input and baseline for completeness check
    with torch.no_grad():
        score_input = model(input_x)[0, target_class].item()
        score_baseline = model(baseline_x)[0, target_class].item()

    delta_score = score_input - score_baseline

    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, n_steps + 1, device=device)
    accumulated_grads = torch.zeros_like(input_x)

    for i in range(1, n_steps + 1):
        alpha = alphas[i]
        interpolated = baseline_x + alpha * (input_x - baseline_x)
        interpolated.requires_grad_(True)

        logits = model(interpolated)
        score = logits[0, target_class]

        model.zero_grad()
        score.backward()

        accumulated_grads += interpolated.grad.detach()

    avg_grads = accumulated_grads / n_steps
    integrated_grads_tensor = (input_x - baseline_x) * avg_grads

    ig_array = integrated_grads_tensor[0, 0].cpu().numpy()
    sum_ig = float(np.sum(ig_array))

    completeness_error = abs(sum_ig - delta_score)

    return ig_array, float(completeness_error)
