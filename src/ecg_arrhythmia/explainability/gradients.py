"""
Input Gradient explainability module for 1D ECG signals.

Computes raw input saliency maps: d(target_class_score) / d(input_waveform).
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


def compute_input_gradients(
    model: nn.Module,
    waveform_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """
    Computes Input Gradient saliency map for a single ECG beat waveform.
    
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
        
    Returns
    -------
    np.ndarray
        Input gradient attribution array of shape (216,).
    """
    model.eval()
    model.to(device)

    if waveform_tensor.dim() == 2:
        waveform_tensor = waveform_tensor.unsqueeze(0)  # [1, 1, 216]
    elif waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0).unsqueeze(0)

    input_x = waveform_tensor.to(device).clone().detach().requires_grad_(True)

    logits = model(input_x)
    score = logits[0, target_class]

    model.zero_grad()
    score.backward()

    grads = input_x.grad[0, 0].cpu().numpy()
    return grads
