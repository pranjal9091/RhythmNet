"""
Aggregate attribution analysis module across classes.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecg_arrhythmia.explainability.integrated_gradients import compute_integrated_gradients
from ecg_arrhythmia.models import DEFAULT_LABEL_MAPPING


def compute_aggregate_attributions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_examples_per_class: int = 100,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Computes mean absolute Integrated Gradients attribution across correctly classified beats
    per AAMI class.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model in eval mode.
    dataloader : DataLoader
        DataLoader for dataset.
    device : torch.device
        Compute device.
    max_examples_per_class : int, default=100
        Maximum examples to average per class.
    seed : int, default=42
        Random seed for reproducibility.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping class name to mean absolute attribution array of shape (216,).
    """
    model.eval()
    model.to(device)

    np.random.seed(seed)
    class_attributions = {cls: [] for cls in ["N", "S", "V", "F"]}

    for batch in dataloader:
        waveforms = batch["waveform"]  # [B, 1, 216]
        labels = batch["label"].cpu().numpy()

        with torch.no_grad():
            logits = model(waveforms.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        for i in range(len(labels)):
            true_label = labels[i]
            pred_label = preds[i]

            # Process only correct predictions
            if true_label == pred_label:
                reverse_map = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
                cls_name = reverse_map[true_label]

                if cls_name in class_attributions and len(class_attributions[cls_name]) < max_examples_per_class:
                    w = waveforms[i : i + 1]
                    ig_att, _ = compute_integrated_gradients(model, w, target_class=true_label, device=device, n_steps=30)
                    class_attributions[cls_name].append(np.abs(ig_att))

        # Check if all targeted classes reached max_examples_per_class
        if all(len(v) >= max_examples_per_class for v in class_attributions.values()):
            break

    result = {}
    for cls, atts in class_attributions.items():
        if len(atts) > 0:
            result[cls] = np.mean(np.array(atts), axis=0)
        else:
            result[cls] = np.zeros(216, dtype=np.float32)

    return result
