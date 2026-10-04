"""
Deterministic example selection module for waveform explainability.

Selects representative correctly classified beats (N, S, V, F, Q) and
dominant confusion pairs (N->V, V->N, S->N, F->N) for post-hoc attribution analysis.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


def select_explainability_examples(
    pred_df: pd.DataFrame,
    seed: int = 42,
    max_examples: int = 10,
) -> List[Dict[str, Any]]:
    """
    Selects a deterministic set of representative and misclassified beats for visualization.
    
    Parameters
    ----------
    pred_df : pd.DataFrame
        Prediction DataFrame containing [beat_id, record_id, sample_index, true_class,
        predicted_class, confidence, probability_N, ...].
    seed : int, default=42
        Random seed for deterministic fallback.
    max_examples : int, default=10
        Maximum total examples to select.
        
    Returns
    -------
    List[Dict[str, Any]]
        List of selected example specification dictionaries.
    """
    selected = []

    # 1. High-confidence correct examples for each class
    for cls in ["N", "V", "S", "F", "Q"]:
        correct_mask = (pred_df["true_class"] == cls) & (pred_df["predicted_class"] == cls)
        subset = pred_df[correct_mask]

        if len(subset) > 0:
            top_row = subset.sort_values(by="confidence", ascending=False).iloc[0]
            selected.append({
                "example_id": f"{cls}_correct",
                "beat_id": top_row["beat_id"],
                "record_id": str(top_row["record_id"]),
                "sample_index": int(top_row["sample_index"]),
                "true_class": cls,
                "predicted_class": cls,
                "confidence": float(top_row["confidence"]),
                "selection_reason": f"High-confidence correct {cls} beat",
                "dataset_index": int(top_row.name),
            })

    # 2. Dominant misclassification / confusion pairs
    confusion_pairs = [("N", "V"), ("V", "N"), ("S", "N"), ("F", "N")]
    for t_cls, p_cls in confusion_pairs:
        conf_mask = (pred_df["true_class"] == t_cls) & (pred_df["predicted_class"] == p_cls)
        subset = pred_df[conf_mask]

        if len(subset) > 0:
            top_row = subset.sort_values(by="confidence", ascending=False).iloc[0]
            selected.append({
                "example_id": f"{t_cls}_to_{p_cls}",
                "beat_id": top_row["beat_id"],
                "record_id": str(top_row["record_id"]),
                "sample_index": int(top_row["sample_index"]),
                "true_class": t_cls,
                "predicted_class": p_cls,
                "confidence": float(top_row["confidence"]),
                "selection_reason": f"Highest-confidence misclassification {t_cls} -> {p_cls}",
                "dataset_index": int(top_row.name),
            })

    return selected[:max_examples]
