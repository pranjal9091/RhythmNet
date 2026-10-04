"""Reproducibility utilities for deterministic experiments."""

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> int:
    """Set random seeds across Python random, NumPy, and PyTorch (if available).

    Args:
        seed: Integer seed value.
        deterministic_torch: If True and PyTorch is installed, configures
            backends for maximum reproducibility.

    Returns:
        The seed integer applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # torch.mps.manual_seed is available in recent PyTorch versions
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
