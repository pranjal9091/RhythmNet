"""
Multiclass Temperature Scaling module for probability calibration.

Temperature scaling optimizes a single scalar parameter T > 0 on validation set logits
by minimizing Negative Log-Likelihood (NLL / CrossEntropyLoss).
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Union
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecg_arrhythmia.utils import get_logger

logger = get_logger("temperature_scaling")


class ModelWithTemperature(nn.Module):
    """
    PyTorch Module wrapper that scales raw output logits by a temperature parameter T > 0.
    
    Parameters
    ----------
    model : nn.Module
        Underlying PyTorch model (e.g. ECG1DCNN).
    temperature : float, default=1.0
        Initial temperature value T.
    """

    def __init__(self, model: nn.Module, temperature: float = 1.0):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        # Store log_temperature parameter to guarantee T = exp(log_T) > 0 unconditionally
        self.log_temperature = nn.Parameter(torch.tensor([float(torch.log(torch.tensor(temperature)))]))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def set_temperature(self, temp: float) -> None:
        if temp <= 0:
            raise ValueError(f"Temperature must be strictly positive, got {temp}")
        with torch.no_grad():
            self.log_temperature.copy_(torch.tensor([float(torch.log(torch.tensor(temp)))]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning scaled logits: z / T.
        """
        logits = self.model(x)
        return logits / torch.exp(self.log_temperature)


def fit_temperature_scaling(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_iter: int = 50,
) -> Tuple[float, float, float]:
    """
    Fits temperature scaling parameter T > 0 strictly on validation set logits.
    
    Parameters
    ----------
    model : nn.Module
        Trained model instance.
    val_loader : DataLoader
        DataLoader for validation set ONLY.
    device : torch.device
        Compute device.
    max_iter : int, default=50
        Maximum optimization iterations for L-BFGS.
        
    Returns
    -------
    Tuple[float, float, float]
        (optimal_temperature, val_nll_before, val_nll_after)
    """
    model.eval()
    model.to(device)

    # Collect all validation logits and labels
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            waveforms = batch["waveform"].to(device)
            labels = batch["label"].to(device)
            logits = model(waveforms)
            logits_list.append(logits)
            labels_list.append(labels)

    val_logits = torch.cat(logits_list, dim=0)
    val_labels = torch.cat(labels_list, dim=0)

    criterion = nn.CrossEntropyLoss()
    val_nll_before = float(criterion(val_logits, val_labels).item())

    # Set up parameter T = exp(log_T)
    log_temp_param = nn.Parameter(torch.zeros(1, device=device))  # log(1.0) = 0
    optimizer = torch.optim.LBFGS([log_temp_param], lr=0.01, max_iter=max_iter)

    def eval_loss():
        optimizer.zero_grad()
        temp = torch.exp(log_temp_param)
        loss = criterion(val_logits / temp, val_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    optimal_temp = float(torch.exp(log_temp_param).item())
    val_nll_after = float(criterion(val_logits / torch.exp(log_temp_param), val_labels).item())

    logger.info(
        f"Fitted Temperature T = {optimal_temp:.4f} | "
        f"Val NLL Before: {val_nll_before:.4f} -> After: {val_nll_after:.4f}"
    )

    return optimal_temp, val_nll_before, val_nll_after


def save_temperature_artifact(
    model_name: str,
    temperature: float,
    val_nll_before: float,
    val_nll_after: float,
    output_path: Union[str, Path],
    n_bins: int = 10,
    seed: int = 42,
) -> None:
    """Saves temperature calibration metadata artifact as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "model": model_name,
        "temperature": float(temperature),
        "fitting_split": "validation",
        "objective": "negative_log_likelihood",
        "val_nll_before": float(val_nll_before),
        "val_nll_after": float(val_nll_after),
        "n_bins": int(n_bins),
        "seed": int(seed),
    }
    
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)


def load_temperature_artifact(input_path: Union[str, Path]) -> float:
    """Loads temperature value from saved JSON artifact."""
    input_path = Path(input_path)
    with open(input_path, "r") as f:
        artifact = json.load(f)
    return float(artifact["temperature"])
