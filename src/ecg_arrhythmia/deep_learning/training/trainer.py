"""
PyTorch training and validation loop module with Early Stopping and Checkpointing.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from ecg_arrhythmia.utils import get_logger

logger = get_logger("deep_learning_trainer")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Executes one training epoch.
    
    Returns
    -------
    Tuple[float, float]
        (mean_train_loss, train_accuracy)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        waveforms = batch["waveform"].to(device)  # [B, 1, 216]
        labels = batch["label"].to(device)        # [B]

        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    total_samples = len(all_labels)
    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(all_labels, all_preds)

    return float(epoch_loss), float(epoch_acc)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Evaluates model performance on validation set.
    
    Returns
    -------
    Tuple[float, float, float, float]
        (val_loss, val_acc, val_balanced_acc, val_macro_f1)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        waveforms = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        logits = model(waveforms)
        loss = criterion(logits, labels)

        running_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    total_samples = len(all_labels)
    val_loss = running_loss / total_samples
    val_acc = accuracy_score(all_labels, all_preds)
    val_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return float(val_loss), float(val_acc), float(val_bal_acc), float(val_macro_f1)


def train_deep_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
    criterion: Optional[nn.Module] = None,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, Any]]:
    """
    Main training routine with ReduceLROnPlateau, Early Stopping on val_macro_f1,
    and checkpoint saving.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch neural network model.
    train_loader : DataLoader
        DataLoader for training set.
    val_loader : DataLoader
        DataLoader for validation set.
    config : Dict[str, Any]
        Configuration parameters dictionary.
    device : torch.device
        Compute device.
    checkpoint_path : Path
        Target filepath for saving the best model checkpoint (.pt).
    criterion : Optional[nn.Module], default=None
        Custom loss module (e.g., weighted CrossEntropyLoss, FocalLoss).
        If None, defaults to nn.CrossEntropyLoss().
        
    Returns
    -------
    Tuple[nn.Module, pd.DataFrame, Dict[str, Any]]
        (best_model, history_df, run_summary_dict)
    """
    model = model.to(device)
    
    cfg_train = config.get("training", {})
    cfg_sched = config.get("scheduler", {})

    lr = cfg_train.get("learning_rate", 1e-3)
    weight_decay = cfg_train.get("weight_decay", 1e-4)
    max_epochs = cfg_train.get("max_epochs", 40)
    patience_early = cfg_train.get("early_stopping_patience", 7)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg_sched.get("factor", 0.5),
        patience=cfg_sched.get("patience", 3),
    )

    best_val_macro_f1 = -1.0
    best_epoch = -1
    no_improve_count = 0

    history = []
    start_total_time = time.time()

    logger.info(f"Starting training for max {max_epochs} epochs on device: {device}")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_bal_acc, val_macro_f1 = evaluate_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_macro_f1)

        epoch_duration = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_balanced_acc": val_bal_acc,
            "val_macro_f1": val_macro_f1,
            "lr": current_lr,
            "epoch_duration_sec": epoch_duration,
        })

        logger.info(
            f"Epoch {epoch:02d}/{max_epochs:02d} [{epoch_duration:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, BalAcc: {val_bal_acc:.4f}, Macro-F1: {val_macro_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Checkpoint if validation Macro-F1 improved
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            no_improve_count = 0

            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_validation_macro_f1": best_val_macro_f1,
                "config": config,
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"--> Saved best checkpoint to {checkpoint_path} (val_macro_f1 = {best_val_macro_f1:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience_early:
                logger.info(f"Early stopping triggered after {patience_early} epochs without val_macro_f1 improvement.")
                break

    total_train_time = time.time() - start_total_time
    history_df = pd.DataFrame(history)

    # Reload best model weights
    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    avg_epoch_time = float(history_df["epoch_duration_sec"].mean())
    run_summary = {
        "total_epochs": len(history_df),
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_macro_f1),
        "total_train_time_sec": float(total_train_time),
        "avg_epoch_time_sec": avg_epoch_time,
    }

    return model, history_df, run_summary
