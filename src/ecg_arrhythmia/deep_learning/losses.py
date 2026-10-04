"""
Loss functions module for Class Imbalance Ablation.

Includes standard CrossEntropyLoss and FocalLoss implementations with support
for class-weighting and probability focal modulation.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss implementation.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter modulating the loss for easy vs hard examples.
    weight : Optional[torch.Tensor], default=None
        Optional class weight tensor of shape (K,).
    reduction : str, default='mean'
        Specifies the reduction to apply to the output ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        logits : torch.Tensor
            Unnormalized logits of shape (N, K).
        targets : torch.Tensor
            Ground truth integer class labels of shape (N,).
            
        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        # Log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Gather target probabilities p_t
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal factor (1 - p_t)^gamma
        focal_factor = (1.0 - pt) ** self.gamma
        focal_loss = -focal_factor * log_pt

        # Apply class weights if provided
        if self.weight is not None:
            weights = self.weight.to(logits.device)
            alpha_t = weights.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
