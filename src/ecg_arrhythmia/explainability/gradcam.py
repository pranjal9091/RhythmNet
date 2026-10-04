"""
1D Grad-CAM (Gradient-weighted Class Activation Mapping) explainability module.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM1D:
    """
    1D Grad-CAM implementation targeting the final convolutional layer of a 1D CNN.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch 1D CNN model.
    target_layer : Optional[nn.Module], default=None
        Target convolutional layer. If None, targets model.block3[3] or model.block3.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "block3"):
                # Target the second Conv1d in Block 3
                if isinstance(model.block3, nn.Sequential) and len(model.block3) >= 4:
                    self.target_layer = model.block3[3]
                else:
                    self.target_layer = model.block3
            else:
                raise ValueError("Target layer could not be determined automatically.")
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        waveform_tensor: torch.Tensor,
        target_class: int,
        device: torch.device,
        target_length: int = 216,
    ) -> np.ndarray:
        """
        Generates 1D Grad-CAM heatmap interpolated to target_length.
        
        Parameters
        ----------
        waveform_tensor : torch.Tensor
            Input waveform tensor of shape (1, 216) or (1, 1, 216).
        target_class : int
            Target integer class index.
        device : torch.device
            Compute device.
        target_length : int, default=216
            Target 1D length to interpolate heatmap.
            
        Returns
        -------
        np.ndarray
            1D Grad-CAM heatmap array of shape (target_length,) scaled to [0, 1].
        """
        self.model.to(device)

        if waveform_tensor.dim() == 2:
            input_x = waveform_tensor.unsqueeze(0).to(device)
        elif waveform_tensor.dim() == 1:
            input_x = waveform_tensor.unsqueeze(0).unsqueeze(0).to(device)
        else:
            input_x = waveform_tensor.to(device)

        self.model.zero_grad()
        logits = self.model(input_x)
        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hook failed to capture activations or gradients.")

        # Channel weights w_k = mean gradient over spatial dimension
        weights = torch.mean(self.gradients, dim=2, keepdim=True)  # [1, C, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, L_conv]

        cam = F.relu(cam)

        # Interpolate to 216 samples
        cam_interp = F.interpolate(cam, size=target_length, mode="linear", align_corners=False)
        heatmap = cam_interp[0, 0].cpu().numpy()

        # Min-max normalize to [0, 1]
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap
