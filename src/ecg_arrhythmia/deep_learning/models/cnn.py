"""
1D Convolutional Neural Network (CNN) baseline model for raw ECG beat classification.
"""

import torch
import torch.nn as nn


class ECG1DCNN(nn.Module):
    """
    1D CNN for 216-sample single-channel ECG heartbeat classification into 5 AAMI classes.
    
    Parameters
    ----------
    in_channels : int, default=1
        Number of input channels (1 for single-lead ECG).
    num_classes : int, default=5
        Number of output classes (N, S, V, F, Q).
    dropout_rates : tuple of float, default=(0.15, 0.20, 0.30)
        Dropout rates for Block 1, Block 2, and Classifier layer.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        dropout_rates: tuple = (0.15, 0.20, 0.30),
    ):
        super(ECG1DCNN, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=7, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rates[0]),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_rates[1]),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rates[2]),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input waveform tensor of shape (batch, in_channels, 216).
            
        Returns
        -------
        torch.Tensor
            Unnormalized logits of shape (batch, num_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, 1)  # [batch, 128]
        logits = self.classifier(x)
        return logits
