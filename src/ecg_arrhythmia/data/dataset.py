"""PyTorch Dataset abstraction for ECG beats (Active in Milestone 2/6)."""

from typing import Any, Callable, Optional, Sequence
import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError:
    # Fallback dummy class if PyTorch is not available
    class Dataset:  # type: ignore
        pass


class ECGBeatDataset(Dataset):
    """PyTorch Dataset container for segmented ECG beat waveforms and labels."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        record_ids: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Args:
            signals: Array of shape (N, samples) or (N, channels, samples)
            labels: Array of shape (N,) integer class indices
            record_ids: Optional array of originating patient/record ID per beat
            transform: Optional transform applied on beat signal
        """
        assert len(signals) == len(labels), "Signals and labels must have matching lengths"
        self.signals = signals
        self.labels = labels
        self.record_ids = record_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Any:
        signal = self.signals[idx]
        label = self.labels[idx]

        if self.transform is not None:
            signal = self.transform(signal)

        return signal, label
