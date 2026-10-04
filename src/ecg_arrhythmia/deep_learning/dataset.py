"""
PyTorch Dataset module for ECG beat waveforms.

Provides a clean PyTorch Dataset wrapper around preprocessed 216-sample ECG beats
and associated metadata for deep learning models.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_LABEL_MAPPING = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}


class ECGBeatDataset(Dataset):
    """
    PyTorch Dataset for single-channel 216-sample ECG beat waveforms.
    
    Parameters
    ----------
    beats : Union[np.ndarray, str, Path]
        Array of shape (N, 216) or filepath to beats.npy.
    metadata : Union[pd.DataFrame, str, Path]
        DataFrame or filepath to metadata.csv.
    label_mapping : Dict[str, int], default=DEFAULT_LABEL_MAPPING
        Mapping from string AAMI class label to integer class ID.
    """

    def __init__(
        self,
        beats: Union[np.ndarray, str, Path],
        metadata: Union[pd.DataFrame, str, Path],
        label_mapping: Dict[str, int] = DEFAULT_LABEL_MAPPING,
    ):
        if isinstance(beats, (str, Path)):
            beats_path = Path(beats)
            if not beats_path.exists():
                raise FileNotFoundError(f"Beats file not found: {beats_path}")
            self.beats = np.load(beats_path).astype(np.float32)
        else:
            self.beats = np.array(beats, dtype=np.float32)

        if isinstance(metadata, (str, Path)):
            meta_path = Path(metadata)
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            self.metadata_df = pd.read_csv(meta_path)
        else:
            self.metadata_df = metadata.copy()

        if len(self.beats) != len(self.metadata_df):
            raise ValueError(
                f"Mismatch between beats count ({len(self.beats)}) "
                f"and metadata rows ({len(self.metadata_df)})."
            )

        self.label_mapping = label_mapping
        labels_str = self.metadata_df["aami_class"].values
        self.labels = np.array([self.label_mapping[c] for c in labels_str], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.beats)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        waveform = torch.tensor(self.beats[idx], dtype=torch.float32).unsqueeze(0)  # [1, 216]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        row = self.metadata_df.iloc[idx]
        meta_dict = {
            "beat_id": row.get("beat_id", idx),
            "record_id": row.get("record_id", ""),
            "sample_index": row.get("sample_index", 0),
            "aami_class": row.get("aami_class", ""),
        }

        return {
            "waveform": waveform,
            "label": label,
            "metadata": meta_dict,
        }
