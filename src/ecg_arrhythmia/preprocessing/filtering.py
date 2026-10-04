"""ECG signal filtering, lead selection, and sampling frequency validation."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import butter, filtfilt
import wfdb

from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


def select_ecg_lead(
    record: wfdb.Record,
    primary_lead: str = "MLII",
    fallback_leads: Optional[List[str]] = None,
) -> Tuple[np.ndarray, str, int]:
    """Select the specified ECG channel from a WFDB record cleanly.

    Args:
        record: Loaded wfdb.Record object.
        primary_lead: Target primary lead name (default: 'MLII').
        fallback_leads: Ordered list of fallback lead names if primary is missing.

    Returns:
        Tuple of (signal_1d, selected_lead_name, channel_index)

    Raises:
        ValueError: If neither primary nor fallback leads are present in the record.
    """
    if fallback_leads is None:
        fallback_leads = ["V5", "V1", "V2", "V4"]

    channels = [c.upper() for c in record.sig_name]

    # Check primary lead first
    if primary_lead.upper() in channels:
        idx = channels.index(primary_lead.upper())
        selected = record.sig_name[idx]
        return record.p_signal[:, idx], selected, idx

    # Check fallback leads sequentially
    for fb in fallback_leads:
        if fb.upper() in channels:
            idx = channels.index(fb.upper())
            selected = record.sig_name[idx]
            logger.info(
                f"Record {record.record_name}: Primary lead '{primary_lead}' unavailable. "
                f"Using configured fallback lead '{selected}' (channel {idx})."
            )
            return record.p_signal[:, idx], selected, idx

    raise ValueError(
        f"Record {record.record_name} has channels {record.sig_name}, but neither primary lead "
        f"'{primary_lead}' nor fallbacks {fallback_leads} were found."
    )


def validate_sampling_rate(record: wfdb.Record, expected_fs: int = 360) -> int:
    """Validate that the record sampling frequency matches expectations.

    Args:
        record: Loaded wfdb.Record object.
        expected_fs: Expected sampling frequency in Hz.

    Returns:
        The verified sampling frequency.

    Raises:
        ValueError: If sampling rate does not match expected_fs.
    """
    fs = int(record.fs)
    if fs != expected_fs:
        raise ValueError(
            f"Record {record.record_name} sampling rate is {fs} Hz, expected {expected_fs} Hz."
        )
    return fs


def filter_ecg(
    signal: np.ndarray,
    fs: int = 360,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter to 1D ECG signal.

    A 0.5 - 40 Hz bandpass filter effectively removes low-frequency baseline wander
    (respiration/motion artifacts < 0.5 Hz) and high-frequency noise (EMG/powerline > 40 Hz),
    while preserving QRS complex morphology and P/T wave dynamics.

    Args:
        signal: 1D numpy array of raw ECG amplitude values.
        fs: Sampling frequency in Hz.
        low_hz: High-pass cut-off frequency in Hz.
        high_hz: Low-pass cut-off frequency in Hz.
        order: Filter order (Butterworth).
        zero_phase: If True, uses scipy.signal.filtfilt for zero phase shift.

    Returns:
        Filtered 1D numpy array of same length as input signal.
    """
    if signal.ndim != 1:
        raise ValueError(f"Signal must be 1-dimensional array, got shape {signal.shape}")

    if len(signal) == 0:
        return np.array([], dtype=np.float32)

    nyquist = 0.5 * fs
    low = low_hz / nyquist
    high = high_hz / nyquist

    if low <= 0 or high >= 1.0 or low >= high:
        raise ValueError(
            f"Invalid cutoff frequencies ({low_hz} Hz, {high_hz} Hz) for sampling rate {fs} Hz."
        )

    b, a = butter(order, [low, high], btype="bandpass")

    if zero_phase:
        # filtfilt applies filter forward and backward to achieve 0 phase distortion
        filtered = filtfilt(b, a, signal)
    else:
        from scipy.signal import lfilter
        filtered = lfilter(b, a, signal)

    return filtered.astype(np.float32)
