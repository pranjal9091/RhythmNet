"""MIT-BIH Annotation definitions and AAMI EC57 standard mapping foundation.

According to ANSI/AAMI EC57:1998(R)2008 standard:
The 5 major heartbeat classes are:
- N: Non-ectopic (Normal and bundle branch block beats)
- S: Supraventricular ectopic beats (SVEB)
- V: Ventricular ectopic beats (VEB)
- F: Fusion of ventricular and normal beats
- Q: Unknown, paced, or unclassifiable beats

Non-beat annotations (such as rhythm changes, signal quality comments, measurement markers)
must be separated from heartbeat annotations and not treated as cardiac beats.
"""

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

# Official AAMI EC57 Beat Class Mapping Dictionary
AAMI_MAPPING: Dict[str, str] = {
    # N Class (Normal / Bundle Branch Block / Escape)
    "N": "N",  # Normal beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat
    "B": "N",  # Left or right bundle branch block (rarely used symbol)

    # S Class (Supraventricular Ectopic Beats - SVEB)
    "A": "S",  # Atrial premature beat
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Supraventricular premature or ectopic beat (atrial or nodal)

    # V Class (Ventricular Ectopic Beats - VEB)
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat

    # F Class (Fusion of Ventricular and Normal)
    "F": "F",  # Fusion of ventricular and normal beat

    # Q Class (Paced / Unknown / Unclassifiable)
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
    "?": "Q",  # Beat not classified during learning
}

# Recognized non-beat annotations in MIT-BIH (auxiliary markers, rhythm annotations, noise)
NON_BEAT_SYMBOLS: Set[str] = {
    "[",  # Start of ventricular flutter/fibrillation
    "!",  # Ventricular flutter wave
    "]",  # End of ventricular flutter/fibrillation
    "x",  # Non-conducted P-wave (blocked APC)
    "(",  # Waveform onset
    ")",  # Waveform end
    "p",  # Peak of P-wave
    "t",  # Peak of T-wave
    "u",  # Peak of U-wave
    "`",  # PQ junction
    "'",  # J-point
    "^",  # Non-captured pacemaker artifact
    "|",  # Isolated QRS-like artifact
    "~",  # Change in signal quality
    "+",  # Rhythm change
    "s",  # ST segment change
    "T",  # T-wave change
    "*",  # Systole
    "D",  # Diastole
    "=",  # Measurement mark
    '"',  # Comment annotation
    "@",  # Link
}

# The 5 standard AAMI classes
AAMI_CLASSES: Tuple[str, ...] = ("N", "S", "V", "F", "Q")


def is_heartbeat(symbol: str) -> bool:
    """Check if an annotation symbol corresponds to a heartbeat (rather than rhythm/noise marker)."""
    return symbol in AAMI_MAPPING


def map_symbol_to_aami(symbol: str) -> Optional[str]:
    """Map a single MIT-BIH annotation symbol to its AAMI EC57 class.

    Returns:
        One of 'N', 'S', 'V', 'F', 'Q', or None if the symbol is a non-beat marker.
    """
    return AAMI_MAPPING.get(symbol, None)


def inspect_annotation_symbols(symbols: List[str]) -> Dict[str, int]:
    """Return a frequency count dictionary of raw annotation symbols."""
    return dict(Counter(symbols))


def summarize_aami_distribution(symbols: List[str]) -> Tuple[Dict[str, int], int]:
    """Summarize the distribution of symbols into AAMI classes and non-beat counts.

    Returns:
        (aami_counts_dict, non_beat_count)
    """
    aami_counts: Dict[str, int] = {cls: 0 for cls in AAMI_CLASSES}
    non_beat_count = 0

    for sym in symbols:
        mapped = map_symbol_to_aami(sym)
        if mapped in aami_counts:
            aami_counts[mapped] += 1
        else:
            non_beat_count += 1

    return aami_counts, non_beat_count
