"""Data access layer for downloading, locating, and reading MIT-BIH records via WFDB."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import wfdb

from ecg_arrhythmia.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RecordMetadata:
    """Structured container for ECG record metadata."""

    record_id: str
    sampling_rate: int
    n_channels: int
    n_samples: int
    duration_sec: float
    channels: Tuple[str, ...]
    units: Tuple[str, ...]
    comments: Tuple[str, ...]
    has_annotations: bool


class MITBIHLoader:
    """Loader and manager for MIT-BIH Arrhythmia Database records."""

    PHYSIONET_DB = "mitdb"

    def __init__(self, raw_data_dir: Union[str, Path]):
        self.raw_data_dir = Path(raw_data_dir).resolve()
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def is_record_present(self, record_id: str) -> bool:
        """Check if header (.hea), signal (.dat), and annotation (.atr) files exist locally."""
        hea = self.raw_data_dir / f"{record_id}.hea"
        dat = self.raw_data_dir / f"{record_id}.dat"
        atr = self.raw_data_dir / f"{record_id}.atr"
        return hea.is_file() and dat.is_file() and atr.is_file()

    def download_record(self, record_id: str, overwrite: bool = False) -> None:
        """Download a single record from PhysioNet mitdb into raw_data_dir if not present."""
        if self.is_record_present(record_id) and not overwrite:
            logger.debug(f"Record {record_id} already exists locally at {self.raw_data_dir}")
            return

        logger.info(f"Downloading record {record_id} from PhysioNet '{self.PHYSIONET_DB}'...")
        wfdb.dl_files(
            db=self.PHYSIONET_DB,
            dl_dir=str(self.raw_data_dir),
            files=[f"{record_id}.hea", f"{record_id}.dat", f"{record_id}.atr"],
            keep_subdirs=False,
            overwrite=overwrite,
        )

    def download_all_records(
        self, record_ids: Optional[List[str]] = None, overwrite: bool = False
    ) -> List[str]:
        """Download specified or all standard MIT-BIH records.

        Returns:
            List of successfully available record IDs.
        """
        if record_ids is None:
            # Query PhysioNet directory listing if available or fall back to standard 48 records
            try:
                record_ids = wfdb.get_record_list(self.PHYSIONET_DB)
            except Exception as e:
                logger.warning(f"Could not fetch online record list: {e}. Using standard 48 records.")
                from ecg_arrhythmia.utils.config import load_config
                cfg = load_config("configs/data.yaml")
                record_ids = cfg.get("records", {}).get("all", [])

        downloaded = []
        for rid in record_ids:
            try:
                self.download_record(str(rid), overwrite=overwrite)
                downloaded.append(str(rid))
            except Exception as exc:
                logger.error(f"Failed to download record {rid}: {exc}")

        return downloaded

    def load_record(
        self, record_id: str, sampfrom: int = 0, sampto: Optional[int] = None
    ) -> wfdb.Record:
        """Load an ECG record using wfdb.rdrecord()."""
        rec_path = str(self.raw_data_dir / str(record_id))
        try:
            record = wfdb.rdrecord(rec_path, sampfrom=sampfrom, sampto=sampto, physical=True)
            return record
        except FileNotFoundError:
            logger.info(f"Local files missing for record {record_id}. Attempting on-demand download...")
            self.download_record(str(record_id))
            return wfdb.rdrecord(rec_path, sampfrom=sampfrom, sampto=sampto, physical=True)

    def load_annotations(
        self, record_id: str, sampfrom: int = 0, sampto: Optional[int] = None
    ) -> wfdb.Annotation:
        """Load annotation file for an ECG record using wfdb.rdann()."""
        rec_path = str(self.raw_data_dir / str(record_id))
        try:
            ann = wfdb.rdann(rec_path, extension="atr", sampfrom=sampfrom, sampto=sampto)
            return ann
        except FileNotFoundError:
            logger.info(f"Local annotations missing for record {record_id}. Attempting on-demand download...")
            self.download_record(str(record_id))
            return wfdb.rdann(rec_path, extension="atr", sampfrom=sampfrom, sampto=sampto)

    def get_record_metadata(self, record_id: str) -> RecordMetadata:
        """Inspect and return typed metadata for a record using its header."""
        rec_path = str(self.raw_data_dir / str(record_id))
        try:
            header = wfdb.rdheader(rec_path)
        except FileNotFoundError:
            logger.info(f"Local header missing for record {record_id}. Downloading record...")
            self.download_record(str(record_id))
            header = wfdb.rdheader(rec_path)

        atr_file = self.raw_data_dir / f"{record_id}.atr"
        has_annotations = atr_file.is_file() and atr_file.stat().st_size > 0

        n_samples = header.sig_len
        fs = header.fs
        duration_sec = float(n_samples) / float(fs) if fs else 0.0

        return RecordMetadata(
            record_id=str(record_id),
            sampling_rate=int(fs),
            n_channels=int(header.n_sig),
            n_samples=int(n_samples),
            duration_sec=duration_sec,
            channels=tuple(header.sig_name or ()),
            units=tuple(header.units or ()),
            comments=tuple(header.comments or ()),
            has_annotations=has_annotations,
        )

    def list_local_records(self) -> List[str]:
        """Discover and return all complete record IDs present in raw_data_dir."""
        hea_files = list(self.raw_data_dir.glob("*.hea"))
        records = []
        for h in sorted(hea_files):
            rid = h.stem
            if self.is_record_present(rid):
                records.append(rid)
        return records
