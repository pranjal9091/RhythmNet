# Data Directory Structure

This directory stores raw and processed ECG records and partition manifests.

```
data/
├── raw/         <- Unprocessed MIT-BIH Arrhythmia Database files (.dat, .hea, .atr)
├── processed/   <- Filtered, segmented beats (beats.npy) and beat metadata (metadata.csv)
│   ├── train/   <- 41,262 beats from 17 DS1 records
│   ├── val/     <- 9,744 beats from 5 DS1 records
│   └── test/    <- 49,694 beats from 22 DS2 records (de Chazal benchmark)
└── manifests/   <- Patient-independent partition lists (train, val, test record IDs)
```

## Downloading Dataset
To download the MIT-BIH Arrhythmia Database from PhysioNet via `wfdb`:
```bash
python scripts/download_data.py --all
```
This downloads the 48 half-hour ambulatory ECG recordings into `data/raw/mitdb/`.

## Preprocessing & Beat Dataset Construction
To execute signal filtering, annotation-centered beat segmentation, and per-beat z-score normalization:
```bash
python scripts/preprocess.py
```

### Preprocessing Specifications
- **ECG Lead Selection**: Primary `MLII` with configured fallback (`V5`, `V1`, `V2`, `V4`).
- **Bandpass Filter**: Zero-phase 4th-order Butterworth filter ($0.5\text{--}40\text{ Hz}$).
- **Beat Window**: 216 samples ($72$ samples / $200\text{ ms}$ prior to R-peak, $144$ samples / $400\text{ ms}$ post R-peak @ $360\text{ Hz}$).
- **Normalization**: Independent per-beat z-score normalization ($(\mathbf{x} - \mu) / (\sigma + \epsilon)$).

## Data Governance & Leakage Prevention
- **Never commit data to version control.** Raw and processed records are ignored in `.gitignore`.
- All record-to-split assignments are governed deterministically by `data/manifests/`.
- Training, validation, and testing partitions are strictly disjoint at the **patient / record level**.
- **Zero Leakage**: Train, Validation, and Test partitions share 0 record IDs.
