# Hyperspectral Skin Cancer — Dataset Schema

## Data Source
Leon et al. 2020 — *Non-Invasive Skin Cancer Diagnosis Using Hyperspectral Imaging*
DOI: https://doi.org/10.3390/jcm9061662

Zenodo dataset (Sensors 2022 version): 76 hyperspectral captures of 61 patients.

---

## Per-Sample File Structure (extracted NPY format)

Each sample lives in its own directory:

```
extracted_dataset/npy_arrays/{SAMPLE_ID}/
├── {SAMPLE_ID}__calibratedHsCube.npy    # calibrated reflectance cube
├── {SAMPLE_ID}__hsCube.npy              # raw (uncalibrated) cube
├── {SAMPLE_ID}__spectralRGB.npy         # RGB reconstruction
├── {SAMPLE_ID}__binaryLabel_rawcodes.npy
├── {SAMPLE_ID}__multiLabel_rawcodes.npy
└── {SAMPLE_ID}__labels.json            # decoded labels (human-readable)
```

---

## Variable Schemas

### `calibratedHsCube`
| Field | Value |
|-------|-------|
| Shape | `(125, 50, 50)` |
| Dtype | `float64` |
| Units | Pseudo-reflectance (dimensionless, roughly 0–1) |
| Bands | 125 spectral bands, ~450–950 nm, ≈4 nm/band |
| Spatial | 50×50 pixel image patch |
| Formula | `PI = (RI - DI) / (WI - DI)` (Eq. 1, Leon et al. 2020) |
| Expected range | [−0.2, 2.0] with most values in [0, 1] |

### `hsCube`
| Field | Value |
|-------|-------|
| Shape | `(125, 50, 50)` |
| Dtype | `float64` |
| Units | Raw sensor DN (dark-not-subtracted) |
| Notes | Used with white/dark references to compute `calibratedHsCube` |

### `spectralRGB`
| Field | Value |
|-------|-------|
| Shape | `(3, 50, 50)` |
| Dtype | `float64` |
| Channels | R, G, B (in that order) |
| Notes | Dataset's own RGB reconstruction; used for visual QC |

### `binaryLabel`
| Field | Value |
|-------|-------|
| Type | `str` |
| Valid values | `"B"` (Benign), `"M"` (Malignant) |

### `multiLabel`
| Field | Value |
|-------|-------|
| Type | `str` |
| Valid values | `"BE"` (Benign Epithelial), `"BM"` (Benign Melanocytic), `"ME"` (Malignant Epithelial), `"MM"` (Malignant Melanocytic) |

---

## Label Consistency Rule
```
B → {BE, BM}
M → {ME, MM}
```
Any other combination is a data-quality error.

---

## Filename Format
```
P{patient_id}_C{capture_id}
```
Example: `P13_C2000` = Patient 13, second capture.

- 76 images from 61 unique patients
- Multi-image patients: P13 (3), P27 (4), P29 (3), P60 (3), P86 (4)

---

## Preprocessing Pipeline Output

After running the full pipeline, each sample produces:

```
preprocessing/outputs/processed/per_pixel/{SAMPLE_ID}/
├── {SAMPLE_ID}__processed.npy   # shape (116, 50, 50), float64, values in [0,1]
└── {SAMPLE_ID}__meta.json       # {sample_id, binaryLabel, multiLabel, n_nan_inf_fixed, bands}
```

Band count after pipeline: **116** (trim first 4, last 5 from original 125).

---

## Paper-Reported Class Distribution (Table 1, Leon et al. 2020)
| Class | Description | Approx. fraction |
|-------|-------------|-----------------|
| BE | Benign Epithelial | 7% |
| BM | Benign Melanocytic | 45% |
| ME | Malignant Epithelial | 32% |
| MM | Malignant Melanocytic | 16% |

---

## Schema Validation Code
See [`hsi_preprocessing/validation.py`](../hsi_preprocessing/validation.py) →
`validate_mat_structure(sample_dir, sample_id)`.
