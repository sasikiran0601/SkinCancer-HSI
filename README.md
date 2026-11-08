# Hyperspectral Skin Cancer Classification

This repository contains a research pipeline for classifying skin-lesion
hyperspectral images (HSI) into four classes:

| Code | Class |
| --- | --- |
| BE | Benign epithelial |
| BM | Benign melanocytic |
| ME | Malignant epithelial |
| MM | Malignant melanocytic |

The project is an executable research prototype. It is not clinically
validated and its outputs must not be used as a standalone medical diagnosis.

## Current results and limitations

The available dataset contains 76 captures from 61 patients. The four-class
distribution is approximately BE 7.9%, BM 44.7%, ME 31.6%, and MM 15.8%.
Schema and label validation pass for all 76 captures, and patient-independent
five-fold splits contain no patient overlap.

Controlled experiments currently achieve approximately 0.49-0.51 macro-F1 and
0.75-0.78 binary macro-F1. BM and ME are more reliable than the minority BE
and MM classes. The dataset validation report contains calibration warnings for
56 captures. Pixel-level evaluation expands each 50x50 capture into 2,500
vectors, but those vectors are not independent patients or lesions.

These limitations mean that code improvements and internal cross-validation
do not replace additional patients, minority-class samples, calibration review,
or external validation.

## Pipeline

```text
125-band calibrated HSI cube
  -> validation and non-finite value repair
  -> remove 4 leading and 5 trailing bands
  -> 11-band moving-average smoothing
  -> per-pixel or per-band normalization
  -> patient-independent splits
  -> PyTorch training and evaluation
  -> ONNX export
  -> Streamlit or Raspberry Pi inference
```

## Repository layout

```text
dataset.py                         Root dataset loader
preprocessing/hsi_preprocessing/   Preprocessing library
preprocessing/scripts/             Validation, splitting, and pipeline commands
preprocessing/tests/               Preprocessing unit tests
model/src/                         Models, training, evaluation, and CLI
model/configs/                     Training configuration files
model/rpi_deploy/                  Raspberry Pi conversion and inference tools
deployment/streamlit/              Streamlit ONNX application
preprocessing/dataset_schema.md    Dataset format and label definitions
model/rpi_deploy/README_RPi4.md    Raspberry Pi deployment guide
```

Generated reports, experiment outputs, presentation results, and model
artifacts are kept separately from the source modules.

## Requirements

Use Python 3.10 or newer where supported by the installed PyTorch and
scientific Python packages. Training requires a CUDA-capable PyTorch
installation. Raspberry Pi inference does not require PyTorch.

Install preprocessing dependencies:

```powershell
cd preprocessing
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install model dependencies in a second environment:

```powershell
cd ..\model
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Preprocessing

The raw dataset is expected under `extracted_dataset/npy_arrays`, with one
directory per capture. Run the split, preprocessing, label, and schema
commands from `preprocessing`:

```powershell
python scripts\run_splitting.py
python scripts\run_full_pipeline.py
python scripts\run_label_checks.py
python scripts\run_dataset_validation.py
```

The pipeline writes processed cubes, metadata, splits, reports, and plots under
`preprocessing/outputs`. Processed per-pixel cubes have shape `(116, 50, 50)`.

## Training and evaluation

From `model` with the model environment active:

```powershell
python ..\dataset.py
python -m src.cli train --model self_attention --epochs 100
python -m src.cli kfold --model self_attention --folds 5 --epochs 30
python evaluate_test.py
```

Training configuration is in `model/configs/default.yaml`. The default
configuration uses the patient-independent splits, multi-class labels, and
per-pixel normalization.

The current evaluation evidence is stored under
`model/scratch/experiments` and `model/scratch/baseline_and_audit_results.json`.
Treat these as research artifacts, not clinical validation.

## ONNX export

Export a trained Self-Attention CNN checkpoint:

```powershell
python -m src.cli export `
  --model outputs\best_model.pth `
  --arch self_attention `
  --output outputs\model.onnx
```

The repository includes ONNX artifacts at
`model/outputs/champion_model.onnx` and
`deployment/streamlit/champion_model.onnx`.

## Raspberry Pi deployment

The Raspberry Pi workflow accepts a calibrated 125-band MATLAB or NumPy cube,
converts it to NumPy when necessary, applies the training preprocessing, and
runs the ONNX model:

```bash
python convert_mat_to_npy.py --input capture.mat --output raw.npy
python preprocess_raw.py --input raw.npy --output processed.npy
python rpi_inference.py --model model.onnx --image processed.npy --output result.json
```

See [model/rpi_deploy/README_RPi4.md](model/rpi_deploy/README_RPi4.md) for
installation, file transfer, batch processing, and optional GPIO instructions.
An uncalibrated `hsCube` requires matching white and dark reference data and
must not be passed directly to the model.

## Streamlit application

Install the deployment dependencies and start the application from the
repository root:

```powershell
pip install -r deployment\streamlit\requirements.txt
streamlit run deployment\streamlit\streamlit_app.py
```

The application accepts a processed 116-band cube and displays pixel-wise
model outputs, class composition, and inference latency. It is a prototype
viewer, not a clinical decision system.

## Testing

Run preprocessing tests from the `preprocessing` directory:

```powershell
python -m pytest tests -q
```

If pytest is not installed, install the preprocessing requirements first.

## Research and deployment boundaries

- Results are based on a small, single-source dataset.
- Minority-class performance is unstable.
- Calibration warnings require investigation.
- Pixel-level counts do not represent independent clinical samples.
- No external clinical test set is included.
- Deployment outputs should be reviewed by qualified professionals and should
  not be interpreted as a diagnosis.
