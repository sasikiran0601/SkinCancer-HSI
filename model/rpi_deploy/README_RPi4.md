# Raspberry Pi 4 Deployment Guide: Hyperspectral Skin Cancer Classification

This directory contains the files needed to run hyperspectral model inference on
a Raspberry Pi 4 Model B (ARM Cortex-A72).

---

## 1. System Architecture

```
[ PC / Workstation (Training) ]
        │
        ▼  PyTorch export (`python -m src.cli export ...`)
   `model.onnx`  (~354 KB)
        │
        ▼  Transfer via SCP / USB
[ Raspberry Pi 4 (Edge Hardware) ]
        │
        ├─► `rpi_inference.py` (ONNX Runtime ARM64 engine)
        ├─► Input: `sample_processed.npy` (116 spectral bands, 50x50 pixels)
        │
        ├──► Terminal: Colored 50x50 ASCII lesion map & statistics
        ├──► Storage : `result.json` (Structured model-output summary)
        └──► Hardware: (Optional) 4x GPIO LEDs indicating class / risk
```

---

## 2. Hardware Requirements

- **Device**: Raspberry Pi 4 Model B (2GB, 4GB, or 8GB RAM)
- **OS**: Raspberry Pi OS (64-bit recommended, Debian Bullseye or Bookworm)
- **Power**: Official 5V 3A USB-C Power Supply
- **Storage**: 16 GB+ MicroSD card
- **Optional Hardware**:
  - Breadboard & jumper wires
  - 4x LEDs (Blue, Green, Yellow, Red)
  - 4x 330Ω resistors

---

## 3. Step-by-Step Deployment Guide

### Step 3.1: Copy Deployment Files to Raspberry Pi

On your **PC** (PowerShell or Terminal), run `scp` to transfer the model and inference scripts to your Pi:

```powershell
# From your PC model directory:
scp model.onnx rpi_deploy/* pi@raspberrypi.local:~/skincancer/

# Also copy a sample HSI cube for testing:
scp ../preprocessing/outputs/processed/per_pixel/P100_C1000/P100_C1000__processed.npy pi@raspberrypi.local:~/skincancer/sample.npy
```

*(Replace `raspberrypi.local` with your Pi's actual IP address if needed, e.g. `192.168.1.100`)*

---

### Step 3.2: Set Up Environment on Raspberry Pi

SSH into your Raspberry Pi:

```bash
ssh pi@raspberrypi.local
cd ~/skincancer
```

Create a virtual environment and install dependencies:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install lightweight dependencies (NO PyTorch needed!)
pip install --upgrade pip
pip install -r requirements_rpi.txt
```

> **Note**: `onnxruntime` installs a pre-built wheel for ARM64 Linux that executes on all 4 Cortex-A72 CPU cores with optimized NEON SIMD instructions.

---

### Step 3.3: Run Inference on Hardware

Run inference with the default parameters:

```bash
python rpi_inference.py --model model.onnx --image sample.npy
```

### Step 3.3a: Demo with a raw calibrated cube

For a presentation demo, copy the original **calibrated** cube with 125 bands
and `preprocess_raw.py` to the Pi. The helper performs the same repair, band
trimming, smoothing, and per-pixel normalization used during training:

```bash
python preprocess_raw.py --input raw_calibrated.npy --output sample_processed.npy
python rpi_inference.py --model model.onnx --image sample_processed.npy --output result.json
```

The raw input must have shape `(125, H, W)`. An uncalibrated `hsCube` cannot be
processed correctly without its matching white and dark reference cubes.

For the supplied MATLAB 7.3 dataset ZIP archives, install the HDF5 reader on
the Pi and convert each MAT capture first:

```bash
pip install h5py
python convert_mat_to_npy.py --input mat_files/P100_C1000.mat \
  --output raw_inputs/P100_C1000.npy
python preprocess_raw.py --input raw_inputs/P100_C1000.npy \
  --output processed_inputs/P100_C1000.npy
```

#### Command-Line Arguments:
| Flag | Default | Description |
|---|---|---|
| `--model` | `model.onnx` | Path to the ONNX model file |
| `--image` | *(Required)* | Path to the `.npy` HSI cube |
| `--output` | `result.json` | Path where JSON diagnostics will be saved |
| `--batch_size` | `512` | Batch size for pixel processing |
| `--gpio` | `False` | Enable physical GPIO LEDs |

---

## 4. Output & Diagnostics

### Terminal Display
When executed, `rpi_inference.py` prints:
1. **Model & Hardware Initialization**: Number of CPU threads, load time.
2. **50x50 spatial ASCII map**:
   - `\033[94m#\033[0m` (Blue) : Benign Epidermal (BE)
   - `\033[92m#\033[0m` (Green): Benign Melanocytic (BM)
   - `\033[93m#\033[0m` (Yellow): Malignant Epidermal (ME)
   - `\033[91m#\033[0m` (Red)  : Malignant Melanoma (MM)
3. **Model-output summary**:
   - Class breakdown with percentages
   - Benign versus malignant model-class composition
   - Inference latency and throughput (pixels/second)

### JSON Output (`result.json`)
```json
{
  "sample": "sample.npy",
  "dominant_class": "BE",
  "dominant_name": "Benign Epidermal (BE)",
  "is_malignant": false,
  "benign_percent": 99.44,
  "malignant_percent": 0.56,
  "class_counts": {
    "BE": 2485,
    "BM": 1,
    "ME": 10,
    "MM": 4
  },
  "latency_ms": 326.17,
  "throughput_px_per_sec": 7664.7
}
```

---

## 5. (Optional) Hardware GPIO Indicator Setup

You can connect 4 status LEDs to the Raspberry Pi GPIO header to provide immediate physical feedback during patient screening:

| Diagnosis | LED Color | Raspberry Pi Physical Pin | BCM Pin | Description |
|---|---|---|---|---|
| **BE** | Blue | Pin 11 | GPIO 17 | Benign Epidermal |
| **BM** | Green | Pin 13 | GPIO 27 | Benign Melanocytic |
| **ME** | Yellow | Pin 15 | GPIO 22 | Malignant Epidermal |
| **MM** | Red | Pin 16 | GPIO 23 | Malignant Melanoma (High Risk) |
| **GND** | Ground | Pin 6, 9, 14, 20 | GND | Common Cathode |

### Wiring:
- Connect the **Anode (+)** of each LED through a **330Ω resistor** to the respective GPIO pin.
- Connect the **Cathode (-)** of all LEDs to any **GND** pin on the Raspberry Pi.

Enable GPIO in the script by adding the `--gpio` flag:
```bash
python rpi_inference.py --model model.onnx --image sample.npy --gpio
```

---

## 6. Performance notes

| Metric | Measured on RPi 4 (4 Cores @ 1.5 GHz) |
|---|---|
| **Model Size** | 354 KB (`model.onnx`) |
| **Memory Footprint (RAM)** | < 120 MB |
| **Full Cube (2,500 pixels) Latency** | Measure on the target Pi |
| **Per-Pixel Latency** | Depends on ONNX Runtime and thermal state |
| **Throughput** | Depends on ONNX Runtime and batch size |
| **Power Consumption** | Depends on the Pi power profile |

The measured presentation run on the available Pi processed 2,500 pixels in
approximately 5.6 seconds. Treat this as a local observation rather than a
hardware guarantee. The output is a model prediction and is not a standalone
medical diagnosis.
