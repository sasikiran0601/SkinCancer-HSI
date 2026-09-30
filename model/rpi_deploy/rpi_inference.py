"""
Raspberry Pi 4 — Hyperspectral Skin Cancer Inference
====================================================
Lightweight, standalone edge deployment script.
Runs directly on Raspberry Pi 4 (ARM64 / Cortex-A72) using ONNX Runtime and NumPy.
Does NOT require PyTorch.

Features:
  - Loads ONNX exported model (model.onnx, ~350 KB)
  - Processes .npy HSI cube (116 bands) in vectorized batches
  - Displays colorized ASCII classification map in the terminal
  - Outputs structured diagnostic summary to terminal and result.json
  - Optional Raspberry Pi GPIO LED indicators for each class (BE, BM, ME, MM)

Usage:
  python rpi_inference.py --model model.onnx --image sample.npy
  python rpi_inference.py --model model.onnx --image sample.npy --output result.json --gpio
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Try importing ONNX Runtime
try:
    import onnxruntime as ort  # type: ignore[import-not-found, import-untyped]
except ImportError:
    print("\n[ERROR] onnxruntime is not installed!")
    print("Please install via: pip install onnxruntime\n")
    sys.exit(1)

# Optional GPIO Support for physical hardware LEDs on Raspberry Pi
import importlib

GPIO_AVAILABLE = False
GPIO: Optional[Any] = None
try:
    GPIO = importlib.import_module("RPi.GPIO")
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False
    GPIO = None


CLASS_NAMES = ["BE", "BM", "ME", "MM"]
CLASS_FULL_NAMES = [
    "Benign Epidermal (BE)",
    "Benign Melanocytic (BM)",
    "Malignant Epidermal (ME)",
    "Malignant Melanoma (MM)",
]

# ANSI Terminal Colors for terminal map
# BE: Cyan/Blue (34), BM: Green (32), ME: Yellow/Orange (33), MM: Red (31)
ANSI_COLORS = {
    0: "\033[94m#\033[0m",  # Blue for BE
    1: "\033[92m#\033[0m",  # Green for BM
    2: "\033[93m#\033[0m",  # Yellow/Orange for ME
    3: "\033[91m#\033[0m",  # Red for MM
}

# Default GPIO BCM pin mapping (configurable via CLI)
# Wire LED anode (+ via 330 ohm resistor) to GPIO pin, cathode (-) to GND
DEFAULT_GPIO_PINS = {
    "BE": 17,  # Pin 11 - Blue LED
    "BM": 27,  # Pin 13 - Green LED
    "ME": 22,  # Pin 15 - Yellow LED
    "MM": 23,  # Pin 16 - Red LED (High Risk)
}


def parse_args():
    parser = argparse.ArgumentParser(description="RPi4 Hyperspectral Skin Cancer Inference")
    parser.add_argument("--model", type=str, default="model.onnx", help="Path to model.onnx")
    parser.add_argument("--image", type=str, default=None, help="Path to .npy HSI cube")
    parser.add_argument("--output", type=str, default="result.json", help="Path to output JSON")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--gpio", action="store_true", help="Enable GPIO LED indicators on RPi4")
    parser.add_argument("--show_map", action="store_true", default=True, help="Display ASCII map")

    # If run without arguments, display help and sample usage
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExample usage:")
        print("  python rpi_inference.py --model model.onnx --image sample.npy")
        print("  python rpi_inference.py --model model.onnx --image sample.npy --gpio\n")
        sys.exit(0)

    args = parser.parse_args()
    if not args.image:
        print("[ERROR] Argument --image is required.")
        sys.exit(1)
    return args


def load_hsi_cube(image_path: str) -> np.ndarray:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {image_path}")

    cube = np.load(image_path)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {cube.shape}")

    if cube.shape[0] == 116:
        cube = np.transpose(cube, (1, 2, 0))
    elif cube.shape[2] != 116:
        raise ValueError(f"Expected 116 bands, but cube has shape {cube.shape}")

    return cube.astype(np.float32)


def setup_gpio(pins: dict) -> bool:
    if not GPIO_AVAILABLE or GPIO is None:
        print("[GPIO] RPi.GPIO library not found or not running on Raspberry Pi. Skipping GPIO setup.")
        return False
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for class_name, pin in pins.items():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        return True
    except Exception as e:
        print(f"[GPIO] Setup warning: {e}")
        return False


def set_gpio_led(pins: dict, dominant_class: str) -> None:
    if not GPIO_AVAILABLE or GPIO is None:
        return
    try:
        for class_name, pin in pins.items():
            if class_name == dominant_class:
                GPIO.output(pin, GPIO.HIGH)
            else:
                GPIO.output(pin, GPIO.LOW)
    except Exception as e:
        print(f"[GPIO] Output warning: {e}")


def display_ascii_map(pred_map: np.ndarray) -> None:
    H, W = pred_map.shape
    print("\n" + "-" * 60)
    print("  ASCII Spatial Classification Map (50 x 50)")
    print("  Legend: \033[94m# BE\033[0m  \033[92m# BM\033[0m  \033[93m# ME\033[0m  \033[91m# MM\033[0m")
    print("-" * 60)
    for r in range(H):
        line = "".join([ANSI_COLORS.get(pred_map[r, c], " ") for c in range(W)])
        print("  " + line)
    print("-" * 60)


def main():
    args = parse_args()

    # Verify model exists
    if not Path(args.model).exists():
        print(f"[ERROR] ONNX model file '{args.model}' not found!")
        sys.exit(1)

    print("\n==============================================================")
    print("      RASPBERRY PI 4 HYPERSPECTRAL INFERENCE ENGINE           ")
    print("==============================================================")
    print(f"  Model       : {args.model}")
    print(f"  Input Cube  : {args.image}")

    # Optional GPIO init
    gpio_active = False
    if args.gpio:
        gpio_active = setup_gpio(DEFAULT_GPIO_PINS)
        if gpio_active:
            print("  GPIO Status : Active (LEDs mapped to BCM 17, 27, 22, 23)")

    # Load data
    t_load_start = time.perf_counter()
    cube = load_hsi_cube(args.image)
    t_load_ms = (time.perf_counter() - t_load_start) * 1000
    H, W, B = cube.shape
    total_pixels = H * W
    print(f"  Cube Loaded : {H}x{W} ({total_pixels:,} pixels), {B} bands in {t_load_ms:.1f} ms")

    # Load ONNX session
    t_init_start = time.perf_counter()
    # Optimizations for Raspberry Pi 4 CPU
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # 4 Cortex-A72 cores
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        args.model,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    t_init_ms = (time.perf_counter() - t_init_start) * 1000
    print(f"  ONNX Engine : Initialized with 4 CPU threads in {t_init_ms:.1f} ms")

    # Run inference
    pixels = cube.reshape(-1, B)
    batch_size = args.batch_size
    all_probs = []

    print(f"  Running inference on {total_pixels} pixels (batch size {batch_size})...")
    t_infer_start = time.perf_counter()
    for i in range(0, total_pixels, batch_size):
        batch = pixels[i : i + batch_size]
        raw_logits = session.run(None, {input_name: batch})[0]
        logits = np.asarray(raw_logits, dtype=np.float32)
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        all_probs.append(probs)

    t_infer_ms = (time.perf_counter() - t_infer_start) * 1000
    all_probs = np.vstack(all_probs)

    # Compute statistics
    preds = np.argmax(all_probs, axis=1)
    pred_map = preds.reshape(H, W)
    mean_probs = np.mean(all_probs, axis=0)

    class_counts = {CLASS_NAMES[c]: int(np.sum(preds == c)) for c in range(4)}
    class_percents = {CLASS_NAMES[c]: float(class_counts[CLASS_NAMES[c]] / total_pixels * 100) for c in range(4)}
    dominant_class = max(class_counts.keys(), key=lambda k: class_counts[k])
    dominant_idx = CLASS_NAMES.index(dominant_class)

    benign_pct = class_percents["BE"] + class_percents["BM"]
    malignant_pct = class_percents["ME"] + class_percents["MM"]
    is_malignant = malignant_pct > benign_pct

    # Display ASCII map
    if args.show_map:
        display_ascii_map(pred_map)

    # Console summary
    print("\n--------------------------------------------------------------")
    print("  DIAGNOSTIC RESULTS:")
    print("--------------------------------------------------------------")
    for c_idx, c_name in enumerate(CLASS_NAMES):
        bar = "#" * int(class_percents[c_name] // 2)
        print(f"    [{c_name}] {CLASS_FULL_NAMES[c_idx]:<25}: {class_counts[c_name]:>5} px ({class_percents[c_name]:>5.1f}%) | {bar}")
    print("--------------------------------------------------------------")
    print(f"  Dominant Diagnosis : {CLASS_FULL_NAMES[dominant_idx]} ({class_percents[dominant_class]:.1f}%)")
    print(f"  Benign Burden      : {benign_pct:.1f}%")
    print(f"  Malignant Burden   : {malignant_pct:.1f}%")
    if is_malignant:
        print("  Clinical Risk      : \033[91m[HIGH RISK] MALIGNANT LESION INDICATED\033[0m")
    else:
        print("  Clinical Risk      : \033[92m[LOW RISK] PREDOMINANTLY BENIGN TISSUE\033[0m")
    print("--------------------------------------------------------------")
    print(f"  Inference Latency  : {t_infer_ms:.2f} ms ({t_infer_ms / total_pixels * 1000:.3f} us/pixel)")
    print(f"  Throughput         : {total_pixels / (t_infer_ms / 1000):,.0f} pixels/second")
    print("==============================================================\n")

    # Update GPIO if enabled
    if gpio_active:
        set_gpio_led(DEFAULT_GPIO_PINS, dominant_class)
        print(f"[GPIO] Activated LED indicator for dominant class: {dominant_class}")

    # Save structured JSON
    result_data = {
        "sample": Path(args.image).name,
        "dominant_class": dominant_class,
        "dominant_name": CLASS_FULL_NAMES[dominant_idx],
        "is_malignant": bool(is_malignant),
        "benign_percent": round(benign_pct, 2),
        "malignant_percent": round(malignant_pct, 2),
        "class_counts": class_counts,
        "class_percentages": {k: round(v, 2) for k, v in class_percents.items()},
        "mean_probabilities": {CLASS_NAMES[i]: round(float(mean_probs[i]), 4) for i in range(4)},
        "latency_ms": round(t_infer_ms, 2),
        "throughput_px_per_sec": round(total_pixels / (t_infer_ms / 1000), 1),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    print(f"Result JSON saved to: {args.output}")


if __name__ == "__main__":
    main()
