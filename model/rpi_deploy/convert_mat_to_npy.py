"""Convert one dataset MATLAB capture into a calibrated NumPy cube."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def find_calibrated_cube(mat_path: Path) -> np.ndarray:
    candidates = []

    def collect(name: str, value: h5py.Dataset) -> None:
        if "calibrated" in name.lower():
            candidates.append(np.asarray(value))

    with h5py.File(mat_path, "r") as mat_file:
        mat_file.visititems(
            lambda name, value: collect(name, value)
            if isinstance(value, h5py.Dataset)
            else None
        )

    if not candidates:
        raise KeyError("No calibrated cube variable was found in the MAT file.")

    cube = np.asarray(candidates[0], dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D calibrated cube, received {cube.shape}")
    if cube.shape[0] == 125:
        return cube
    if cube.shape[-1] == 125:
        return np.moveaxis(cube, -1, 0)
    raise ValueError(f"Expected 125 spectral bands, received {cube.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert calibrated MAT to NPY.")
    parser.add_argument("--input", required=True, help="Input .mat file")
    parser.add_argument("--output", required=True, help="Output calibrated .npy file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cube = find_calibrated_cube(input_path)
    np.save(args.output, cube)
    print(f"Saved {args.output}: shape={cube.shape}, dtype={cube.dtype}")


if __name__ == "__main__":
    main()
