"""
Shared pytest configuration and fixture exports for the test suite.
"""

import sys
from pathlib import Path

# Make sure the preprocessing package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Re-export all fixtures from synthetic_data so they're auto-discovered
from tests.fixtures.synthetic_data import (  # noqa: F401
    cube_small,
    cube_full_bands,
    pixel_mono,
    pixel_const,
    signal_1d,
    calibration_inputs,
    cube_with_nans,
    inf_calibration_case,
)
