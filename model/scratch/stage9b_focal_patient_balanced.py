"""Stage 9b: focal loss (gamma 1.5) plus patient-balanced sampling."""
from ._stage_helpers import run_named

if __name__ == "__main__":
    run_named("exp4_focal_patient_balanced")
