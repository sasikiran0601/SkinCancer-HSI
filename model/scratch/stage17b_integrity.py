"""Stage 17b: verify patient-disjoint folds and record the locked-fold hash."""
from ._stage_helpers import ROOT, verify_locked_folds, write_json

if __name__ == "__main__":
    write_json(ROOT / "stage17b_patient_integrity.json", verify_locked_folds())
