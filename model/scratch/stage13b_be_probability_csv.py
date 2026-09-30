"""Stage 13b: emit true-BE probability CSVs (no thresholds or test data)."""
from .stage6_17_controlled_experiments import _probability_report, _read_summaries

if __name__ == "__main__":
    _probability_report(_read_summaries()[:4])
