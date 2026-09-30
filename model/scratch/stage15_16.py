"""Compatibility entry point for stages 15--16 prediction/report generation."""
from .stage15_16_predictions_reports import _probability_report, _read_summaries

if __name__ == "__main__":
    _probability_report(_read_summaries()[:4])
