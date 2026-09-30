"""Compatibility entry point for stage 13 BE probability analysis."""
from .stage13_probability_analysis import _probability_report, _read_summaries

if __name__ == "__main__":
    _probability_report(_read_summaries()[:4])
