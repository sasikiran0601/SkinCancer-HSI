"""Stages 15/16: verify validation predictions and generate BE analysis."""
from .stage6_17_controlled_experiments import _probability_report, _read_summaries

if __name__ == "__main__":
    _probability_report(_read_summaries()[:4])
