"""Stage 13: analyze BE probabilities using validation predictions only."""
from .stage6_17_controlled_experiments import _probability_report, _read_summaries

if __name__ == "__main__":
    _probability_report(_read_summaries()[:4])
