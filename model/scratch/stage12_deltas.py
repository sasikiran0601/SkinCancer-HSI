"""Stage 12: calculate all changes relative to the clean CE baseline."""
from .stage6_17_controlled_experiments import _read_summaries, _reports

if __name__ == "__main__":
    _reports(_read_summaries())
