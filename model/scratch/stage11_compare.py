"""Stage 11: create the controlled experiment comparison from persisted data."""
from .stage6_17_controlled_experiments import _read_summaries, _reports

if __name__ == "__main__":
    _reports(_read_summaries())
