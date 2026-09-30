"""Stage 10b: persist mean, standard deviation, min and max aggregates."""
from ._stage_helpers import ROOT, load_metrics, write_json

if __name__ == "__main__":
    aggregates = {}
    for path in ROOT.glob("exp*/metrics.json"):
        data = load_metrics(path.parent.name)
        aggregates[path.parent.name] = data["summary"]
    write_json(ROOT / "stage10b_aggregates.json", aggregates)
