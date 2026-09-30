"""Stage 10: validate required per-fold metric fields are persisted."""
from ._stage_helpers import ROOT, load_metrics, write_json

REQUIRED = ("macro_f1", "weighted_f1", "accuracy", "per_class", "binary",
            "confusion_matrix", "normalized_confusion_matrix")

if __name__ == "__main__":
    names = [p.name for p in ROOT.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    result = {name: {"folds": len(load_metrics(name)["fold_metrics"]),
                     "required_fields": list(REQUIRED),
                     "valid": all(all(k in fold for k in REQUIRED)
                                  for fold in load_metrics(name)["fold_metrics"])}
              for name in names}
    write_json(ROOT / "stage10_metric_validation.json", result)
