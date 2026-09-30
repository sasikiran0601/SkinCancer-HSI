"""Stage 14: verify every fold has an immutable best checkpoint and metadata."""
from ._stage_helpers import ROOT, load_metrics, write_json

if __name__ == "__main__":
    result = {}
    for path in ROOT.glob("exp*/metrics.json"):
        data = load_metrics(path.parent.name)
        result[path.parent.name] = [
            {"fold": fold["fold"], "checkpoint": fold["checkpoint"],
             "exists": (ROOT / path.parent.name / f"fold_{fold['fold']}" / "best_model.pth").exists(),
             "best_epoch": fold["best_epoch"],
             "best_validation_macro_f1": fold["best_validation_macro_f1"]}
            for fold in data["fold_metrics"]
        ]
    write_json(ROOT / "stage14_checkpoint_manifest.json", result)
