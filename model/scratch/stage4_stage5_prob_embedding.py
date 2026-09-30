import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# Allow running from scratch/ subdirectory or from the model root
_MODEL_ROOT = Path(__file__).resolve().parent.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from src.cli import get_model  # type: ignore[import]
from src.hsi_dataset import HSIPixelDataset, make_dataloader  # type: ignore[import]

def extract_features_and_probs():
    print("=" * 80)
    print("STAGE 4: SOFTMAX PROBABILITY ANALYSIS")
    print("STAGE 5: PATIENT-LEVEL EMBEDDING ANALYSIS")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "best_finetuned_self_attention.pth"
    print(f"Loading checkpoint: {checkpoint_path} on {device}")

    model = get_model("self_attention", dropout=0.4, attn_dim=128)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Hook or modified forward to get both logits and 128-D embedding
    def forward_with_embedding(m, x):
        x = x.unsqueeze(1)
        x = m.block1(x)
        x = m.se1(x)
        x = m.attention1(x)
        x = m.pool1(x)
        x = m.block2(x)
        x = m.se2(x)
        x = m.pool2(x)
        x = m.block3(x)
        x = m.se3(x)
        x = m.attention2(x)
        x = m.block4(x)
        x_gap = x.mean(dim=-1)
        x_flat = x.flatten(1)
        x = torch.cat([x_flat, x_gap], dim=1)
        # In eval mode dropout is identity
        x = m.relu(m.fc1(x))
        emb = m.relu(m.fc2(x)) # 128-dim penultimate embedding
        logits = m.out(emb)
        return logits, emb

    patient_re = re.compile(r"^P(\d+)_C\d+$", re.IGNORECASE)
    def get_pid(sid):
        m = patient_re.match(sid.strip())
        return int(m.group(1)) if m else sid

    class_names = ["BE", "BM", "ME", "MM"]
    prob_stats_report = {}

    collected_data = {}

    for split_name in ["val", "test"]:
        ds = HSIPixelDataset(
            data_dir="../preprocessing/outputs",
            split=split_name,
            label_mode="multi",
            normalization="per_pixel"
        )

        loader = torch.utils.data.DataLoader(
            ds, batch_size=128, shuffle=False, num_workers=0
        )

        all_probs = []
        all_embs = []
        all_labels = []
        all_sids = []
        all_pids = []

        # Map idx to sample_id: ds.sample_ids has one entry per pixel
        total_pixels = len(ds)
        assert len(ds.sample_ids) == total_pixels

        with torch.no_grad():
            for x, y, sids in tqdm(loader, desc=f"Inference on {split_name.upper()}"):
                x = x.to(device)
                logits, emb = forward_with_embedding(model, x)
                probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_embs.append(emb.cpu().numpy())
                all_labels.extend(y.numpy())
                all_sids.extend(list(sids))
                all_pids.extend([get_pid(s) for s in sids])

        all_probs = np.vstack(all_probs)
        all_embs = np.vstack(all_embs)
        all_labels = np.array(all_labels)
        all_sids = np.array(all_sids)
        all_pids = np.array(all_pids)

        collected_data[split_name] = {
            "probs": all_probs,
            "embs": all_embs,
            "labels": all_labels,
            "sids": all_sids,
            "pids": all_pids
        }

        # STAGE 4: PROBABILITY ANALYSIS FOR BE (class 0) AND MM (class 3)
        prob_stats_report[split_name] = {}

        for target_cls_idx, target_name in [(0, "BE"), (3, "MM")]:
            mask = (all_labels == target_cls_idx)
            cls_probs = all_probs[mask] # shape: (N_pixels, 4)
            n_samples = np.sum(mask)

            stats = {}
            for col_idx, pred_cname in enumerate(class_names):
                p_col = cls_probs[:, col_idx]
                stats[pred_cname] = {
                    "mean": float(np.mean(p_col)),
                    "median": float(np.median(p_col)),
                    "std": float(np.std(p_col)),
                    "p95": float(np.percentile(p_col, 95)),
                    "max": float(np.max(p_col))
                }
            prob_stats_report[split_name][target_name] = {
                "support_pixels": int(n_samples),
                "unique_patients": [int(p) for p in set(all_pids[mask].tolist())],
                "probabilities": stats
            }

            print(f"\n--- {split_name.upper()} SET: True Class = {target_name} ({n_samples} pixels, Patients: {set(all_pids[mask])}) ---")
            print(f"{'Pred Prob':<12} | {'Mean':<8} | {'Median':<8} | {'Std':<8} | {'95th %':<8} | {'Max':<8}")
            print("-" * 60)
            for pred_cname in class_names:
                st = stats[pred_cname]
                print(f"P({pred_cname}){'':<7} | {st['mean']:.4f}   | {st['median']:.4f}   | {st['std']:.4f}   | {st['p95']:.4f}   | {st['max']:.4f}")

    # STAGE 5: PATIENT-LEVEL EMBEDDING ANALYSIS
    print("\n" + "=" * 80)
    print("STAGE 5: GENERATING PCA AND t-SNE EMBEDDING PROJECTIONS")
    print("=" * 80)

    # Subsample 150 points per capture across Val and Test to maintain balanced representation in plot
    sampled_indices_val = []
    for sid in np.unique(collected_data["val"]["sids"]):
        idx = np.where(collected_data["val"]["sids"] == sid)[0]
        np.random.seed(42)
        chosen = np.random.choice(idx, size=min(150, len(idx)), replace=False)
        sampled_indices_val.extend(chosen)

    sampled_indices_test = []
    for sid in np.unique(collected_data["test"]["sids"]):
        idx = np.where(collected_data["test"]["sids"] == sid)[0]
        np.random.seed(42)
        chosen = np.random.choice(idx, size=min(150, len(idx)), replace=False)
        sampled_indices_test.extend(chosen)

    val_embs = collected_data["val"]["embs"][sampled_indices_val]
    val_labels = collected_data["val"]["labels"][sampled_indices_val]
    val_pids = collected_data["val"]["pids"][sampled_indices_val]

    test_embs = collected_data["test"]["embs"][sampled_indices_test]
    test_labels = collected_data["test"]["labels"][sampled_indices_test]
    test_pids = collected_data["test"]["pids"][sampled_indices_test]

    combined_embs = np.vstack([val_embs, test_embs])
    combined_labels = np.concatenate([val_labels, test_labels])
    combined_pids = np.concatenate([val_pids, test_pids])
    split_tags = np.array(["val"] * len(val_embs) + ["test"] * len(test_embs))

    print(f"Running PCA on {len(combined_embs)} sampled embedding vectors...")
    pca = PCA(n_components=2, random_state=42)
    pca_2d: np.ndarray = pca.fit_transform(combined_embs.astype(np.float32))  # type: ignore[assignment]

    print(f"Running t-SNE on {len(combined_embs)} sampled embedding vectors...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_2d: np.ndarray = tsne.fit_transform(combined_embs.astype(np.float32))  # type: ignore[assignment]

    # Create Plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: PCA colored by Class with special markers for BE Patients
    colors = {"BE": "#1f77b4", "BM": "#2ca02c", "ME": "#ff7f0e", "MM": "#d62728"}
    for idx, cname in enumerate(class_names):
        mask = (combined_labels == idx)
        axes[0].scatter(
            pca_2d[mask, 0], pca_2d[mask, 1],
            c=colors[cname], label=f"Class {cname}", alpha=0.5, s=20
        )

    # Highlight Val BE Patient and Test BE Patient
    val_be_mask = (combined_labels == 0) & (split_tags == "val")
    test_be_mask = (combined_labels == 0) & (split_tags == "test")

    axes[0].scatter(
        pca_2d[val_be_mask, 0], pca_2d[val_be_mask, 1],
        facecolors='none', edgecolors='cyan', s=70, linewidths=1.5, label="Val BE (Patient 110)"
    )
    axes[0].scatter(
        pca_2d[test_be_mask, 0], pca_2d[test_be_mask, 1],
        facecolors='none', edgecolors='black', s=90, linewidths=2.0, marker='^', label="Test BE (Patient 32 - MISSED)"
    )

    axes[0].set_title(f"PCA of 128-D Embeddings\n(Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontsize=13)
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: t-SNE colored by Class with Patient Annotations
    for idx, cname in enumerate(class_names):
        mask = (combined_labels == idx)
        axes[1].scatter(
            tsne_2d[mask, 0], tsne_2d[mask, 1],
            c=colors[cname], label=f"Class {cname}", alpha=0.5, s=20
        )

    axes[1].scatter(
        tsne_2d[val_be_mask, 0], tsne_2d[val_be_mask, 1],
        facecolors='none', edgecolors='cyan', s=70, linewidths=1.5, label="Val BE (Patient 110)"
    )
    axes[1].scatter(
        tsne_2d[test_be_mask, 0], tsne_2d[test_be_mask, 1],
        facecolors='none', edgecolors='black', s=90, linewidths=2.0, marker='^', label="Test BE (Patient 32 - MISSED)"
    )

    axes[1].set_title("t-SNE of 128-D Embeddings\n(Val + Test Embeddings)", fontsize=13)
    axes[1].set_xlabel("t-SNE Dim 1")
    axes[1].set_ylabel("t-SNE Dim 2")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "scratch/embedding_projections_pca_tsne.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[OK] Embedding visualization saved to {plot_path}")

    with open("scratch/probability_analysis_report.json", "w") as f:
        json.dump(prob_stats_report, f, indent=2)
    print("[OK] Probability analysis report saved to scratch/probability_analysis_report.json")

if __name__ == "__main__":
    extract_features_and_probs()
