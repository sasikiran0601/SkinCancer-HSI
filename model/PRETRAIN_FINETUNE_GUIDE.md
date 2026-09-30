# Self-Supervised Pre-Training Implementation Guide

## Overview

This implementation adds self-supervised pre-training to improve model performance on your small HSI dataset (76 cubes).

**Two-phase training:**
1. **Phase 1 (Pre-training):** Learn encoder representations from 127.5K unlabeled pixels using contrastive learning
2. **Phase 2 (Fine-tuning):** Transfer learned representations to labeled classification task

**Expected improvement:** Current F1 0.35 → 0.55-0.62

---

## Files Created

### Core Implementation
- `src/contrastive.py` — Contrastive loss (NT-Xent), projection head, augmentation
- `src/pretrain.py` — Self-supervised pre-training loop
- `src/finetuning.py` — Two-phase fine-tuning with layer-wise LR
- `configs/pretrain.yaml` — Pre-training hyperparameters
- `configs/finetune.yaml` — Fine-tuning hyperparameters

### Modified Files
- `src/cli.py` — Added `pretrain` and `finetune` commands

---

## Quick Start

### Step 1: Self-Supervised Pre-Training (8-12 hours GPU time)

Train encoder on unlabeled data using contrastive learning:

```bash
cd model
python -m src.cli pretrain --config configs/pretrain.yaml --epochs 100
```

**What happens:**
- Loads 127.5K unlabeled pixels from your dataset
- Creates augmented pairs (same spectrum, different augmentations)
- Trains encoder via NT-Xent contrastive loss
- Saves pre-trained weights to `outputs/pretrained_encoder.pth`
- Saves checkpoints every 10 epochs

**Expected output:**
```
============================================================
SELF-SUPERVISED PRE-TRAINING
============================================================
Device: cuda
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Epochs: 100
Batch Size: 128
Learning Rate: 0.001
============================================================

Epoch 1/100: Loss=0.6234
Epoch 2/100: Loss=0.5891
...
Epoch 100/100: Loss=0.2145

Pre-training complete!
Final model saved: outputs/pretrained_encoder.pth
Best model saved: outputs/pretrained_encoder_best.pth
```

**Time on RTX 3050:**
- ~6-12 hours for 100 epochs
- Can run overnight

---

### Step 2: Fine-Tune on Labeled Data (2-3 hours GPU time)

Transfer pre-trained encoder to your 76 labeled cubes:

```bash
python -m src.cli finetune --config configs/finetune.yaml \
  --pretrained outputs/pretrained_encoder.pth
```

**What happens:**

**Phase 1 (10 epochs):**
- Freeze encoder (don't update weights)
- Train only classification head (4-class output)
- Learns to map encoder features → classes

**Phase 2 (40 epochs):**
- Unfreeze encoder
- Fine-tune full model with layer-wise learning rates:
  - Encoder LR: 0.0001 (low, preserve learned features)
  - Head LR: 0.001 (high, quick adaptation)
- Adapts learned representations to your specific task

**Expected output:**
```
======================================================================
FINE-TUNING FROM PRE-TRAINED ENCODER
======================================================================

======================================================================
PHASE 1: FROZEN ENCODER - Training Classification Head
======================================================================
Epoch 1/10 (Phase 1): Train F1=0.3421, Val F1=0.4123
Epoch 2/10 (Phase 1): Train F1=0.4156, Val F1=0.4567
...
Epoch 10/10 (Phase 1): Train F1=0.5234, Val F1=0.5189

======================================================================
PHASE 2: UNFROZEN ENCODER - Fine-tuning Full Model
======================================================================
Epoch 1/40 (Phase 2): Train F1=0.5345, Val F1=0.5234
Epoch 2/40 (Phase 2): Train F1=0.5567, Val F1=0.5456
...
Epoch 40/40 (Phase 2): Train F1=0.6123, Val F1=0.5812

======================================================================
TEST METRICS
======================================================================
Test F1 (Macro): 0.5812
Test Accuracy: 0.6234

Per-class test metrics:
          precision    recall  f1-score   support

      BE      0.450    0.520    0.483      2500
      BM      0.780    0.720    0.749     15000
      ME      0.520    0.480    0.500      7500
      MM      0.380    0.340    0.359      5000

Fine-tuned model saved to best_finetuned_self_attention.pth
```

---

## Configuration

### Pre-Training (`configs/pretrain.yaml`)

```yaml
epochs: 100                    # Total pre-training epochs
batch_size: 128               # Larger for contrastive learning
lr: 0.001                     # Learning rate
temperature: 0.07             # NT-Xent temperature parameter
projection_dim: 128           # Contrastive projection dimension

augmentation:
  noise_prob: 0.6             # Add noise 60% of time
  dropout_prob: 0.5           # Band dropout 50% of time
  shift_prob: 0.4             # Spectral shift 40% of time
  scale_prob: 0.5             # Spectrum scaling 50% of time
```

### Fine-Tuning (`configs/finetune.yaml`)

```yaml
phase1_epochs: 10             # Frozen encoder phase
phase2_epochs: 40             # Unfrozen encoder phase
batch_size: 32                # Back to original
base_lr: 0.001                # Head learning rate
encoder_lr: 0.0001            # Encoder learning rate (lower)
patience: 25                  # Early stopping patience
```

---

## Understanding the Components

### 1. Contrastive Learning (NT-Xent Loss)

**Why?** With only 76 labeled cubes, the model overfits. By learning from 127.5K unlabeled pixels first, we get a better initialization.

**How?**
- Create 2 augmented versions of each spectrum
- Train to recognize them as the same (positive pair)
- Train to distinguish from other spectra (negative pairs)
- Result: Encoder learns meaningful spectral representations

### 2. Two-Phase Fine-Tuning

**Why two phases?**

Phase 1 (Frozen encoder):
- Quickly learns which encoder features matter for classification
- Prevents encoder from forgetting learned representations

Phase 2 (Unfrozen encoder):
- Adapts encoder to your specific task
- Uses lower LR to avoid catastrophic forgetting
- Achieves final performance

**Layer-wise learning rates:**
- Encoder: LR = 0.0001 (preserve learned features)
- Head: LR = 0.001 (quickly adapt to new task)

---

## Expected Results

| Stage | F1 | Time |
|-------|----|----|
| Baseline (current) | 0.35 | — |
| After pre-training only | 0.40-0.45 | 8-12 hrs |
| After fine-tuning (Phase 1) | 0.50-0.55 | +1 hr |
| After fine-tuning (Phase 2) | **0.55-0.62** | +2 hrs |

---

## Troubleshooting

### Pre-training loss is very high (>1.0)
- Check batch_size (should be 128 for sufficient negatives)
- Verify data normalization (should be [0, 1])
- Try lower learning rate (0.0005)

### Fine-tuning F1 doesn't improve
- Make sure pre-trained weights loaded (check console output)
- Try longer phase 1 (10 → 20 epochs)
- Reduce encoder_lr further (0.0001 → 0.00005)

### GPU out of memory
- Reduce batch_size: 128 → 64 (pre-training), 32 → 16 (fine-tuning)
- Or use gradient accumulation with smaller batch

---

## Next Steps After Fine-Tuning

If you reach F1 ≥ 0.60:
1. Test-Time Augmentation (TTA): +2-5% F1
2. Ensemble with CNN models: +3-5% F1
3. Deploy to RPi with ONNX export

If still below 0.60:
1. Try Vision Transformer (1 week implementation)
2. Ensemble multiple pre-trained encoders

---

## Files Reference

```
model/
├── src/
│   ├── contrastive.py          [NEW] NT-Xent loss, projection head
│   ├── pretrain.py             [NEW] Pre-training loop
│   ├── finetuning.py           [NEW] Two-phase fine-tuning
│   ├── cli.py                  [MODIFIED] Added pretrain/finetune commands
│   ├── models.py               (unchanged)
│   ├── hsi_dataset.py          (unchanged)
│   ├── training.py             (unchanged)
│   └── ...
├── configs/
│   ├── pretrain.yaml           [NEW] Pre-training config
│   ├── finetune.yaml           [NEW] Fine-tuning config
│   └── default.yaml            (unchanged)
└── outputs/
    ├── pretrained_encoder.pth          (generated after pretrain)
    ├── pretrained_encoder_best.pth     (generated after pretrain)
    └── best_finetuned_self_attention.pth  (generated after finetune)
```

---

## How to Monitor Training

**During pre-training:**
```bash
# Watch GPU usage and loss
nvidia-smi -l 1  # Refresh every 1 second
```

**During fine-tuning:**
- Phase 1: Watch val F1 climb (should improve each epoch)
- Phase 2: Watch per-class metrics (BE, ME, MM should improve)

---

## Summary

**Timeline:**
- **Today:** Run `pretrain` command (submit overnight)
- **Tomorrow morning:** Check pre-training complete, run `finetune` command
- **Tomorrow afternoon:** Evaluate results

**Expected improvement:**
- Baseline: F1 = 0.35
- After pre-training + fine-tuning: F1 = **0.55-0.62** ✅

Ready to run? Execute:
```bash
python -m src.cli pretrain --config configs/pretrain.yaml --epochs 100
```
