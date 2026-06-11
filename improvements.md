# HAR Pipeline Improvements

## Overview

This document diagnoses the root causes of the current ~30–35% subject-held-out accuracy and describes the solutions implemented in `notebooks_v2`. The system performs Human Activity Recognition (HAR) on the InHARD dataset: 12 action classes, ~3400 clips from 16 subjects recorded via 3-view composite CCTV (1280×720 mosaic). The backbone is a frozen V-JEPA2 or DINOv2 ViT-L; a small MLP head performs classification.

The core issue is not the backbone — it is a combination of data starvation, temporal information loss, premature training, and YOLO operating on the wrong input. Each problem is addressed in turn below.

---

## Problem Analysis

### Problem 1 — Artificial data cap wastes majority-class data

**`clips_per_class=100` discards clips from well-represented classes while minority classes are already scarce.**

The dataset is naturally imbalanced:

| Class | Available clips |
|---|---|
| Take subsystem | 39 |
| Put down measuring | 74 |
| Take measuring | 76 |
| (majority classes) | 200–400+ |

Capping every class at 100 does not balance the dataset — it merely reduces the majority classes while the minority classes still have far fewer samples. The result is a smaller training set than necessary and no actual balance. The model sees at most ~1200 total training clips (100 × 12) when several hundred more are available at no additional cost.

### Problem 2 — Subject split reduces effective training set to ~75 samples/class

With 16 subjects and a leave-2-out subject split, roughly 12.5% of subjects are held out per fold. After the cap and the split, each class has on average ~75 training samples. The frozen backbone produces 1024-dimensional CLS token embeddings. Fitting a reliable decision boundary in 1024 dimensions from 75 examples per class is geometrically underdetermined. The classifier memorizes subject appearance rather than action semantics.

The 60% (random split) vs 30% (subject split) accuracy gap is the clearest evidence: the model has learned who is performing the action, not what action is being performed.

### Problem 3 — YOLO runs on the full 1280×720 mosaic

The 3-view composite frame is a mosaic of three camera angles:
- Top-left quadrant: overhead/top-down view (best for detecting motion direction)
- Top-right quadrant: side view
- Bottom-right quadrant: frontal view

Running YOLO on the full mosaic means the detector sees the person three times simultaneously. Bounding boxes can span quadrant boundaries or select the wrong sub-view. The backbone then encodes a crop that mixes visual information from multiple perspectives, degrading the geometric consistency of the embedding.

### Problem 4 — Temporal mean-pooling destroys sequence order

The current pipeline computes a per-frame CLS token and then takes the temporal mean across all frames in the clip. This produces a single 1024-dim vector. Two consequences:

1. "Take screwdriver" and "Put down screwdriver" produce nearly identical mean embeddings because the hand-to-object proximity is the same — only the motion direction differs.
2. Any action that has a distinctive temporal arc (reach → grasp → lift vs lift → lower → release) is collapsed to the same centroid.

Sequence order carries discriminative information that mean-pooling permanently discards.

### Problem 5 — Premature stopping at 25 epochs

Training loss: 1.58, validation loss: 2.04 at epoch 25. The gap between train and val loss is still closing — the model has not yet converged. With a small dataset and a fixed learning rate scheduler not tuned for 25 epochs, training terminates in the steep descent phase rather than the plateau. More epochs with a proper warm schedule would close ~0.3–0.5 nats of the gap.

### Problem 6 — Subject-appearance memorization confirmed

The 60% vs 30% split gap is a ~30 percentage-point drop when subjects are held out. A model that generalizes to unseen subjects would show a much smaller gap. This confirms the backbone embeddings encode subject-specific appearance features (body proportions, clothing, gait) at least as strongly as action features, and the MLP head has learned to exploit them.

---

## Solutions Implemented

### Solution 1 — Remove cap, use WeightedRandomSampler + Focal Loss

All available clips are used for training with no per-class ceiling. Class imbalance is addressed at two levels:

- **WeightedRandomSampler**: each training batch is drawn such that every class has equal expected representation, without discarding any clips.
- **Focal Loss (gamma=2)**: downweights easy, well-classified examples and focuses gradient signal on the hard minority-class clips. This is particularly important for "Take subsystem" (39 clips) which would otherwise be overwhelmed in a uniform cross-entropy regime.

This typically doubles or triples effective minority-class coverage without collecting new data.

### Solution 2 — Explicit 3-view extraction before YOLO

Before any detection or backbone inference, each composite frame is split into its three constituent sub-views by pixel coordinates:

```
Full frame (1280×720)
├── Top-left  [0:360, 0:640]    → overhead view  (best directional cue)
├── Top-right [0:360, 640:1280] → side view
└── Bottom-right [360:720, 640:1280] → frontal view
```

YOLO runs independently on each sub-view at its native resolution. The person crop from each view is passed to the backbone separately, yielding three independent 1024-dim embeddings per frame. The model can then learn view-specific weights or fuse them explicitly.

The overhead view is particularly valuable for disambiguating "Take" vs "Put down" actions because the hand trajectory is visible as a top-down arc rather than a foreshortened movement.

### Solution 3 — Temporal attention pooling

A learnable scalar weight is assigned to each frame position before aggregation:

```
alpha_t = softmax(W * h_t)   for t in 1..T
embedding = sum(alpha_t * h_t)
```

Where `h_t` is the CLS token at frame `t` and `W` is a learned projection vector. This allows the model to upweight the discriminative frames (the moment of contact, the peak of lift) and downweight neutral transitional frames. The weight vector is trained end-to-end with the classifier head, adding only `T` parameters.

### Solution 4 — Optional Bi-GRU sequential head

Instead of aggregating frame embeddings before classification, the T×1024 sequence is passed through a bidirectional GRU:

```
Input:  T × 1024  (sequence of per-frame CLS tokens)
Bi-GRU: hidden_size=256, bidirectional → T × 512
Pool:   last hidden state or attention over hidden states → 512-dim
Head:   Linear(512 → 12)
```

The bidirectional pass allows each frame's representation to be conditioned on both past and future context. This captures the "reach then grasp" vs "release then retract" temporal arc that is invisible to mean- or attention-pooling over independent frame representations.

The GRU head is optional — it adds inference latency. The attention-pooling head (Solution 3) should be tried first.

### Solution 5 — Supervised Contrastive Learning pre-training (SupCon)

Before training the final classifier, the embedding space is shaped by a contrastive objective:

```
Stage 1 (SupCon):
  Backbone (frozen) → projector MLP (1024 → 128 L2-normalized)
  Loss: SupConLoss — pull same-class clips together, push different-class apart
  Epochs: 50

Stage 2 (Classification):
  Backbone (frozen) → projector weights dropped
  Trained attention-pool or GRU head → classifier
  Loss: Focal Loss
  Epochs: 100
```

SupCon is particularly effective in the low-data regime because it generates O(N²) pairwise supervision signals from N clips rather than N classification signals. Minority classes receive proportionally more gradient because every pair involving a minority-class clip contributes to the loss.

### Solution 6 — Mixup augmentation in embedding space

Three virtual training samples are synthesized per real sample by interpolating pairs of embeddings from the same class:

```
x_mix = lambda * x_i + (1 - lambda) * x_j,   lambda ~ Beta(0.4, 0.4)
y_mix = lambda * y_i + (1 - lambda) * y_j
```

Interpolation occurs on the pre-aggregated 1024-dim CLS tokens, not on raw pixels, so no additional backbone forward passes are required. This triples the effective training set size for minority classes at essentially zero compute cost. It also acts as a regularizer that smooths the decision boundary between similar classes.

### Solution 7 — 100 epochs with CosineAnnealingLR

Training is extended to 100 epochs with a cosine annealing schedule:

```
lr(t) = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(pi * t / T_max))
eta_min = 1e-5
```

The cosine schedule reduces oscillation near convergence and allows the optimizer to settle into a flatter minimum. Early stopping with patience=15 epochs on validation loss prevents overfitting while allowing the model to train past the premature 25-epoch cutoff.

### Solution 8 — Extended analysis toolkit

New analysis capabilities added to `notebooks_v2`:

- **UMAP/t-SNE cluster plots**: visualize whether same-class clips cluster regardless of subject identity
- **Per-view frame comparison strip**: side-by-side frames from all three views for a given clip, to visually verify crop quality
- **YOLO detection overlays**: bounding box visualization per sub-view to catch detection failures
- **Per-person accuracy heatmaps**: 12×16 confusion matrix (class × subject) to identify which subjects or actions drive the performance gap
- **Confidence calibration curves**: reliability diagrams to detect overconfident predictions on low-data classes

---

## Architecture Diagram

```
                        RAW COMPOSITE FRAME (1280×720)
                               |
              +----------------+----------------+
              |                |                |
         TOP-LEFT         TOP-RIGHT       BOTTOM-RIGHT
         [overhead]         [side]          [front]
              |                |                |
           YOLO             YOLO             YOLO
         crop/pad          crop/pad         crop/pad
              |                |                |
        ViT-L backbone   ViT-L backbone   ViT-L backbone
          (frozen)         (frozen)         (frozen)
              |                |                |
        CLS tokens T×1024  CLS tokens T×1024  CLS tokens T×1024
              |                |                |
              +----------------+----------------+
                               |
                    View fusion (concat / learned)
                               |
                    T × (1024 * n_views)
                               |
              +----------------+----------------+
              |                                 |
       Attention Pool                       Bi-GRU
       (lightweight)                    (sequential)
              |                                 |
          1024-dim                          512-dim
              |                                 |
              +----------------+----------------+
                               |
                    Focal Loss (gamma=2)
                    + SupCon pre-training
                    + Mixup augmentation
                               |
                        12-class output
```

---

## Expected Results

| Configuration | Accuracy (subject-split) | Notes |
|---|---|---|
| Baseline (current) | 30–35% | 25 epochs, mean-pool, full mosaic, 100-cap |
| + Data fix + view extraction + epochs (P1–P3) | 55–65% | Remove cap, 3-view crops, 100 epochs |
| + Temporal attention + Mixup + Focal Loss (P4–P6) | 70–80% | Sequence-aware pooling, augmentation |
| + SupCon pre-training + Bi-GRU (P7) | 85–90%+ | Contrastive embedding shaping |

Note: the 60% figure previously reported is on a random (non-subject-held-out) split and is not comparable to the subject-split figures above. Random-split accuracy inflates results because the model can memorize subject appearance. All targets above are subject-held-out.

---

## Future Work

### Data collection
- Collect additional clips specifically for the three scarce classes ("Take subsystem", "Put down measuring", "Take measuring") — even 50 additional clips per class would halve the imbalance ratio.
- Consider data collection from multiple clothing/appearance variants for existing subjects to decorrelate appearance from action.

### Backbone fine-tuning
- Unfreeze the last 2–4 transformer blocks of the ViT-L backbone with a very small learning rate (1e-6) after the classifier has converged. This allows the backbone to specialize its representations to industrial actions while preserving the general visual features learned during pre-training.

### Multi-view attention fusion
- Replace the simple concatenation of view embeddings with a cross-view attention module: each view's tokens attend over the other views' tokens. This is particularly useful when one view is occluded or the worker is off-center.

### Semi-supervised learning
- If unlabeled production floor footage is available, use the SupCon projector to generate pseudo-labels on unlabeled clips and add them to training with a confidence threshold.

### Online inference optimization
- Profile per-view YOLO + backbone inference latency on the target edge hardware.
- If latency is a constraint, use only the overhead view (best single-view discriminative cue) and a lightweight MobileViT or EfficientViT backbone instead of ViT-L.

### Calibration
- Apply temperature scaling post-training to correct the overconfident predictions observed on minority classes. Well-calibrated confidence scores are important for production floor alerting (false alarms have real operational cost).

### Confusion class pairs
- "Take screwdriver" vs "Put down screwdriver" and "Take measuring" vs "Put down measuring" are the hardest pairs. Consider a hierarchical classifier: first predict the object (screwdriver / measuring / subsystem), then predict the direction (take / put down). The direction is where temporal attention adds the most value.
