"""v2 HAR training — Focal Loss + WeightedSampler + Mixup + SupCon + CosineAnnealingLR."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from lib.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_HEAD_ARCH,
    DEFAULT_LR,
    DEFAULT_MIXUP_ALPHA,
    DEFAULT_MIXUP_N_AUG,
    DEFAULT_SEED,
    DEFAULT_SPLIT_MODE,
    DEFAULT_SUPCON_EPOCHS,
    DEFAULT_SUPCON_LR,
    DEFAULT_SUPCON_PROJ_DIM,
    DEFAULT_SUPCON_TEMPERATURE,
    DEFAULT_TEST_SIZE,
    DEFAULT_USE_FOCAL_LOSS,
    DEFAULT_USE_MIXUP,
    DEFAULT_USE_WEIGHTED_SAMPLER,
    DEFAULT_WEIGHT_DECAY,
)
from lib.har_model import HarGRU, HarMLP, SupConProjector, save_checkpoint

logger = logging.getLogger(__name__)


# ── Loss functions ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss — down-weights easy examples, focuses on hard/rare ones.

    Particularly effective for imbalanced InHARD classes.
    gamma=2 is standard; higher values → more focus on hard examples.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020).

    Pulls same-class embeddings together, pushes different-class apart.
    Especially effective for separating visually-similar action pairs
    (e.g. 'Take screwdriver' vs 'Put down screwdriver').
    """
    def __init__(self, temperature: float = DEFAULT_SUPCON_TEMPERATURE):
        super().__init__()
        self.T = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: (B, proj_dim) — must be L2-normalized
        B = features.size(0)
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()   # (B, B)
        sim  = torch.matmul(features, features.T) / self.T            # (B, B)
        # exclude diagonal (self-similarity)
        sim.fill_diagonal_(-1e9)
        mask.fill_diagonal_(0)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        n_pos = mask.sum(1).clamp(min=1)
        loss  = -(mask * log_prob).sum(1) / n_pos
        return loss.mean()


# ── Data utilities ─────────────────────────────────────────────────────────────

def _split_indices(
    y: np.ndarray,
    subjects: np.ndarray | None,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    split_mode: str = DEFAULT_SPLIT_MODE,
) -> tuple[np.ndarray, np.ndarray]:
    n   = len(y)
    idx = np.arange(n)
    if split_mode == "subject" and subjects is not None and len(subjects) == n:
        uniq = np.unique(subjects)
        if len(uniq) >= 2:
            rng = np.random.RandomState(seed)
            rng.shuffle(uniq)
            n_test = max(1, int(round(len(uniq) * test_size)))
            test_subj  = set(uniq[:n_test])
            test_mask  = np.array([s in test_subj for s in subjects])
            train_idx, test_idx = idx[~test_mask], idx[test_mask]
            if len(train_idx) > 0 and len(test_idx) > 0:
                return train_idx, test_idx
    stratify = None
    if len(set(y)) > 1:
        counts = np.bincount(y)
        if counts[counts > 0].min() >= 2:
            stratify = y
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=stratify)
    return np.array(train_idx), np.array(test_idx)


def _class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    present  = np.unique(y_train)
    weights  = np.ones(n_classes, dtype=np.float32)
    if len(present):
        weights[present] = compute_class_weight("balanced", classes=present, y=y_train).astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32)


def _weighted_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    counts  = np.bincount(y_train)
    class_w = 1.0 / np.maximum(counts, 1).astype(np.float64)
    sample_w = class_w[y_train]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_w).float(),
        num_samples=len(y_train),
        replacement=True,
    )


def mixup_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = DEFAULT_MIXUP_ALPHA,
    n_aug: int = DEFAULT_MIXUP_N_AUG,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolation-based augmentation in embedding space."""
    rng = np.random.default_rng(seed)
    X_all, y_all = [X], [y]
    for _ in range(n_aug):
        lam  = rng.beta(alpha, alpha, size=len(X))[:, None].astype(np.float32)
        perm = rng.permutation(len(X))
        X_all.append(lam * X + (1 - lam) * X[perm])
        y_all.append(y)  # keep the dominant label
    return np.vstack(X_all), np.concatenate(y_all)


# ── SupCon pre-training stage ─────────────────────────────────────────────────

def train_supcon(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    emb_dim: int,
    proj_dim: int = DEFAULT_SUPCON_PROJ_DIM,
    epochs: int = DEFAULT_SUPCON_EPOCHS,
    lr: float = DEFAULT_SUPCON_LR,
    temperature: float = DEFAULT_SUPCON_TEMPERATURE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
) -> SupConProjector:
    """Train projector head with SupCon loss; returns trained projector.

    After this stage, pass X_train through projector to get refined embeddings
    for the downstream MLP/GRU classifier.
    """
    torch.manual_seed(seed)
    projector = SupConProjector(emb_dim, proj_dim).to(device)
    opt  = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=1e-4)
    crit = SupConLoss(temperature=temperature)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    sampler = _weighted_sampler(y_train)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    print(f"SupCon pre-training: {epochs} epochs, proj_dim={proj_dim}, T={temperature}")
    for ep in range(1, epochs + 1):
        projector.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            z = projector(xb)
            loss = crit(z, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        if ep == 1 or ep % max(1, epochs // 5) == 0 or ep == epochs:
            print(f"  SupCon ep {ep:3d}/{epochs}  loss={np.mean(losses):.4f}  lr={sched.get_last_lr()[0]:.2e}")

    projector.eval()
    return projector


def apply_supcon_projection(
    projector: SupConProjector,
    X: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Project raw embeddings through trained SupCon projector."""
    projector.eval()
    with torch.inference_mode():
        z = projector(torch.from_numpy(X).float().to(device))
    return z.cpu().numpy().astype(np.float32)


# ── Main MLP / GRU training ───────────────────────────────────────────────────

def train_har_head(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    *,
    exclude_labels: list[str],
    subjects: np.ndarray | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    dropout: float = DEFAULT_DROPOUT,
    seed: int = DEFAULT_SEED,
    split_mode: str = DEFAULT_SPLIT_MODE,
    test_size: float = DEFAULT_TEST_SIZE,
    head_arch: str = DEFAULT_HEAD_ARCH,
    use_focal_loss: bool = DEFAULT_USE_FOCAL_LOSS,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    use_weighted_sampler: bool = DEFAULT_USE_WEIGHTED_SAMPLER,
    use_mixup: bool = DEFAULT_USE_MIXUP,
    mixup_alpha: float = DEFAULT_MIXUP_ALPHA,
    mixup_n_aug: int = DEFAULT_MIXUP_N_AUG,
    use_supcon: bool = False,
    supcon_epochs: int = DEFAULT_SUPCON_EPOCHS,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(class_names)

    train_idx, val_idx = _split_indices(y, subjects, test_size=test_size, seed=seed, split_mode=split_mode)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # ── SupCon pre-training ────────────────────────────────────────────────
    if use_supcon:
        projector = train_supcon(
            X_train, y_train,
            emb_dim=X.shape[1], epochs=supcon_epochs,
            device=device, seed=seed,
        )
        X_train = apply_supcon_projection(projector, X_train, device=device)
        X_val   = apply_supcon_projection(projector, X_val,   device=device)
        emb_dim = X_train.shape[1]
    else:
        emb_dim = X.shape[1]

    # ── Mixup augmentation ─────────────────────────────────────────────────
    if use_mixup:
        X_train_aug, y_train_aug = mixup_embeddings(
            X_train, y_train, alpha=mixup_alpha, n_aug=mixup_n_aug, seed=seed,
        )
        print(f"Mixup: {len(X_train)} → {len(X_train_aug)} training samples")
    else:
        X_train_aug, y_train_aug = X_train, y_train

    # ── Loss ───────────────────────────────────────────────────────────────
    cw = _class_weights(y_train, n_classes).to(device)
    if use_focal_loss:
        crit: nn.Module = FocalLoss(gamma=focal_gamma, weight=cw)
        print(f"FocalLoss(gamma={focal_gamma}) + class weights")
    else:
        crit = nn.CrossEntropyLoss(weight=cw)
        print("CrossEntropyLoss + class weights")

    # ── Model ──────────────────────────────────────────────────────────────
    if head_arch == "gru":
        model = HarGRU(emb_dim, n_classes, dropout=dropout).to(device)
        print(f"HarGRU  emb_dim={emb_dim}  n_classes={n_classes}")
    else:
        model = HarMLP(emb_dim, n_classes, dropout=dropout).to(device)
        print(f"HarMLP  emb_dim={emb_dim}  n_classes={n_classes}")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    # ── DataLoader ─────────────────────────────────────────────────────────
    train_ds = TensorDataset(
        torch.from_numpy(X_train_aug).float(),
        torch.from_numpy(y_train_aug).long(),
    )
    if use_weighted_sampler:
        sampler = _weighted_sampler(y_train_aug)
        loader  = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    else:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # ── Training loop ──────────────────────────────────────────────────────
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    history: list[dict] = []
    best_val, best_state = float("inf"), None

    print(f"\nTraining {head_arch.upper()} head | {epochs} epochs | device={device}")
    print(f"  train={len(X_train_aug)} (+aug) val={len(X_val)} classes={n_classes}")
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
            n += len(xb)
        sched.step()
        train_loss = ep_loss / max(n, 1)

        model.eval()
        with torch.inference_mode():
            val_logits = model(X_val_t)
            val_loss   = float(crit(val_logits, y_val_t).item())
            pred       = val_logits.argmax(dim=1).cpu().numpy()

        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": sched.get_last_lr()[0],
        })
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            acc = (pred == y_val).mean()
            print(f"  ep {ep:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  val_acc={acc:.1%}  lr={sched.get_last_lr()[0]:.2e}")

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        final_pred = model(X_val_t).argmax(dim=1).cpu().numpy()

    report = classification_report(
        y_val, final_pred,
        labels=list(range(n_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    split_info = {
        "split_mode": split_mode,
        "n_train_orig": int(len(train_idx)),
        "n_train_aug":  int(len(X_train_aug)),
        "n_val":        int(len(val_idx)),
        "use_mixup":    use_mixup,
        "use_supcon":   use_supcon,
        "use_focal":    use_focal_loss,
        "use_sampler":  use_weighted_sampler,
        "head_arch":    head_arch,
        "best_val_loss": best_val,
    }
    if subjects is not None:
        split_info["n_subjects_train"] = int(len(set(subjects[train_idx])))
        split_info["n_subjects_val"]   = int(len(set(subjects[val_idx])))

    return model, {
        "history": history,
        "val_report": report,
        "device": device,
        "split_info": split_info,
        "y_val": y_val,
        "y_pred": final_pred,
        "val_idx": val_idx,
        "train_idx": train_idx,
        "emb_dim": emb_dim,
    }


def train_from_npz(
    npz_path: Path,
    ckpt_path: Path,
    *,
    exclude_labels: list[str],
    epochs: int = DEFAULT_EPOCHS,
    split_mode: str = DEFAULT_SPLIT_MODE,
    head_arch: str = DEFAULT_HEAD_ARCH,
    use_focal_loss: bool = DEFAULT_USE_FOCAL_LOSS,
    use_weighted_sampler: bool = DEFAULT_USE_WEIGHTED_SAMPLER,
    use_mixup: bool = DEFAULT_USE_MIXUP,
    use_supcon: bool = False,
    backbone: str = "vjepa",
) -> dict:
    from lib.session_log import model_tag_from_checkpoint

    data        = np.load(npz_path, allow_pickle=True)
    X           = data["X"].astype(np.float32)
    y           = data["y"].astype(np.int64)
    class_names = list(data["class_names"])
    subjects    = data["subjects"].astype(str) if "subjects" in data else None

    model, stats = train_har_head(
        X, y, class_names,
        exclude_labels=exclude_labels,
        subjects=subjects,
        epochs=epochs,
        split_mode=split_mode,
        head_arch=head_arch,
        use_focal_loss=use_focal_loss,
        use_weighted_sampler=use_weighted_sampler,
        use_mixup=use_mixup,
        use_supcon=use_supcon,
    )
    tag = model_tag_from_checkpoint(ckpt_path)
    save_checkpoint(
        ckpt_path,
        model=model,
        class_names=class_names,
        exclude_labels=exclude_labels,
        emb_dim=stats["emb_dim"],
        head_arch=head_arch,
        meta={
            "epochs": epochs,
            "n_samples": len(X),
            "n_classes": len(class_names),
            "classifier_version": tag,
            "embedding_version": tag,
            "npz_path": str(npz_path),
            "backbone": backbone,
            **stats.get("split_info", {}),
        },
    )
    stats["checkpoint"] = str(ckpt_path)
    stats["class_names"] = class_names
    return stats
