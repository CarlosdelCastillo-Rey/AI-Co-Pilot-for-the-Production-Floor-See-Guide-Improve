"""Train HAR MLP on precomputed embeddings (v2: class weights + subject split)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from lib.har_model import HarMLP, save_checkpoint


def _split_indices(
    y: np.ndarray,
    subjects: np.ndarray | None,
    *,
    test_size: float,
    seed: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    idx = np.arange(n)
    if split_mode == "subject" and subjects is not None and len(subjects) == n:
        uniq = np.unique(subjects)
        if len(uniq) >= 2:
            rng = np.random.RandomState(seed)
            rng.shuffle(uniq)
            n_test_subj = max(1, int(round(len(uniq) * test_size)))
            test_subj = set(uniq[:n_test_subj])
            test_mask = np.array([s in test_subj for s in subjects])
            test_idx = idx[test_mask]
            train_idx = idx[~test_mask]
            if len(train_idx) > 0 and len(test_idx) > 0:
                return train_idx, test_idx

    stratify = None
    if len(set(y)) > 1:
        counts = np.bincount(y)
        if len(counts) and counts[counts > 0].min() >= 2:
            stratify = y
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return np.array(train_idx), np.array(test_idx)


def _class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    """Balanced weights; missing train classes get weight 1.0 (subject split edge case)."""
    present = np.unique(y_train)
    weights = np.ones(n_classes, dtype=np.float32)
    if len(present) > 0:
        computed = compute_class_weight("balanced", classes=present, y=y_train)
        weights[present] = computed.astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32)


def train_har_mlp(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    *,
    exclude_labels: list[str],
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    split_mode: str = "subject",
    test_size: float = 0.2,
    use_class_weights: bool = True,
    subjects: np.ndarray | None = None,
) -> tuple[HarMLP, dict]:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(class_names)

    train_idx, val_idx = _split_indices(
        y,
        subjects,
        test_size=test_size,
        seed=seed,
        split_mode=split_mode,
    )
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = HarMLP(X.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if use_class_weights:
        weight = _class_weights(y_train, n_classes).to(device)
        crit: nn.Module = nn.CrossEntropyLoss(weight=weight)
    else:
        crit = nn.CrossEntropyLoss()

    def _batches(xa, ya):
        idx = np.arange(len(xa))
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            j = idx[i : i + batch_size]
            yield torch.from_numpy(xa[j]).float().to(device), torch.from_numpy(ya[j]).long().to(device)

    history: list[dict] = []
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for xb, yb in _batches(X_train, y_train):
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(xb)
            n += len(xb)
        train_loss = loss_sum / max(n, 1)

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_val).float().to(device)
            yv = torch.from_numpy(y_val).long().to(device)
            val_logits = model(xv)
            val_loss = float(crit(val_logits, yv).item())
            pred = val_logits.argmax(dim=1).cpu().numpy()

        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})
        if ep == epochs or ep % max(1, epochs // 5) == 0:
            print(f"epoch {ep}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

    report = classification_report(
        y_val,
        pred,
        labels=list(range(n_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    split_info = {
        "split_mode": split_mode,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "use_class_weights": use_class_weights,
    }
    if subjects is not None:
        split_info["n_subjects_train"] = int(len(set(subjects[train_idx])))
        split_info["n_subjects_val"] = int(len(set(subjects[val_idx])))

    return model, {
        "history": history,
        "val_report": report,
        "device": device,
        "split_info": split_info,
    }


def train_from_npz(
    npz_path: Path,
    ckpt_path: Path,
    *,
    exclude_labels: list[str],
    epochs: int = 25,
    split_mode: str = "subject",
    use_class_weights: bool = True,
    backbone: str = "vjepa",
) -> dict:
    from lib.session_log import model_tag_from_checkpoint

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    class_names = list(data["class_names"])
    subjects = data["subjects"].astype(str) if "subjects" in data else None

    model, stats = train_har_mlp(
        X,
        y,
        class_names,
        exclude_labels=exclude_labels,
        epochs=epochs,
        split_mode=split_mode,
        use_class_weights=use_class_weights,
        subjects=subjects,
    )
    tag = model_tag_from_checkpoint(ckpt_path)
    save_checkpoint(
        ckpt_path,
        model=model,
        class_names=class_names,
        exclude_labels=exclude_labels,
        emb_dim=X.shape[1],
        meta={
            "epochs": epochs,
            "n_samples": len(X),
            "n_classes": len(class_names),
            "classifier_version": tag,
            "embedding_version": tag,
            "npz_path": str(npz_path),
            "backbone": backbone,
            "split_mode": split_mode,
            "use_class_weights": use_class_weights,
            **stats.get("split_info", {}),
        },
    )
    stats["checkpoint"] = str(ckpt_path)
    stats["class_names"] = class_names
    return stats
