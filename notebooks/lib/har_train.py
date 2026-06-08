"""Train HAR MLP on precomputed embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from lib.har_model import HarMLP, save_checkpoint


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
) -> tuple[HarMLP, dict]:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(class_names)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if len(set(y)) > 1 else None
    )

    model = HarMLP(X.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    return model, {"history": history, "val_report": report, "device": device}


def train_from_npz(
    npz_path: Path,
    ckpt_path: Path,
    *,
    exclude_labels: list[str],
    epochs: int = 25,
) -> dict:
    from lib.session_log import model_tag_from_checkpoint

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    class_names = list(data["class_names"])
    model, stats = train_har_mlp(
        X, y, class_names, exclude_labels=exclude_labels, epochs=epochs
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
        },
    )
    stats["checkpoint"] = str(ckpt_path)
    stats["class_names"] = class_names
    return stats
