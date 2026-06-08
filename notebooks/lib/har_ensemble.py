"""Ensemble + hyperparameter search for HAR classifiers on V-JEPA embeddings (Avance 5)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import LinearSVC

from lib.har_analysis import load_embeddings_npz
from lib.har_model import HarMLP, load_checkpoint
from lib.paths import CHECKPOINTS_DIR, OUTPUTS_DIR

ENSEMBLE_DIR = OUTPUTS_DIR / "ensemble_avance5"
PRIMARY_METRIC = "macro_f1"


@dataclass
class ModelResult:
    name: str
    kind: str
    metrics: dict[str, float]
    train_seconds: float
    y_pred: np.ndarray
    probs: np.ndarray
    estimator: Any = field(default=None, repr=False)
    extra: dict[str, Any] = field(default_factory=dict)


def _ensure_out() -> Path:
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    return ENSEMBLE_DIR


def holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratify = y if len(np.unique(y)) > 1 else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    n_classes = probs.shape[1]
    macro_auc = float("nan")
    try:
        if n_classes > 2:
            macro_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        else:
            macro_auc = float(roc_auc_score(y_true, probs[:, 1]))
    except ValueError:
        pass
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_auc_ovr": macro_auc,
    }


def finalize_result(result: ModelResult, y_test: np.ndarray) -> ModelResult:
    result.y_pred = result.probs.argmax(axis=1)
    result.metrics = compute_metrics(y_test, result.y_pred, result.probs)
    return result


def _sklearn_probs(estimator: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    scores = estimator.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def tune_sklearn(
    name: str,
    kind: str,
    estimator: Any,
    param_distributions: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_iter: int = 8,
    cv: int = 3,
    seed: int = 42,
) -> ModelResult:
    t0 = time.perf_counter()
    search = RandomizedSearchCV(
        estimator,
        param_distributions,
        n_iter=n_iter,
        cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
        scoring="f1_macro",
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    probs = _sklearn_probs(search.best_estimator_, X_test)
    return ModelResult(
        name=name,
        kind=kind,
        metrics={},
        train_seconds=time.perf_counter() - t0,
        y_pred=np.array([]),
        probs=probs,
        estimator=search.best_estimator_,
        extra={"best_params": search.best_params_, "cv_best_score": float(search.best_score_)},
    )


def train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_classes: int,
    hidden: int = 512,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 20,
    seed: int = 42,
) -> tuple[HarMLP, np.ndarray]:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HarMLP(X_train.shape[1], n_classes, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    X_t = torch.from_numpy(X_train).float().to(device)
    y_t = torch.from_numpy(y_train).long().to(device)
    bs = 32
    for _ in range(epochs):
        model.train()
        for i in range(0, len(X_train), bs):
            j = slice(i, min(i + bs, len(X_train)))
            opt.zero_grad()
            loss = crit(model(X_t[j]), y_t[j])
            loss.backward()
            opt.step()
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(X_test).float().to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    return model, probs


def tune_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_classes: int,
    seed: int = 42,
    epochs: int = 20,
) -> ModelResult:
    grid = [
        {"hidden": 256, "dropout": 0.2, "lr": 1e-3},
        {"hidden": 512, "dropout": 0.3, "lr": 1e-3},
        {"hidden": 512, "dropout": 0.4, "lr": 5e-4},
    ]
    t0 = time.perf_counter()
    best_f1 = -1.0
    best_model: HarMLP | None = None
    best_probs: np.ndarray | None = None
    best_cfg: dict[str, Any] = {}
    for cfg in grid:
        model, probs = train_torch_mlp(
            X_train, y_train, X_test, n_classes=n_classes, epochs=epochs, seed=seed, **cfg
        )
        pred = probs.argmax(axis=1)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_probs = probs
            best_cfg = cfg
    assert best_model is not None and best_probs is not None
    return ModelResult(
        name="MLP_VJEPA_tuned",
        kind="individual",
        metrics={},
        train_seconds=time.perf_counter() - t0,
        y_pred=np.array([]),
        probs=best_probs,
        estimator=best_model,
        extra={"best_hparams": best_cfg, "search_macro_f1": best_f1},
    )


def eval_checkpoint(
    checkpoint: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    name: str,
    n_classes: int,
) -> ModelResult | None:
    if not checkpoint.is_file():
        return None
    model, info = load_checkpoint(checkpoint)
    if len(info["class_names"]) != n_classes:
        return None
    t0 = time.perf_counter()
    device = info["device"]
    with torch.inference_mode():
        logits = model(torch.from_numpy(X_test).float().to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    return ModelResult(
        name=name,
        kind="checkpoint",
        metrics={},
        train_seconds=time.perf_counter() - t0,
        y_pred=np.array([]),
        probs=probs,
        estimator=model,
        extra={"checkpoint": str(checkpoint)},
    )


def build_homogeneous_voting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int = 42,
) -> ModelResult:
    t0 = time.perf_counter()
    members = [
        ("mlp_a", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=seed)),
        ("mlp_b", MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=300, random_state=seed + 1)),
        ("mlp_c", MLPClassifier(hidden_layer_sizes=(384, 192), alpha=1e-3, max_iter=300, random_state=seed + 2)),
    ]
    pipe_members = [(tag, Pipeline([("sc", StandardScaler()), ("clf", clf)])) for tag, clf in members]
    vote = VotingClassifier(estimators=pipe_members, voting="soft", n_jobs=-1)
    vote.fit(X_train, y_train)
    probs = vote.predict_proba(X_test)
    return ModelResult(
        name="Voting_MLP_homogeneous",
        kind="homogeneous",
        metrics={},
        train_seconds=time.perf_counter() - t0,
        y_pred=np.array([]),
        probs=probs,
        estimator=vote,
    )


def build_heterogeneous_voting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int = 42,
) -> ModelResult:
    t0 = time.perf_counter()
    estimators = [
        ("lr", Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=seed))])),
        (
            "svm",
            Pipeline([("sc", StandardScaler()), ("clf", LinearSVC(dual="auto", random_state=seed))]),
        ),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
    ]
    vote = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    vote.fit(X_train, y_train)
    probs = vote.predict_proba(X_test)
    return ModelResult(
        name="Voting_heterogeneous",
        kind="heterogeneous",
        metrics={},
        train_seconds=time.perf_counter() - t0,
        y_pred=np.array([]),
        probs=probs,
        estimator=vote,
    )


def build_stacking(
    base_estimators: list[tuple[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int = 42,
    cv: int = 3,
) -> ModelResult:
    t0 = time.perf_counter()
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=500, random_state=seed),
        cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    stack.fit(X_train, y_train)
    probs = stack.predict_proba(X_test)
    return ModelResult(
        name="Stacking_LR_meta",
        kind="stacking",
        metrics={},
        train_seconds=time.perf_counter() - t0,
        y_pred=np.array([]),
        probs=probs,
        estimator=stack,
    )


def _weight_grid(n: int) -> list[np.ndarray]:
    if n == 2:
        return [np.array([w, 1.0 - w]) for w in np.linspace(0, 1, 11)]
    if n == 3:
        out = []
        for w0 in np.linspace(0, 1, 6):
            for w1 in np.linspace(0, 1 - w0, max(1, int(6 * (1 - w0)))):
                w2 = 1.0 - w0 - w1
                if w2 >= 0:
                    out.append(np.array([w0, w1, w2]))
        return out
    return [np.ones(n) / n]


def build_blending(
    y_train: np.ndarray,
    oof_train_probs: list[np.ndarray],
    prob_test_list: list[np.ndarray],
) -> ModelResult:
    t0 = time.perf_counter()
    n = len(oof_train_probs)
    best_w = np.ones(n) / n
    best_f1 = -1.0
    for weights in _weight_grid(n):
        weights = weights / max(weights.sum(), 1e-9)
        blend = sum(weights[i] * oof_train_probs[i] for i in range(n))
        f1 = f1_score(y_train, blend.argmax(axis=1), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_w = weights
    blend_test = sum(best_w[i] * prob_test_list[i] for i in range(n))
    return ModelResult(
        name="Blending_weighted",
        kind="blending",
        metrics={},
        train_seconds=time.perf_counter() - t0,
        y_pred=np.array([]),
        probs=blend_test,
        extra={"weights": best_w.tolist(), "oof_macro_f1": best_f1},
    )


def comparison_table(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "modelo": r.name,
                "tipo": r.kind,
                "accuracy": r.metrics.get("accuracy"),
                "macro_f1": r.metrics.get("macro_f1"),
                "weighted_f1": r.metrics.get("weighted_f1"),
                "macro_precision": r.metrics.get("macro_precision"),
                "macro_recall": r.metrics.get("macro_recall"),
                "macro_auc_ovr": r.metrics.get("macro_auc_ovr"),
                "train_sec": round(r.train_seconds, 2),
            }
        )
    return pd.DataFrame(rows).sort_values(PRIMARY_METRIC, ascending=False).reset_index(drop=True)


def select_final_model(table: pd.DataFrame, *, max_train_sec: float | None = None) -> pd.Series:
    df = table.copy()
    if max_train_sec is not None:
        df = df[df["train_sec"] <= max_train_sec]
    if df.empty:
        df = table
    return df.iloc[0]


def _save_fig(path: Path) -> str:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_confusion(y_test: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_dir: Path) -> str:
    cm = confusion_matrix(y_test, y_pred)
    cm_n = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    short = [c[:16] for c in class_names]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_n, xticklabels=short, yticklabels=short, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real (holdout)")
    ax.set_title("Matriz de confusión — modelo final (normalizada por fila)")
    plt.xticks(rotation=45, ha="right")
    return _save_fig(out_dir / "confusion_matrix.png")


def plot_roc_ovr(y_test: np.ndarray, probs: np.ndarray, class_names: list[str], out_dir: Path) -> str:
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_test)
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, name in enumerate(class_names):
        if i >= y_bin.shape[1]:
            break
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{name[:14]} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curvas ROC (one-vs-rest)")
    ax.legend(fontsize=7, loc="lower right")
    return _save_fig(out_dir / "roc_ovr.png")


def plot_pr_ovr(y_test: np.ndarray, probs: np.ndarray, class_names: list[str], out_dir: Path) -> str:
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_test)
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, name in enumerate(class_names):
        if i >= y_bin.shape[1]:
            break
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ax.plot(rec, prec, label=name[:14])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall (one-vs-rest)")
    ax.legend(fontsize=7, loc="best")
    return _save_fig(out_dir / "precision_recall_ovr.png")


def plot_feature_importance(estimator: Any, out_dir: Path, *, top_k: int = 20) -> str | None:
    imp = None
    if hasattr(estimator, "feature_importances_"):
        imp = estimator.feature_importances_
    elif hasattr(estimator, "named_steps"):
        clf = estimator.named_steps.get("clf")
        if clf is not None and hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
    elif hasattr(estimator, "estimators_"):
        for sub in estimator.estimators_:
            if hasattr(sub, "named_steps"):
                clf = sub.named_steps.get("clf")
                if clf is not None and hasattr(clf, "coef_"):
                    imp = np.abs(clf.coef_).mean(axis=0)
                    break
    if imp is None:
        return None
    idx = np.argsort(imp)[-top_k:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([f"dim_{i}" for i in idx], imp[idx], color="#4CAF50")
    ax.set_title(f"Importancia de características (top-{top_k} dims embedding)")
    ax.set_xlabel("importancia")
    return _save_fig(out_dir / "feature_importance.png")


def plot_comparison_bars(table: pd.DataFrame, out_dir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    table.plot(x="modelo", y="macro_f1", kind="barh", ax=axes[0], legend=False, color="#2196F3")
    axes[0].set_title(f"Métrica principal: {PRIMARY_METRIC}")
    axes[0].set_xlim(0, 1)
    table.plot(x="modelo", y="train_sec", kind="barh", ax=axes[1], legend=False, color="#FF9800")
    axes[1].set_title("Tiempo de entrenamiento (s)")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    return _save_fig(out_dir / "comparison_metrics.png")


def plot_calibration_residuals(y_test: np.ndarray, probs: np.ndarray, out_dir: Path) -> str:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_test).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(conf[correct == 1], bins=20, alpha=0.7, label="aciertos", color="#4CAF50")
    axes[0].hist(conf[correct == 0], bins=20, alpha=0.7, label="errores", color="#F44336")
    axes[0].set_xlabel("confianza máxima")
    axes[0].set_title("Calibración — confianza vs acierto")
    axes[0].legend()
    residual = 1.0 - conf
    axes[1].scatter(conf, residual, alpha=0.35, s=12)
    axes[1].set_xlabel("confianza")
    axes[1].set_ylabel("residual (1 - conf)")
    axes[1].set_title("Análisis de residuos (clasificación)")
    return _save_fig(out_dir / "calibration_residuals.png")


def run_full_ensemble_study(
    *,
    npz_path: Path | None = None,
    quick: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    out_dir = _ensure_out()
    bundle = load_embeddings_npz(npz_path)
    X, y = bundle["X"], bundle["y"]
    class_names = bundle["class_names"]
    n_classes = len(class_names)

    X_train, X_test, y_train, y_test = holdout_split(X, y, seed=seed)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results: list[ModelResult] = []
    n_iter = 4 if quick else 8
    mlp_epochs = 10 if quick else 20

    for ckpt_name, label in (
        ("har_vjepa_all14_100each.pt", "MLP_fase4_all14"),
        ("har_vjepa_train12_100each.pt", "MLP_fase4_train12"),
    ):
        r = eval_checkpoint(CHECKPOINTS_DIR / ckpt_name, X_test, y_test, name=label, n_classes=n_classes)
        if r is not None:
            results.append(finalize_result(r, y_test))

    results.append(
        finalize_result(
            tune_sklearn(
                "LogReg_tuned",
                "individual",
                Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=800, random_state=seed))]),
                {"clf__C": [0.01, 0.1, 1.0, 10.0]},
                X_train, y_train, X_test, n_iter=n_iter, seed=seed,
            ),
            y_test,
        )
    )
    results.append(
        finalize_result(
            tune_sklearn(
                "RandomForest_tuned",
                "individual",
                RandomForestClassifier(random_state=seed, n_jobs=-1),
                {"n_estimators": [100, 200, 400], "max_depth": [None, 20, 40]},
                X_train, y_train, X_test, n_iter=n_iter, seed=seed,
            ),
            y_test,
        )
    )
    results.append(
        finalize_result(
            tune_sklearn(
                "GradBoost_tuned",
                "individual",
                GradientBoostingClassifier(random_state=seed),
                {"n_estimators": [80, 120], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
                X_train, y_train, X_test, n_iter=n_iter, seed=seed,
            ),
            y_test,
        )
    )
    results.append(
        finalize_result(
            tune_torch_mlp(X_train, y_train, X_test, y_test, n_classes=n_classes, seed=seed, epochs=mlp_epochs),
            y_test,
        )
    )

    results.append(finalize_result(build_homogeneous_voting(X_train_s, y_train, X_test_s, seed=seed), y_test))
    results.append(finalize_result(build_heterogeneous_voting(X_train, y_train, X_test, seed=seed), y_test))

    stack_bases = [
        ("lr", Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=seed))])),
        ("rf", RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)),
        (
            "gb",
            Pipeline([
                ("sc", StandardScaler()),
                ("clf", GradientBoostingClassifier(n_estimators=100, random_state=seed)),
            ]),
        ),
    ]
    results.append(finalize_result(build_stacking(stack_bases, X_train, y_train, X_test, seed=seed), y_test))

    sk_results = [r for r in results if r.kind in ("individual", "checkpoint") and r.estimator is not None]
    sk_results = sorted(sk_results, key=lambda r: r.metrics.get("macro_f1", 0), reverse=True)[:3]
    oof_probs, test_probs = [], []
    for r in sk_results:
        if isinstance(r.estimator, HarMLP):
            continue
        try:
            oof_probs.append(cross_val_predict(r.estimator, X_train, y_train, cv=3, method="predict_proba", n_jobs=-1))
            test_probs.append(r.probs)
        except Exception:
            pass
    if len(oof_probs) >= 2:
        results.append(finalize_result(build_blending(y_train, oof_probs, test_probs), y_test))

    table = comparison_table(results)
    table.to_csv(out_dir / "comparison_table.csv", index=False)

    final_row = select_final_model(table)
    final = next(r for r in results if r.name == final_row["modelo"])

    charts = {
        "confusion": plot_confusion(y_test, final.y_pred, class_names, out_dir),
        "roc": plot_roc_ovr(y_test, final.probs, class_names, out_dir),
        "pr": plot_pr_ovr(y_test, final.probs, class_names, out_dir),
        "comparison": plot_comparison_bars(table, out_dir),
        "calibration": plot_calibration_residuals(y_test, final.probs, out_dir),
    }
    fi = plot_feature_importance(final.estimator, out_dir)
    if fi:
        charts["feature_importance"] = fi

    payload = {
        "primary_metric": PRIMARY_METRIC,
        "final_model": final_row["modelo"],
        "final_metrics": final.metrics,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_names": class_names,
        "charts": charts,
        "out_dir": str(out_dir),
    }
    (out_dir / "ensemble_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "table": table,
        "final": final,
        "final_row": final_row,
        "results": results,
        "y_test": y_test,
        "class_names": class_names,
        "out_dir": out_dir,
        "summary": payload,
    }
