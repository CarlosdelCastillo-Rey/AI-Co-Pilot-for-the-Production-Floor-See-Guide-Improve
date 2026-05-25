from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_live_stream_baseline():
    """
    Trains the Predictive Baseline Model optimized for live streaming pipelines.
    Aligned with digital twin literature for lag mitigation and
    frame-by-frame postural/kinematic edge analysis.
    """
    # 1. Configure relative paths based on the project structure
    ROOT_DIR = Path(__file__).resolve().parents[2]
    MODEL_OUTPUT_PATH = ROOT_DIR / "vision-ops-backend" / "baseline_tree.joblib"

    print(f"[*] Starting Baseline model training for Live Stream...")
    print(f"[*] The model will be exported directly to: {MODEL_OUTPUT_PATH}")

    # 2. Generate Synthetic dataset simulating the feature distributions from features.py
    np.random.seed(42)
    n_samples = 3000

    s = 3000

    spatial_aspect_ratio = np.concatenate(
        [
            np.random.normal(0.85, 0.20, 1800),
            np.random.normal(1.20, 0.30, 600),
            np.random.normal(0.65, 0.12, 600),
        ]
    )
    mov_intensity_z = np.concatenate(
        [
            np.random.normal(0.0, 0.8, 1800),
            np.random.normal(1.6, 0.9, 600),
            np.random.normal(-0.6, 0.4, 600),
        ]
    )
    camera_proximity_index = np.random.uniform(0.001, 0.03, n_samples)
    lighting_stability_index = np.random.uniform(0.1, 0.8, n_samples)

    tool_proximity_proxy = np.abs(mov_intensity_z) * (lighting_stability_index + 0.2)

    df = pd.DataFrame(
        {
            "spatial_aspect_ratio": spatial_aspect_ratio,
            "mov_intensity_z": mov_intensity_z,
            "camera_proximity_index": camera_proximity_index,
            "lighting_stability_index": lighting_stability_index,
            "tool_proximity_proxy": tool_proximity_proxy,
        }
    )

    labels = []
    for _, row in df.iterrows():
        noise = np.random.normal(1.0, 0.15)

        kinematic_score = row["mov_intensity_z"] * noise
        postural_score = row["spatial_aspect_ratio"] * noise
        proxy_score = row["tool_proximity_proxy"]

        if kinematic_score > 1.4 and postural_score > 1.1:
            labels.append("CRITICAL_ANOMALY")
        elif kinematic_score < -0.4 and proxy_score < 0.25:
            labels.append("LINE_IDLE")
        elif 0.8 < postural_score < 1.3 and 0.5 < kinematic_score < 1.5:
            labels.append(np.random.choice(["NOMINAL_OPERATION", "CRITICAL_ANOMALY"], p=[0.7, 0.3]))
        else:
            labels.append("NOMINAL_OPERATION")

    df["target_state"] = labels

    # 2. Feature splitting and validation partitioning
    feature_cols = [
        "spatial_aspect_ratio",
        "mov_intensity_z",
        "camera_proximity_index",
        "lighting_stability_index",
        "tool_proximity_proxy",
    ]
    X = df[feature_cols]
    y = df["target_state"]

    # 3. 80-20% split to evaluate & mitigate under/overfitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Decision Tree Classifier set up
    # Max_depth=4 restricts overgrowth, guaranteeing a lightweight model
    # to avoid blocking the background thread (threading.Thread) of the camera pipeline.
    model = DecisionTreeClassifier(
        max_depth=8,
        criterion="gini",
        min_samples_leaf=5,
        min_samples_split=2,
        splitter="best",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 5. Multiclass metric evaluation (Precision, Recall, F1-Score)
    y_pred = model.predict(X_test)
    print(f"Model Metrics on Synthetic Dataset:\n")
    print(classification_report(y_test, y_pred))

    # 6. Feature Importance extraction to verify variance response
    print("Feature Importances:\n")
    for col, importance in zip(feature_cols, model.feature_importances_):
        print(f" Feature: {col:<25} | Relative Importance: {importance:.4f}")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nSuccess! Baseline model binary saved to backend root.")


if __name__ == "__main__":
    train_live_stream_baseline()
