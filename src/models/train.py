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
    n_samples = 2500

    spatial_ratio_nominal = np.random.normal(0.65, 0.1, 1500)
    spatial_ratio_anomaly = np.random.uniform(1.6, 2.5, 500)
    spatial_ratio_idle = np.random.normal(0.4, 0.05, 500)
    spatial_aspect_ratio = np.concatenate(
        [spatial_ratio_nominal, spatial_ratio_anomaly, spatial_ratio_idle]
    )

    z_mov_nominal = np.random.normal(0.0, 0.5, 1500)
    z_mov_anomaly = np.random.normal(3.5, 0.8, 500)
    z_mov_idle = np.random.normal(-1.2, 0.2, 500)
    mov_intensity_z = np.concatenate([z_mov_nominal, z_mov_anomaly, z_mov_idle])

    camera_proximity_index = np.random.uniform(0.001, 0.03, n_samples)
    lighting_stability_index = np.random.uniform(0.1, 0.8, n_samples)
    tool_proximity_proxy = mov_intensity_z * (lighting_stability_index + 1e-5)

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
        if row["mov_intensity_z"] > 2.5 or row["spatial_aspect_ratio"] > 1.5:
            labels.append("CRITICAL_ANOMALY")
        elif row["mov_intensity_z"] < -1.0:
            labels.append("LINE_IDLE")
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
        max_depth=4, criterion="gini", min_samples_split=10, random_state=42
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
