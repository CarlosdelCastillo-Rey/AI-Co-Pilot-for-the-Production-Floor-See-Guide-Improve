from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report


def model_experimentation_pipeline():
    """
    Executes a structured machine learning experimentation pipeline for the Predictive Baseline Model.
    Trains 6 distinct models, evaluates streaming performance, identifies the top 2,
    and exports the best model.
    """
    ROOT_DIR = Path(__file__).resolve().parents[2]
    MODEL_OUTPUT_PATH = ROOT_DIR / "vision-ops-backend" / "baseline_tree.joblib"

    # 1. Generate Synthetic dataset simulating the feature distributions from features.py
    np.random.seed(42)
    n_samples = 3000

    spatial_aspect_ratio = np.concatenate(
        [
            np.random.normal(0.85, 0.20, 1800),
            np.random.uniform(1.20, 0.30, 600),
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

    feature_cols = [
        "spatial_aspect_ratio",
        "mov_intensity_z",
        "camera_proximity_index",
        "lighting_stability_index",
        "tool_proximity_proxy",
    ]
    X = df[feature_cols]
    y = df["target_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_models = {
        "Model_1_Decision_Tree_Gini": DecisionTreeClassifier(
            criterion="gini", max_depth=4, random_state=42
        ),
        "Model_2_Decision_Tree_Entropy": DecisionTreeClassifier(
            criterion="entropy", max_depth=4, random_state=42
        ),
        "Model_3_Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Model_4_SVM_Linear": SVC(kernel="linear", probability=True, random_state=42),
        "Model_5_SVM_RBF": SVC(kernel="rbf", probability=True, random_state=42),
        "Model_6_KNN": KNeighborsClassifier(n_neighbors=5),
    }

    model_scores = {}
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = f1_score(y_test, preds, average="macro")
        model_scores[name] = score
        print(f"- {name:<35} | F1-Score: {score:.4f}")

    sorted_models = sorted(model_scores.items(), key=lambda item: item[1], reverse=True)
    top_1_name, top_2_name = sorted_models[0][0], sorted_models[1][0]

    print(f"\nTop 2 Selected Models for Fine-Tuning: {top_1_name} and {top_2_name}")

    tuned_models = {}

    param_grids = {
        "Decision_Tree": {
            "max_depth": [4, 6, 8, 12, None],
            "min_samples_split": [2, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 4, 5, 10],
            "splitter": ["best", "random"],
        },
        "SVM": {"C": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0], "gamma": ["scale", "auto", 0.01, 0.1, 1.0]},
        "Logistic_Regression": {
            "C": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0],
            "solver": ["lbfgs", "liblinear", "saga"],
            "penalty": ["l2", "l1", "elasticnet", None],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "euclidean", "manhattan"],
        },
    }

    for candidate_name in [top_1_name, top_2_name]:
        print(f"Grid search for hyperparameters optimization for: {candidate_name}...")

        if "Decision_Tree" in candidate_name:
            family = "Decision_Tree"
            estimator = DecisionTreeClassifier(random_state=42)
        elif "SVM" in candidate_name:
            family = "SVM"
            estimator = SVC(probability=True, random_state=42)
        elif "Logistic_Regression" in candidate_name:
            family = "Logistic_Regression"
            estimator = LogisticRegression(max_iter=1000, random_state=42)
        else:
            family = "KNN"
            estimator = KNeighborsClassifier()

        grid_search = GridSearchCV(
            estimator=estimator, param_grid=param_grids[family], scoring="f1_macro", cv=3, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_tuned_score = f1_score(y_test, grid_search.predict(X_test), average="macro")
        tuned_models[candidate_name] = {
            "estimator": grid_search.best_estimator_,
            "score": best_tuned_score,
            "params": grid_search.best_params_,
        }
        print(f"\nBest Config found: {grid_search.best_params_}")
        print(f"\nF1-Score: {best_tuned_score:.4f}\n")

    best_model_name = max(tuned_models, key=lambda k: tuned_models[k]["score"])
    best_model = tuned_models[best_model_name]["estimator"]

    print(f"Model Chosen for Production: {best_model_name}")
    print(f"Hyperparameters: {tuned_models[best_model_name]['params']}")
    print("\nFinal Metrics:")
    print(classification_report(y_test, best_model.predict(X_test)))

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"Best model compiled to production root: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    model_experimentation_pipeline()
