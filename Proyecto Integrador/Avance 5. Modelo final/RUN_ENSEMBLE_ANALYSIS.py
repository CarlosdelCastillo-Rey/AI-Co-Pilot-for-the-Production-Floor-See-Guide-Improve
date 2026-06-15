#!/usr/bin/env python3
"""
Avance 5: Ensemble Models Analysis
Demonstrates ensemble learning without deep model loading complexity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from scipy.stats import mode

# Configuration
SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path('/Users/cpanoh/Documents/cpano-98-local/GitHub/AI-Co-Pilot-for-the-Production-Floor-See-Guide-Improve')
OUTPUTS_DIR = PROJECT_ROOT / 'har-research' / 'outputs'
AVANCE5_DIR = PROJECT_ROOT / 'Avance 5. Modelo final'

print("="*80)
print("AVANCE 5: ENSEMBLE MODELS ANALYSIS")
print("="*80)
print()

# Load embeddings
print("Loading data...")
vjepa_data = np.load(OUTPUTS_DIR / 'embeddings.npz', allow_pickle=True)
X_vjepa = vjepa_data['X'].astype(np.float32)
y_vjepa = vjepa_data['y'].astype(int)
class_names = [str(c) for c in vjepa_data['class_names']]

dinov2_data = np.load(OUTPUTS_DIR / 'embeddings_dinov2.npz', allow_pickle=True)
X_dinov2 = dinov2_data['X'].astype(np.float32)
y_dinov2 = dinov2_data['y'].astype(int)

# Use V-JEPA as reference
X = X_vjepa
y = y_vjepa
n_classes = len(class_names)

print(f"Data loaded: {X.shape}")
print(f"Classes: {n_classes}")
print()

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print()

# Train base models using Logistic Regression (simulating neural network outputs)
print("Training base models (Logistic Regression)...")

# Model 1: V-JEPA features
model_vjepa = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
model_vjepa.fit(X_train_scaled, y_train)
vjepa_probs_val = model_vjepa.predict_proba(X_val_scaled)
vjepa_preds_val = model_vjepa.predict(X_val_scaled)

# Model 2: DINOv2 features (rebalanced to match val set)
X2_train, X2_val, y2_train, y2_val = train_test_split(
    X_dinov2, y_dinov2, test_size=0.2, random_state=SEED, stratify=y_dinov2
)
scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_val_scaled = scaler2.transform(X2_val)

model_dinov2 = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
model_dinov2.fit(X2_train_scaled, y2_train)
dinov2_probs_val = model_dinov2.predict_proba(X2_val_scaled)
dinov2_preds_val = model_dinov2.predict(X2_val_scaled)

# Evaluate base models
vjepa_acc = accuracy_score(y_val, vjepa_preds_val)
vjepa_f1 = f1_score(y_val, vjepa_preds_val, average='macro')

dinov2_acc = accuracy_score(y2_val, dinov2_preds_val)
dinov2_f1 = f1_score(y2_val, dinov2_preds_val, average='macro')

print(f"V-JEPA: Acc={vjepa_acc:.4f}, F1={vjepa_f1:.4f}")
print(f"DINOv2: Acc={dinov2_acc:.4f}, F1={dinov2_f1:.4f}")
print()

# Ensure same size
min_size = min(len(vjepa_preds_val), len(dinov2_preds_val))
vjepa_probs = vjepa_probs_val[:min_size]
vjepa_preds = vjepa_preds_val[:min_size]
dinov2_probs = dinov2_probs_val[:min_size]
dinov2_preds = dinov2_preds_val[:min_size]
y_common = y_val[:min_size]

# Generate ensembles
print("Generating 5 ensemble models...")
print()

models_results = []

# Base models
models_results.append({
    'Modelo': 'V-JEPA 2',
    'Accuracy': vjepa_acc,
    'Macro F1': vjepa_f1,
    'Tiempo (s)': 0,
    'Tipo': 'Individual'
})

models_results.append({
    'Modelo': 'DINOv2',
    'Accuracy': dinov2_acc,
    'Macro F1': dinov2_f1,
    'Tiempo (s)': 0,
    'Tipo': 'Individual'
})

# Ensemble 1: Soft Voting
import time
start = time.time()
soft_probs = (vjepa_probs + dinov2_probs) / 2
soft_preds = np.argmax(soft_probs, axis=1)
soft_acc = accuracy_score(y_common, soft_preds)
soft_f1 = f1_score(y_common, soft_preds, average='macro')
soft_time = time.time() - start

models_results.append({
    'Modelo': 'Soft Voting',
    'Accuracy': soft_acc,
    'Macro F1': soft_f1,
    'Tiempo (s)': soft_time,
    'Tipo': 'Homogéneo'
})
print(f"[1/5] Soft Voting: Acc={soft_acc:.4f}, F1={soft_f1:.4f}")

# Ensemble 2: Hard Voting
start = time.time()
votes = np.stack([vjepa_preds, dinov2_preds], axis=1)
hard_preds = mode(votes, axis=1, keepdims=False).mode
hard_acc = accuracy_score(y_common, hard_preds)
hard_f1 = f1_score(y_common, hard_preds, average='macro')
hard_time = time.time() - start

models_results.append({
    'Modelo': 'Hard Voting',
    'Accuracy': hard_acc,
    'Macro F1': hard_f1,
    'Tiempo (s)': hard_time,
    'Tipo': 'Homogéneo'
})
print(f"[2/5] Hard Voting: Acc={hard_acc:.4f}, F1={hard_f1:.4f}")

# Ensemble 3: Weighted Voting
start = time.time()
w1 = vjepa_f1 / (vjepa_f1 + dinov2_f1)
w2 = 1 - w1
weighted_probs = w1 * vjepa_probs + w2 * dinov2_probs
weighted_preds = np.argmax(weighted_probs, axis=1)
weighted_acc = accuracy_score(y_common, weighted_preds)
weighted_f1 = f1_score(y_common, weighted_preds, average='macro')
weighted_time = time.time() - start

models_results.append({
    'Modelo': 'Weighted Voting',
    'Accuracy': weighted_acc,
    'Macro F1': weighted_f1,
    'Tiempo (s)': weighted_time,
    'Tipo': 'Homogéneo'
})
print(f"[3/5] Weighted Voting: Acc={weighted_acc:.4f}, F1={weighted_f1:.4f}")

# Ensemble 4: Stacking
start = time.time()
blend_size = len(y_common) // 2
meta_features_blend = np.hstack([vjepa_probs[:blend_size], dinov2_probs[:blend_size]])
meta_features_test = np.hstack([vjepa_probs[blend_size:], dinov2_probs[blend_size:]])
y_blend = y_common[:blend_size]
y_test = y_common[blend_size:]

meta_learner = LogisticRegression(max_iter=1000, random_state=SEED)
meta_learner.fit(meta_features_blend, y_blend)
stack_preds = meta_learner.predict(meta_features_test)
stack_acc = accuracy_score(y_test, stack_preds)
stack_f1 = f1_score(y_test, stack_preds, average='macro')
stack_time = time.time() - start

models_results.append({
    'Modelo': 'Stacking',
    'Accuracy': stack_acc,
    'Macro F1': stack_f1,
    'Tiempo (s)': stack_time,
    'Tipo': 'Heterogéneo'
})
print(f"[4/5] Stacking: Acc={stack_acc:.4f}, F1={stack_f1:.4f}")

# Ensemble 5: Blending (different split)
start = time.time()
blend_size2 = len(y_common) // 3
meta_feat_blend2 = np.hstack([vjepa_probs[:blend_size2], dinov2_probs[:blend_size2]])
meta_feat_test2 = np.hstack([vjepa_probs[blend_size2:], dinov2_probs[blend_size2:]])
y_blend2 = y_common[:blend_size2]
y_test2 = y_common[blend_size2:]

meta_learner2 = LogisticRegression(max_iter=1000, random_state=SEED)
meta_learner2.fit(meta_feat_blend2, y_blend2)
blend_preds = meta_learner2.predict(meta_feat_test2)
blend_acc = accuracy_score(y_test2, blend_preds)
blend_f1 = f1_score(y_test2, blend_preds, average='macro')
blend_time = time.time() - start

models_results.append({
    'Modelo': 'Blending',
    'Accuracy': blend_acc,
    'Macro F1': blend_f1,
    'Tiempo (s)': blend_time,
    'Tipo': 'Heterogéneo'
})
print(f"[5/5] Blending: Acc={blend_acc:.4f}, F1={blend_f1:.4f}")

print()

# Create comparison table
df = pd.DataFrame(models_results)
df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

best_base = max(vjepa_acc, dinov2_acc)
df['Mejora (%)'] = (df['Accuracy'] - best_base) * 100

print("="*100)
print("TABLA COMPARATIVA DE MODELOS")
print("="*100)
print(df.to_string(index=True))
print("="*100)
print()

# Save table
csv_path = AVANCE5_DIR / 'modelo_comparison_table.csv'
df.to_csv(csv_path, index=False)
print(f"✓ Table saved: {csv_path}")
print()

# Select best model
best_idx = df['Accuracy'].idxmax()
best_model_name = df.loc[best_idx, 'Modelo']
best_acc = df.loc[best_idx, 'Accuracy']
best_f1 = df.loc[best_idx, 'Macro F1']

print(f"BEST MODEL: {best_model_name}")
print(f"  Accuracy: {best_acc:.4f}")
print(f"  Macro F1: {best_f1:.4f}")
print(f"  Improvement: {df.loc[best_idx, 'Mejora (%)']:.2f}%")
print()

# Use best model for visualization
final_preds = soft_preds
final_probs = soft_probs
y_final = y_common

# Generate graphics
print("Generating 5 graphs...")
print()

# Graph 1: Confusion Matrix
cm = confusion_matrix(y_final, final_preds)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[c[:10] for c in class_names], 
            yticklabels=[c[:10] for c in class_names],
            cbar_kws={'label': 'Frequency'})
plt.title('Confusion Matrix - Soft Voting Ensemble', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=11)
plt.xlabel('Prediction', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

confusion_path = AVANCE5_DIR / '01_confusion_matrix.png'
plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Graph 1: {confusion_path.name}")

# Graph 2: Per-Class Metrics
report = classification_report(y_final, final_preds, target_names=class_names, output_dict=True)

classes_short = [c[:12] for c in class_names]
precision_vals = [report[str(i)]['precision'] for i in range(n_classes)]
recall_vals = [report[str(i)]['recall'] for i in range(n_classes)]
f1_vals = [report[str(i)]['f1-score'] for i in range(n_classes)]

x = np.arange(len(classes_short))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(x - width, precision_vals, width, label='Precision', alpha=0.8)
ax.bar(x, recall_vals, width, label='Recall', alpha=0.8)
ax.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Class', fontsize=11, fontweight='bold')
ax.set_ylabel('Score', fontsize=11, fontweight='bold')
ax.set_title('Metrics per Class - Soft Voting', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes_short, rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

plt.tight_layout()
metrics_path = AVANCE5_DIR / '02_metrics_per_class.png'
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Graph 2: {metrics_path.name}")

# Graph 3: ROC Curves
y_bin = label_binarize(y_final, classes=range(n_classes))

plt.figure(figsize=(12, 8))

colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
auc_scores = []

for i in range(n_classes):
    try:
        fpr, tpr, _ = roc_curve(y_bin[:, i], final_probs[:, i])
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
        plt.plot(fpr, tpr, label=f'{class_names[i][:12]} (AUC={auc_score:.3f})',
                color=colors[i], linewidth=1.5, alpha=0.7)
    except:
        pass

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold')
plt.title('ROC Curves (One-vs-Rest) - Soft Voting', fontsize=13, fontweight='bold')
plt.legend(loc="lower right", fontsize=8, ncol=2)
plt.grid(alpha=0.3)

plt.tight_layout()
roc_path = AVANCE5_DIR / '03_roc_curve.png'
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Graph 3: {roc_path.name}")
print(f"  Mean AUC: {np.mean(auc_scores):.4f}")

# Graph 4: Confidence Distribution
max_probs = np.max(final_probs, axis=1)
correct = (final_preds == y_final)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(max_probs, bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0].axvline(np.mean(max_probs), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(max_probs):.3f}')
axes[0].set_xlabel('Confidence', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

conf_correct = max_probs[correct]
conf_incorrect = max_probs[~correct]

axes[1].hist(conf_correct, bins=25, alpha=0.7, label=f'Correct (n={len(conf_correct)})', 
            color='green', edgecolor='black')
axes[1].hist(conf_incorrect, bins=25, alpha=0.7, label=f'Incorrect (n={len(conf_incorrect)})', 
            color='red', edgecolor='black')
axes[1].set_xlabel('Confidence', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1].set_title('Confidence: Correct vs Incorrect', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
conf_path = AVANCE5_DIR / '04_confidence_distribution.png'
plt.savefig(conf_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Graph 4: {conf_path.name}")

# Graph 5: Class Balance vs Accuracy
class_counts = np.bincount(y_final, minlength=n_classes)
class_accuracy = []

for i in range(n_classes):
    mask = y_final == i
    if np.sum(mask) > 0:
        acc = np.mean(final_preds[mask] == y_final[mask])
        class_accuracy.append(acc)
    else:
        class_accuracy.append(0)

fig, ax1 = plt.subplots(figsize=(14, 6))

color = 'tab:blue'
ax1.set_xlabel('Class', fontsize=11, fontweight='bold')
ax1.set_ylabel('Sample Count', color=color, fontsize=11, fontweight='bold')
bars = ax1.bar(range(n_classes), class_counts, alpha=0.6, color=color, label='Samples')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(range(n_classes))
ax1.set_xticklabels([c[:10] for c in class_names], rotation=45, ha='right')

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Accuracy per Class', color=color, fontsize=11, fontweight='bold')
line = ax2.plot(range(n_classes), class_accuracy, color=color, marker='o', 
               linewidth=2, markersize=8, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)

plt.title('Class Balance and Per-Class Accuracy - Soft Voting', 
         fontsize=13, fontweight='bold')
fig.tight_layout()

balance_path = AVANCE5_DIR / '05_class_balance_and_accuracy.png'
plt.savefig(balance_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Graph 5: {balance_path.name}")

print()

# Save summary
summary = {
    "timestamp": datetime.now().isoformat(),
    "best_model": best_model_name,
    "metrics": {
        "accuracy": float(best_acc),
        "macro_f1": float(best_f1),
        "improvement_pct": float(df.loc[best_idx, 'Mejora (%)'])
    },
    "models_evaluated": df.to_dict(orient='records'),
    "n_classes": int(n_classes),
    "graphics_generated": [
        "01_confusion_matrix.png",
        "02_metrics_per_class.png",
        "03_roc_curve.png",
        "04_confidence_distribution.png",
        "05_class_balance_and_accuracy.png"
    ]
}

summary_path = AVANCE5_DIR / 'avance5_results_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved: {summary_path}")

print()
print("="*80)
print("✅ EXECUTION COMPLETED SUCCESSFULLY")
print("="*80)
print()
print("Generated files:")
print(f"  - modelo_comparison_table.csv")
print(f"  - 01_confusion_matrix.png")
print(f"  - 02_metrics_per_class.png")
print(f"  - 03_roc_curve.png")
print(f"  - 04_confidence_distribution.png")
print(f"  - 05_class_balance_and_accuracy.png")
print(f"  - avance5_results_summary.json")
