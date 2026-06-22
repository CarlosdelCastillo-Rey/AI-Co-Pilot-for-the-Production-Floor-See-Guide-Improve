"""
VisionOps HAR — Visualizaciones Completas de Resultados
Equipo #56 MNA-V — Tec de Monterrey
"""

import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.manifold import TSNE

t0 = time.time()

# ─── Rutas ───────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path('/Users/cpanoh/Documents/cpano-98-local/GitHub/AI-Co-Pilot-for-the-Production-Floor-See-Guide-Improve')
EMB_VJEPA     = PROJECT_ROOT / 'har-research/outputs/embeddings.npz'
EMB_DINOV2    = PROJECT_ROOT / 'har-research/outputs/embeddings_dinov2.npz'
PIPELINE_JSON = PROJECT_ROOT / 'har-research/outputs/pipeline_v2_run_summary.json'
OUT_DIR       = PROJECT_ROOT / 'Proyecto Integrador/Visualizaciones'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOOTER = "Equipo #56 MNA-V — VisionOps HAR — Tec de Monterrey"

# ─── Paleta ──────────────────────────────────────────────────────────────────
NAVY   = '#1B2A4A'
GOLD   = '#8A8C3E'
BLUE   = '#0059BB'
ORANGE = '#E07B39'
GREEN  = '#2E7D32'
RED    = '#C62828'
LGRAY  = '#F5F5F5'
MGRAY  = '#9E9E9E'

TAB12 = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
    '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
    '#aec7e8','#ffbb78'
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 15,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': LGRAY,
    'axes.grid': True,
    'grid.alpha': 0.4,
})

# ─── Datos embebidos ──────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Consult sheets','Picking in front','Picking left',
    'Put down component','Put down measuring rod','Put down screwdriver',
    'Put down subsystem','Take component','Take measuring rod',
    'Take screwdriver','Take subsystem','Turn sheets'
]
CLASS_SHORT = [
    'Consult','Pick front','Pick left',
    'PD comp.','PD meas.rod','PD screw.',
    'PD subsys.','Take comp.','Take meas.rod',
    'Take screw.','Take subsys.','Turn sheets'
]
N = len(CLASS_NAMES)

CLASS_COUNTS = {
    'Consult sheets':132,'Turn sheets':224,'Take screwdriver':420,
    'Put down screwdriver':416,'Picking in front':456,'Picking left':641,
    'Take component':485,'Put down component':385,'Take measuring rod':76,
    'Put down measuring rod':74,'Take subsystem':39,'Put down subsystem':77
}

MODEL_RESULTS = {
    'PowerMean-7':     {'accuracy':0.8781,'macro_f1':0.8610,'type':'Ensemble★'},
    'V-JEPA 2':        {'accuracy':0.7854,'macro_f1':0.7063,'type':'Individual'},
    'DINOv2':          {'accuracy':0.7635,'macro_f1':0.7220,'type':'Individual'},
    'Stacking':        {'accuracy':0.7580,'macro_f1':0.6037,'type':'Heterogéneo'},
    'Blending':        {'accuracy':0.7484,'macro_f1':0.5606,'type':'Heterogéneo'},
    'Soft Voting':     {'accuracy':0.5153,'macro_f1':0.4443,'type':'Homogéneo'},
    'Hard Voting':     {'accuracy':0.4876,'macro_f1':0.3697,'type':'Homogéneo'},
    'Weighted Voting': {'accuracy':0.4029,'macro_f1':0.3452,'type':'Homogéneo'},
}

TYPE_COLOR = {
    'Ensemble★': GREEN,
    'Individual': BLUE,
    'Heterogéneo': ORANGE,
    'Homogéneo': RED,
}

VJEPA_PC = {
    'Consult sheets':         {'p':0.189,'r':0.625,'f1':0.290,'sup':48},
    'Picking in front':       {'p':0.698,'r':0.366,'f1':0.480,'sup':82},
    'Picking left':           {'p':0.875,'r':0.066,'f1':0.123,'sup':106},
    'Put down component':     {'p':0.364,'r':0.125,'f1':0.186,'sup':32},
    'Put down measuring rod': {'p':0.128,'r':0.417,'f1':0.196,'sup':12},
    'Put down screwdriver':   {'p':0.581,'r':0.295,'f1':0.391,'sup':61},
    'Put down subsystem':     {'p':0.310,'r':0.750,'f1':0.439,'sup':12},
    'Take component':         {'p':0.300,'r':0.088,'f1':0.136,'sup':34},
    'Take measuring rod':     {'p':0.078,'r':0.545,'f1':0.136,'sup':11},
    'Take screwdriver':       {'p':0.839,'r':0.426,'f1':0.565,'sup':61},
    'Take subsystem':         {'p':0.000,'r':0.000,'f1':0.000,'sup':6},
    'Turn sheets':            {'p':0.628,'r':0.598,'f1':0.613,'sup':87},
}
DINOV2_PC = {
    'Consult sheets':         {'p':0.340,'r':0.667,'f1':0.451,'sup':48},
    'Picking in front':       {'p':0.771,'r':0.329,'f1':0.462,'sup':82},
    'Picking left':           {'p':1.000,'r':0.113,'f1':0.203,'sup':106},
    'Put down component':     {'p':0.571,'r':0.500,'f1':0.533,'sup':32},
    'Put down measuring rod': {'p':0.188,'r':0.750,'f1':0.300,'sup':12},
    'Put down screwdriver':   {'p':0.667,'r':0.492,'f1':0.566,'sup':61},
    'Put down subsystem':     {'p':0.400,'r':1.000,'f1':0.571,'sup':12},
    'Take component':         {'p':0.438,'r':0.206,'f1':0.280,'sup':34},
    'Take measuring rod':     {'p':0.088,'r':0.727,'f1':0.157,'sup':11},
    'Take screwdriver':       {'p':0.821,'r':0.377,'f1':0.517,'sup':61},
    'Take subsystem':         {'p':0.333,'r':0.167,'f1':0.222,'sup':6},
    'Turn sheets':            {'p':0.770,'r':0.678,'f1':0.721,'sup':87},
}

# ─── Carga de datos reales ────────────────────────────────────────────────────
print("Cargando embeddings y pipeline summary...")
vj  = np.load(EMB_VJEPA,  allow_pickle=True)
dn  = np.load(EMB_DINOV2, allow_pickle=True)
X_vj, y_vj, subj_vj = vj['X'].astype(np.float32), vj['y'].astype(int), vj['subjects']
X_dn, y_dn           = dn['X'].astype(np.float32), dn['y'].astype(int)

with open(PIPELINE_JSON) as f:
    ps = json.load(f)

vjepa_hist  = ps['steps']['03_train']['vjepa']['history']
dinov2_hist = ps['steps']['03_train']['dinov2']['history']
def parse_arr(s):
    import re
    return np.array([int(x) for x in re.findall(r'\d+', str(s))], dtype=int)

y_val_vj    = parse_arr(ps['steps']['03_train']['vjepa']['y_val'])
y_pred_vj   = parse_arr(ps['steps']['03_train']['vjepa']['y_pred'])
y_val_dn    = parse_arr(ps['steps']['03_train']['dinov2']['y_val'])
y_pred_dn   = parse_arr(ps['steps']['03_train']['dinov2']['y_pred'])
print(f"  V-JEPA: {X_vj.shape}, DINOv2: {X_dn.shape}")

# LogReg para probabilidades (ROC / PR curves)
print("Entrenando LogReg para curvas ROC/PR...")
X_tr, X_te, y_tr, y_te = train_test_split(X_vj, y_vj, test_size=0.2,
                                            random_state=42, stratify=y_vj)
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
lr.fit(X_tr, y_tr)
y_prob = lr.predict_proba(X_te)
y_bin  = label_binarize(y_te, classes=list(range(N)))
print(f"  LogReg acc={lr.score(X_te, y_te):.4f} n_test={len(y_te)}")

# t-SNE (fallback from UMAP)
print("Calculando t-SNE (puede tardar ~60s)...")
idx_tsne = np.random.RandomState(42).choice(len(X_vj), size=min(2000, len(X_vj)), replace=False)
X_tsne_in = X_vj[idx_tsne]
y_tsne    = y_vj[idx_tsne]
X_2d = TSNE(n_components=2, random_state=42, perplexity=40,
             max_iter=1000, init='pca').fit_transform(X_tsne_in)
print("  t-SNE listo.")


def save(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ {name}")


def add_footer(fig):
    fig.text(0.5, 0.01, FOOTER, ha='center', va='bottom',
             fontsize=7, color=MGRAY, style='italic')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 01 — Distribución de clases
# ═══════════════════════════════════════════════════════════════════════════════
def fig01():
    counts = {k: CLASS_COUNTS[k] for k in CLASS_NAMES}
    names  = list(counts.keys())
    vals   = list(counts.values())
    order  = np.argsort(vals)[::-1]
    names_s = [CLASS_SHORT[i] for i in order]
    vals_s  = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 7))
    norm = plt.Normalize(min(vals_s), max(vals_s))
    colors = plt.cm.Blues(norm(vals_s))

    bars = ax.barh(names_s, vals_s, color=colors, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals_s):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                str(v), va='center', fontsize=9, fontweight='bold')

    med = np.median(vals_s)
    ax.axvline(med, color=GOLD, ls='--', lw=1.5, label=f'Mediana ({int(med)})')

    rare_threshold = 80
    for i, (n, v) in enumerate(zip(names_s, vals_s)):
        if v < rare_threshold:
            ax.text(v + 8, i, ' ⚠', va='center', fontsize=11, color=RED)

    ax.set_xlabel('Número de Clips')
    ax.set_title('Distribución de Clips por Clase de Acción\n(InHARD, 12 clases, n=3,425 clips totales)', fontweight='bold')
    ax.legend(loc='lower right')
    ax.text(0.99, 0.02, '⚠ Clases con <80 clips (raras)', transform=ax.transAxes,
            ha='right', fontsize=8, color=RED)
    ax.invert_yaxis()
    add_footer(fig)
    save(fig, 'FIG_01_class_distribution.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 02 — Distribución por sujeto
# ═══════════════════════════════════════════════════════════════════════════════
def fig02():
    subs, cnts = np.unique(subj_vj, return_counts=True)
    val_subs   = {'P02', 'P07', 'P11'}
    colors_s   = [RED if s in val_subs else BLUE for s in subs]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(subs, cnts, color=colors_s, edgecolor='white', linewidth=0.5)
    for bar, c in zip(bars, cnts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(c), ha='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Sujeto')
    ax.set_ylabel('Clips')
    ax.set_title('Clips por Sujeto — Split por Sujeto para Validación No-Sesgada\n(13 entrenamiento / 3 validación)', fontweight='bold')
    ax.legend(handles=[
        mpatches.Patch(color=BLUE, label='Entrenamiento (13 sujetos)'),
        mpatches.Patch(color=RED,  label='Validación (P02, P07, P11)'),
    ], loc='upper right')
    add_footer(fig)
    save(fig, 'FIG_02_subject_distribution.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 03 — Curvas de entrenamiento (train/val loss)
# ═══════════════════════════════════════════════════════════════════════════════
def fig03():
    def extract(hist):
        ep  = [h['epoch']      for h in hist]
        tr  = [h['train_loss'] for h in hist]
        vl  = [h['val_loss']   for h in hist]
        return np.array(ep), np.array(tr), np.array(vl)

    ep_vj, tr_vj, vl_vj = extract(vjepa_hist)
    ep_dn, tr_dn, vl_dn = extract(dinov2_hist)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (ep_v, tr_v, ep_d, tr_d, ylabel, title) in zip(axes, [
        (ep_vj, tr_vj, ep_dn, tr_dn, 'Train Loss', 'Pérdida de Entrenamiento'),
        (ep_vj, vl_vj, ep_dn, vl_dn, 'Val Loss',   'Pérdida de Validación'),
    ]):
        ax.plot(ep_v, tr_v, color=BLUE,   lw=2, label='V-JEPA')
        ax.plot(ep_d, tr_d, color=ORANGE, lw=2, label='DINOv2')

        best_vj = ep_v[np.argmin(tr_v if 'Train' in ylabel else vl_vj)]
        best_dn = ep_d[np.argmin(tr_d if 'Train' in ylabel else vl_dn)]

        ax.set_xlabel('Época')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend()

    # Marcar mejor epoch val en subplot de val
    ax2 = axes[1]
    best_vj_ep = ep_vj[np.argmin(vl_vj)]
    best_dn_ep = ep_dn[np.argmin(vl_dn)]
    ax2.axvline(best_vj_ep, color=BLUE,   ls=':', lw=1.2, alpha=0.7)
    ax2.axvline(best_dn_ep, color=ORANGE, ls=':', lw=1.2, alpha=0.7)
    ax2.annotate(f'Min ep={best_vj_ep}', xy=(best_vj_ep, min(vl_vj)),
                 xytext=(best_vj_ep+5, min(vl_vj)+0.3),
                 fontsize=8, color=BLUE,
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=1))
    ax2.annotate(f'Min ep={best_dn_ep}', xy=(best_dn_ep, min(vl_dn)),
                 xytext=(best_dn_ep+5, min(vl_dn)+0.15),
                 fontsize=8, color=ORANGE,
                 arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1))

    fig.suptitle('Curvas de Entrenamiento: Focal Loss vs Época (100 épocas, split por sujeto)',
                 fontweight='bold', fontsize=15)
    plt.tight_layout()
    add_footer(fig)
    save(fig, 'FIG_03_training_curves.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 04 — Learning rate schedule
# ═══════════════════════════════════════════════════════════════════════════════
def fig04():
    eps_vj = [h['epoch'] for h in vjepa_hist]
    lrs_vj = [h['lr']    for h in vjepa_hist]
    eps_dn = [h['epoch'] for h in dinov2_hist]
    lrs_dn = [h['lr']    for h in dinov2_hist]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eps_vj, lrs_vj, color=BLUE,   lw=2, label='V-JEPA')
    ax.plot(eps_dn, lrs_dn, color=ORANGE, lw=2, label='DINOv2', ls='--')
    ax.set_xlabel('Época')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Programación del Learning Rate\n(Cosine Annealing, lr₀ = 1×10⁻³)', fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.legend()
    add_footer(fig)
    save(fig, 'FIG_04_learning_rate_schedule.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 05 — F1 por clase: V-JEPA vs DINOv2
# ═══════════════════════════════════════════════════════════════════════════════
def fig05():
    f1_vj = [VJEPA_PC[c]['f1']  for c in CLASS_NAMES]
    f1_dn = [DINOV2_PC[c]['f1'] for c in CLASS_NAMES]

    order = np.argsort(f1_vj)[::-1]
    names_o = [CLASS_SHORT[i] for i in order]
    f1_vj_o = [f1_vj[i] for i in order]
    f1_dn_o = [f1_dn[i] for i in order]

    x = np.arange(len(names_o))
    w = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - w/2, f1_vj_o, w, color=BLUE,   label='V-JEPA 2', edgecolor='white')
    b2 = ax.bar(x + w/2, f1_dn_o, w, color=ORANGE, label='DINOv2',   edgecolor='white')

    ax.axhline(0.70, color=GOLD, ls='--', lw=1.5, label='Umbral objetivo (F1=0.70)')

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names_o, rotation=40, ha='right', fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score por Clase: V-JEPA 2 vs DINOv2\n(LogisticRegression sobre embeddings, n_val=501)', fontweight='bold')
    ax.legend()
    add_footer(fig)
    save(fig, 'FIG_05_per_class_f1_comparison.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 06 — Precision y Recall por clase
# ═══════════════════════════════════════════════════════════════════════════════
def fig06():
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    x = np.arange(N)
    w = 0.38

    for ax, metric, title in zip(axes,
        ['p', 'r'], ['Precision', 'Recall']):
        vals_vj = [VJEPA_PC[c][metric]  for c in CLASS_NAMES]
        vals_dn = [DINOV2_PC[c][metric] for c in CLASS_NAMES]

        b1 = ax.bar(x - w/2, vals_vj, w, color=BLUE,   label='V-JEPA 2', edgecolor='white')
        b2 = ax.bar(x + w/2, vals_dn, w, color=ORANGE, label='DINOv2',   edgecolor='white')

        for bar, v in zip(list(b1)+list(b2), vals_vj+vals_dn):
            if v < 0.20:
                ax.add_patch(plt.Rectangle(
                    (bar.get_x(), 0), bar.get_width(), bar.get_height(),
                    fill=False, edgecolor=RED, lw=2, zorder=5))

        ax.axhline(0.20, color=RED, ls=':', lw=1.2, alpha=0.6, label='Umbral bajo (<0.20)')
        ax.set_ylabel(title)
        ax.set_title(f'{title} por Clase — V-JEPA 2 vs DINOv2', fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASS_SHORT, rotation=40, ha='right', fontsize=8)
    fig.suptitle('Precision y Recall por Clase: Comparativa de Modelos\n(Bordes rojos = valores críticos <0.20)',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    add_footer(fig)
    save(fig, 'FIG_06_precision_recall_per_class.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 07 — Radar chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig07():
    f1_vj = [VJEPA_PC[c]['f1']  for c in CLASS_NAMES]
    f1_dn = [DINOV2_PC[c]['f1'] for c in CLASS_NAMES]

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    f1_vj_p = f1_vj + [f1_vj[0]]
    f1_dn_p = f1_dn + [f1_dn[0]]
    angles_p = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles_p, f1_vj_p, color=BLUE,   lw=2,   label='V-JEPA 2')
    ax.fill(angles_p, f1_vj_p, color=BLUE,   alpha=0.15)
    ax.plot(angles_p, f1_dn_p, color=ORANGE, lw=2,   label='DINOv2')
    ax.fill(angles_p, f1_dn_p, color=ORANGE, alpha=0.15)

    ax.set_xticks(angles)
    ax.set_xticklabels(CLASS_SHORT, fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8'], fontsize=7)

    ax.set_title('Comparativa F1 por Clase: Diagrama de Radar\nV-JEPA 2 vs DINOv2',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    add_footer(fig)
    save(fig, 'FIG_07_radar_chart.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 08 — Matriz de confusión V-JEPA
# ═══════════════════════════════════════════════════════════════════════════════
def fig_confusion(y_v, y_p, model_name, fname, acc_label):
    cm = confusion_matrix(y_v, y_p, labels=list(range(N)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT,
                ax=ax, linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Recall por celda'})

    ax.set_xlabel('Predicción', fontsize=11)
    ax.set_ylabel('Real', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_title(f'Matriz de Confusión — {model_name}\n(Normalizada por fila | {acc_label})',
                 fontweight='bold')
    add_footer(fig)
    save(fig, fname)


def fig08():
    acc = f'Accuracy={np.mean(y_val_vj==y_pred_vj):.1%}'
    fig_confusion(y_val_vj, y_pred_vj, 'V-JEPA (MLP PyTorch)',
                  'FIG_08_confusion_matrix_vjepa.png', acc)

def fig09():
    acc = f'Accuracy={np.mean(y_val_dn==y_pred_dn):.1%}'
    fig_confusion(y_val_dn, y_pred_dn, 'DINOv2 (MLP PyTorch)',
                  'FIG_09_confusion_matrix_dinov2.png', acc)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Comparativa de modelos
# ═══════════════════════════════════════════════════════════════════════════════
def fig10():
    models = list(MODEL_RESULTS.keys())
    accs   = [MODEL_RESULTS[m]['accuracy'] for m in models]
    f1s    = [MODEL_RESULTS[m]['macro_f1'] for m in models]
    types  = [MODEL_RESULTS[m]['type']     for m in models]
    colors = [TYPE_COLOR[t] for t in types]

    order  = np.argsort(accs)
    models_o = [models[i] for i in order]
    accs_o   = [accs[i]   for i in order]
    f1s_o    = [f1s[i]    for i in order]
    colors_o = [colors[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, vals, xlabel, thresh, label in zip(axes,
        [accs_o, f1s_o],
        ['Accuracy', 'Macro F1-Score'],
        [0.75,       0.70],
        ['Accuracy', 'Macro F1']):

        bars = ax.barh(models_o, vals, color=colors_o, edgecolor='white', linewidth=0.5)

        # Destacar PowerMean-7
        for i, (bar, m) in enumerate(zip(bars, models_o)):
            if 'PowerMean' in m:
                bar.set_edgecolor(GOLD)
                bar.set_linewidth(3)
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{vals[i]:.3f} ★', va='center', fontsize=9, fontweight='bold', color=GREEN)
            elif m == 'V-JEPA 2':
                bar.set_edgecolor(BLUE)
                bar.set_linewidth(2.5)
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{vals[i]:.3f} ✓', va='center', fontsize=9, fontweight='bold', color=BLUE)
            else:
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{vals[i]:.3f}', va='center', fontsize=8)

        ax.axvline(thresh, color=GOLD, ls='--', lw=1.5, label=f'Meta ({thresh})')
        ax.set_xlabel(xlabel)
        ax.set_title(f'Comparativa: {label}', fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.legend(loc='lower right', fontsize=8)

    legend_handles = [mpatches.Patch(color=v, label=k) for k, v in TYPE_COLOR.items()]
    fig.legend(handles=legend_handles, title='Tipo de modelo',
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle('Comparativa de Todos los Modelos — Accuracy y Macro F1-Score\n(8 modelos: 2 base + 5 ensambles de Avance 5 + PowerMean-7)',
                 fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    add_footer(fig)
    save(fig, 'FIG_10_model_comparison.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11 — t-SNE embeddings V-JEPA
# ═══════════════════════════════════════════════════════════════════════════════
def fig11():
    fig, ax = plt.subplots(figsize=(12, 9))
    for ci, cname in enumerate(CLASS_NAMES):
        mask = y_tsne == ci
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=TAB12[ci], label=CLASS_SHORT[ci],
                   s=14, alpha=0.65, edgecolors='none')

    ax.set_title('t-SNE de Embeddings V-JEPA (1024-dim → 2D)\n'
                 f'n={len(idx_tsne)} clips | AUC promedio = 0.9573',
                 fontweight='bold')
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=8, markerscale=1.8)
    plt.tight_layout()
    add_footer(fig)
    save(fig, 'FIG_11_tsne_vjepa.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 12 — t-SNE comparativa (V-JEPA vs DINOv2)
# ═══════════════════════════════════════════════════════════════════════════════
def fig12():
    idx_dn = np.random.RandomState(42).choice(len(X_dn), size=len(idx_tsne), replace=False)
    X_dn_sub = X_dn[idx_dn]
    y_dn_sub = y_dn[idx_dn]
    print("  Calculando t-SNE DINOv2...")
    X_2d_dn = TSNE(n_components=2, random_state=42, perplexity=40,
                    max_iter=1000, init='pca').fit_transform(X_dn_sub)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, X2, ys, title in zip(axes,
        [X_2d, X_2d_dn],
        [y_tsne, y_dn_sub],
        ['V-JEPA (AUC=0.9573)', 'DINOv2']):
        for ci, cname in enumerate(CLASS_NAMES):
            mask = ys == ci
            ax.scatter(X2[mask,0], X2[mask,1], c=TAB12[ci],
                       label=CLASS_SHORT[ci], s=10, alpha=0.6, edgecolors='none')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')

    handles = [mpatches.Patch(color=TAB12[i], label=CLASS_SHORT[i]) for i in range(N)]
    fig.legend(handles=handles, bbox_to_anchor=(0.5, -0.02),
               loc='lower center', ncol=6, fontsize=8)
    fig.suptitle('Espacio de Embeddings: V-JEPA vs DINOv2 (t-SNE 2D)\n'
                 f'n={len(idx_tsne)} clips por backbone', fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    add_footer(fig)
    save(fig, 'FIG_12_tsne_comparison.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 13 — Curvas ROC por clase
# ═══════════════════════════════════════════════════════════════════════════════
def fig13():
    fig, ax = plt.subplots(figsize=(12, 9))
    aucs = []
    for ci, cname in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, ci], y_prob[:, ci])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=TAB12[ci], lw=1.5,
                label=f'{CLASS_SHORT[ci]} (AUC={roc_auc:.3f})')

    mean_auc = np.mean(aucs)
    ax.plot([0,1],[0,1], color=MGRAY, ls='--', lw=1, label='Azar (AUC=0.50)')
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    ax.set_title(f'Curvas ROC por Clase — V-JEPA 2 (LogReg, one-vs-rest)\n'
                 f'AUC Promedio Macro = {mean_auc:.4f}', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.01,1), loc='upper left', fontsize=8)
    ax.text(0.55, 0.08, f'AUC Promedio = {mean_auc:.4f}\n(Excelente discriminación)',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            color=GREEN, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    add_footer(fig)
    save(fig, 'FIG_13_roc_curves.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 14 — Curvas Precision-Recall
# ═══════════════════════════════════════════════════════════════════════════════
def fig14():
    fig, ax = plt.subplots(figsize=(12, 9))
    for ci, cname in enumerate(CLASS_NAMES):
        prec, rec, _ = precision_recall_curve(y_bin[:, ci], y_prob[:, ci])
        ap = average_precision_score(y_bin[:, ci], y_prob[:, ci])
        ax.plot(rec, prec, color=TAB12[ci], lw=1.5,
                label=f'{CLASS_SHORT[ci]} (AP={ap:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Curvas Precision-Recall por Clase — V-JEPA 2\n(LogReg one-vs-rest | AP = Average Precision)', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.01,1), loc='upper left', fontsize=8)
    plt.tight_layout()
    add_footer(fig)
    save(fig, 'FIG_14_precision_recall_curves.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 15 — Dashboard ejecutivo
# ═══════════════════════════════════════════════════════════════════════════════
def fig15():
    fig = plt.figure(figsize=(22, 16))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # [0,0] Model comparison accuracy
    ax00 = fig.add_subplot(gs[0, 0])
    models = list(MODEL_RESULTS.keys())
    accs   = [MODEL_RESULTS[m]['accuracy'] for m in models]
    types  = [MODEL_RESULTS[m]['type']     for m in models]
    order  = np.argsort(accs)
    m_o    = [models[i] for i in order]
    a_o    = [accs[i]   for i in order]
    c_o    = [TYPE_COLOR[types[i]] for i in order]
    bars   = ax00.barh(m_o, a_o, color=c_o, edgecolor='white')
    ax00.axvline(0.75, color=GOLD, ls='--', lw=1)
    for b, v in zip(bars, a_o):
        ax00.text(b.get_width()+0.005, b.get_y()+b.get_height()/2,
                  f'{v:.3f}', va='center', fontsize=7)
    ax00.set_title('Accuracy por Modelo', fontweight='bold', fontsize=11)
    ax00.set_xlim(0, 1.0)

    # [0,1] F1 por clase V-JEPA
    ax01 = fig.add_subplot(gs[0, 1])
    f1s  = [VJEPA_PC[c]['f1'] for c in CLASS_NAMES]
    clrs = [RED if v < 0.3 else BLUE for v in f1s]
    ax01.bar(CLASS_SHORT, f1s, color=clrs, edgecolor='white')
    ax01.axhline(0.70, color=GOLD, ls='--', lw=1)
    ax01.set_xticklabels(CLASS_SHORT, rotation=60, ha='right', fontsize=7)
    ax01.set_ylim(0, 0.85)
    ax01.set_title('F1 por Clase — V-JEPA 2', fontweight='bold', fontsize=11)

    # [0,2] AUC por clase (barras)
    ax02 = fig.add_subplot(gs[0, 2])
    aucs_cls = []
    for ci in range(N):
        _, _, _ = roc_curve(y_bin[:, ci], y_prob[:, ci])
        aucs_cls.append(auc(*roc_curve(y_bin[:, ci], y_prob[:, ci])[:2]))
    ax02.bar(CLASS_SHORT, aucs_cls, color=TAB12, edgecolor='white')
    ax02.axhline(np.mean(aucs_cls), color=NAVY, ls='--', lw=1.5,
                 label=f'Prom={np.mean(aucs_cls):.3f}')
    ax02.set_xticklabels(CLASS_SHORT, rotation=60, ha='right', fontsize=7)
    ax02.set_ylim(0.5, 1.05)
    ax02.set_title('AUC por Clase (ROC)', fontweight='bold', fontsize=11)
    ax02.legend(fontsize=8)

    # [1,0] Class distribution
    ax10 = fig.add_subplot(gs[1, 0])
    vals = [CLASS_COUNTS[c] for c in CLASS_NAMES]
    ax10.barh(CLASS_SHORT, vals, color=plt.cm.Blues(
        np.array(vals)/max(vals)), edgecolor='white')
    ax10.axvline(np.median(vals), color=GOLD, ls='--', lw=1)
    ax10.set_title('Clips por Clase', fontweight='bold', fontsize=11)
    ax10.invert_yaxis()

    # [1,1] Training loss curves
    ax11 = fig.add_subplot(gs[1, 1])
    ep_v = [h['epoch'] for h in vjepa_hist]
    tl_v = [h['train_loss'] for h in vjepa_hist]
    vl_v = [h['val_loss']   for h in vjepa_hist]
    ep_d = [h['epoch'] for h in dinov2_hist]
    tl_d = [h['train_loss'] for h in dinov2_hist]
    ax11.plot(ep_v, tl_v, color=BLUE,   lw=1.5, label='VJEPA train')
    ax11.plot(ep_v, vl_v, color=BLUE,   lw=1.5, ls='--', label='VJEPA val', alpha=0.7)
    ax11.plot(ep_d, tl_d, color=ORANGE, lw=1.5, label='DINOv2 train')
    ax11.set_xlabel('Época', fontsize=9)
    ax11.set_ylabel('Loss', fontsize=9)
    ax11.set_title('Curvas de Pérdida', fontweight='bold', fontsize=11)
    ax11.legend(fontsize=7)

    # [1,2] Radar (simplificado como scatter)
    ax12 = fig.add_subplot(gs[1, 2], polar=True)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    f1_vj_p = [VJEPA_PC[c]['f1']  for c in CLASS_NAMES] + [VJEPA_PC[CLASS_NAMES[0]]['f1']]
    f1_dn_p = [DINOV2_PC[c]['f1'] for c in CLASS_NAMES] + [DINOV2_PC[CLASS_NAMES[0]]['f1']]
    ang_p = angles + [angles[0]]
    ax12.plot(ang_p, f1_vj_p, color=BLUE,   lw=1.5)
    ax12.fill(ang_p, f1_vj_p, color=BLUE,   alpha=0.15)
    ax12.plot(ang_p, f1_dn_p, color=ORANGE, lw=1.5)
    ax12.fill(ang_p, f1_dn_p, color=ORANGE, alpha=0.15)
    ax12.set_xticks(angles)
    ax12.set_xticklabels(CLASS_SHORT, fontsize=6)
    ax12.set_ylim(0, 0.85)
    ax12.set_title('Radar F1', fontweight='bold', fontsize=11, pad=15)

    # [2,0] Confusion matrix mini (V-JEPA)
    ax20 = fig.add_subplot(gs[2, 0])
    cm = confusion_matrix(y_val_vj, y_pred_vj, labels=list(range(N)))
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    sns.heatmap(cm_n, ax=ax20, cmap='Blues', cbar=False,
                xticklabels=False, yticklabels=False, linewidths=0)
    ax20.set_title('Matriz Confusión V-JEPA', fontweight='bold', fontsize=11)
    ax20.set_xlabel('Predicción'); ax20.set_ylabel('Real')

    # [2,1] Precision vs Recall scatter (V-JEPA, por clase)
    ax21 = fig.add_subplot(gs[2, 1])
    for ci, c in enumerate(CLASS_NAMES):
        p = VJEPA_PC[c]['p']; r = VJEPA_PC[c]['r']; s = VJEPA_PC[c]['sup']
        ax21.scatter(r, p, color=TAB12[ci], s=max(s*0.8, 30), zorder=3, edgecolors='white')
        ax21.annotate(CLASS_SHORT[ci], (r, p), fontsize=6,
                      xytext=(3, 2), textcoords='offset points')
    ax21.set_xlabel('Recall')
    ax21.set_ylabel('Precision')
    ax21.set_xlim(-0.05, 1.05)
    ax21.set_ylim(-0.05, 1.1)
    ax21.axhline(0.5, color=MGRAY, ls=':', lw=0.8)
    ax21.axvline(0.5, color=MGRAY, ls=':', lw=0.8)
    ax21.set_title('Precision vs Recall (V-JEPA 2)', fontweight='bold', fontsize=11)

    # [2,2] KPIs summary table
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.axis('off')
    kpis = [
        ['Métrica',              'Valor'],
        ['Accuracy V-JEPA 2',   '78.54%'],
        ['Macro F1 V-JEPA 2',   '0.7063'],
        ['AUC Promedio',         '0.9573'],
        ['PowerMean-7 Accuracy','87.81%'],
        ['PowerMean-7 F1',       '0.8610'],
        ['Latencia inferencia',  '< 5 ms'],
        ['Clases reconocidas',   '12'],
        ['Clips dataset',        '3,425'],
        ['Sujetos train/val',    '13 / 3'],
        ['Backbone V-JEPA 2',   'ViT-L (1024d)'],
        ['ROI payback',          '1.7 meses'],
    ]
    tbl = ax22.table(cellText=kpis[1:], colLabels=kpis[0],
                     loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#E8EEF7')
    ax22.set_title('KPIs del Proyecto', fontweight='bold', fontsize=11)

    fig.suptitle('VisionOps HAR — Dashboard Ejecutivo de Resultados\n'
                 'AI Co-Pilot para el Piso de Producción: Ver, Guiar, Mejorar',
                 fontweight='bold', fontsize=16, color=NAVY)
    add_footer(fig)
    save(fig, 'FIG_15_executive_dashboard.png')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*60)
    print("VisionOps HAR — Generando 15 visualizaciones")
    print("="*60 + "\n")

    steps = [
        ("FIG 01 — Distribución de clases",       fig01),
        ("FIG 02 — Distribución por sujeto",       fig02),
        ("FIG 03 — Curvas de entrenamiento",        fig03),
        ("FIG 04 — Learning rate schedule",         fig04),
        ("FIG 05 — F1 por clase comparativo",       fig05),
        ("FIG 06 — Precision y Recall por clase",   fig06),
        ("FIG 07 — Radar chart F1",                 fig07),
        ("FIG 08 — Confusión V-JEPA",               fig08),
        ("FIG 09 — Confusión DINOv2",               fig09),
        ("FIG 10 — Comparativa de modelos",         fig10),
        ("FIG 11 — t-SNE V-JEPA",                  fig11),
        ("FIG 12 — t-SNE comparativa",              fig12),
        ("FIG 13 — Curvas ROC",                     fig13),
        ("FIG 14 — Curvas Precision-Recall",        fig14),
        ("FIG 15 — Dashboard ejecutivo",            fig15),
    ]

    ok, fail = 0, 0
    for name, fn in steps:
        print(f"\n→ {name}")
        try:
            fn()
            ok += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            fail += 1

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ {ok} figuras generadas   ❌ {fail} errores")
    print(f"⏱  Tiempo total: {elapsed:.1f}s")
    print(f"📁 Guardadas en: {OUT_DIR}")
    print("="*60)
