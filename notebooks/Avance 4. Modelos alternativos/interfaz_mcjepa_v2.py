#!/usr/bin/env python3
"""
interfaz_mcjepa_v2.py
═══════════════════════════════════════════════════════════════════════════════
Interfaz local para comparar 5 clasificadores de actividad sobre video en
tiempo real. Cada clasificador usa una representación distinta del video:

  ┌──────────────────────────────┬─────────────────────────────────────────┐
  │ Modelo                       │ Descripción                             │
  ├──────────────────────────────┼─────────────────────────────────────────┤
  │ DINOv2_puro                  │ Promedio de CLS tokens por frame        │
  │ DINOv2_to_MCJEPA             │ Secuencia DINOv2 → encoder multi-escala │
  │ VJEPA2_puro                  │ Promedio de tokens latentes V-JEPA2     │
  │ VJEPA2_MCJEPA_frozen         │ Tokens V-JEPA2 → MC-JEPA Head (frozen)  │
  │ VJEPA2_MCJEPA_partial        │ Tokens V-JEPA2 → MC-JEPA Head (finetune)│
  └──────────────────────────────┴─────────────────────────────────────────┘

Los checkpoints se generan con los notebooks Parte1 y Parte2 en Colab.

Overlay sobre el video:
  • Heatmap de movimiento entre frames (aproximación visual de zonas activas)
  • Cajas de detección de personas (YOLOv8n)
  • Recuadro por modelo con clase predicha, confianza y barra visual

Requisitos:
  - Python 3.10 (V-JEPA2 requiere esta versión específica de transformers)
  - GPU recomendada; en CPU la inferencia puede tardar varios segundos por clip

Cómo ejecutar:
  py -3.10 interfaz_mcjepa_v2.py
"""

# ── Auto-instalación de dependencias faltantes ────────────────────────────────
# Se ejecuta una sola vez al inicio; si el paquete ya está instalado, no hace nada
import sys, subprocess

def _pip(pkg):
    """Instala un paquete usando el mismo Python que corre este script."""
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                   check=False, capture_output=True)

for mod, pkg in {"cv2": "opencv-python", "PIL": "pillow",
                 "ultralytics": "ultralytics", "transformers": "transformers",
                 "huggingface_hub": "huggingface_hub"}.items():
    try:
        __import__(mod)
    except ImportError:
        print(f"Instalando {pkg}..."); _pip(pkg)

# ── Imports estándar ──────────────────────────────────────────────────────────
import os, json, time, threading, traceback
from pathlib import Path
from collections import deque

# GUI: tkinter (incluido con Python)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Visión y tensores
import cv2
import numpy as np
from PIL import Image, ImageTk

import torch
import torch.nn as nn
import torch.nn.functional as F

# Modelos preentrenados de HuggingFace
from transformers import AutoImageProcessor, AutoModel, AutoVideoProcessor


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — ajusta estas variables antes de ejecutar
# ══════════════════════════════════════════════════════════════════════════════

# Carpeta donde están los archivos .pt generados por los notebooks de Colab.
# Estructura esperada (los archivos pueden estar en la raíz o en subcarpetas):
#   CHECKPOINTS/
#     dino_to_mcjepa_encoder.pt
#     DINOv2_puro_classifier.pt
#     DINOv2_to_MCJEPA_classifier.pt
#     VJEPA2_puro_classifier.pt
#     VJEPA2_MCJEPA_frozen.pt           (solo si corriste Parte 2)
#     VJEPA2_MCJEPA_partial_finetune.pt (solo si corriste Parte 2)
CHECKPOINT_DIR = Path(r"C:\Users\52614\Documents\Claude\Projects\Ssss\CHECKPOINTS")   # ← CAMBIA

# Token de HuggingFace para descargar V-JEPA2 (facebook/vjepa2-vitl-fpc64-256).
# IMPORTANTE: nunca subas tu token real a GitHub.
# Alternativa segura: export HF_TOKEN=hf_xxx  y dejar esta línea como está.
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # lee del entorno; pon tu token ahí

# IDs de los modelos en HuggingFace Hub
DINO_MODEL_NAME = "facebook/dinov2-small"          # ~300 MB, se cachea en ~/.cache/huggingface
VJEPA2_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256" # ~1.3 GB, requiere Python 3.10

# Número de frames que se samplea de cada clip para inferencia.
# El modelo fue entrenado con 64, pero 16 basta para inferencia en tiempo real.
NUM_FRAMES = 16

# Tamaño al que se redimensiona cada frame antes de pasarlo al modelo.
IMAGE_SIZE = 224

# Inferencia cada N frames leídos del video.
# Valor bajo = más actualizaciones pero más lento; valor alto = más fluido pero menos frecuente.
INF_EVERY = 16

# ── Paleta de colores de la interfaz ─────────────────────────────────────────
C_BG    = "#1a1a2e"   # fondo general (azul muy oscuro)
C_PANEL = "#16213e"   # fondo del panel lateral
C_DARK  = "#0d0d1a"   # fondo de elementos internos (barras, canvas)

# Color identificador de cada modelo en la UI y en el overlay del video
MODEL_COLORS = {
    "DINOv2_puro":                    "#4FC3F7",   # azul claro
    "DINOv2_to_MCJEPA":               "#FFD54F",   # amarillo
    "VJEPA2_puro":                    "#81C784",   # verde
    "VJEPA2_MCJEPA_frozen":           "#FF8A65",   # naranja
    "VJEPA2_MCJEPA_partial_finetune": "#CE93D8",   # violeta
}

# Seleccionar dispositivo: GPU si está disponible, CPU como fallback
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] {DEVICE}" + (f" — {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else " (CPU: inferencia lenta)"))


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURAS DE RED
#  Estas clases deben coincidir exactamente con las definidas en los notebooks,
#  ya que los pesos (.pt) se cargan sobre estas mismas estructuras.
# ══════════════════════════════════════════════════════════════════════════════

class DinoToMCJEPAEncoder(nn.Module):
    """
    Encoder multi-escala temporal sobre secuencias de embeddings DINOv2.

    Entrada:  [B, T, D_dino]  — B videos, T frames, D_dino dimensiones por frame
    Salida:   [B, hidden * 3] — embedding del video combinando 3 escalas temporales

    Usa tres encoders Transformer en paralelo, cada uno sobre una ventana distinta:
      - short_encoder:  últimos T//4 frames  → captura la acción más reciente
      - mid_encoder:    últimos T//2 frames  → captura contexto intermedio
      - global_encoder: todos los frames     → captura el clip completo

    Las tres representaciones se concatenan y normalizan para formar el embedding final.
    Este encoder se usa como extractor de características (sin entrenamiento en Parte 1).
    """
    def __init__(self, d_in, hidden=256, nhead=4, layers=1):
        super().__init__()
        # Proyección lineal: mapea D_dino → hidden para que los encoders operen
        # en un espacio de dimensión fija independientemente del backbone usado
        self.proj = nn.Linear(d_in, hidden)

        # Tres encoders Transformer, uno por escala temporal
        self.short_encoder  = self._mk(hidden, nhead, layers)
        self.mid_encoder    = self._mk(hidden, nhead, layers)
        self.global_encoder = self._mk(hidden, nhead, layers)

        # Normalización final sobre la concatenación de las 3 representaciones
        self.norm = nn.LayerNorm(hidden * 3)

    def _mk(self, h, n, l):
        """Crea un TransformerEncoder con los parámetros dados."""
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h, nhead=n, dim_feedforward=h * 4,
                dropout=0.1, batch_first=True, activation="gelu"),
            num_layers=l)

    def forward(self, dino_seq):
        x = self.proj(dino_seq)           # [B, T, hidden]
        T = x.shape[1]

        # Longitudes de ventana para cada escala
        sl, ml = max(1, T // 4), max(1, T // 2)

        # Encodificar cada ventana y hacer mean pooling sobre la dimensión temporal
        zs = self.short_encoder(x[:, -sl:]).mean(1)   # [B, hidden]
        zm = self.mid_encoder(x[:, -ml:]).mean(1)     # [B, hidden]
        zg = self.global_encoder(x).mean(1)           # [B, hidden]

        # Concatenar las 3 escalas y normalizar → [B, hidden*3]
        return self.norm(torch.cat([zs, zm, zg], dim=-1))


class EmbeddingMLPClassifier(nn.Module):
    """
    Clasificador MLP ligero que opera sobre embeddings de video ya extraídos.

    Entrada:  [B, input_dim]  — embedding del video (producido por DINOv2, V-JEPA2, etc.)
    Salida:   [B, num_classes] — logits (sin softmax) por clase de actividad

    Arquitectura:
      LayerNorm → Linear(hidden) → GELU → Dropout
               → Linear(hidden//2) → GELU → Dropout
               → Linear(num_classes)

    Este clasificador se entrena en Parte 1 sobre embeddings congelados de los backbones.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),                                          # normaliza la entrada
            nn.Linear(input_dim, hidden_dim),   nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)                           # logits finales
        )

    def forward(self, x):
        return self.net(x)


class MCJEPAHead(nn.Module):
    """
    Cabeza MC-JEPA que opera directamente sobre tokens latentes de V-JEPA2.

    Entrada:  tokens [B, N, D_vjepa2]  — N tokens spatio-temporales del backbone
    Salida:   dict con logits, embeddings de contexto y predicciones JEPA

    Implementa dos objetivos simultáneos:
      1. Clasificación supervisada:
           tokens → encoder short/mid/global → fusión → clasificador → logits
      2. Objetivo predictivo JEPA (auto-supervisión):
           contexto (short+mid) → predictor → z_pred_global
           target: z_global (contexto completo, detenido como objetivo)
           loss_jepa = MSE(z_pred_global, z_global.detach())

    La loss total durante entrenamiento es:
      loss = CrossEntropy(logits, labels) + lambda_jepa * MSE(z_pred, z_global)

    Este módulo se entrena en Parte 2, con V-JEPA2 congelado o parcialmente descongelado.
    """
    def __init__(self, d_in, hidden=256, nhead=4, layers=1, num_classes=14):
        super().__init__()
        # Proyección de tokens V-JEPA2 al espacio interno de la cabeza
        self.proj = nn.Linear(d_in, hidden)

        # Encoders para las 3 escalas temporales (igual que DinoToMCJEPAEncoder)
        self.short_encoder  = self._mk(hidden, nhead, layers)
        self.mid_encoder    = self._mk(hidden, nhead, layers)
        self.global_encoder = self._mk(hidden, nhead, layers)

        # Fusión de contexto corto + medio → representación compacta z_context
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.2))

        # Predictor JEPA: estima z_global a partir de z_context
        # (objetivo de auto-supervisión: predecir el contexto completo desde el parcial)
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden))

        # Clasificador final: z_context → logits de actividad
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes))

    def _mk(self, h, n, l):
        """Crea un TransformerEncoder con los parámetros dados."""
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h, nhead=n, dim_feedforward=h * 4,
                dropout=0.1, batch_first=True, activation="gelu"),
            num_layers=l)

    def forward(self, tokens):
        """
        tokens: [B, N, D_vjepa2]
        Retorna dict con:
          - logits:        [B, num_classes]  para clasificación
          - z_context:     [B, hidden]       embedding de contexto (short+mid)
          - z_pred_global: [B, hidden]       predicción del contexto global
          - z_global:      [B, hidden]       contexto global real (target JEPA)
        """
        x = self.proj(tokens)   # [B, N, hidden]
        B, N, _ = x.shape

        # Ventanas temporales
        sl, ml = max(1, N // 4), max(1, N // 2)

        # Codificar cada escala con mean pooling temporal
        zs = self.short_encoder(x[:, -sl:]).mean(1)   # [B, hidden] — contexto corto
        zm = self.mid_encoder(x[:, -ml:]).mean(1)     # [B, hidden] — contexto medio
        zg = self.global_encoder(x).mean(1)           # [B, hidden] — contexto global (target)

        # Fusionar short + mid para obtener el embedding de contexto
        z_ctx = self.fusion(torch.cat([zs, zm], dim=-1))   # [B, hidden]

        return {
            "logits":        self.classifier(z_ctx),   # predicción de clase
            "z_context":     z_ctx,                    # embedding usado para clasificar
            "z_pred_global": self.predictor(z_ctx),    # predicción del contexto global (JEPA)
            "z_global":      zg                        # contexto global real (target JEPA)
        }


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACTORES DE FEATURES
#  Wrappers sobre los backbones preentrenados para obtener representaciones
#  de los frames/videos en el formato que esperan los clasificadores.
# ══════════════════════════════════════════════════════════════════════════════

class DINOv2Extractor(nn.Module):
    """
    Extractor de features usando DINOv2 (facebook/dinov2-small).

    Procesa cada frame individualmente y extrae el token CLS del ViT,
    que resume el contenido visual del frame en un vector.

    Entrada:  batch_frames — lista de B videos, cada uno con T imágenes PIL
    Salida:   [B, T, D_dino] — embedding CLS por frame (D_dino=384 para dinov2-small)

    El modelo se mantiene congelado (sin gradientes) durante toda la inferencia.
    """
    def __init__(self, model_name):
        super().__init__()
        # Processor: normaliza y tokeniza imágenes para el ViT
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        # Congelar todos los parámetros: solo se usa como extractor, no se entrena
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, batch_frames):
        B, T = len(batch_frames), len(batch_frames[0])

        # Aplanar todos los frames en una sola lista para procesarlos en batch
        flat = [img for vf in batch_frames for img in vf]   # B*T imágenes

        inp  = self.processor(images=flat, return_tensors="pt").to(DEVICE)
        out  = self.model(**inp)

        # Extraer el token CLS (índice 0 de la secuencia) como embedding del frame
        feats = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out.pooler_output

        # Reorganizar de [B*T, D] a [B, T, D]
        return feats.view(B, T, -1)


class VJEPA2TokenExtractor(nn.Module):
    """
    Extractor de tokens latentes usando V-JEPA2 (facebook/vjepa2-vitl-fpc64-256).

    En lugar de usar AutoVideoProcessor (incompatible con algunas versiones de
    transformers), realiza el preprocesamiento manualmente:
      - Resize a IMAGE_SIZE × IMAGE_SIZE
      - Normalización ImageNet estándar (mean/std de ImageNet)
      - Apilado en tensor (B, T, 3, H, W)

    Entrada:  batch_frames — lista de B videos, cada uno con T imágenes PIL
    Salida:   [B, N, D_vjepa2] — N tokens spatio-temporales por video (D=1024 para ViT-L)

    El modelo se mantiene congelado durante toda la inferencia.
    """
    # Estadísticas de normalización ImageNet (mismas usadas durante el preentrenamiento)
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        # Congelar: V-JEPA2 es muy grande (~1.3 GB), no se fine-tunea en inferencia
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _parse(outputs):
        """
        Normaliza la salida de V-JEPA2 a forma [B, N, D] independientemente
        del formato que devuelva el modelo (tensor, tuple, dict, etc.).
        """
        if hasattr(outputs, "last_hidden_state"):
            t = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            t = outputs[0]
        elif isinstance(outputs, dict):
            # Buscar la clave estándar de tokens
            for k in ("last_hidden_state", "x_norm_patchtokens"):
                if k in outputs:
                    t = outputs[k]
                    break
            else:
                t = list(outputs.values())[0]
        else:
            t = outputs

        # Normalizar dimensiones al formato [B, N, D]
        if t.ndim == 2:
            t = t.unsqueeze(1)          # [B, D] → [B, 1, D]
        elif t.ndim == 4:
            B, T, N, D = t.shape
            t = t.reshape(B, T * N, D)  # [B, T, N, D] → [B, T*N, D]
        return t

    def _preprocess(self, batch_frames):
        """
        Convierte una lista de frames PIL a tensor (B, T, 3, H, W) normalizado.
        Implementación manual para evitar dependencia de AutoVideoProcessor.
        """
        clips = []
        for video_frames in batch_frames:
            # Convertir cada frame: PIL → numpy → float [0,1] → tensor
            arr = np.stack([
                np.array(img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
                for img in video_frames
            ]).astype(np.float32) / 255.0           # [T, H, W, 3]
            t = torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T, 3, H, W]
            clips.append(t)

        video = torch.stack(clips).to(DEVICE)       # [B, T, 3, H, W]

        # Aplicar normalización ImageNet
        mean = self._MEAN.to(DEVICE)
        std  = self._STD.to(DEVICE)
        return (video - mean) / std

    @torch.inference_mode()
    def forward(self, batch_frames):
        video = self._preprocess(batch_frames)      # [B, T, 3, H, W]

        # skip_predictor=True evita computar la cabeza predictiva del modelo
        # (solo necesitamos el encoder, no el predictor de JEPA)
        try:
            out = self.model(video, skip_predictor=True)
        except TypeError:
            out = self.model(video)   # fallback si skip_predictor no está soportado

        return self._parse(out)       # [B, N, D_vjepa2]


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE VIDEO
# ══════════════════════════════════════════════════════════════════════════════

def build_motion_heatmap(frames_rgb, out_h, out_w):
    """
    Genera un heatmap de movimiento comparando los dos últimos frames del buffer.

    Calcula la diferencia absoluta entre frames consecutivos en escala de grises,
    la suaviza con un filtro gaussiano y la normaliza a [0, 1].

    Retorna None si hay menos de 2 frames o si no hay movimiento detectable.
    El resultado es un array float32 de forma (out_h, out_w) que se usa
    para colorear el video con cv2.COLORMAP_JET.
    """
    if len(frames_rgb) < 2:
        return None

    # Redimensionar y convertir a escala de grises con suavizado para reducir ruido
    prev = cv2.GaussianBlur(
        cv2.cvtColor(cv2.resize(frames_rgb[-2], (out_w, out_h)), cv2.COLOR_RGB2GRAY
                     ).astype(np.float32), (0, 0), 3)
    curr = cv2.GaussianBlur(
        cv2.cvtColor(cv2.resize(frames_rgb[-1], (out_w, out_h)), cv2.COLOR_RGB2GRAY
                     ).astype(np.float32), (0, 0), 3)

    # Diferencia absoluta entre frames consecutivos
    diff = cv2.absdiff(prev, curr)

    # Suavizado adicional para obtener regiones coherentes en lugar de píxeles aislados
    diff = cv2.GaussianBlur(diff, (0, 0), 9)

    mx = diff.max()
    if mx < 1e-6:
        return None   # Sin movimiento detectable

    return (diff / mx).astype(np.float32)   # Normalizado a [0, 1]


def frames_to_pil(frames_rgb, num_frames, image_size):
    """
    Submuestrea y redimensiona frames del buffer para la inferencia.

    Si el buffer tiene más frames de los necesarios, selecciona num_frames
    uniformemente distribuidos. Si tiene menos, repite el último frame.

    Entrada:  frames_rgb — lista de arrays numpy RGB del buffer de video
    Salida:   lista de num_frames imágenes PIL de tamaño image_size × image_size
    """
    if len(frames_rgb) >= num_frames:
        # Submuestrear: tomar num_frames frames distribuidos uniformemente
        idx = np.linspace(0, len(frames_rgb) - 1, num_frames).astype(int)
        selected = [frames_rgb[i] for i in idx]
    else:
        # Rellenar: repetir el último frame hasta llegar a num_frames
        selected = list(frames_rgb) + [frames_rgb[-1]] * (num_frames - len(frames_rgb))

    return [Image.fromarray(cv2.resize(f, (image_size, image_size))) for f in selected]


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════

def load_all_models(ckpt_dir: Path, status_cb):
    """
    Carga todos los modelos disponibles desde CHECKPOINT_DIR.

    Para cada modelo, verifica que TANTO el extractor de features (backbone)
    COMO el clasificador (.pt) estén disponibles. Si falta cualquiera de los
    dos, el modelo se marca como no disponible en models['_ready'].

    Retorna un dict con:
      - 'dino':       DINOv2Extractor
      - 'dmc_enc':    DinoToMCJEPAEncoder
      - 'vj_tok':     VJEPA2TokenExtractor
      - 'DINOv2_puro', 'DINOv2_to_MCJEPA', etc.: clasificadores MLP o MCJEPAHead
      - 'class_names': lista de nombres de clases
      - '_ready':     dict {model_key: bool} indicando qué modelos están operativos
    """
    models = {}

    # Imprimir diagnóstico de la carpeta de checkpoints
    print("\n" + "=" * 60)
    print(f"CHECKPOINT_DIR: {ckpt_dir}")
    print(f"  Existe: {ckpt_dir.exists()}")
    if ckpt_dir.exists():
        print("  Archivos encontrados:")
        for f in sorted(ckpt_dir.rglob("*.pt")):
            print(f"    {f.relative_to(ckpt_dir)}")
    print("=" * 60 + "\n")

    # ── Login en HuggingFace ──────────────────────────────────────────────────
    # Necesario para descargar V-JEPA2 la primera vez (modelo con acceso restringido)
    if HF_TOKEN:
        try:
            from huggingface_hub import login
            login(token=HF_TOKEN, add_to_git_credential=False)
            print("HuggingFace login OK")
        except Exception as e:
            print(f"[WARN] HF login: {e}")

    # ── Cargar config.json ────────────────────────────────────────────────────
    # El config.json es generado por los notebooks y contiene los nombres de clases,
    # NUM_CLASSES y otros hiperparámetros del experimento.
    cfg_path = ckpt_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        class_names = cfg.get("class_names", [])
        num_classes  = cfg.get("NUM_CLASSES", len(class_names))
    else:
        # Si no hay config.json, los nombres de clases se intentan leer desde los .pt
        print("[WARN] config.json no encontrado — los nombres de clase se leerán desde los .pt")
        class_names, num_classes = [], 2

    models["class_names"] = class_names
    models["num_classes"]  = num_classes
    print(f"Clases ({num_classes}): {class_names}")

    # ── Cargar DINOv2 ─────────────────────────────────────────────────────────
    # Backbone para los modelos DINOv2_puro y DINOv2_to_MCJEPA
    status_cb("Cargando DINOv2...")
    try:
        dino = DINOv2Extractor(DINO_MODEL_NAME)

        # Verificar que el modelo funciona con un batch de prueba
        _dummy = [[Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))] * 4]
        with torch.inference_mode():
            _out = dino(_dummy)
        D_DINO = _out.shape[-1]   # dimensión del embedding (384 para dinov2-small)

        models["dino"]  = dino
        models["D_DINO"] = D_DINO
        print(f"DINOv2 OK — D_DINO={D_DINO}")
    except Exception as e:
        print(f"[ERROR] DINOv2: {e}")
        models["dino"] = None
        D_DINO = 384   # valor por defecto para no bloquear la carga del resto

    # ── Cargar DinoToMCJEPAEncoder ────────────────────────────────────────────
    # Encoder multi-escala que transforma la secuencia DINOv2 en un embedding compacto.
    # Se genera al final de la Celda de guardado del notebook Parte 1.
    status_cb("Cargando DinoToMCJEPAEncoder...")
    try:
        # Buscar el archivo en la raíz o en la subcarpeta embeddings_parte1/
        enc_candidates = [
            ckpt_dir / "dino_to_mcjepa_encoder.pt",
            ckpt_dir / "embeddings_parte1" / "dino_to_mcjepa_encoder.pt",
        ]
        enc_path = next((p for p in enc_candidates if p.exists()), None)
        if enc_path is None:
            raise FileNotFoundError("dino_to_mcjepa_encoder.pt no encontrado")

        enc_ckpt = torch.load(enc_path, map_location=DEVICE, weights_only=False)
        d_in     = enc_ckpt.get("D_DINO", D_DINO)   # dimensión de entrada guardada en el .pt

        dmc_enc = DinoToMCJEPAEncoder(d_in=d_in, hidden=256, nhead=4, layers=1).to(DEVICE)
        dmc_enc.load_state_dict(enc_ckpt["state_dict"])
        dmc_enc.eval()

        models["dmc_enc"] = dmc_enc
        print("DinoToMCJEPAEncoder OK")
    except Exception as e:
        print(f"[ERROR] DinoToMCJEPAEncoder: {e}")
        models["dmc_enc"] = None

    # ── Cargar V-JEPA2 ────────────────────────────────────────────────────────
    # Backbone para los modelos VJEPA2_puro y VJEPA2_MCJEPA_*.
    # Requiere Python 3.10 y transformers compatible con el tipo 'vjepa2'.
    # Se intentan múltiples métodos de carga para maximizar compatibilidad.
    status_cb("Cargando V-JEPA2 (puede tardar ~30 s la primera vez)...")
    vj_enc = None

    for attempt, kwargs in enumerate([
        {"trust_remote_code": True},                                    # usa código del repo HF
        {"trust_remote_code": True, "ignore_mismatched_sizes": True},  # ignora diferencias de forma
        {"trust_remote_code": True, "model_type": "vjepa2"},           # fuerza tipo explícito
    ]):
        try:
            print(f"  V-JEPA2 intento {attempt + 1}: {kwargs}")
            vj_enc = AutoModel.from_pretrained(VJEPA2_MODEL_ID, **kwargs).to(DEVICE)
            print(f"  V-JEPA2 cargado con intento {attempt + 1}")
            break
        except Exception as e:
            print(f"  Intento {attempt + 1} falló: {str(e)[:120]}")
            vj_enc = None

    if vj_enc is not None:
        try:
            vj_tok = VJEPA2TokenExtractor(vj_enc)

            # Verificar con batch dummy que el forward funciona correctamente
            _dummy_v = [[Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))] * 4]
            with torch.inference_mode():
                _tv = vj_tok(_dummy_v)
            D_VJEPA2 = _tv.shape[-1]   # dimensión de tokens (1024 para ViT-L)

            models["vj_tok"]   = vj_tok
            models["D_VJEPA2"] = D_VJEPA2
            print(f"V-JEPA2 OK — D_VJEPA2={D_VJEPA2}")
        except Exception as e:
            print(f"[ERROR] V-JEPA2 forward: {e}")
            models["vj_tok"] = None
            D_VJEPA2 = 1024
    else:
        print("[ERROR] V-JEPA2: todos los intentos fallaron — usa Python 3.10")
        models["vj_tok"] = None
        D_VJEPA2 = 1024

    # ── Cargar clasificadores MLP (generados en Parte 1) ──────────────────────
    # Cada .pt contiene: state_dict, input_dim, num_classes y class_names.
    # Se busca primero en subcarpeta embedding_classifiers/, luego en la raíz.
    clf_dir = ckpt_dir / "embedding_classifiers"

    for rep in ["DINOv2_puro", "DINOv2_to_MCJEPA", "VJEPA2_puro"]:
        status_cb(f"Cargando MLP: {rep}...")
        try:
            candidates = [
                clf_dir / f"{rep}_classifier.pt",   # ubicación estándar (subcarpeta)
                ckpt_dir / f"{rep}_classifier.pt",  # ubicación alternativa (raíz)
            ]
            pt = next((p for p in candidates if p.exists()), None)
            if pt is None:
                raise FileNotFoundError(f"No encontrado en: {[str(c) for c in candidates]}")

            ck  = torch.load(pt, map_location=DEVICE, weights_only=False)
            mlp = EmbeddingMLPClassifier(ck["input_dim"], ck["num_classes"]).to(DEVICE)
            mlp.load_state_dict(ck["state_dict"])
            mlp.eval()

            # Si todavía no tenemos nombres de clases, tomarlos de este checkpoint
            if not class_names and ck.get("class_names"):
                class_names            = ck["class_names"]
                models["class_names"]  = class_names
                num_classes            = len(class_names)
                models["num_classes"]  = num_classes

            models[rep] = mlp
            print(f"  {rep} OK  (input_dim={ck['input_dim']}, clases={ck['num_classes']}, archivo={pt.name})")
        except Exception as e:
            print(f"  [WARN] {rep}: {e}")
            models[rep] = None

    # ── Cargar MCJEPAHead (generados en Parte 2) ──────────────────────────────
    # Cada .pt contiene: head_state_dict, D_VJEPA2, num_classes y el historial de entrenamiento.
    # Se busca en subcarpeta vjepa2_mcjepa_experiments/ o en la raíz.
    mc_dir = ckpt_dir / "vjepa2_mcjepa_experiments"

    for key, fname in [
        ("VJEPA2_MCJEPA_frozen",           "VJEPA2_MCJEPA_frozen.pt"),
        ("VJEPA2_MCJEPA_partial_finetune", "VJEPA2_MCJEPA_partial_finetune.pt"),
    ]:
        status_cb(f"Cargando MCJEPAHead: {key}...")
        try:
            candidates = [mc_dir / fname, ckpt_dir / fname]
            pt = next((p for p in candidates if p.exists()), None)
            if pt is None:
                raise FileNotFoundError(f"No encontrado: {fname}")

            ck   = torch.load(pt, map_location=DEVICE, weights_only=False)
            head = MCJEPAHead(
                d_in=ck.get("D_VJEPA2", D_VJEPA2),
                hidden=256, nhead=4, layers=1,
                num_classes=ck.get("num_classes", num_classes)
            ).to(DEVICE)
            head.load_state_dict(ck["head_state_dict"])
            head.eval()

            models[key] = head
            print(f"  {key} OK")
        except Exception as e:
            print(f"  [WARN] {key}: {e}")
            models[key] = None

    # ── Cargar YOLOv8n para detección de personas ─────────────────────────────
    # Se descarga automáticamente (~6 MB) si no está en caché local.
    # Solo detecta la clase 0 (persona) para dibujar cajas en el video.
    status_cb("Cargando YOLOv8n...")
    try:
        from ultralytics import YOLO
        models["yolo"] = YOLO("yolov8n.pt")
        print("YOLO OK")
    except Exception as e:
        print(f"[WARN] YOLO: {e}")
        models["yolo"] = None

    # ── Verificar qué modelos están completamente operativos ──────────────────
    # Un modelo está "listo" solo si TANTO el extractor de features COMO el
    # clasificador se cargaron correctamente. Si falta cualquiera, no puede predecir.
    def _model_ready(key):
        """Verifica si el modelo 'key' tiene todas sus dependencias cargadas."""
        if not models.get(key):
            return False, "clasificador .pt no cargado"
        if key == "DINOv2_puro" and not models.get("dino"):
            return False, "DINOv2 extractor falló"
        if key == "DINOv2_to_MCJEPA":
            if not models.get("dino"):    return False, "DINOv2 extractor falló"
            if not models.get("dmc_enc"): return False, "falta dino_to_mcjepa_encoder.pt"
        if key in ("VJEPA2_puro", "VJEPA2_MCJEPA_frozen", "VJEPA2_MCJEPA_partial_finetune"):
            if not models.get("vj_tok"):  return False, "V-JEPA2 extractor falló (usa py -3.10)"
        return True, "OK"

    models["_ready"] = {}
    print("\n" + "=" * 60)
    print("RESUMEN DE CARGA:")
    for key in MODEL_ORDER:
        ok, reason = _model_ready(key)
        models["_ready"][key] = ok
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {MODEL_LABELS[key]:35s}  {reason}")
    print(f"\n  Clases ({len(models.get('class_names', []))}): {models.get('class_names', [])}")
    print("=" * 60 + "\n")

    return models


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def run_inference(frames_rgb, frame_bgr, models):
    """
    Ejecuta todos los clasificadores disponibles sobre el clip actual.

    Flujo:
      1. Convertir buffer de frames a PIL (con submuestreo a NUM_FRAMES)
      2. Extraer features con DINOv2 (si está disponible)
      3. Extraer tokens con V-JEPA2 (si está disponible)
      4. Pasar por cada clasificador activo según sus dependencias
      5. Detectar personas con YOLO sobre el frame actual
      6. Calcular heatmap de movimiento

    Los extractores son compartidos entre clasificadores:
      DINOv2 → DINOv2_puro y DINOv2_to_MCJEPA
      V-JEPA2 → VJEPA2_puro, VJEPA2_MCJEPA_frozen, VJEPA2_MCJEPA_partial_finetune

    Retorna:
      result  — dict {model_key: {"probs", "pred", "conf"}} con predicciones
      boxes   — lista de (x1, y1, x2, y2, conf) de personas detectadas
      heatmap — array float32 [H, W] normalizado a [0,1] o None
    """
    H, W = frame_bgr.shape[:2]

    # Preparar el clip: submuestrear y convertir a PIL
    pil_frames = frames_to_pil(frames_rgb, NUM_FRAMES, IMAGE_SIZE)
    batch = [pil_frames]   # batch de tamaño 1 para inferencia online

    result = {}

    # ── Paso 1: extraer features base (compartidas entre clasificadores) ──────
    dino_seq  = None   # [1, T, D_dino]   — secuencia de CLS tokens DINOv2
    vj_tokens = None   # [1, N, D_vjepa2] — tokens spatio-temporales V-JEPA2

    if models.get("dino"):
        try:
            dino_seq = models["dino"](batch)
        except Exception as e:
            print("DINOv2 fwd:", e)

    if models.get("vj_tok"):
        try:
            vj_tokens = models["vj_tok"](batch)
        except Exception as e:
            print("VJEPA2 fwd:", e)

    ready = models.get("_ready", {})

    # ── Paso 2: clasificar con cada modelo activo ─────────────────────────────

    # DINOv2_puro: promedio temporal de CLS tokens → MLP
    if ready.get("DINOv2_puro") and dino_seq is not None:
        try:
            emb    = dino_seq.mean(dim=1)             # [1, D_dino] — promedio sobre frames
            logits = models["DINOv2_puro"](emb)
            probs  = F.softmax(logits, dim=-1)[0].cpu()
            result["DINOv2_puro"] = {
                "probs": probs, "pred": probs.argmax().item(), "conf": probs.max().item()}
        except Exception as e:
            print("DINOv2_puro clf:", e)

    # DINOv2_to_MCJEPA: secuencia DINOv2 → encoder multi-escala → MLP
    if ready.get("DINOv2_to_MCJEPA") and dino_seq is not None:
        try:
            emb    = models["dmc_enc"](dino_seq)      # [1, hidden*3] — embedding multi-escala
            logits = models["DINOv2_to_MCJEPA"](emb)
            probs  = F.softmax(logits, dim=-1)[0].cpu()
            result["DINOv2_to_MCJEPA"] = {
                "probs": probs, "pred": probs.argmax().item(), "conf": probs.max().item()}
        except Exception as e:
            print("DINOv2_to_MCJEPA clf:", e)

    # VJEPA2_puro: promedio de tokens V-JEPA2 → MLP
    if ready.get("VJEPA2_puro") and vj_tokens is not None:
        try:
            emb    = vj_tokens.mean(dim=1)            # [1, D_vjepa2] — promedio sobre tokens
            logits = models["VJEPA2_puro"](emb)
            probs  = F.softmax(logits, dim=-1)[0].cpu()
            result["VJEPA2_puro"] = {
                "probs": probs, "pred": probs.argmax().item(), "conf": probs.max().item()}
        except Exception as e:
            print("VJEPA2_puro clf:", e)

    # VJEPA2_MCJEPA_frozen: tokens V-JEPA2 → MCJEPAHead (backbone congelado en entrenamiento)
    if ready.get("VJEPA2_MCJEPA_frozen") and vj_tokens is not None:
        try:
            out   = models["VJEPA2_MCJEPA_frozen"](vj_tokens)
            probs = F.softmax(out["logits"], dim=-1)[0].cpu()
            result["VJEPA2_MCJEPA_frozen"] = {
                "probs": probs, "pred": probs.argmax().item(), "conf": probs.max().item()}
        except Exception as e:
            print("MCJEPA_frozen:", e)

    # VJEPA2_MCJEPA_partial_finetune: igual pero con últimas capas de V-JEPA2 descongeladas
    if ready.get("VJEPA2_MCJEPA_partial_finetune") and vj_tokens is not None:
        try:
            out   = models["VJEPA2_MCJEPA_partial_finetune"](vj_tokens)
            probs = F.softmax(out["logits"], dim=-1)[0].cpu()
            result["VJEPA2_MCJEPA_partial_finetune"] = {
                "probs": probs, "pred": probs.argmax().item(), "conf": probs.max().item()}
        except Exception as e:
            print("MCJEPA_partial:", e)

    # ── Paso 3: detección de personas con YOLO ────────────────────────────────
    boxes = []
    if models.get("yolo"):
        try:
            # classes=[0]: solo detectar personas (clase 0 en COCO)
            res = models["yolo"](frame_bgr, classes=[0], verbose=False)[0]
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        except Exception as e:
            print("YOLO:", e)

    # ── Paso 4: heatmap de movimiento ─────────────────────────────────────────
    heatmap = build_motion_heatmap(frames_rgb, H, W)

    return result, boxes, heatmap


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES DE LA INTERFAZ
# ══════════════════════════════════════════════════════════════════════════════

# Orden en que se muestran los modelos en el panel lateral y en los recuadros del video
MODEL_ORDER = [
    "DINOv2_puro",
    "DINOv2_to_MCJEPA",
    "VJEPA2_puro",
    "VJEPA2_MCJEPA_frozen",
    "VJEPA2_MCJEPA_partial_finetune",
]

# Etiquetas legibles para mostrar en la UI (más cortas que los keys internos)
MODEL_LABELS = {
    "DINOv2_puro":                    "DINOv2 puro",
    "DINOv2_to_MCJEPA":               "DINOv2 → MCJEPA",
    "VJEPA2_puro":                    "VJEPA2 puro",
    "VJEPA2_MCJEPA_frozen":           "VJEPA2 + MCJEPA frozen",
    "VJEPA2_MCJEPA_partial_finetune": "VJEPA2 + MCJEPA partial",
}


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ GRÁFICA
# ══════════════════════════════════════════════════════════════════════════════

class VideoAnalyzerApp:
    """
    Aplicación principal de análisis de video.

    Layout:
      ┌─────────────────────────────────┬────────────────┐
      │  Canvas de video (con overlay)  │  Panel lateral │
      │  + heatmap + cajas YOLO         │  5 tarjetas    │
      │  + recuadros de predicción      │  + consenso    │
      ├─────────────────────────────────┤                │
      │  Controles: abrir, play, seek   │                │
      └─────────────────────────────────┴────────────────┘

    La inferencia corre en un hilo separado para no bloquear la UI.
    Los resultados se actualizan en el hilo principal via root.after().
    """

    # Tamaño del buffer circular de frames para inferencia
    WIN_FRAMES = 32

    def __init__(self, root):
        self.root = root
        tag = f" [GPU: {torch.cuda.get_device_name(0)}]" if DEVICE == "cuda" else " [CPU]"
        self.root.title(f"MC-JEPA — 5 Clasificadores{tag}")
        self.root.configure(bg=C_BG)
        self.root.minsize(1500, 820)

        # Estado del video
        self.cap          = None    # cv2.VideoCapture activo
        self.playing      = False
        self.frame_buf    = deque(maxlen=self.WIN_FRAMES)   # buffer circular de frames RGB
        self.cur_idx      = 0       # frame actual
        self.total_frames = 0
        self.native_fps   = 25.0

        # Modelos y sincronización
        self.models       = {}
        self._lock        = threading.Lock()   # protege acceso a last_result, last_boxes, last_heatmap
        self._inf_running = False              # semáforo: evita lanzar dos inferencias simultáneas

        # Últimos resultados de inferencia (compartidos entre hilo de inferencia y hilo UI)
        self.last_result  = {}
        self.last_boxes   = []
        self.last_heatmap = None

        # Variables de control de la UI
        self.show_heatmap = tk.BooleanVar(value=True)
        self.show_boxes   = tk.BooleanVar(value=True)
        self.status_var   = tk.StringVar(value="Iniciando...")
        self._tk_img      = None   # referencia a la imagen actual (evita que el GC la elimine)

        self._build_ui()

        # Cargar modelos en hilo separado para no bloquear la ventana al inicio
        threading.Thread(target=self._load_models, daemon=True).start()

    # ── Construcción de la interfaz ───────────────────────────────────────────

    def _build_ui(self):
        """Construye todos los widgets de la interfaz."""
        main = tk.Frame(self.root, bg=C_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ── Panel izquierdo: video + controles ────────────────────────────────
        left = tk.Frame(main, bg=C_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas donde se renderiza el video con todos los overlays
        self.canvas = tk.Canvas(left, bg=C_DARK, highlightthickness=1,
                                highlightbackground="#2a2a4a")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._img_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        # Barra de controles: abrir, play/pause, checkboxes, seekbar, contador
        ctrl = tk.Frame(left, bg=C_PANEL, pady=5)
        ctrl.pack(fill=tk.X, pady=(5, 0))

        self._btn(ctrl, "📂 Abrir video", self._open_video, "#0f3460").pack(side=tk.LEFT, padx=6)
        self.btn_play = self._btn(ctrl, "▶", self._toggle_play, "#533483",
                                  font=("Segoe UI", 12), state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(ctrl, text="Heatmap",   variable=self.show_heatmap,
                       bg=C_PANEL, fg="#aaa", selectcolor=C_PANEL,
                       activebackground=C_PANEL).pack(side=tk.LEFT, padx=8)
        tk.Checkbutton(ctrl, text="Detección", variable=self.show_boxes,
                       bg=C_PANEL, fg="#aaa", selectcolor=C_PANEL,
                       activebackground=C_PANEL).pack(side=tk.LEFT)

        self.seek_var = tk.DoubleVar()
        ttk.Scale(ctrl, from_=0, to=100, orient=tk.HORIZONTAL,
                  variable=self.seek_var, command=self._on_seek
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.lbl_frame = tk.Label(ctrl, text="0/0", bg=C_PANEL, fg="#555",
                                  font=("Segoe UI", 8))
        self.lbl_frame.pack(side=tk.RIGHT, padx=6)

        # Barra de estado inferior (mensajes de carga, info del video)
        tk.Label(left, textvariable=self.status_var, bg=C_BG, fg="#666",
                 font=("Segoe UI", 8), anchor=tk.W).pack(fill=tk.X, pady=(3, 0))

        # ── Panel derecho: tarjetas de clasificadores ─────────────────────────
        right_outer = tk.Frame(main, bg=C_PANEL, width=340,
                               highlightthickness=1, highlightbackground="#2a2a4a")
        right_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right_outer.pack_propagate(False)

        tk.Label(right_outer, text="CLASIFICADORES", bg=C_PANEL, fg="#666",
                 font=("Segoe UI", 8, "bold"), pady=8).pack()

        # Una tarjeta por modelo, con indicador de estado, predicción, barra y top-3
        self._cards = {}
        for key in MODEL_ORDER:
            self._cards[key] = self._build_model_card(right_outer, key)
            tk.Frame(right_outer, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=4)

        # Sección de consenso: clase con más votos entre los modelos activos
        tk.Label(right_outer, text="CONSENSO", bg=C_PANEL, fg="#666",
                 font=("Segoe UI", 8, "bold")).pack(pady=(4, 2))
        self.lbl_consensus = tk.Label(right_outer, text="—", bg=C_PANEL, fg="white",
                                      font=("Consolas", 16, "bold"))
        self.lbl_consensus.pack()
        self.lbl_consensus_detail = tk.Label(right_outer, text="esperando inferencia",
                                             bg=C_PANEL, fg="#666", font=("Segoe UI", 8))
        self.lbl_consensus_detail.pack(pady=(0, 6))

    def _build_model_card(self, parent, key):
        """
        Crea la tarjeta visual de un clasificador en el panel lateral.

        Contiene:
          - Punto de color + nombre del modelo
          - Indicador de estado (⏳ cargando / ✅ listo / ❌ error)
          - Etiqueta con clase predicha y confianza
          - Barra de confianza (canvas)
          - Top-3 clases con sus probabilidades
        """
        color = MODEL_COLORS.get(key, "#aaa")
        label = MODEL_LABELS.get(key, key)

        card = tk.Frame(parent, bg=C_PANEL)
        card.pack(fill=tk.X, padx=10, pady=2)

        # Encabezado: punto de color + nombre + indicador de estado
        hdr = tk.Frame(card, bg=C_PANEL)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="●", bg=C_PANEL, fg=color, font=("Segoe UI", 10)).pack(side=tk.LEFT)
        tk.Label(hdr, text=f"  {label}", bg=C_PANEL, fg="white",
                 font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)

        # Indicador de estado: ⏳ mientras carga, ✅ si listo, ❌ si falló
        lbl_status = tk.Label(hdr, text="⏳", bg=C_PANEL, fg="#888",
                              font=("Segoe UI", 9))
        lbl_status.pack(side=tk.RIGHT, padx=4)

        # Clase predicha con confianza (se actualiza en cada inferencia)
        lbl_pred = tk.Label(card, text="cargando...", bg=C_PANEL, fg="#555",
                            font=("Consolas", 12, "bold"), anchor=tk.W)
        lbl_pred.pack(fill=tk.X, pady=(3, 1))

        # Barra de confianza visual (canvas dibujado programáticamente)
        bar = tk.Canvas(card, bg=C_DARK, height=16, highlightthickness=0)
        bar.pack(fill=tk.X, pady=(0, 2))

        # Top-3 clases con sus probabilidades (texto pequeño debajo de la barra)
        lbl_top = tk.Label(card, text="", bg=C_PANEL, fg="#777",
                           font=("Segoe UI", 7), anchor=tk.W, justify=tk.LEFT)
        lbl_top.pack(fill=tk.X)

        return {"pred": lbl_pred, "bar": bar, "top": lbl_top,
                "color": color, "status": lbl_status}

    def _btn(self, parent, text, cmd, bg, font=("Segoe UI", 10), **kw):
        """Crea un botón estilizado con el tema oscuro de la app."""
        return tk.Button(parent, text=text, command=cmd, bg=bg, fg="white",
                         activebackground="#334", relief=tk.FLAT,
                         padx=10, pady=4, font=font, cursor="hand2", **kw)

    # ── Carga de modelos (hilo de fondo) ──────────────────────────────────────

    def _load_models(self):
        """
        Carga todos los modelos en un hilo de fondo para no congelar la UI.
        Al terminar, actualiza los indicadores de estado de cada tarjeta.
        """
        self.models = load_all_models(
            CHECKPOINT_DIR,
            status_cb=lambda msg: self.root.after(0, self.status_var.set, msg)
        )
        n_loaded = sum(1 for k in MODEL_ORDER if self.models.get("_ready", {}).get(k))
        mode = "GPU" if DEVICE == "cuda" else "CPU"
        self.root.after(0, self.status_var.set,
                        f"Listo [{mode}] — {n_loaded}/5 modelos cargados — abre un video")

        # Actualizar indicadores visuales en el hilo principal
        self.root.after(0, self._update_card_status)

    def _update_card_status(self):
        """
        Actualiza los indicadores ✅/❌ de cada tarjeta según el resultado de la carga.
        Se ejecuta en el hilo principal de Tkinter después de que los modelos cargan.
        """
        ready = self.models.get("_ready", {})
        for key in MODEL_ORDER:
            card = self._cards[key]
            ok   = ready.get(key, False)

            if ok:
                card["status"].configure(text="✅", fg="#81C784")
                card["pred"].configure(text="—", fg=card["color"])
            else:
                # Mensaje específico según qué dependencia falta
                if key == "DINOv2_to_MCJEPA" and not self.models.get("dmc_enc"):
                    msg = "falta dino_to_mcjepa_encoder.pt"
                elif key in ("VJEPA2_puro", "VJEPA2_MCJEPA_frozen", "VJEPA2_MCJEPA_partial_finetune") \
                        and not self.models.get("vj_tok"):
                    msg = "V-JEPA2 no cargó — usa py -3.10"
                elif not self.models.get(key):
                    msg = "checkpoint .pt no encontrado"
                else:
                    msg = "no disponible"
                card["status"].configure(text="❌", fg="#ef5350")
                card["pred"].configure(text=msg, fg="#555")

    # ── Control de video ──────────────────────────────────────────────────────

    def _open_video(self):
        """
        Abre un diálogo de selección de archivo y carga el video seleccionado.
        Prueba múltiples backends de OpenCV para maximizar compatibilidad de formatos.
        """
        path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("Todos", "*.*")])
        if not path:
            return

        # Liberar el video anterior si hay uno abierto
        if self.cap:
            self.playing = False
            self.cap.release()

        # Intentar abrir con diferentes backends (compatibilidad con códecs)
        for backend in [cv2.CAP_ANY, cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_DSHOW]:
            cap = cv2.VideoCapture(path, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.cap = cap
                    break
                cap.release()
        else:
            messagebox.showerror("Error", "No se puede abrir el video. Convierte a MP4/H.264.")
            return

        # Inicializar estado del video
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.native_fps   = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.cur_idx      = 0
        self.frame_buf.clear()
        self.last_result  = {}
        self.last_boxes   = []
        self.last_heatmap = None

        self.seek_var.set(0)
        self.btn_play.configure(state=tk.NORMAL)
        self._read_frame()
        self.status_var.set(
            f"{Path(path).name}  |  {self.total_frames} frames  |  {self.native_fps:.1f} fps")

    def _toggle_play(self):
        """Alterna entre play y pause."""
        if not self.cap: return
        self.playing = not self.playing
        self.btn_play.configure(text="⏸" if self.playing else "▶")
        if self.playing: self._play_loop()

    def _play_loop(self):
        """
        Loop de reproducción usando root.after() para no bloquear el hilo UI.
        Calcula el delay dinámicamente para mantener el FPS nativo del video.
        """
        if not self.playing or not self.cap: return
        t0 = time.perf_counter()
        self._read_frame()
        elapsed = (time.perf_counter() - t0) * 1000
        # Programar el siguiente frame respetando el FPS nativo
        self.root.after(max(1, int(1000 / self.native_fps - elapsed)), self._play_loop)

    def _on_seek(self, val):
        """Salta al frame indicado por la seekbar."""
        if not self.cap: return
        idx = int(float(val))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self.cur_idx = idx
        self.frame_buf.clear()   # limpiar buffer: el contexto temporal cambió
        if not self.playing: self._read_frame()

    def _read_frame(self):
        """
        Lee el siguiente frame del video, lo agrega al buffer y decide si
        lanzar una inferencia. Actualiza la UI con el frame renderizado.
        """
        if not self.cap: return

        ret, bgr = self.cap.read()
        if not ret:
            # Fin del video
            self.playing = False
            self.btn_play.configure(text="▶")
            return

        self.cur_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.lbl_frame.configure(text=f"{self.cur_idx}/{self.total_frames}")
        self.seek_var.set(min(self.cur_idx, self.total_frames - 1))

        # Convertir BGR (OpenCV) a RGB (PIL/numpy) y agregar al buffer circular
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.frame_buf.append(rgb.copy())

        # Lanzar inferencia si:
        #   - hay suficientes frames en el buffer (al menos WIN_FRAMES//2)
        #   - es el turno según INF_EVERY
        #   - no hay otra inferencia corriendo
        #   - los modelos ya cargaron
        should_infer = (
            len(self.frame_buf) >= self.WIN_FRAMES // 2
            and self.cur_idx % max(1, INF_EVERY) == 0
            and not self._inf_running
            and bool(self.models)
        )
        if should_infer:
            buf_copy = list(self.frame_buf)
            bgr_copy = bgr.copy()
            threading.Thread(
                target=self._run_inference_thread,
                args=(buf_copy, bgr_copy), daemon=True).start()

        # Renderizar y mostrar el frame actual (con overlays del último resultado)
        self._show_frame(self._render(rgb))

    # ── Inferencia en hilo separado ───────────────────────────────────────────

    def _run_inference_thread(self, frames, bgr):
        """
        Ejecuta la inferencia en un hilo de fondo para no bloquear la UI.
        Actualiza last_result, last_boxes y last_heatmap con lock para thread safety.
        Al terminar, programa la actualización del sidebar en el hilo principal.
        """
        self._inf_running = True
        try:
            result, boxes, heatmap = run_inference(frames, bgr, self.models)

            # Actualizar resultados compartidos con lock para evitar condiciones de carrera
            with self._lock:
                self.last_result  = result
                self.last_boxes   = boxes
                self.last_heatmap = heatmap

            # Actualizar sidebar en el hilo principal de Tkinter
            self.root.after(0, self._update_sidebar, result)
        except Exception:
            print(traceback.format_exc())
        finally:
            self._inf_running = False

    # ── Render del frame con overlays ─────────────────────────────────────────

    def _render(self, frame_rgb):
        """
        Renderiza el frame con todos los overlays:
          1. Heatmap de movimiento (semi-transparente, colormap JET)
          2. Cajas de detección de personas (YOLO)
          3. Recuadros de predicción por modelo (anclados sobre las personas o esquina)
        """
        disp = frame_rgb.copy()

        # Leer los últimos resultados de forma thread-safe
        with self._lock:
            boxes   = list(self.last_boxes)
            heatmap = None if self.last_heatmap is None else self.last_heatmap.copy()
            result  = dict(self.last_result)

        H_f, W_f = disp.shape[:2]

        # ── Overlay 1: heatmap de movimiento ─────────────────────────────────
        if self.show_heatmap.get() and heatmap is not None:
            try:
                hm   = cv2.resize(heatmap, (W_f, H_f))
                # Aplicar colormap JET: azul=poco movimiento, rojo=mucho movimiento
                hm_c = cv2.cvtColor(
                    cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET),
                    cv2.COLOR_BGR2RGB)
                # Mezclar con el frame original (62% original, 38% heatmap)
                disp = cv2.addWeighted(disp, 0.62, hm_c, 0.38, 0)
            except Exception:
                pass

        # ── Overlay 2: cajas de detección de personas ─────────────────────────
        if self.show_boxes.get():
            for x1, y1, x2, y2, conf in boxes:
                # Rectángulo azul claro alrededor de la persona
                cv2.rectangle(disp, (x1, y1), (x2, y2), (79, 195, 247), 2)
                # Etiqueta con fondo sólido encima de la caja
                lbl = f"Persona {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(disp, (x1, y1 - th - 7), (x1 + tw + 6, y1), (79, 195, 247), -1)
                cv2.putText(disp, lbl, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 30), 1)

        # ── Overlay 3: recuadros de predicción por modelo ─────────────────────
        class_names = self.models.get("class_names", [])
        active = [k for k in MODEL_ORDER if k in result]   # modelos con predicción actual

        if self._inf_running:
            # Indicador de que hay una inferencia en curso
            cv2.rectangle(disp, (8, 8), (180, 30), (13, 13, 26), -1)
            cv2.rectangle(disp, (8, 8), (180, 30), (80, 80, 120), 1)
            cv2.putText(disp, "Analizando...", (14, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 200), 1)

        elif active:
            # Posicionar recuadros:
            #   - Si hay personas detectadas: encima de la primera persona
            #   - Si no: esquina superior izquierda
            if boxes and self.show_boxes.get():
                x1_p    = min(b[0] for b in boxes)
                y1_p    = min(b[1] for b in boxes)
                anchor_x = max(8, x1_p)
                anchor_y = max(8, y1_p - len(active) * 36 - 8)
            else:
                anchor_x, anchor_y = 8, 8

            # Dimensiones y estilo de cada recuadro
            BOX_W  = 260    # ancho del recuadro en píxeles
            BOX_H  = 32     # alto del recuadro en píxeles
            PAD    = 4      # espacio entre recuadros
            FONT   = cv2.FONT_HERSHEY_SIMPLEX
            FSCALE = 0.44   # escala del texto de predicción
            FTHICK = 1

            for i, key in enumerate(active):
                r        = result[key]
                cls_name = class_names[r["pred"]] if r["pred"] < len(class_names) else str(r["pred"])
                conf     = r["conf"]
                color_bgr = self._hex_to_bgr(MODEL_COLORS.get(key, "#ffffff"))

                # Acortar etiqueta del modelo para que quepa en el recuadro
                short_lbl = MODEL_LABELS[key].replace("DINOv2 →", "DINO→") \
                                             .replace("VJEPA2 + MCJEPA", "VJEPA2")

                bx = anchor_x
                by = anchor_y + i * (BOX_H + PAD)

                # Fondo semitransparente: copiar frame, dibujar rectángulo oscuro y mezclar
                overlay = disp.copy()
                cv2.rectangle(overlay, (bx, by), (bx + BOX_W, by + BOX_H), (13, 13, 26), -1)
                cv2.addWeighted(overlay, 0.75, disp, 0.25, 0, disp)

                # Borde del color del modelo para identificación visual rápida
                cv2.rectangle(disp, (bx, by), (bx + BOX_W, by + BOX_H), color_bgr, 2)

                # Barra de confianza al fondo del recuadro (proporcional a conf)
                bar_w = int((BOX_W - 4) * conf)
                cv2.rectangle(disp, (bx + 2, by + BOX_H - 5),
                              (bx + 2 + bar_w, by + BOX_H - 2), color_bgr, -1)

                # Nombre del modelo (fila superior, gris claro)
                cv2.putText(disp, short_lbl, (bx + 6, by + 12),
                            FONT, 0.38, (180, 180, 180), FTHICK)

                # Clase predicha + confianza (fila inferior, color del modelo)
                pred_txt = f"{cls_name[:22]}  {conf:.0%}"
                cv2.putText(disp, pred_txt, (bx + 6, by + 26),
                            FONT, FSCALE, color_bgr, FTHICK)

        return disp

    @staticmethod
    def _hex_to_bgr(hex_color):
        """Convierte color hexadecimal '#RRGGBB' a tupla BGR para OpenCV."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)   # OpenCV usa BGR, no RGB

    def _show_frame(self, frame_rgb):
        """
        Escala el frame para ajustarse al canvas manteniendo el aspect ratio,
        lo centra sobre fondo negro y lo muestra en la UI.
        """
        cw = self.canvas.winfo_width()  or 960
        ch = self.canvas.winfo_height() or 540
        h, w = frame_rgb.shape[:2]

        # Escalar manteniendo aspect ratio
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        img = Image.fromarray(frame_rgb).resize((nw, nh), Image.LANCZOS)

        # Centrar sobre fondo negro del tamaño del canvas
        bg = Image.new("RGB", (cw, ch), (13, 13, 26))
        bg.paste(img, ((cw - nw) // 2, (ch - nh) // 2))

        # Mantener referencia al PhotoImage para evitar que el GC lo elimine
        self._tk_img = ImageTk.PhotoImage(bg)
        self.canvas.itemconfig(self._img_id, image=self._tk_img)

    # ── Actualización del panel lateral ──────────────────────────────────────

    def _update_sidebar(self, result):
        """
        Actualiza todas las tarjetas del panel lateral con los últimos resultados
        de inferencia y calcula el consenso por mayoría de votos.
        """
        class_names = self.models.get("class_names", [])

        for key in MODEL_ORDER:
            card = self._cards[key]

            if key not in result:
                # Modelo sin resultado: no cargó o aún no tiene suficientes frames
                card["pred"].configure(
                    text="No cargado" if not self.models.get(key) else "—")
                continue

            r        = result[key]
            probs    = r["probs"]
            pred_idx = r["pred"]
            conf     = r["conf"]
            cls_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

            # Actualizar predicción y barra de confianza
            card["pred"].configure(text=f"{cls_name}  {conf:.0%}")
            self._draw_bar(card["bar"], conf, card["color"])

            # Mostrar top-3 clases con probabilidades
            top_k   = min(3, len(probs))
            top_idx = probs.topk(top_k).indices.tolist()
            top_lines = []
            for i in top_idx:
                cn = class_names[i] if i < len(class_names) else str(i)
                top_lines.append(f"  {probs[i]:.1%}  {cn}")
            card["top"].configure(text="\n".join(top_lines))

        # ── Consenso: mayoría de votos entre modelos activos ──────────────────
        # Cada modelo activo vota por su clase predicha; gana la más votada.
        votes = {}
        for key in MODEL_ORDER:
            if key not in result: continue
            pred_idx = result[key]["pred"]
            cls_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
            votes[cls_name] = votes.get(cls_name, 0) + 1

        if votes:
            winner = max(votes, key=votes.get)
            total  = sum(votes.values())
            self.lbl_consensus.configure(text=winner)
            # Detalle: "ClaseA: 3/5  ClaseB: 2/5"
            detail = "  ".join(f"{c}: {n}/{total}" for c, n in
                                sorted(votes.items(), key=lambda x: -x[1]))
            self.lbl_consensus_detail.configure(text=detail)

    def _draw_bar(self, canvas, value, color):
        """
        Dibuja una barra de progreso horizontal en el canvas dado.
        La barra tiene una línea blanca en el borde superior para efecto de brillo.
        """
        canvas.delete("all")
        w = canvas.winfo_width()  or 300
        h = canvas.winfo_height() or 16

        # Fondo oscuro
        canvas.create_rectangle(0, 0, w, h, fill=C_DARK, outline="")

        # Barra rellena proporcional al valor de confianza
        filled = max(2, int(w * float(value)))
        canvas.create_rectangle(0, 0, filled, h, fill=color, outline="")

        # Línea blanca superior como detalle visual
        canvas.create_rectangle(0, 0, filled, 2, fill="white", outline="")

        # Texto centrado con el porcentaje
        canvas.create_text(w // 2, h // 2, text=f"{value:.1%}",
                           fill="white", font=("Segoe UI", 7, "bold"))


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # IMPORTANTE: ejecutar con Python 3.10 para compatibilidad con V-JEPA2
    #   py -3.10 interfaz_mcjepa_v2.py
    root = tk.Tk()
    root.geometry("1500x860")
    app = VideoAnalyzerApp(root)
    root.mainloop()
