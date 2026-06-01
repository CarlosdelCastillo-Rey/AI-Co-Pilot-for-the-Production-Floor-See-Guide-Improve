# Checkpoints

Esta carpeta almacena los pesos entrenados generados por los notebooks de Colab.

## Archivos esperados

```
checkpoints/
├── config.json                           ← nombres de clases, hiperparámetros
├── dino_to_mcjepa_encoder.pt             ← encoder multi-escala sobre DINOv2 (Parte 1)
├── DINOv2_puro_classifier.pt             ← MLP sobre embeddings DINOv2 promediados
├── DINOv2_to_MCJEPA_classifier.pt        ← MLP sobre embeddings del encoder MC-JEPA
├── VJEPA2_puro_classifier.pt             ← MLP sobre tokens promediados de V-JEPA2
├── VJEPA2_MCJEPA_frozen.pt               ← MCJEPAHead con V-JEPA2 congelado (Parte 2)
└── VJEPA2_MCJEPA_partial_finetune.pt     ← MCJEPAHead con últimas capas descongeladas (Parte 2)
```

## Cómo obtenerlos

1. Correr `notebooks/Parte1_Extraccion_Baselines.ipynb` en Colab → genera los primeros 4 archivos
2. Correr `notebooks/Parte2_Clasificacion_MCJEPA.ipynb` en Colab → genera los últimos 2
3. Descargar la carpeta `MC_JEPA_INHARD_CHECKPOINTS/` de Google Drive
4. Colocar los archivos `.pt` y `config.json` en esta carpeta

