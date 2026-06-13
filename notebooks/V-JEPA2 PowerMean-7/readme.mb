# V-JEPA2 PowerMean-7

## Descripción general

Este modelo corresponde al mejor clasificador obtenido para reconocimiento de acciones humanas en videos del dataset InHARD.
La arquitectura final utiliza embeddings extraídos con **V-JEPA2** y un ensamble de siete clasificadores MLP.

El flujo general es:

```text
Video / clip T=8
        ↓
V-JEPA2
        ↓
Embedding fused mean [1024]
        ↓
7 clasificadores MLP
        ↓
Power Mean Ensemble q=0.5
        ↓
Acción predicha
```

El modelo final fue nombrado:

```text
V-JEPA2 PowerMean-7
```

## Mejor resultado obtenido

| Métrica           | Resultado |
| ----------------- | --------: |
| Accuracy          |  0.878141 |
| Balanced Accuracy |  0.848224 |
| Precision Macro   |  0.886422 |
| Recall Macro      |  0.848224 |
| F1 Macro          |  0.860986 |
| F1 Weighted       |  0.874309 |

## Entrada del modelo

El clasificador final no recibe directamente el video.
Primero se debe extraer un embedding usando V-JEPA2.

La entrada esperada para el clasificador final es:

```python
z_fused.shape == [1024]
```

o para varios ejemplos:

```python
z_fused.shape == [B, 1024]
```

Donde:

* `B` es el batch size.
* `1024` es la dimensión del embedding generado por V-JEPA2.
* `z_fused` representa el embedding fusionado por promedio entre vistas.

## Salida del modelo

La función de predicción regresa:

```python
{
    "predicted_class_id": int,
    "predicted_class": str,
    "confidence": float,
    "top_k": [
        {
            "class_id": int,
            "class_name": str,
            "probability": float
        }
    ]
}
```

Ejemplo de salida:

```python
{
    "predicted_class_id": 8,
    "predicted_class": "Take component",
    "confidence": 0.87,
    "top_k": [
        {"class_id": 8, "class_name": "Take component", "probability": 0.87},
        {"class_id": 0, "class_name": "Assemble system", "probability": 0.06},
        {"class_id": 2, "class_name": "No action", "probability": 0.03}
    ]
}
```

## Clases disponibles

El modelo predice una de las siguientes 12 clases:

```text
0. Assemble system
1. Consult sheets
2. No action
3. Picking in front
4. Picking left
5. Put down component
6. Put down measuring rod
7. Put down screwdriver
8. Take component
9. Take measuring rod
10. Take screwdriver
11. Turn sheets
```

## Archivos necesarios

Para correr el modelo final se requieren los siguientes siete checkpoints:

```text
best_OFFICIAL_vjepa2_t8_fused_mean_mlp_plain.pt
Exp5A_plain_MLP_h512_drop0.25_lr0.0001_ls0.05.pt
Exp6A_mlp_seed11_h512_drop025_lr1e4_ls005.pt
Exp6A_mlp_seed22_h512_drop030_lr1e4_ls005.pt
Exp6A_mlp_seed33_h512_drop025_lr3e4_ls003.pt
Exp6A_mlp_seed44_h256_drop025_lr1e4_ls005.pt
Exp6A_mlp_seed55_h512_drop020_lr1e4_ls003.pt
```

Estos archivos son las cabezas clasificadoras MLP.
No son siete modelos V-JEPA2 completos; todos reciben el mismo embedding V-JEPA2 `[1024]`.

## Lógica del ensamble final

Cada MLP genera una distribución de probabilidad sobre las 12 clases:

```text
MLP 1 → [12 probabilidades]
MLP 2 → [12 probabilidades]
...
MLP 7 → [12 probabilidades]
```

Después, las probabilidades se combinan con **Power Mean Ensemble** usando:

```text
q = 0.5
```

La fórmula usada es:

```text
p_final,c = normalize( mean(p_model,c ^ q) ^ (1/q) )
```

Con `q = 0.5`, esto equivale a:

```text
1. Tomar la raíz cuadrada de cada probabilidad.
2. Promediar las raíces.
3. Elevar el resultado al cuadrado.
4. Normalizar para que las probabilidades sumen 1.
```

Este método dio mejor resultado que el promedio simple, el promedio geométrico y el hard voting.

## Ejemplo de uso

```python
# z_fused debe ser un embedding V-JEPA2 T=8 fused mean de tamaño [1024]

prediction = predict_action_final(z_fused)

print("Clase predicha:", prediction["predicted_class"])
print("Confianza:", prediction["confidence"])
print("Top-k:", prediction["top_k"])
```

## Resumen del pipeline

```text
1. Tomar un clip de video con T=8 frames.
2. Procesar el clip con V-JEPA2.
3. Extraer el embedding de 1024 dimensiones.
4. Fusionar las vistas usando fused mean.
5. Pasar el embedding por los siete MLPs.
6. Combinar probabilidades con Power Mean q=0.5.
7. Seleccionar la clase con mayor probabilidad final.
```

## Nombre recomendado del modelo

```text
V-JEPA2 PowerMean-7
```

Nombre de archivo sugerido:

```text
FINAL_VJEPA2_T8_FusedMean_PowerMean7_q05.pt
```
