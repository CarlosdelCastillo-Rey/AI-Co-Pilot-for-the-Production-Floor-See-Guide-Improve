# Avance 4 — Modelos Alternativos para Reconocimiento de Actividades en Video

Este repositorio contiene notebooks, checkpoints y una interfaz local para evaluar modelos alternativos de reconocimiento de actividades humanas en video.  
El objetivo principal fue comparar distintas representaciones visuales usando modelos preentrenados como DINOv2 y V-JEPA2, además de una arquitectura inspirada en MC-JEPA.

---

## Estructura del proyecto

```text
Avance 4. Modelos alternativos/
│
├── Checkpoints/
│   ├── .gitkeep
│   ├── DINOv2_puro_classifier.pt
│   ├── DINOv2_to_MCJEPA_classifier.pt
│   ├── VJEPA2_puro_classifier.pt
│   ├── VJEPA2_MCJEPA_frozen.pt
│   ├── VJEPA2_MCJEPA_partial_finetune.pt
│   └── dino_to_mcjepa_encoder.pt
│
├── Parte1_Extraccion_Baselines.ipynb
├── Parte2_Clasificacion_MCJEPA.ipynb
├── interfaz_mcjepa_v2.py
├── README.md
```

---

## Descripción general

Este avance forma parte de un proyecto de reconocimiento de actividades humanas en video.

La idea principal fue probar diferentes modelos de visión y video para extraer características visuales y después entrenar clasificadores sobre esas representaciones.

Se trabajó con tres enfoques principales:

1. **Modelos preentrenados como extractores de características**
   - DINOv2
   - V-JEPA2

2. **Encoder temporal multi-escala inspirado en MC-JEPA**
   - Representación corta
   - Representación media
   - Representación global

3. **Interfaz local de inferencia**
   - Carga de video
   - Predicción por modelo
   - Heatmap de movimiento
   - Detección de personas
   - Consenso entre clasificadores

---

## Archivos principales

### `Parte1_Extraccion_Baselines.ipynb`

Este notebook corresponde a la primera etapa del experimento.

En esta parte se extraen representaciones visuales usando modelos preentrenados y se entrenan clasificadores ligeros sobre esos embeddings.

Modelos evaluados:

---

### 1. DINOv2 puro

DINOv2 es usado como extractor de características por frame.

Cada frame del video se pasa por el modelo y se obtiene una representación visual.  
Después, las representaciones de los frames se promedian para obtener un embedding del video completo.

Flujo general:

```text
Video → Frames → DINOv2 → Embeddings por frame → Promedio temporal → Clasificador MLP
```

---

### 2. DINOv2 → MC-JEPA

En este enfoque se usan los embeddings generados por DINOv2, pero en lugar de promediarlos directamente, se pasan por un encoder temporal multi-escala inspirado en MC-JEPA.

Este encoder toma la secuencia temporal de embeddings y genera tres representaciones:

- contexto corto;
- contexto medio;
- contexto global.

Después, estas representaciones se combinan para clasificar la actividad.

Flujo general:

```text
Video → Frames → DINOv2 → Secuencia de embeddings → Encoder multi-escala → Clasificador MLP
```

---

### 3. V-JEPA2 puro

V-JEPA2 es un modelo preentrenado directamente sobre video.

A diferencia de DINOv2, que trabaja principalmente sobre imágenes individuales, V-JEPA2 genera tokens latentes con información espacio-temporal.

En este enfoque, los tokens de V-JEPA2 se promedian y se usan como representación del video.

Flujo general:

```text
Video → V-JEPA2 → Tokens latentes → Promedio de tokens → Clasificador MLP
```

---

## Estrategias de suavizado

En la Parte 1 también se probaron estrategias de suavizado temporal sobre las probabilidades de salida de los modelos.

Las estrategias evaluadas fueron:

- **No smoothing**
- **SMA probs**
- **EMA probs**
- **Adaptive EMA**

---

### No smoothing

Se usa directamente la predicción del modelo, sin modificar las probabilidades.

```text
Predicción final = Probabilidades actuales del modelo
```

---

### SMA — Simple Moving Average

Promedia las probabilidades recientes usando una ventana temporal.

```text
Predicción suavizada = promedio de las últimas N predicciones
```

Este método puede ayudar cuando las predicciones fluctúan mucho, pero puede introducir retraso.

---

### EMA — Exponential Moving Average

Da más peso a las predicciones recientes y menos peso a las antiguas.

```text
EMA_t = α * p_t + (1 - α) * EMA_{t-1}
```

donde:

- `p_t` es la probabilidad actual;
- `EMA_t` es la probabilidad suavizada actual;
- `α` controla qué tanto se confía en la predicción más reciente.

---

### Adaptive EMA

Es una variante del EMA donde el factor de suavizado puede cambiar de acuerdo con la confianza o estabilidad de la predicción.

La idea es que el suavizado sea más flexible:

- si el modelo está seguro, se puede actualizar más rápido;
- si el modelo está inestable, se puede suavizar más.

---

## `Parte2_Clasificacion_MCJEPA.ipynb`

Este notebook corresponde a la segunda etapa del experimento.

Aquí se implementa una cabeza MC-JEPA sobre tokens de V-JEPA2.

La arquitectura intenta combinar dos objetivos:

1. **Clasificación supervisada**

   El modelo predice la clase de actividad del video.

2. **Objetivo predictivo tipo JEPA**

   El modelo intenta predecir una representación global del video usando solamente información parcial o contextual.

---

## Arquitectura MC-JEPA implementada

La cabeza MC-JEPA trabaja con tokens latentes de V-JEPA2.

A partir de esos tokens, se generan tres representaciones:

- `z_short`: representación de contexto corto;
- `z_mid`: representación de contexto medio;
- `z_global`: representación global del video.

Después, el modelo fusiona el contexto corto y medio para formar un embedding de contexto:

```text
z_context = Fusion(z_short, z_mid)
```

Ese embedding se usa para dos cosas:

1. Clasificar la actividad.
2. Predecir la representación global.

La pérdida total combina clasificación y predicción JEPA:

```text
Loss total = CrossEntropy(logits, labels) + λ * MSE(z_pred_global, z_global)
```

donde:

- `CrossEntropy` mide el error de clasificación;
- `MSE` mide qué tan bien el modelo predice la representación global;
- `λ` controla el peso de la pérdida JEPA.

---

## Variantes evaluadas en Parte 2

Se evaluaron dos variantes principales:

### 1. VJEPA2_MCJEPA_frozen

En esta variante, V-JEPA2 permanece congelado.  
Solo se entrena la cabeza MC-JEPA.

Flujo general:

```text
Video → V-JEPA2 congelado → Tokens → MC-JEPA Head → Clasificación
```

Ventajas:

- requiere menos recursos;
- reduce riesgo de sobreajuste;
- entrena más rápido.

Desventaja:

- el backbone no se adapta al dataset específico.

---

### 2. VJEPA2_MCJEPA_partial_finetune

En esta variante, se descongelan parcialmente algunas capas de V-JEPA2 para permitir ajuste fino limitado.

Flujo general:

```text
Video → V-JEPA2 parcialmente ajustado → Tokens → MC-JEPA Head → Clasificación
```

Ventajas:

- permite adaptar parte del modelo al dataset;
- puede mejorar si hay suficientes datos.

Desventajas:

- requiere más memoria;
- puede sobreajustarse si hay pocos datos;
- necesita mejor control de hiperparámetros.

---

## `interfaz_mcjepa_v2.py`

Este script implementa una interfaz local para probar los modelos entrenados sobre video.

La interfaz permite cargar un video y comparar visualmente hasta cinco clasificadores:

- DINOv2 puro
- DINOv2 → MC-JEPA
- V-JEPA2 puro
- V-JEPA2 + MC-JEPA frozen
- V-JEPA2 + MC-JEPA partial finetune

---

## Funcionalidades de la interfaz

La interfaz incluye:

- carga de video local;
- inferencia periódica sobre clips de video;
- predicción por cada modelo;
- nivel de confianza por predicción;
- barra visual de confianza;
- top de clases probables;
- consenso entre modelos;
- heatmap de movimiento;
- detección de personas usando YOLOv8;
- overlay visual sobre el video.

Flujo general de inferencia:

```text
Video → Buffer de frames → Extracción de features → Clasificadores → Predicciones → Overlay visual
```

---

## Checkpoints

Los checkpoints se generan desde los notebooks de entrenamiento.

La carpeta `Checkpoints/` contiene los pesos entrenados necesarios para ejecutar la interfaz.

Estructura actual:

```text
Checkpoints/
│
├── .gitkeep
├── DINOv2_puro_classifier.pt
├── DINOv2_to_MCJEPA_classifier.pt
├── VJEPA2_puro_classifier.pt
├── VJEPA2_MCJEPA_frozen.pt
├── VJEPA2_MCJEPA_partial_finetune.pt
└── dino_to_mcjepa_encoder.pt
```

Descripción de archivos:

| Archivo | Descripción |
|---|---|
| `DINOv2_puro_classifier.pt` | Clasificador entrenado sobre embeddings promedio de DINOv2 |
| `DINOv2_to_MCJEPA_classifier.pt` | Clasificador entrenado sobre embeddings DINOv2 procesados por encoder multi-escala |
| `VJEPA2_puro_classifier.pt` | Clasificador entrenado sobre tokens promedio de V-JEPA2 |
| `VJEPA2_MCJEPA_frozen.pt` | Cabeza MC-JEPA entrenada con V-JEPA2 congelado |
| `VJEPA2_MCJEPA_partial_finetune.pt` | Cabeza MC-JEPA entrenada con ajuste parcial de V-JEPA2 |
| `dino_to_mcjepa_encoder.pt` | Encoder temporal multi-escala usado para DINOv2 → MC-JEPA |

---

## Resultados principales

Los resultados actuales son preliminares.

Aunque el dataset completo contiene más videos, los embeddings usados en esta corrida fueron una muestra reducida.

| Split | Muestras usadas |
|---|---:|
| Train | 120 |
| Val | 40 |
| Test | 40 |

Por esta razón, las métricas deben interpretarse como una primera validación experimental, no como resultado final del modelo.

---

## Comparación de modelos baseline

| Modelo | Suavizado | Accuracy | F1 weighted | F1 macro |
|---|---|---:|---:|---:|
| VJEPA2 puro | No smoothing | **0.475** | **0.3923** | **0.1664** |
| DINOv2 → MC-JEPA | No smoothing | 0.300 | 0.2699 | 0.0821 |
| VJEPA2 puro | EMA probs | 0.375 | 0.2582 | 0.0738 |
| VJEPA2 puro | SMA probs | 0.300 | 0.2115 | 0.0604 |
| DINOv2 puro | No smoothing | 0.300 | 0.2107 | 0.0765 |
| VJEPA2 puro | Adaptive EMA | 0.250 | 0.1643 | 0.0469 |
| DINOv2 puro | SMA probs | 0.250 | 0.1000 | 0.0286 |
| DINOv2 puro | EMA probs | 0.250 | 0.1000 | 0.0286 |
| DINOv2 puro | Adaptive EMA | 0.250 | 0.1000 | 0.0286 |

---

## Mejor resultado obtenido

El mejor resultado fue obtenido por:

```text
Modelo:       VJEPA2 puro
Suavizado:    No smoothing
Accuracy:     0.475
F1 weighted:  0.3923
F1 macro:     0.1664
```

Esto sugiere que V-JEPA2 generó mejores representaciones para este problema que DINOv2 en esta configuración inicial.

---

## Interpretación de los resultados baseline

El modelo **VJEPA2 puro sin suavizado** obtuvo el mejor desempeño general.

Esto tiene sentido porque V-JEPA2 fue preentrenado sobre video, por lo que puede capturar mejor información temporal y espacial que un modelo basado en imágenes individuales como DINOv2.

Sin embargo, el F1 macro fue bajo.  
Esto indica que el modelo no está clasificando bien todas las clases por igual.

Posibles causas:

- pocas muestras usadas en la corrida;
- desbalance entre clases;
- acciones visualmente similares;
- clips con poca información clara de la acción;
- algunas clases dominan más que otras;
- el modelo puede estar aprendiendo mejor las clases más frecuentes.

---

## Resultados con suavizado

Los métodos de suavizado no mejoraron el mejor resultado.

Por ejemplo:

| Modelo | Suavizado | Accuracy |
|---|---|---:|
| VJEPA2 puro | No smoothing | **0.475** |
| VJEPA2 puro | EMA probs | 0.375 |
| VJEPA2 puro | SMA probs | 0.300 |
| VJEPA2 puro | Adaptive EMA | 0.250 |

Esto indica que, para esta evaluación, usar suavizado redujo el desempeño.

Una posible explicación es que las muestras evaluadas corresponden a clips independientes.  
El suavizado es más útil cuando hay una secuencia continua de video, pero puede afectar negativamente si se aplica entre clips que no pertenecen a la misma acción continua.

---

## Resultados de MC-JEPA

| Modelo | Accuracy | F1 weighted | F1 macro | Loss cls | Loss JEPA |
|---|---:|---:|---:|---:|---:|
| VJEPA2_MCJEPA_frozen | 0.250 | 0.1000 | 0.0286 | 2.2241 | 0.0811 |
| VJEPA2_MCJEPA_partial_finetune | 0.250 | 0.1000 | 0.0286 | 2.3991 | 0.1321 |

---

## Interpretación de MC-JEPA

Las variantes MC-JEPA todavía no superaron al baseline de V-JEPA2 puro.

Esto no significa que la arquitectura no funcione.  
Más bien indica que la configuración actual todavía necesita ajuste.

Posibles causas:

- pocas muestras usadas para entrenamiento;
- número limitado de épocas;
- hiperparámetros no optimizados;
- balance inadecuado entre pérdida de clasificación y pérdida JEPA;
- posible necesidad de un mejor manejo temporal de tokens;
- fine-tuning parcial insuficiente;
- learning rate posiblemente alto o bajo para el ajuste fino;
- riesgo de sobreajuste por el tamaño reducido de la muestra.

---

## Resultados de clustering

También se evaluó la separación de las representaciones mediante métricas de clustering.

| Representación | ARI | NMI | Silhouette |
|---|---:|---:|---:|
| DINOv2 → MC-JEPA | 0.0251 | 0.5088 | 0.0695 |
| DINOv2 puro | 0.0067 | 0.5083 | 0.0966 |
| VJEPA2 puro | -0.0095 | 0.4892 | 0.1047 |

---

## Interpretación de clustering

Las métricas de clustering muestran que las representaciones tienen cierta estructura, pero las clases todavía no están claramente separadas en el espacio de embeddings.

El valor de NMI muestra que existe algo de relación entre los clusters y las clases reales.  
Sin embargo, ARI y Silhouette son bajos, lo que indica que las clases se mezclan bastante.

En términos simples:

```text
Los embeddings capturan algunos patrones,
pero todavía no separan claramente todas las acciones.
```

---

## Comparación general

| Enfoque | Resultado general |
|---|---|
| DINOv2 puro | Baseline simple basado en imágenes; desempeño bajo-medio |
| DINOv2 → MC-JEPA | Mejora frente a DINOv2 puro en F1 weighted |
| VJEPA2 puro | Mejor modelo preliminar |
| VJEPA2 + MC-JEPA frozen | No superó el baseline |
| VJEPA2 + MC-JEPA partial finetune | No superó el baseline |

---

## Conclusiones del avance

En este avance se logró construir un pipeline completo para comparar representaciones de video usando modelos preentrenados.

El resultado más sólido fue el de **VJEPA2 puro**, lo que sugiere que un modelo preentrenado directamente sobre video captura mejor la información temporal que un modelo de imagen aplicado frame por frame como DINOv2.

La arquitectura MC-JEPA ya fue implementada y probada, pero todavía requiere mejoras para superar los baselines.

El trabajo deja preparado un pipeline completo:

```text
video → extracción de features → clasificación → evaluación → interfaz de inferencia
```

---

## Puntos a mejorar

Para las siguientes iteraciones se recomienda:

### 1. Usar más datos

Los resultados actuales usan una muestra pequeña.  
Es necesario correr los experimentos con más videos del dataset completo.

---

### 2. Revisar el balance de clases

El F1 macro bajo sugiere que algunas clases tienen muy bajo desempeño.

Se debe analizar:

- matriz de confusión;
- precisión por clase;
- recall por clase;
- F1 por clase;
- cantidad de muestras por clase.

---

### 3. Ajustar hiperparámetros

Especialmente:

- learning rate;
- batch size;
- número de épocas;
- dropout;
- `lambda_jepa`;
- tamaño de embedding;
- número de capas del encoder;
- número de heads de atención;
- capas descongeladas en partial finetuning.

---

### 4. Mejorar el uso del suavizado

El suavizado no ayudó en clips independientes.  
Puede tener más sentido aplicarlo en video continuo o streaming en tiempo real.

Para próximas pruebas se puede evaluar:

- suavizado solo dentro del mismo video;
- reiniciar el estado del suavizado entre clips;
- comparar smoothing por ventana temporal real;
- usar filtros adaptativos solo cuando la confianza sea baja.

---

### 5. Medir tiempo de inferencia

V-JEPA2 es más pesado que DINOv2.

Por eso se debe reportar:

- segundos por clip;
- FPS de inferencia;
- uso de GPU;
- uso de memoria;
- diferencia entre CPU y GPU.

---

### 6. Evaluar por clase

Además de accuracy, es importante analizar qué acciones se detectan correctamente y cuáles se confunden.

Esto permitirá identificar si el modelo falla en clases específicas o si el error es general.

---

### 7. Optimizar la interfaz

La interfaz ya funciona como demo, pero puede mejorarse con:

- selección dinámica de checkpoints;
- exportación de resultados;
- registro de predicciones por frame;
- gráfico temporal de confianza;
- selección de modelo activo;
- comparación visual entre modelos;
- modo webcam;
- modo video en tiempo real.

---

## Requisitos generales

Para ejecutar los notebooks se recomienda usar Google Colab con GPU.

Para ejecutar la interfaz local se recomienda:

- Python 3.10
- GPU NVIDIA si se usa V-JEPA2
- Checkpoints generados previamente desde los notebooks
- Token de Hugging Face si el modelo V-JEPA2 requiere acceso

---

## Instalación de dependencias

Dependencias principales:

```bash
pip install torch torchvision transformers opencv-python pillow ultralytics huggingface_hub
```

También puede requerirse:

```bash
pip install scikit-learn pandas numpy matplotlib tqdm
```

---

## Ejecución de notebooks

Primero ejecutar:

```text
Parte1_Extraccion_Baselines.ipynb
```

Este notebook genera:

- embeddings;
- clasificadores baseline;
- métricas de evaluación;
- checkpoints de clasificadores;
- encoder DINOv2 → MC-JEPA.

Después ejecutar:

```text
Parte2_Clasificacion_MCJEPA.ipynb
```

Este notebook genera:

- modelos MC-JEPA sobre V-JEPA2;
- métricas de clasificación;
- checkpoints de las variantes frozen y partial finetune.

---

## Ejecución de la interfaz

Para ejecutar la interfaz local:

```bash
python interfaz_mcjepa_v2.py
```

Antes de correrla, revisar dentro del archivo:

```python
CHECKPOINT_DIR = Path("ruta/a/Checkpoints")
```

Esta ruta debe apuntar a la carpeta donde se encuentran los checkpoints generados por los notebooks.

---

## Nota sobre la ruta de checkpoints

La estructura actual guarda los checkpoints directamente dentro de la carpeta `Checkpoints/`.

Por lo tanto, la interfaz debe buscar archivos como:

```text
Checkpoints/DINOv2_puro_classifier.pt
Checkpoints/DINOv2_to_MCJEPA_classifier.pt
Checkpoints/VJEPA2_puro_classifier.pt
Checkpoints/VJEPA2_MCJEPA_frozen.pt
Checkpoints/VJEPA2_MCJEPA_partial_finetune.pt
Checkpoints/dino_to_mcjepa_encoder.pt
```

---

## Notas importantes

- V-JEPA2 puede requerir Python 3.10.
- V-JEPA2 puede requerir acceso desde Hugging Face.
- En CPU la inferencia puede ser lenta.
- Para mejor rendimiento se recomienda GPU.
- Los checkpoints no deben incluir tokens personales.
- No subir tokens de Hugging Face al repositorio.

---

## Resumen para presentación

En este avance se construyó un pipeline experimental para reconocimiento de actividades en video usando modelos preentrenados.

Se compararon DINOv2, DINOv2 con un encoder temporal tipo MC-JEPA y V-JEPA2.

El mejor resultado preliminar fue **VJEPA2 puro sin suavizado**, con:

```text
Accuracy:     47.5%
F1 weighted:  0.3923
F1 macro:     0.1664
```

La arquitectura MC-JEPA fue implementada y evaluada, pero todavía no superó al baseline de V-JEPA2 puro.

El siguiente paso es correr los experimentos con más datos, revisar desempeño por clase, ajustar hiperparámetros y mejorar la evaluación en video continuo.

---

## Conclusión final

Este avance implementa una primera comparación entre modelos de representación visual para reconocimiento de actividades en video.

El mejor modelo preliminar fue **VJEPA2 puro**, con mejor desempeño que DINOv2 y DINOv2 con encoder multi-escala.

La arquitectura MC-JEPA fue integrada y evaluada, pero requiere optimización para mejorar su desempeño.

El proyecto queda preparado para continuar con:

- evaluación sobre más datos;
- ajuste de hiperparámetros;
- análisis por clase;
- pruebas en video continuo;
- optimización de inferencia;
- mejora de la interfaz local.
