# AI Co-Pilot para el Piso de Producción: Ver, Guiar, Mejorar

> Sistema de visión multi-cámara orientado a Industria 4.0 que convierte cámaras IP existentes en un **gemelo digital operativo** del piso de producción — asistiendo a supervisores de planta sin sustituirlos.

**Titular:** Alignity IQ Edge, LLC — Houston, Texas, EUA  
**Equipo #56 (MNA-V · Tec de Monterrey):** Landy Haydee Schlebach Osorio · Carlos Pano Hernández · Carlos Fernando Del Castillo Rey  
**Asesor académico:** Dr. Gerardo Camacho  
**Patrocinador industrial:** Dr. José Jacobo Eluani Vázquez (Representante Legal, Alignity IQ Edge, LLC)

---

## Resumen ejecutivo

En planta, la supervisión suele depender de recorridos físicos o de mirar monitores de forma pasiva. Eso dificulta cubrir varias zonas a la vez y hace costosa la reconstrucción de lo ocurrido al cierre de turno.

Este proyecto plantea un entorno **VisionOps**: ingestión de video multi-cámara (p. ej. RTSP/ONVIF), extracción de señales visuales con **modelos fundacionales** y orquestación mediante **NLP** para convertir hallazgos en alertas, bitácoras y reportes accionables — con foco en **tiempo casi real** (seguridad, asistencia) y **análisis post-turno** (auditoría, mejora continua, KPIs).

## Pregunta de investigación

> ¿Puede un sistema de analítica de video multi-cámara basado en IA generar una **bitácora visual automatizada** que identifique patrones operativos, cuellos de botella y eventos críticos, reduciendo la necesidad de supervisión humana continua frente a los monitores?

## Objetivos del proyecto

1. **Validar viabilidad** de un pipeline de analítica de video multi-cámara que produzca una bitácora visual automatizada (patrones, cuellos de botella, eventos críticos).
2. **Avanzar hacia un gemelo digital operativo** del piso de producción, combinando comprensión **espacial** (inventario de escena, fronteras de seguridad, activos) y **temporal** (acciones, secuencias, anomalías en cadena), integrando hallazgos con capas de **lenguaje** para alertas y reportes ejecutivos.
3. **Gestionar riesgos de despliegue industrial**: dominio (laboratorio vs. CCTV real), privacidad (p. ej. marcos regulatorios locales), deriva visual y coste computacional — con estrategias documentadas (edge vs. nube, cuantización, transfer learning).

## Arquitectura conceptual: modelos y roles

El marco técnico actual prioriza **DINOv3** y **V-JEPA 2.x** como motores de representación, complementados por una **capa NLP** (RAG, estructuración JSON, integración MES/ERP bajo enfoques tipo ISA-95 donde aplique). La idea es un pipeline **modular** que evite duplicar cómputo: escena a baja tasa donde baste con imagen; clips temporales donde importe la dinámica.

| Componente | Rol principal | Entrada típica | Ventaja clave |
|------------|---------------|----------------|----------------|
| **DINOv3** | Segmentación / comprensión espacial densa | Frame o muestreo de frames (alta resolución cuando haga falta) | Representaciones SSL; útil para piezas, herramientas, personas y zonas sin etiquetado masivo |
| **V-JEPA 2.x** | Dinámica temporal, acciones, anticipación | Clips de video (p. ej. decenas de frames) | Modelado predictivo en espacio latente (no reconstrucción pixel a pixel) |
| **Fusión + NLP** | Contexto (“qué” + “cuándo” + “dónde”) y orquestación | Embeddings / eventos estructurados | Alertas, consultas a bitácora, tickets y trazabilidad |

**Dataset de referencia (I+D):** el repositorio incluye material de referencia del **InHARD** (*Industrial Human Action Recognition Dataset*): acciones industriales, multimodalidad RGB + esqueleto 3D, útil para experimentar reconocimiento de acciones y generalización. La documentación interna advierte límites al extrapolar de laboratorio a planta real (ángulos de cámara, oclusiones, diversidad demográfica).

> **Nota sobre dependencias (`pyproject.toml`):** el entorno Python actual incluye también **Ultralytics (YOLOv8)**, **DeepSORT**, **Gemini** y utilidades de video — útiles para prototipos, baselines y alertas VLM. La línea de producto e investigación descrita arriba evoluciona hacia el stack **DINOv3 + V-JEPA + NLP** descrito en `Equipo56/Project/`.

## Los tres pilares (Ver · Guiar · Mejorar)

| Pilar | Qué aporta | Tecnologías / enfoque |
|-------|------------|------------------------|
| **Ver** | Ingesta multi-cámara y comprensión de escena y acción | DINOv3 (espacial) + V-JEPA (temporal); datos InHARD para experimentación |
| **Guiar** | Alertas y asistencia en ventanas de baja latencia | Edge / modelos compactos; reglas y comparadores vs. SOP; capa de lenguaje para mensajes accionables |
| **Mejorar** | Bitácora visual, búsqueda semántica de intervalos, KPIs post-turno | Indexación de eventos, RAG con filtros de tiempo/ubicación, informes e integración con sistemas de planta |

## Metodología

El ciclo de vida del componente de ML se alinea con **CRISP-ML(Q)** (Visengeriyeva et al., 2023), con marco de investigación acorde a Hernández-Sampieri & Mendoza (2023).

## Estructura del repositorio

```text
.
├── src/                      # Código fuente del co-piloto (en evolución; ver .gitkeep)
├── notebooks/                # Experimentación y análisis exploratorio
├── data_sample/              # Muestras y metadatos permitidos (sin video propietario)
│   ├── InHARD-master/        # Documentación y recursos de referencia del dataset InHARD
│   └── InHARD-master.zip     # Archivo comprimido de respaldo (no sustituye dataset completo)
├── models/                   # Referencias de modelos / código upstream para I+D
│   ├── dinov3-main/          # Código de referencia DINOv3 (Meta)
│   ├── vjepa2-main/         # Código de referencia V-JEPA 2 (Meta FAIR)
│   └── *.zip                 # Copias comprimidas opcionales
├── Equipo56/Project/         # Contexto y ensayos académicos del proyecto (Markdown)
├── pyproject.toml            # Metadatos y dependencias Python (uv)
├── uv.lock                   # Árbol de dependencias resuelto (generado por uv)
├── .python-version           # Python 3.11
├── .gitignore
└── README.md
```

Los **pesos** de modelos (`.pt`, `.pth`, `.onnx`, etc.) están **excluidos del control de versiones** por tamaño y licencia; clones bajo `models/dinov3-main/` y `models/vjepa2-main/` siguen las licencias de sus respectivos proyectos.

## Primeros pasos

El proyecto usa [uv](https://docs.astral.sh/uv/) para gestionar Python y dependencias.

```bash
# Instalar uv (si no lo tienes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clonar e instalar
git clone <url-del-repositorio>
cd AI-Co-Pilot-for-the-Production-Floor-See-Guide-Improve
uv sync --all-groups
```

Para notebooks: `uv sync --group notebooks`.

### Demo local (Live + identidad facial)

Desde la raíz del repo, con **uv**, **npm** y acceso a internet en el primer arranque:

```bash
./run-local.sh
```

- Arranca **vision-ops-backend** (puerto 8000) y **vision-ops-app** (3000).
- Si faltan los ONNX de YuNet/SFace, ejecuta `models/install_face_models.sh` (~40 MB, solo en tu máquina; ver `models/README.md`).
- UI: `http://localhost:3000/live` y `http://localhost:3000/identity`.

Los pesos `.onnx` y los datos de enrolamiento (`vision-ops-backend/data/faces/`) **no van en git**; cada desarrollador los obtiene con el script anterior.

## Documentación de contexto académico

- `Equipo56/Project/Project-context.md` — datos generales, dominio, tecnologías integradas e impacto MNA-V.  
- `Equipo56/Project/Visión por Computador Industrial: IA Co-Piloto.md` — marco VisionOps, InHARD, sinergia DINOv3 / V-JEPA, riesgos y capa NLP.

## Licencia

Privado — © Alignity IQ Edge, LLC. Todos los derechos reservados. El código de terceros bajo `models/` conserva sus licencias originales.
