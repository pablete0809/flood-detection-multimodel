# flood-detection-multimodel

Pipeline Python STAC-first para detección de inundaciones con Sentinel-1 y Sentinel-2. Dado un área geográfica (AOI) y fechas de interés, descarga imágenes SAR y ópticas automáticamente, extrae features por píxel, entrena modelos de clasificación y genera mapas GeoTIFF de probabilidad de inundación.

La probabilidad final se descompone mediante la Regla de la Probabilidad Total:

```
P(inundación) = P(agua) × P(inundación | agua)
```

Incluye un **framework de comparación de múltiples modelos** (Threshold, LogReg, Random Forest, XGBoost, MLP, U-Net, Attention U-Net, DeepLabV3+) con métricas estándar y figuras publicables.

---

## Contenido

- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Configuración STAC](#configuración-stac)
- [Ejecución end-to-end](#ejecución-end-to-end)
- [Comparación de modelos](#comparación-de-modelos-floodprob-compare)
- [Salidas](#salidas)
- [DEM](#dem)
- [Notas técnicas](#notas-técnicas)
- [Referencia de comandos CLI](#referencia-de-comandos-cli)

---

## Requisitos

- Python 3.10+
- GDAL / rasterio instalados correctamente en el sistema
- Acceso de red al endpoint STAC (CDSE o Planetary Computer)

---

## Instalación

### 1. Crear entorno virtual

```bash
python -m venv .venv
```

### 2. Activar el entorno

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Actualizar pip

```bash
pip install --upgrade pip
```

### 4. Instalar el paquete base

```bash
pip install -e .
```

### 5. Instalar dependencias opcionales (recomendado)

Instala **todas** las dependencias del framework de comparación (XGBoost + PyTorch + segmentation-models-pytorch):

```bash
pip install -e ".[compare]"
```

O instálalas por separado:

```bash
# Solo XGBoost
pip install -e ".[xgboost]"

# Solo PyTorch (U-Net, Attention U-Net, DeepLabV3+, requiere ~2 GB)
pip install -e ".[torch]"

# Notebooks Jupyter
pip install -e ".[notebook]"
```

> **Nota:** Si `pip install -e ".[...]"` falla en tu shell, prueba con comillas simples: `pip install -e '.[compare]'`

> **Nota:** Asegúrate de usar **siempre el Python del entorno virtual** (`source .venv/bin/activate`) antes de ejecutar cualquier comando `floodprob`. Si instalaste dependencias fuera del venv, los comandos fallarán.

---

## Estructura del proyecto

```text
.
├── configs/
│   ├── aoi_valencia.geojson          # AOI de ejemplo
│   ├── template_cdse.yaml            # Plantilla CDSE
│   ├── template_planetary_computer.yaml
│   └── valencia_example.yaml         # Config completa de ejemplo
├── data/
│   └── labels/
│       └── emsr773_flood_mask.geojson  # Ground truth EMSR773 (DANA Valencia oct-2024)
├── src/floodprob/
│   ├── cli.py                        # CLI: todos los comandos
│   ├── config.py                     # Configuración Pydantic + YAML
│   ├── preprocess.py                 # Orquesta preproceso
│   ├── train.py                      # Orquesta entrenamiento (un modelo)
│   ├── predict.py                    # Orquesta inferencia
│   ├── evaluate.py                   # Métricas de evaluación
│   ├── compare.py                    # ★ Orquesta comparación multi-modelo
│   ├── compare_metrics.py            # ★ Métricas píxel-aware (AUC, F1, IoU…)
│   ├── visualize_compare.py          # ★ Figuras publicables (PNG 300dpi + PDF)
│   ├── visualize.py                  # Vista rápida Folium
│   ├── features/
│   │   ├── s1.py                     # Features Sentinel-1 (VV, VH, delta…)
│   │   ├── s2.py                     # Features Sentinel-2 (NDWI, MNDWI)
│   │   └── dem.py                    # Features DEM (elevation, slope, dist_permwater)
│   ├── ingest/
│   │   ├── stac_search.py            # Búsqueda en catálogo STAC
│   │   └── stack.py                  # Construcción de stacks zarr
│   ├── labels/
│   │   └── pseudolabels.py           # Generación de pseudo-etiquetas
│   ├── models/
│   │   ├── train_water.py            # Modelo P(agua): Threshold + LogReg
│   │   ├── train_flood_given_water.py# Modelo P(inund|agua): LogReg + XGBoost
│   │   ├── train_rf.py               # ★ Random Forest (agua y flood)
│   │   ├── train_xgboost_water.py    # ★ XGBoost para la cabeza de agua
│   │   ├── train_mlp.py              # ★ MLP sklearn (agua y flood)
│   │   ├── train_unet.py             # ★ U-Net PyTorch sobre patches
│   │   ├── train_attention_unet.py   # ★ Attention U-Net PyTorch (Oktay et al. 2018)
│   │   └── train_deeplabv3plus.py    # ★ DeepLabV3+ via segmentation-models-pytorch
│   └── utils/
│       ├── geospatial.py             # Reproyección, clip, save_geotiff
│       └── io.py                     # Caché, retries, JSON helpers
└── pyproject.toml
```

> ★ = ficheros del framework de comparación

---

## Configuración STAC

Usa el fichero de config recomendado: [`configs/valencia_example.yaml`](configs/valencia_example.yaml)

### Opción A — CDSE (Copernicus Data Space Ecosystem)

```yaml
stac:
  api_url: "https://catalogue.dataspace.copernicus.eu/stac"
  collections:
    s1: "SENTINEL-1"
    s2: "SENTINEL-2"
  auth:
    type: bearer
    token_env: CDSE_TOKEN   # define esta variable de entorno
```

### Opción B — Microsoft Planetary Computer

```yaml
stac:
  api_url: "https://planetarycomputer.microsoft.com/api/stac/v1"
  collections:
    s1: "sentinel-1-grd"
    s2: "sentinel-2-l2a"
  auth:
    type: none   # PC firma los assets automáticamente si tienes planetary_computer instalado
```

Verifica el acceso al catálogo antes de descargar:

```bash
floodprob stac-check --config configs/valencia_example.yaml
```

---

## Ejecución end-to-end

Hay dos flujos según el objetivo:

### Flujo A — modelo único (train/predict/evaluate)

```bash
# 1. Verificar acceso al catálogo STAC
floodprob stac-check --config configs/valencia_example.yaml

# 2. Descargar imágenes (stacks zarr en cache/)
floodprob ingest --config configs/valencia_example.yaml

# 3. Calcular features y pseudo-etiquetas
floodprob preprocess --config configs/valencia_example.yaml

# 4. Entrenar el modelo principal (un solo modelo según config)
floodprob train --config configs/valencia_example.yaml

# 5. Generar mapas de probabilidad
floodprob predict --config configs/valencia_example.yaml

# 6. Evaluar métricas
floodprob evaluate --config configs/valencia_example.yaml

# 7. Vista interactiva Folium (opcional)
floodprob folium --config configs/valencia_example.yaml
```

### Flujo B — comparación multi-modelo ★ (recomendado)

Entrena y evalúa todos los modelos en una sola ejecución, generando figuras publicables. Solo requiere `preprocess` como paso previo.

```bash
# 1–3. Igual que el Flujo A (stac-check, ingest, preprocess)

# 4. Comparar todos los modelos y generar figuras
floodprob compare --config configs/valencia_example.yaml

# Forzar re-entrenamiento aunque existan bundles cacheados
floodprob compare --config configs/valencia_example.yaml --force-retrain
```

---

## Comparación de modelos (`floodprob compare`)

El comando `compare` entrena y evalúa múltiples modelos en una sola ejecución y genera figuras publicables.

### Modelos disponibles

| Modelo | Tarea | Notas |
|---|---|---|
| `threshold_change` | agua | Regla fija (baseline), sin entrenamiento |
| `logreg` | agua + flood | Regresión logística calibrada (sklearn) |
| `random_forest` | agua + flood | Random Forest con CalibratedClassifierCV |
| `xgboost` | agua + flood | XGBoost; requiere `pip install -e ".[xgboost]"` |
| `mlp` | agua + flood | MLP 3 capas ocultas (sklearn, CPU) |
| `unet` | flood | U-Net PyTorch 3-nivel con MC-Dropout; requiere `pip install -e ".[torch]"` |
| `attention_unet` | flood | **★ NUEVO** Attention U-Net (Oktay et al. 2018) — puertas de atención en skip connections; requiere `.[torch]` |
| `deeplabv3plus` | flood | **★ NUEVO** DeepLabV3+ encoder ResNet-34 con ASPP; requiere `pip install -e ".[compare]"` |

### Arquitecturas de red neural

#### U-Net (base)
Encoder–decoder 3 niveles con skip connections directas. MC-Dropout para estimación de incertidumbre.
Canales: `C_in → 16 → 32 → 64` (encoder), simétrico en decoder.

#### Attention U-Net ★
Igual que U-Net pero con **puertas de atención** (attention gates) en cada skip connection del decoder. La puerta aprende qué regiones del encoder son relevantes para la predicción de inundación, suprimiendo el ruido de fondo y amplificando las zonas inundadas. Implementación: Oktay et al. (2018) — formulación aditiva.

#### DeepLabV3+ ★
Encoder ResNet-34 con **ASPP** (Atrous Spatial Pyramid Pooling) para captura de contexto multi-escala sin reducir resolución, más un decoder ligero con skip connection desde el encoder. Entrenado desde cero (`encoder_weights=None`) dado que los datos SAR difieren de ImageNet. Usa cosine LR annealing. Implementado vía `segmentation-models-pytorch`.

### Configurar en YAML

```yaml
model:
  compare:
    models_water:  ["threshold_change", "logreg", "random_forest", "xgboost", "mlp"]
    models_flood:  ["logreg", "random_forest", "xgboost", "mlp", "unet", "attention_unet", "deeplabv3plus"]
    random_forest:
      n_estimators: 200
      max_depth: null
      min_samples_leaf: 5
    mlp:
      hidden_layer_sizes: [128, 64, 32]
      max_iter: 500
      alpha: 0.0001
    unet:
      epochs: 30
      lr: 0.001
      batch_size: 8
      dropout: 0.2
      mc_samples: 20
    attention_unet:       # mismos hiperparámetros que unet
      epochs: 30
      lr: 0.001
      batch_size: 8
      dropout: 0.2
      mc_samples: 20
    deeplabv3plus:
      encoder: "resnet34"
      epochs: 20
      lr: 0.0005
      batch_size: 4
```

### Salidas de la comparación

```
outputs/<run_name>/
├── metadata/
│   └── manifest_compare.json            # métricas por modelo y ventana
├── models/
│   ├── compare_<model>_<task>.joblib    # bundles cacheados (sklearn)
│   ├── unet_flood.pt                    # pesos PyTorch U-Net
│   ├── attention_unet_flood.pt          # pesos PyTorch Attention U-Net ★
│   └── deeplabv3plus_flood.pt           # pesos PyTorch DeepLabV3+ ★
├── predictions/
│   ├── p_agua_<model>_<key>.tif
│   ├── p_inundacion_<model>_<key>.tif
│   └── p_inundacion_final_<model>_<key>.tif
└── figures/                             # PNG (300 dpi) + PDF
    ├── sar_context_<key>.png            # ★ SAR antes/después/cambio + contorno GT
    ├── model_predictions_<key>.png      # ★ Grid ground truth + predicciones por modelo
    ├── ensemble_analysis_<key>.png      # ★ Acuerdo, incertidumbre, mejor/peor modelo
    ├── roc_water.png / roc_flood.png    # curvas ROC por tarea
    ├── pr_water.png / pr_flood.png      # curvas Precision-Recall
    ├── metrics_bar_water.png / metrics_bar_flood.png  # barras F1, IoU, AUC
    ├── calibration_water.png / calibration_flood.png  # reliability diagrams
    ├── feature_importance_water.png / feature_importance_flood.png
    └── error_map_<key>.png              # solo si external_labels.flood está definido
```

### Figuras nuevas ★

| Figura | Descripción |
|---|---|
| `sar_context_<key>.png` | SAR antes (baseline) / después (evento) / cambio (Δ) para VV y VH. Superpone contorno rojo con el ground truth de inundación. |
| `model_predictions_<key>.png` | Grid completo: columna ground truth + una columna por modelo. Fila superior P(agua), fila inferior P(inundación\|agua). Escala fija [0,1], contorno blanco al umbral 0.5, métricas AUC/F1/IoU en título. |
| `ensemble_analysis_<key>.png` | 4 paneles: acuerdo entre modelos, incertidumbre (σ pixel a pixel), predicción del mejor modelo (F1 máx), predicción del peor modelo (F1 mín). |

---

## Salidas del pipeline principal

En `outputs/<run_name>/predictions/`:

| Fichero | Descripción |
|---|---|
| `p_agua_<window>.tif` | P(agua) — probabilidad de presencia de agua |
| `p_inundacion_dado_agua_<window>.tif` | P(inundación \| agua) |
| `p_inundacion_<window>.tif` | P(inundación) = P(agua) × P(inund.\|agua) |

En `outputs/<run_name>/evaluation/`:
- `metrics.json` — AUC-ROC, F1, IoU, Brier, ECE
- `reliability_water.png` / `reliability_flood.png` — diagramas de calibración

---

## DEM

El pipeline no descarga DEM automáticamente. Coloca un GeoTIFF local y apúntalo en el YAML:

```yaml
dem:
  path: data/dem/valencia_dem.tif
```

Si no hay DEM, el pipeline continúa pero con features topográficas limitadas (sin `slope`, `elevation`, `dist_to_permwater`).

---

## Notas técnicas

| Parámetro | Descripción |
|---|---|
| `crs_salida` | CRS de salida (default `EPSG:25830`) |
| `resolution_m` | Resolución en metros. `10 m` = más detalle, mayor coste; `20 m` = más rápido |
| `seed` | Semilla para reproducibilidad de todos los modelos |
| `model.tile_size` | Tamaño de tile para holdout espacial y patches U-Net/Attention U-Net/DeepLabV3+ (default `128`) |
| `model.calibration` | Método de calibración: `isotonic` o `sigmoid` |
| `model.ensemble_n` | Número de modelos en el ensemble (default `1`) |

- El pipeline usa caché por hash en `cache/`. Si cambias los parámetros de descarga, borra `cache/` para invalidarla.
- Los modelos PyTorch se entrenan en GPU si está disponible (`cuda`), con fallback automático a CPU.
- `quantile(0.5)` se usa en lugar de `median(skipna=True)` para la reducción temporal del SAR, evitando un bug de `np.ma.sort` en numpy ≥ 2.0 con arrays dask.
- Si un asset de Sentinel no es COG, rasterio puede seguir leyendo via VSI según el servidor; si falla, revisa permisos del endpoint o cambia la colección/asset.

---

## Referencia de comandos CLI

```bash
floodprob stac-check  --config <yaml>                          # verifica catálogo STAC
floodprob ingest      --config <yaml>                          # descarga imágenes
floodprob preprocess  --config <yaml>                          # features + pseudo-etiquetas
floodprob train       --config <yaml>                          # entrena un modelo
floodprob predict     --config <yaml>                          # genera mapas de prob.
floodprob evaluate    --config <yaml>                          # calcula métricas
floodprob compare     --config <yaml> [--force-retrain]        # compara todos los modelos ★
floodprob folium      --config <yaml> [--key <window_key>]     # vista interactiva

# Todos los comandos aceptan --verbose / -v para logging detallado
floodprob --verbose compare --config configs/valencia_example.yaml
```

---

## Resultados experimentales — DANA Valencia (octubre 2024)

Evaluación sobre ground truth real del **Copernicus Emergency Management Service** (EMSR773, 190 polígonos de inundación validados, 32.8 km² afectados). No hay imágenes Sentinel-2 disponibles (100% nubosidad durante el evento), por lo que todos los modelos operan **únicamente con SAR** (delta VV/VH, VV/VH en dB, pendiente y distancia a agua permanente).

### Cabeza de Agua — detección de agua en la imagen de evento

| Modelo | AUC-ROC | F1 | IoU | Brier |
|---|---|---|---|---|
| Threshold (baseline) | 0.761 | 0.00 | 0.00 | 0.102 |
| Logistic Regression | 0.999 | 0.930 | 0.869 | 0.026 |
| Random Forest | **1.000** | **0.997** | **0.994** | 0.001 |
| XGBoost | 1.000 | 0.993 | 0.986 | 0.002 |
| MLP | 1.000 | 0.996 | 0.993 | 0.001 |

### Cabeza de Inundación — P(inundación | agua), evaluada vs. EMSR773

| Modelo | AUC-ROC | F1 | IoU | Brier |
|---|---|---|---|---|
| Logistic Regression | 0.618 | 0.039 | 0.020 | 0.023 |
| Random Forest | 0.740 | 0.127 | 0.068 | 0.018 |
| XGBoost | 0.717 | **0.156** | **0.085** | 0.011 |
| MLP | **0.750** | 0.134 | 0.072 | 0.012 |
| U-Net | 0.689 | 0.027 | 0.014 | 0.203 |
| Attention U-Net | — | — | — | — |
| DeepLabV3+ | — | — | — | — |

> Los resultados de Attention U-Net y DeepLabV3+ se actualizarán tras completar la evaluación.

**Notas metodológicas:**
- Las etiquetas de entrenamiento son **pseudo-etiquetas** generadas automáticamente desde cambio SAR + frecuencia histórica de agua; las etiquetas de evaluación son el ground truth externo EMSR773.
- Los F1/IoU bajos en flood reflejan la dificultad real del evento: la DANA afectó zonas históricamente secas, sin Sentinel-2 disponible y con geometría SAR no óptima para inundaciones en zonas agrícolas planas.
- La cabeza de agua muestra AUC≈1.0 porque se entrena y evalúa sobre pseudo-etiquetas derivadas del mismo SAR (no sobre EMSR directamente); es un indicador de consistencia interna, no de generalización.

---

## Licencia

MIT.
