# floodprob-stac

Pipeline Python STAC-first para detección de inundaciones con Sentinel-1 y Sentinel-2. Dado un área geográfica (AOI) y fechas de interés, descarga imágenes SAR y ópticas automáticamente, extrae features por píxel, entrena modelos de clasificación y genera mapas GeoTIFF de probabilidad de inundación.

La probabilidad final se descompone mediante la Regla de la Probabilidad Total:

```
P(inundación) = P(agua) × P(inundación | agua)
```

Incluye un **framework de comparación de múltiples modelos** (Threshold, LogReg, Random Forest, XGBoost, MLP, U-Net) con métricas estándar y figuras publicables.

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

- Python 3.11+
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

Instala **todas** las dependencias del framework de comparación (XGBoost + PyTorch):

```bash
pip install -e ".[compare]"
```

O instálalas por separado:

```bash
# Solo XGBoost
pip install -e ".[xgboost]"

# Solo PyTorch (modelo U-Net, requiere ~2 GB)
pip install -e ".[torch]"

# Notebooks Jupyter
pip install -e ".[notebook]"
```

> **Nota:** Si `pip install -e ".[...]"` falla en tu shell, prueba con comillas simples: `pip install -e '.[compare]'`

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
│   │   └── train_unet.py             # ★ U-Net PyTorch sobre patches
│   └── utils/
│       ├── geospatial.py             # Reproyección, clip, save_geotiff
│       └── io.py                     # Caché, retries, JSON helpers
└── pyproject.toml
```

> ★ = ficheros nuevos del framework de comparación

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

Ejecuta los pasos **en orden**:

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
| `unet` | flood | U-Net PyTorch sobre patches; requiere `pip install -e ".[torch]"` |

### Configurar en YAML

```yaml
model:
  compare:
    models_water:  ["threshold_change", "logreg", "random_forest", "xgboost", "mlp"]
    models_flood:  ["logreg", "random_forest", "xgboost", "mlp", "unet"]
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
```

### Ejecutar la comparación

```bash
# Requiere haber completado `preprocess` primero
floodprob compare --config configs/valencia_example.yaml

# Forzar re-entrenamiento aunque existan bundles cacheados
floodprob compare --config configs/valencia_example.yaml --force-retrain
```

### Salidas de la comparación

```
outputs/<run_name>/
├── metadata/
│   └── manifest_compare.json          # métricas por modelo y ventana
├── models/
│   └── compare_<model>_<task>.joblib  # bundles cacheados
├── predictions/
│   ├── p_agua_<model>_<key>.tif
│   ├── p_inundacion_<model>_<key>.tif
│   └── p_inundacion_final_<model>_<key>.tif
└── figures/                           # PNG (300 dpi) + PDF
    ├── spatial_grid_<key>.png/.pdf    # cuadrícula espacial todos los modelos
    ├── roc_water.png/.pdf             # curvas ROC — cabeza agua
    ├── roc_flood.png/.pdf             # curvas ROC — cabeza flood
    ├── pr_water.png/.pdf              # curvas Precision-Recall
    ├── pr_flood.png/.pdf
    ├── metrics_bar_water.png/.pdf     # barras F1, IoU, AUC
    ├── metrics_bar_flood.png/.pdf
    ├── calibration_water.png/.pdf     # reliability diagrams
    ├── calibration_flood.png/.pdf
    ├── feature_importance_water.png/.pdf  # RF + XGBoost
    ├── feature_importance_flood.png/.pdf
    └── error_map_<key>.png/.pdf       # solo si external_labels.flood está definido
```

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
| `model.tile_size` | Tamaño de tile para holdout espacial y patches U-Net (default `128`) |
| `model.calibration` | Método de calibración: `isotonic` o `sigmoid` |
| `model.ensemble_n` | Número de modelos en el ensemble (default `1`) |

- El pipeline usa caché por hash en `cache/`. Si cambias los parámetros de descarga, borra `cache/` para invalidarla.
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

**Notas metodológicas:**
- Las etiquetas de entrenamiento son **pseudo-etiquetas** generadas automáticamente desde cambio SAR + frecuencia histórica de agua; las etiquetas de evaluación son el ground truth externo EMSR773.
- Los F1/IoU bajos en flood reflejan la dificultad real del evento: la DANA afectó zonas históricamente secas, sin Sentinel-2 disponible y con geometría SAR no óptima para inundaciones en zonas agrícolas planas.
- La cabeza de agua muestra AUC≈1.0 porque se entrena y evalúa sobre pseudo-etiquetas derivadas del mismo SAR (no sobre EMSR directamente); es un indicador de consistencia interna, no de generalización.

---

## Licencia

MIT.
