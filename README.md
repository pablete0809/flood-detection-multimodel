# Cartografía probabilística de inundaciones con Sentinel-1: supervisión débil frente a supervisión completa

Trabajo de Fin de Grado. Caso de estudio principal: **DANA de Valencia (octubre de 2024)**.

Este repositorio contiene un sistema reproducible en Python que genera mapas de probabilidad de inundación a partir de imágenes radar de **Sentinel-1**, imágenes ópticas de **Sentinel-2** y un **modelo digital de elevaciones**. Dado un área (AOI) y unas fechas, descarga las imágenes desde catálogos STAC, calcula variables por píxel, entrena y compara ocho modelos y produce mapas GeoTIFF.

La pregunta que guía el trabajo es:

> **¿Cuánto rendimiento se pierde al entrenar con etiquetas generadas automáticamente a partir del radar en lugar de con la cartografía oficial de la inundación, y dónde se produce esa pérdida?**

---

## Contenido

- [Objetivos](#objetivos)
- [Diseño experimental](#diseño-experimental)
- [Resultados preliminares](#resultados-preliminares-dana-de-valencia)
- [Estado del trabajo y hoja de ruta](#estado-del-trabajo-y-hoja-de-ruta)
- [Entregables](#entregables)
- [Instalación](#instalación)
- [Reproducir el experimento](#reproducir-el-experimento)
- [Configuración STAC](#configuración-stac)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Comparación de modelos: detalles](#comparación-de-modelos-detalles)
- [Notas técnicas](#notas-técnicas)
- [Referencia de comandos CLI](#referencia-de-comandos-cli)

---

## Objetivos

**Objetivo primario.** Desarrollar un sistema reproducible capaz de generar mapas de probabilidad de inundación combinando Sentinel-1, Sentinel-2 y un modelo digital de elevaciones, y medir cuánto rendimiento se pierde cuando, en lugar de entrenar con la cartografía oficial de la inundación, se usan etiquetas generadas de forma automática.

**Objetivos secundarios.**

1. Comprobar si el sistema se puede aplicar a otros eventos y regiones, empezando por la DANA de Málaga de 2025.
2. Determinar si unas pocas zonas etiquetadas a mano bastan para acercar los resultados de la supervisión débil a los de la completa.
3. Identificar en qué tipos de terreno fallan las etiquetas automáticas, como arrozales o zonas urbanas.
4. Valorar si las probabilidades y la incertidumbre que ofrece el sistema son fiables para apoyar decisiones en una emergencia.
5. Explorar mejoras como el uso de imágenes ópticas sin nubes o de modelos preentrenados.
6. Publicar el sistema como software libre reutilizable por otros investigadores.

**Hipótesis de trabajo** (a partir de los resultados preliminares): con pseudo-etiquetas, los modelos aprenden a **ordenar** bien los píxeles según su probabilidad de inundación (AUC alto), pero fallan al **elegir el umbral de decisión**. Una pequeña muestra etiquetada para fijar ese umbral recuperaría la mayor parte de la diferencia con la supervisión completa.

---

## Diseño experimental

### Casos de estudio

| Caso | Evento | Referencia | Uso en el TFG |
|---|---|---|---|
| Valencia | DANA, 29-oct-2024. Imagen S1 del evento: 2024-11-01, órbita relativa 110 (descendente) | Copernicus EMS **EMSR773** (1117 geometrías) | Evaluación cuantitativa |
| Málaga | DANA del Guadalhorce, dic-2025 | Sin referencia por ahora ([`malaga_flood_mask.geojson`](data/labels/) vacío) | Aplicación a otro evento: cualitativa salvo que se consiga una máscara EMS |

### Datos de entrada

| Fuente | Variables | Notas |
|---|---|---|
| Sentinel-1 RTC (VV, VH) | Retrodispersión del evento, del periodo de referencia y su diferencia (Δ) | El periodo de referencia se restringe a la **misma órbita relativa** que el evento |
| Sentinel-2 L2A | NDWI, MNDWI | En Valencia no hay imágenes útiles durante el evento por las nubes, así que el experimento usa **SAR + DEM** |
| DEM | Elevación, pendiente, distancia al agua permanente | GeoTIFF local |
| ESA WorldCover 2021 (clase 80) | Agua permanente | Referencia para las pseudo-etiquetas, no para entrenar |
| JRC Global Surface Water | Frecuencia histórica de agua | **Pendiente** (sustituirá a la frecuencia calculada con una sola fecha; ver A-01) |

### Dos regímenes de entrenamiento

| Régimen | `model.train_on` | Etiquetas de entrenamiento |
|---|---|---|
| **Supervisión débil** | `pseudo` (por defecto) | Pseudo-etiquetas automáticas: cambio de retrodispersión SAR + referencias de agua permanente. EMSR773 **no** se usa para entrenar. |
| **Supervisión completa** | `gt` | EMSR773, solo en los bloques de entrenamiento |

Ambos regímenes se evalúan **contra EMSR773** sobre los mismos píxeles de test.

### Modelos

Ocho modelos de complejidad creciente:

| Modelo | Tipo | Tarea |
|---|---|---|
| `threshold_change` | Regla fija sobre el cambio SAR (línea base, sin entrenamiento) | agua |
| `logreg` | Regresión logística calibrada | agua + inundación |
| `random_forest` | Random Forest calibrado | agua + inundación |
| `xgboost` | Gradient boosting | agua + inundación |
| `mlp` | Perceptrón multicapa (3 capas ocultas) | agua + inundación |
| `unet` | U-Net de 3 niveles con MC-Dropout | inundación |
| `attention_unet` | U-Net con puertas de atención (Oktay et al., 2018) | inundación |
| `deeplabv3plus` | DeepLabV3+ con encoder ResNet-34 y ASPP | inundación |

### Protocolo de evaluación

- **Partición espacial por bloques** ([`splits.py`](src/floodprob/splits.py)): la imagen se divide en tiles de `model.tile_size` píxeles que se reparten al azar (semilla `seed`) en `model.n_folds` folds. Uno es test (`test_fold`), otro validación (`val_fold`) y el resto entrenamiento. Así se evita evaluar en zonas contiguas a las de entrenamiento, que estarían autocorrelacionadas (Roberts et al., 2017). Las etiquetas fuera del fold de entrenamiento se ocultan antes de entrenar.
- **Umbral de decisión:** se elige maximizando F1 en el fold de validación, con etiquetas del mismo tipo que las de entrenamiento. En modo `pseudo` nunca se consulta EMSR773 para elegirlo.
- **Máscara común:** todos los modelos se evalúan sobre **los mismos píxeles de test** (fold de test ∩ referencia válida ∩ variables finitas).
- **Métricas:** AUC-ROC, PR-AUC (métrica principal con clases desbalanceadas), F1 e IoU con el umbral de validación, F1@0,5, Brier y ECE (calibración).
- **Repetibilidad:** las redes neuronales se entrenan con varias semillas y se reportan como media ± desviación típica.

Los fallos metodológicos corregidos para llegar a este protocolo (etiquetas de evaluación usadas para entrenar, holdout degenerado, fuga entre entrenamiento y test, etc.) están documentados en [`docs/registro_metodologico.md`](docs/registro_metodologico.md).

---

## Resultados preliminares: DANA de Valencia

> **Versión v1** (2026-10-28): un solo fold de test y una sola semilla de partición. Las cifras cambiarán al completar los puntos A-01 y A-03 de la [hoja de ruta](#estado-del-trabajo-y-hoja-de-ruta). Los resultados publicados antes en este README (AUC ≈ 1,0 en agua; F1 ≈ 0 en inundación) eran artefactos de fallos de etiquetado y evaluación.

AOI de 685 × 666 píxeles a 20 m. EMSR773 marca 82 801 píxeles inundados (18,1 % del AOI). Test: 112 571 píxeles (18,0 % inundados), los mismos para todos los modelos. Configuración: [`configs/valencia_quicktest.yaml`](configs/valencia_quicktest.yaml).

### Detección de inundación frente a EMSR773

| Modelo | AUC (débil) | PR-AUC (débil) | F1 (débil) | AUC (completa) | PR-AUC (completa) | F1 (completa) |
|---|---|---|---|---|---|---|
| Regresión logística | 0,921 | 0,806 | **0,736** | 0,950 | 0,908 | 0,861 |
| Random Forest | 0,836 | 0,510 | 0,327 | 0,973 | 0,945 | 0,886 |
| XGBoost | 0,755 | 0,371 | 0,322 | 0,973 | 0,945 | 0,891 |
| MLP | 0,879 | 0,527 | 0,326 | 0,973 | 0,945 | 0,885 |
| U-Net | 0,933 | 0,743 | 0,247 | **0,987** | **0,967** | **0,922** |
| Attention U-Net | 0,913 | 0,647 | 0,193 | 0,973 | 0,955 | 0,921 |
| DeepLabV3+ ¹ | 0,946 | 0,831 | 0,763 | 0,973 | 0,958 | 0,921 |

*Débil* = entrenado solo con pseudo-etiquetas. *Completa* = entrenado con EMSR773 en los bloques de entrenamiento.
¹ No determinista e inestable entre semillas (F1 con supervisión completa entre 0,35 y 0,92 en 5 semillas).

Referencia: la propia regla de pseudo-etiquetado, en los píxeles que etiqueta, obtiene F1 = 0,25 frente a EMSR773 en test.

### Primeras lecturas

1. **Techo de supervisión completa:** las U-Net alcanzan F1 = 0,923 ± 0,002 en 5 semillas.
2. **Supervisión débil:** las U-Net mantienen un AUC de 0,92 ± 0,04, pero su F1 cae a 0,50 ± 0,28. Los modelos ordenan bien los píxeles; lo que falla es el umbral.
3. **El umbral es el cuello de botella:** si el umbral de la U-Net con pseudo-etiquetas se fija con unos pocos tiles de validación etiquetados con EMSR773, su F1 pasa de 0,24–0,86 a 0,63–0,85 según la semilla (registro, A-08).
4. **Las pseudo-etiquetas fallan sobre todo en los arrozales de la Albufera:** el 64,5 % de los píxeles inundados según EMSR773 se marcan como agua permanente porque la frecuencia histórica de agua sale de una sola fecha de noviembre (registro, A-01).
5. **Señal SAR:** el cambio de retrodispersión por sí solo alcanza AUC 0,91 frente a EMSR773. Comparar solo pasadas de la misma órbita reduce la dispersión del cambio en zonas estables de 3,49 a 2,74 dB.

Para regenerar las tablas desde los manifiestos:

```bash
python scripts/resumen_metricas.py outputs/valencia_quicktest_rtc/compare_{pseudo,gt}/metadata/manifest_compare.json
```

---

## Estado del trabajo y hoja de ruta

Los identificadores remiten al [registro metodológico](docs/registro_metodologico.md). El calendario completo está en el [cronograma](docs/anteproyecto/cronograma.png).

| ID | Tarea | Objetivo relacionado | Estado |
|---|---|---|---|
| F-01 … F-11 | Corrección del protocolo de entrenamiento y evaluación | Primario | Hecho |
| A-01 | Sustituir la frecuencia histórica de agua de una sola fecha por JRC Global Surface Water y decidir cómo tratar el arrozal inundado estacionalmente | Primario, secundario 3 | Pendiente |
| A-03 | Validación cruzada rotando los 5 folds de test, con varias semillas | Primario | Pendiente |
| A-08 | Experimento de umbral con pocas zonas etiquetadas (curva F1 frente a número de tiles etiquetados) | Secundario 2 | Pendiente |
| — | Análisis de errores por cobertura del suelo (WorldCover) | Secundario 3 | Pendiente |
| A-02 | Calibración de las CNN (temperature scaling) y análisis de la incertidumbre MC-Dropout | Secundario 4 | Pendiente |
| A-05 | Málaga: conseguir una máscara de referencia o presentar una aplicación cualitativa | Secundario 1 | Pendiente |
| A-09 | DeepLabV3+ estable: encoder preentrenado o más pequeño, aumento de datos | Secundario 5 | Pendiente |
| A-04 | Decidir si se mantiene la descomposición P(agua) × P(inundación \| agua) o se presenta como ablación | Primario | Pendiente |
| A-06, A-07 | Comprobar con qué imagen se delimitó EMSR773; franja de separación entre bloques | Primario | Pendiente |

---

## Entregables

| Entregable | Ubicación |
|---|---|
| Memoria del TFG | Pendiente |
| Código fuente (paquete `floodprob`) con instrucciones de reproducción | Este repositorio |
| Mapas de probabilidad e incertidumbre (GeoTIFF) y visor web interactivo | `outputs/<run_name>/` (no versionado) |
| Tablas de métricas, figuras y registro metodológico | [`docs/resultados/`](docs/resultados/), [`docs/registro_metodologico.md`](docs/registro_metodologico.md) |
| Artículo científico | Pendiente |
| Anteproyecto y cronograma | [`docs/anteproyecto/`](docs/anteproyecto/) |

---

## Instalación

Requisitos: Python 3.10+, GDAL/rasterio instalados en el sistema y acceso de red al endpoint STAC (CDSE o Planetary Computer).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Paquete base
pip install -e .

# Todas las dependencias de la comparación (XGBoost + PyTorch + segmentation-models-pytorch)
pip install -e ".[compare]"
```

Extras por separado: `.[xgboost]`, `.[torch]` (U-Net, Attention U-Net, DeepLabV3+; ~2 GB) y `.[notebook]`.

> Si `pip install -e ".[...]"` falla en tu shell, prueba con comillas simples: `pip install -e '.[compare]'`. Usa siempre el Python del entorno virtual antes de ejecutar cualquier comando `floodprob`.

---

## Reproducir el experimento

### Experimento principal: supervisión débil frente a completa

```bash
CFG=configs/valencia_quicktest.yaml

# 1. Datos externos: DEM, WorldCover y máscara EMS
python scripts/download_external_data.py --config $CFG

# 2. Verificar el catálogo STAC y descargar las imágenes (stacks zarr en cache/)
floodprob stac-check --config $CFG
floodprob ingest     --config $CFG

# 3. Variables por píxel, pseudo-etiquetas y referencia (flood_gt, water_gt)
floodprob preprocess --config $CFG

# 4. Entrenar y evaluar los ocho modelos en los dos regímenes
floodprob compare --config $CFG --train-on pseudo    # → outputs/<run>/compare_pseudo/
floodprob compare --config $CFG --train-on gt        # → outputs/<run>/compare_gt/

# Forzar el reentrenamiento aunque haya modelos en caché
floodprob compare --config $CFG --train-on pseudo --force-retrain
```

Guarda los manifiestos de cada versión de resultados en [`docs/resultados/`](docs/resultados/) y anota los cambios en el registro metodológico.

### Pipeline de un solo modelo (generar mapas)

```bash
floodprob train    --config $CFG    # entrena el modelo indicado en la config
floodprob predict  --config $CFG    # mapas de probabilidad GeoTIFF
floodprob evaluate --config $CFG    # métricas
floodprob folium   --config $CFG    # visor interactivo (opcional)
```

---

## Configuración STAC

### Opción A: Microsoft Planetary Computer (la usada en el experimento)

```yaml
stac:
  api_url: "https://planetarycomputer.microsoft.com/api/stac/v1"
  collections:
    s1: "sentinel-1-rtc"     # con corrección radiométrica del terreno
    s2: "sentinel-2-l2a"
  auth:
    type: none   # PC firma los assets automáticamente si tienes planetary_computer instalado
```

### Opción B: CDSE (Copernicus Data Space Ecosystem)

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

Plantillas: [`template_planetary_computer.yaml`](configs/template_planetary_computer.yaml) y [`template_cdse.yaml`](configs/template_cdse.yaml).

---

## Estructura del proyecto

```text
.
├── configs/
│   ├── valencia_quicktest.yaml       # Experimento principal (AOI reducido, 20 m)
│   ├── valencia_example.yaml         # AOI completo de Valencia, 10 m
│   ├── malaga_dana_2025.yaml         # DANA de Málaga (sin referencia todavía)
│   ├── aoi_*.geojson                 # Áreas de interés
│   └── template_*.yaml               # Plantillas CDSE / Planetary Computer
├── data/labels/
│   └── emsr773_flood_mask.geojson    # Referencia EMSR773 (DANA Valencia)
├── docs/
│   ├── registro_metodologico.md      # Fallos, correcciones, hallazgos y puntos abiertos
│   ├── resultados/                   # Manifiestos de métricas de cada versión
│   └── anteproyecto/                 # Cronograma del TFG (script + PNG/PDF/SVG)
├── scripts/
│   ├── download_external_data.py     # DEM, WorldCover y máscaras EMS
│   └── resumen_metricas.py           # manifest_compare.json → tablas Markdown
├── src/floodprob/
│   ├── cli.py                        # CLI: todos los comandos
│   ├── config.py                     # Configuración Pydantic + YAML
│   ├── preprocess.py                 # Variables, pseudo-etiquetas y referencia
│   ├── splits.py                     # Partición espacial por bloques
│   ├── compare.py                    # Experimento multi-modelo
│   ├── compare_metrics.py            # Métricas y elección del umbral
│   ├── visualize_compare.py          # Figuras (PNG 300 dpi + PDF)
│   ├── train.py / predict.py / evaluate.py / visualize.py   # Pipeline de un modelo
│   ├── features/                     # s1.py (VV, VH, Δ), s2.py (NDWI, MNDWI), dem.py
│   ├── ingest/                       # Búsqueda STAC y stacks zarr
│   ├── labels/pseudolabels.py        # Generación de pseudo-etiquetas
│   ├── models/                       # Un fichero por modelo
│   └── utils/                        # Geoespacial e I/O
└── pyproject.toml
```

---

## Comparación de modelos: detalles

### Arquitecturas de red neuronal

- **U-Net:** encoder–decoder de 3 niveles (`C_in → 16 → 32 → 64`) con skip connections. MC-Dropout para estimar la incertidumbre.
- **Attention U-Net:** U-Net con puertas de atención aditivas en cada skip connection (Oktay et al., 2018), que aprenden a resaltar las zonas relevantes del encoder.
- **DeepLabV3+:** encoder ResNet-34 con ASPP (convoluciones dilatadas multiescala) y decoder ligero, vía `segmentation-models-pytorch`. Entrenada desde cero (`encoder_weights=None`) con cosine LR annealing. Con pocos patches es inestable (A-09).

Las entradas de las CNN se estandarizan por canal con estadísticas del fold de entrenamiento. Al terminar el entrenamiento se recalibran las estadísticas de BatchNorm.

### Configurar en YAML

```yaml
model:
  train_on: "pseudo"     # "pseudo" | "gt"; también --train-on
  tile_size: 128
  n_folds: 5
  test_fold: 0
  val_fold: 1
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

### Salidas

```
outputs/<run_name>/compare_<train_on>/   # compare_pseudo/ o compare_gt/
├── metadata/
│   └── manifest_compare.json            # protocolo + métricas por modelo y ventana
├── models/                              # bundles sklearn (.joblib) y pesos PyTorch (.pt)
├── predictions/
│   ├── p_agua_<model>_<key>.tif
│   ├── p_inundacion_<model>_<key>.tif
│   └── p_inundacion_final_<model>_<key>.tif
└── figures/                             # PNG (300 dpi) + PDF
    ├── sar_context_<key>.png            # SAR antes / después / cambio + contorno de la referencia
    ├── model_predictions_<key>.png      # referencia + predicción de cada modelo
    ├── ensemble_analysis_<key>.png      # acuerdo entre modelos, incertidumbre, mejor y peor modelo
    ├── roc_*.png / pr_*.png             # curvas ROC y precisión-exhaustividad
    ├── metrics_bar_*.png                # barras F1, IoU, AUC
    ├── calibration_*.png                # diagramas de fiabilidad
    ├── feature_importance_*.png
    └── error_map_<key>.png              # solo si hay referencia de inundación
```

El pipeline de un solo modelo escribe en `outputs/<run_name>/predictions/` (`p_agua`, `p_inundacion_dado_agua` y `p_inundacion` = P(agua) × P(inundación | agua)) y en `outputs/<run_name>/evaluation/` (`metrics.json` y diagramas de fiabilidad).

> **Nota sobre P(agua) × P(inundación | agua):** ahora mismo la cabeza de inundación se entrena sobre todos los píxeles, como un clasificador directo, así que el producto no es estrictamente la regla de la probabilidad total (A-04). El experimento principal evalúa la cabeza de inundación directamente.

---

## Notas técnicas

| Parámetro | Descripción |
|---|---|
| `crs_salida` | CRS de salida (por defecto `EPSG:25830`) |
| `resolution_m` | Resolución en metros: `10` = más detalle y más coste; `20` = más rápido |
| `seed` | Semilla de la partición y de todos los modelos |
| `model.tile_size` | Tamaño de los bloques de la partición y de los patches de las CNN (por defecto `128`) |
| `model.train_on` | `pseudo` (por defecto) o `gt` |
| `model.n_folds`, `test_fold`, `val_fold` | Partición espacial (por defecto 5 folds, test = 0, validación = 1) |
| `filters.s1_match_target_orbit` | Restringe el periodo de referencia y los históricos S1 a la órbita relativa del evento (por defecto `true`) |
| `model.calibration` | `isotonic` o `sigmoid` |
| `model.ensemble_n` | Número de modelos en el ensemble (por defecto `1`) |

- **DEM:** el pipeline no lo descarga dentro de `ingest`. Usa `scripts/download_external_data.py` o coloca un GeoTIFF local en `dem.path`. Sin DEM, el pipeline continúa sin `elevation`, `slope` ni `dist_to_permwater`.
- **Caché:** se guarda por hash en `cache/`. Si cambias los parámetros de descarga, borra `cache/`.
- **GPU:** los modelos PyTorch usan `cuda` si está disponible y, si no, CPU.
- **Reproducibilidad:** U-Net y Attention U-Net son deterministas con la misma semilla; DeepLabV3+ no del todo, por operaciones CUDA no deterministas (F-11).
- **numpy ≥ 2.0:** la reducción temporal del SAR usa `quantile(0.5)` en lugar de `median(skipna=True)` para evitar un fallo de `np.ma.sort` con arrays dask.
- **Assets no COG:** rasterio puede leerlos vía VSI según el servidor; si falla, revisa los permisos del endpoint o cambia de colección.

---

## Referencia de comandos CLI

```bash
floodprob stac-check  --config <yaml>                                   # verifica el catálogo STAC
floodprob ingest      --config <yaml>                                   # descarga imágenes
floodprob preprocess  --config <yaml>                                   # variables + pseudo-etiquetas + referencia
floodprob compare     --config <yaml> [--train-on pseudo|gt] [--force-retrain]   # experimento multi-modelo
floodprob train       --config <yaml>                                   # entrena un modelo
floodprob predict     --config <yaml>                                   # genera mapas de probabilidad
floodprob evaluate    --config <yaml>                                   # calcula métricas
floodprob folium      --config <yaml> [--key <window_key>]              # visor interactivo

# Todos los comandos aceptan --verbose / -v
floodprob --verbose compare --config configs/valencia_quicktest.yaml
```
