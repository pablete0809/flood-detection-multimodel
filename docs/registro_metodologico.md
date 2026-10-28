# Registro metodológico del proyecto

Este documento recoge los fallos metodológicos encontrados en el pipeline, cómo se detectaron, qué impacto tenían en los resultados y cómo se corrigieron. Sirve para redactar la memoria del TFG y el paper (secciones de *Metodología*, *Validación* y *Limitaciones*) sin tener que reconstruirlo después desde el código o el historial de git.

**Cómo mantenerlo:** cada vez que encuentres o corrijas algo, añade una entrada copiando la plantilla del final. Guarda las cifras *en el momento*: luego es muy difícil volver a calcularlas con el código antiguo. Los manifiestos de resultados de cada versión se guardan en [`docs/resultados/`](resultados/) y se resumen con:

```bash
python scripts/resumen_metricas.py docs/resultados/<manifiesto>.json
```

---

## Resumen

| ID | Fecha | Fallo | Gravedad | Estado |
|---|---|---|---|---|
| [H-01](#h-01--dependencia-circular-en-la-cabeza-de-inundación) | 2026 (commit `aad86f3`) | Dependencia circular al entrenar la cabeza de inundación | Alta | Corregido |
| [H-02](#h-02--mediana-temporal-con-numpy-2x) | 2026 (commit `98f94a3`) | `median(skipna)` falla con numpy 2.x y dask | Técnica | Corregido |
| [F-01](#f-01--las-etiquetas-externas-sustituían-a-las-pseudo-etiquetas) | 2026-10-28 | Las etiquetas externas sustituían a las pseudo-etiquetas: se entrenaba con el ground truth | **Crítica** | Corregido |
| [F-02](#f-02--la-cabeza-de-agua-aprendía-agua-permanente-no-agua-del-evento) | 2026-10-28 | La cabeza de agua aprendía agua *permanente* (WorldCover), no agua del evento | **Crítica** | Corregido |
| [F-03](#f-03--holdout-espacial-degenerado) | 2026-10-28 | Holdout espacial degenerado: 2 franjas verticales con 0,7 % de inundación | **Crítica** | Corregido |
| [F-04](#f-04--el-entrenamiento-incluía-los-píxeles-de-test) | 2026-10-28 | El entrenamiento incluía los píxeles de test (fuga) | **Crítica** | Corregido |
| [F-05](#f-05--cada-modelo-se-evaluaba-sobre-píxeles-distintos) | 2026-10-28 | Cada modelo se evaluaba sobre un conjunto de píxeles distinto | Alta | Corregido |
| [F-06](#f-06--umbral-fijo-de-05) | 2026-10-28 | Umbral fijo de 0,5 para F1/IoU | Alta | Corregido |
| [F-07](#f-07--mc-dropout-con-batchnorm-en-modo-entrenamiento) | 2026-10-28 | MC-Dropout con BatchNorm en modo entrenamiento (U-Net, Attention U-Net) | Media | Corregido |
| [F-08](#f-08--entradas-de-las-cnn-sin-normalizar) | 2026-10-28 | Entradas de las CNN sin normalizar | Media | Corregido |
| [F-09](#f-09--baseline-sar-con-órbitas-mezcladas) | 2026-10-28 | Baseline SAR con órbitas mezcladas | Media | Corregido |
| [F-10](#f-10--estadísticas-de-batchnorm-inservibles-en-deeplabv3) | 2026-10-28 | Estadísticas de BatchNorm inservibles en DeepLabV3+ (AUC 0,31) | Alta | Corregido |
| [F-11](#f-11--entrenamiento-de-las-cnn-sin-semilla) | 2026-10-28 | Entrenamiento de las CNN sin semilla: resultados no reproducibles | Alta | Corregido (salvo DeepLabV3+) |
| [A-01](#a-01--la-frecuencia-histórica-de-agua-viene-de-una-sola-fecha) | 2026-10-28 | La "frecuencia histórica de agua" viene de una sola fecha (arrozales) | Alta | **Abierto** |
| [A-08](#a-08--variabilidad-entre-semillas-y-elección-del-umbral-con-supervisión-débil) | 2026-10-28 | Gran variabilidad entre semillas y umbral mal elegido con pseudo-etiquetas | Alta | **Abierto** (hallazgo) |
| [A-09](#a-09--deeplabv3-inestable-con-pocos-datos) | 2026-10-28 | DeepLabV3+ desde cero es inestable con 23 patches | Media | **Abierto** |
| [A-02 … A-07](#otros-puntos-abiertos) | 2026-10-28 | Calibración CNN, una sola partición, semántica P(inund.\|agua), etc. | — | **Abierto** |

---

## Correcciones históricas (antes de esta revisión)

### H-01 — Dependencia circular en la cabeza de inundación

- **Qué era:** la cabeza P(inundación | agua) se entrenaba solo sobre píxeles con `p_agua > 0.35`, calculado con el propio pipeline, y además usaba como features variables de las que derivaban las etiquetas.
- **Estado:** corregido en el commit inicial (`aad86f3`, "correcciones de dependencia circular y feature leakage"). El detalle exacto no se puede reconstruir desde git porque el código anterior no está versionado.
- **Para la memoria:** puede mencionarse como primera iteración del diseño, pero sin cifras.

### H-02 — Mediana temporal con numpy 2.x

- **Qué era:** `DataArray.median(skipna=True)` sobre arrays dask lanza `IndexError` en `np.ma.sort` con numpy ≥ 2.0.
- **Corrección:** `quantile(0.5, skipna=True)` y reescritura explícita del CRS, que `quantile` elimina ([features/s1.py](../src/floodprob/features/s1.py)). Commit `98f94a3`.
- **Para la memoria:** detalle técnico. No afecta a la metodología; basta como nota de implementación, si acaso.

---

## Revisión del 2026-10-28

Contexto: Valencia, DANA de octubre de 2024. AOI de 685 × 666 píxeles a 20 m. Una sola fecha S1 de evento (2024-11-01, órbita relativa 110, descendente). Sin Sentinel-2 por nubes. Ground truth: EMSR773 rasterizado (82 801 píxeles, 18,1 % del AOI). La configuración es [`configs/valencia_quicktest.yaml`](../configs/valencia_quicktest.yaml).

### F-01 — Las etiquetas externas sustituían a las pseudo-etiquetas

- **Síntoma:** el README describía el entrenamiento con "pseudo-etiquetas generadas automáticamente", pero los ficheros de etiquetas no tenían ni un píxel sin etiquetar (-1).
- **Causa:** en `generate_*_pseudolabels` ([labels/pseudolabels.py](../src/floodprob/labels/pseudolabels.py)), cuando había etiqueta externa se aplicaba `xr.where(external == 1, 1, …)` **y** `xr.where(external == 0, 0, …)`. La etiqueta externa vale 0 en todo píxel fuera del polígono, así que sobrescribía el 100 % de la imagen.
- **Evidencia:** `flood_label` era idéntico a EMSR773 rasterizado (82 801 positivos). `water_label` era idéntico a WorldCover clase 80 (147 046 positivos).
- **Impacto:** (1) no había supervisión débil, sino supervisión completa con el GT; (2) combinado con [F-04](#f-04--el-entrenamiento-incluía-los-píxeles-de-test), los modelos se entrenaban con las mismas etiquetas con las que luego se evaluaban.
- **Corrección:**
  - Las pseudo-etiquetas ya no se tocan.
  - El GT se guarda en variables separadas (`flood_gt`, `water_gt`) en `labels_*.nc` ([preprocess.py](../src/floodprob/preprocess.py), `build_ground_truth`).
  - Nueva opción `model.train_on: pseudo | gt` (también `floodprob compare --train-on`) que decide con qué se entrena. La evaluación usa siempre el GT si existe (`reference_labels`).
  - Cada modo escribe en su carpeta: `outputs/<run>/compare_pseudo/` y `outputs/<run>/compare_gt/`.
- **Para la memoria:** *"Las etiquetas de referencia (EMSR773) se mantienen estrictamente separadas de las de entrenamiento. Se evalúan dos regímenes: supervisión débil, en la que los modelos se entrenan solo con pseudo-etiquetas derivadas del cambio SAR y de referencias de agua permanente, y supervisión completa, en la que se entrenan con EMSR773 en los bloques de entrenamiento. Ambos se evalúan contra EMSR773 en bloques espaciales independientes."*

### F-02 — La cabeza de agua aprendía agua permanente, no agua del evento

- **Síntoma:** AUC ≈ 1,000 en la cabeza de agua para RF, XGBoost y MLP.
- **Causa:** por F-01, `water_label` = WorldCover clase 80, que es agua **permanente**. Además, `freq_agua_hist` (agua histórica) era una feature de entrada, así que el problema se volvía trivial.
- **Evidencia:** solo el **1,4 %** de los píxeles inundados según EMSR estaban etiquetados como agua. La cabeza de agua aprendía que la inundación *no* es agua, lo que contradice la descomposición P(inund.) = P(agua) × P(inund. | agua).
- **Corrección:**
  - WorldCover se usa como `perm_water_ref`, una referencia independiente del evento, dentro de las reglas de pseudo-etiquetado (`_permanent_water`).
  - La referencia de evaluación del agua es `water_gt = EMSR ∪ agua permanente` (agua presente en la fecha del evento). Solo se construye si hay GT de inundación.
- **Para la memoria:** distinguir explícitamente entre agua permanente y agua del evento. Es un error conceptual fácil de cometer y vale la pena una frase en la sección de datos.

### F-03 — Holdout espacial degenerado

- **Síntoma:** las métricas de inundación eran inestables y el PR-AUC de algunos modelos era de 0,003.
- **Causa:** el holdout era `((ty * 10000 + tx) % 5) == 0`, con `ty, tx` = índice de tile. Como 10000 es múltiplo de 5, la condición **solo depende de `tx`**: el "test" eran dos columnas de tiles (tx = 0 y tx = 5), es decir, dos franjas verticales en los bordes.
- **Evidencia:** test = 23,1 % de la imagen con solo un **0,7 %** de inundación (≈ 740 píxeles), frente al 18,1 % del AOI completo.
- **Corrección:** nuevo módulo [splits.py](../src/floodprob/splits.py). Los tiles de 128 px se asignan aleatoriamente (semilla fija) a 5 folds equilibrados: fold 0 = test, fold 1 = validación y el resto = entrenamiento. Lo usan `compare`, `train`, `evaluate` y las figuras. Los tiles están alineados con los patches de las CNN, así que ningún patch mezcla entrenamiento y test.
- **Evidencia tras la corrección:** test = 118 016 píxeles con un 17,2 % de inundación (20 312 positivos); validación = 91 008 píxeles; entrenamiento = 247 186 píxeles.
- **Para la memoria:** describir la validación por bloques espaciales (*spatial block cross-validation*) y justificarla por la autocorrelación espacial (Roberts et al., 2017, *Ecography*).

### F-04 — El entrenamiento incluía los píxeles de test

- **Causa:** `flatten_samples` (modelos por píxel) y la extracción de patches (CNN) usaban toda la imagen; el holdout solo se aplicaba al evaluar.
- **Impacto:** fuga de etiquetas. Junto con F-01, los modelos veían en entrenamiento las etiquetas EMSR de los píxeles de test.
- **Corrección:** `mask_to_train` ([splits.py](../src/floodprob/splits.py)) pone a -1 las etiquetas fuera del fold de entrenamiento antes de entrenar. Todos los entrenadores ya ignoraban el -1, y las CNN descartan los patches sin ningún píxel etiquetado.
- **Verificación:** las etiquetas de entrenamiento en los tiles de test son todas -1.

### F-05 — Cada modelo se evaluaba sobre píxeles distintos

- **Causa:** `predict_flood_probability` ponía NaN donde `p_agua ≤ 0.05`, usando la cabeza de agua de F-02 (agua permanente). Así, los modelos sklearn de inundación se evaluaban casi solo sobre agua permanente, mientras que las CNN, que no usan esa máscara, se evaluaban sobre toda la imagen.
- **Evidencia:** PR-AUC de 0,003 a 0,011 en los modelos sklearn frente a 0,34–0,52 en las CNN. Las cifras no eran comparables.
- **Corrección:** se elimina la máscara ([train_flood_given_water.py](../src/floodprob/models/train_flood_given_water.py)). La cabeza de inundación se entrena sobre todos los píxeles etiquetados, así que predice en todos. La evaluación usa una **máscara común**: fold de test ∩ etiqueta de referencia válida ∩ todas las features finitas (`_feature_valid_mask` en [compare.py](../src/floodprob/compare.py)). El manifiesto registra `n_test`, `n_test_pos` y `n_missing_pred` para comprobarlo.
- **Para la memoria:** *"Todos los modelos se evalúan sobre el mismo conjunto de 112 571 píxeles de test."*

### F-06 — Umbral fijo de 0,5

- **Causa:** F1 e IoU se calculaban binarizando a 0,5, pero los modelos entrenados con `class_weight="balanced"` o `pos_weight`, y los calibrados sobre una prevalencia distinta, no tienen su óptimo en 0,5.
- **Corrección:**
  - El umbral se elige maximizando F1 en el **fold de validación**, con las etiquetas del mismo tipo que las de entrenamiento (pseudo o GT): nunca se mira el GT en modo `pseudo`. Ver `best_f1_threshold` en [compare_metrics.py](../src/floodprob/compare_metrics.py).
  - Se reportan también `f1_at_05` / `iou_at_05` y las métricas independientes del umbral: AUC-ROC y, como principal con clases desbalanceadas, PR-AUC.
- **Evidencia:** LogReg con pseudo-etiquetas tiene F1@0,5 = 0,173 frente a F1 = 0,736 con el umbral de validación (0,055).
- **Para la memoria:** *"El umbral de decisión se selecciona en un conjunto de validación espacialmente independiente."*

### F-07 — MC-Dropout con BatchNorm en modo entrenamiento

- **Causa:** para muestrear MC-Dropout, la U-Net y la Attention U-Net llamaban a `model.train()` en inferencia. Eso activa el dropout, pero también hace que BatchNorm use las estadísticas *del batch*, así que la predicción de un patch dependía de qué otros patches compartían batch.
- **Evidencia:** ECE de 0,28–0,31 en la v0, frente a 0,07 de DeepLabV3+, que sí usaba `eval()`.
- **Corrección:** `_enable_mc_dropout` pone el modelo en `eval()` y reactiva solo los módulos Dropout ([train_unet.py](../src/floodprob/models/train_unet.py)).
- **Nota:** el ECE de las CNN sigue siendo alto (~0,25) después de corregirlo. La causa principal es otra: `pos_weight`, ver [A-02](#otros-puntos-abiertos).

### F-08 — Entradas de las CNN sin normalizar

- **Causa:** las CNN recibían dB (de -30 a +37), metros (`dist_to_permwater`, de 0 a 510) y grados (`slope`) sin estandarizar, con los NaN rellenados a 0, que es un valor con significado físico en dB.
- **Corrección:** estandarización por canal con media y desviación calculadas **solo en píxeles de entrenamiento**, guardadas en el bundle y aplicadas en inferencia. Los NaN pasan a 0 *después* de estandarizar, es decir, a la media del canal (`_channel_stats`, `_normalize`, `_prepare_inputs`).

### F-09 — Baseline SAR con órbitas mezcladas

- **Causa:** el baseline (ago–sep 2024) mezclaba 20 adquisiciones de órbitas ascendentes y descendentes y de tres órbitas relativas distintas, mientras que el evento es una sola pasada (órbita relativa 110, descendente, 06:02). El cambio `delta = evento − baseline` incluía diferencias de geometría de observación y no solo cambio en superficie.
- **Corrección:** `filters.s1_match_target_orbit: true` (por defecto). `ingest` busca primero las órbitas relativas de las ventanas de evento y filtra el baseline y los históricos a esas órbitas ([preprocess.py](../src/floodprob/preprocess.py)). El manifiesto de ingest registra `s1_relative_orbits`.
- **Evidencia** (imagen completa, frente a EMSR773):

  | Baseline | AUC (−Δ VV) | AP (−Δ VV) | AUC (−Δ VH) | σ(Δ VV) en zonas no inundadas |
  |---|---|---|---|---|
  | Órbitas mezcladas (20 imágenes) | 0,887 | 0,838 | 0,887 | 3,49 dB |
  | Solo órbita 110 (8 imágenes, 4 fechas) | **0,913** | **0,869** | **0,907** | **2,74 dB** |

- **Para la memoria:** buena frase para la sección de preprocesado: *"La detección de cambios se realiza entre adquisiciones de la misma órbita relativa, lo que reduce la dispersión del cambio en zonas estables de 3,49 a 2,74 dB."*

### F-10 — Estadísticas de BatchNorm inservibles en DeepLabV3+

- **Síntoma:** al corregir F-03/F-04, DeepLabV3+ entrenado con GT dio **AUC = 0,312 en test**, peor que el azar.
- **Causa:** las estadísticas acumuladas (*running mean/var*) de BatchNorm son una media móvil de activaciones de pesos anteriores. Con ResNet-34 entrenada desde cero, solo 23 patches, batch de 4 y ~120 actualizaciones, acaban muy lejos de las activaciones finales, y en `eval()` la red colapsa.
- **Evidencia:** el mismo modelo da AUC 0,312 en `eval()` y 0,982 con estadísticas del batch. Probablemente también afectaba a la v0, aunque con menos intensidad.
- **Corrección:** `_recalibrate_batchnorm` hace una pasada sin gradientes sobre los patches de entrenamiento con media acumulada (`momentum=None`) al terminar el entrenamiento. Se aplica a las tres CNN por homogeneidad.
- **Evidencia tras la corrección:** DeepLabV3+ entrenado con GT pasa de AUC 0,312 a 0,911 en test con la misma semilla de partición. Sigue siendo inestable entre semillas: ver [A-09](#a-09--deeplabv3-inestable-con-pocos-datos).
- **Para la memoria:** limitación práctica de las redes profundas con BatchNorm cuando hay pocos datos. Es un buen argumento para justificar arquitecturas ligeras en este contexto.

### F-11 — Entrenamiento de las CNN sin semilla

- **Síntoma:** dos ejecuciones idénticas de `compare --train-on pseudo` dieron, para la U-Net, F1 = 0,863 y F1 = 0,597.
- **Causa:** `seed` solo llegaba a los modelos sklearn. En PyTorch no había `manual_seed`, así que la inicialización de pesos, el orden de los batches y el dropout cambiaban en cada ejecución.
- **Corrección:** `_seed_torch(cfg.seed)` antes de construir el modelo y el DataLoader, con cuDNN determinista; semilla fija también para el muestreo MC-Dropout en inferencia ([train_unet.py](../src/floodprob/models/train_unet.py)).
- **Verificación:** dos entrenamientos con la misma semilla dan métricas idénticas en U-Net y Attention U-Net (diferencia 0,0000). **DeepLabV3+ no es totalmente determinista** (diferencia de hasta 0,18 en AUC) por operaciones CUDA no deterministas de `segmentation-models-pytorch`; habría que activar `torch.use_deterministic_algorithms(True)` o reportarlo como limitación.
- **Para la memoria:** *"Todos los resultados de redes neuronales se reportan como media ± desviación típica sobre 5 semillas."* No vale reportar una sola ejecución (ver A-08).

---

## Puntos abiertos

### A-01 — La "frecuencia histórica de agua" viene de una sola fecha

- **Qué pasa:** `freq_agua_hist` se calcula aplicando la regla de cambio a las ventanas históricas (`n_hist_windows: 1` → nov-2023). Tras F-09 queda **una sola fecha** (2023-11-07), así que no es una frecuencia sino una máscara binaria de esa fecha.
- **Evidencia:**
  - El **64,5 %** de los píxeles inundados según EMSR tienen `freq_agua_hist > 0.5`, es decir, la regla los trata como agua permanente y los etiqueta como **no inundados**.
  - Del 72,8 % del GT inundado que acaba etiquetado como 0, el 88,7 % lo es por este criterio; solo el 17,5 % porque el SAR no ve agua.
  - Calidad de las pseudo-etiquetas de inundación frente a EMSR (solo píxeles etiquetados): **precisión 0,696; recall 0,195**.
- **Hipótesis a verificar:** en noviembre, los arrozales de la Albufera están inundados de forma estacional (*perellonà*), así que en la fecha histórica parecen agua. Hay que comprobarlo superponiendo los falsos negativos con el mapa de cultivos o WorldCover.
- **Siguiente paso (semana 2):** sustituir por JRC Global Surface Water (*occurrence*, 1984–2021) o usar varias fechas históricas de varias estaciones. Decidir y justificar si el arrozal inundado estacionalmente cuenta como "inundación" (EMSR lo incluye).
- **Para la memoria:** es un hallazgo en sí mismo. Definir "agua permanente" en una zona con agua agrícola estacional no es trivial, y condiciona cualquier método de etiquetado automático.

### A-08 — Variabilidad entre semillas y elección del umbral con supervisión débil

Experimento: las 3 CNN × 5 semillas (1, 2, 3, 4, 42) × los 2 regímenes de etiquetas, misma partición y mismo protocolo que `compare`. Datos: [`resultados/v1_valencia_cnn_semillas.json`](resultados/v1_valencia_cnn_semillas.json). Test = 112 571 píxeles (18,0 % inundados).

| Entrenamiento | Modelo | AUC-ROC (media ± sd) | PR-AUC (media) | F1 (media ± sd) | F1 mín–máx |
|---|---|---|---|---|---|
| GT (EMSR) | U-Net | 0,980 ± 0,005 | 0,963 | 0,923 ± 0,002 | 0,921–0,926 |
| GT (EMSR) | Attention U-Net | 0,980 ± 0,004 | 0,963 | 0,924 ± 0,001 | 0,921–0,925 |
| GT (EMSR) | DeepLabV3+ | 0,664 ± 0,298 | 0,611 | 0,599 ± 0,286 | 0,346–0,919 |
| Pseudo | U-Net | 0,918 ± 0,038 | 0,681 | 0,496 ± 0,278 | 0,244–0,862 |
| Pseudo | Attention U-Net | 0,917 ± 0,043 | 0,685 | 0,276 ± 0,093 | 0,193–0,434 |
| Pseudo | DeepLabV3+ | 0,563 ± 0,209 | 0,425 | 0,407 ± 0,143 | 0,259–0,592 |

Descomposición del F1 de la U-Net entrenada con pseudo-etiquetas: umbral elegido con pseudo-etiquetas de validación, frente a umbral elegido con GT de validación, frente al mejor umbral posible en test (oráculo, solo como referencia):

| Semilla | Umbral (val. pseudo) | F1 | Umbral (val. GT) | F1 | F1 oráculo |
|---|---|---|---|---|---|
| 1 | 0,664 | 0,862 | 0,634 | 0,850 | 0,875 |
| 2 | 0,797 | 0,244 | 0,582 | 0,627 | 0,632 |
| 3 | 0,866 | 0,418 | 0,664 | 0,763 | 0,773 |
| 4 | 0,786 | 0,707 | 0,663 | 0,770 | 0,794 |
| 42 | 0,858 | 0,247 | 0,678 | 0,744 | 0,792 |

**Lectura:**
1. Con GT, las U-Net son muy estables y alcanzan F1 ≈ 0,92: es el techo de supervisión completa en este evento.
2. Con pseudo-etiquetas, la capacidad de **ordenar** píxeles es buena y estable (AUC ≈ 0,92), pero el F1 es muy inestable.
3. La mayor parte de la pérdida de F1 viene de **elegir el umbral con pseudo-etiquetas**. Como estas marcan los arrozales inundados como "no inundado" (A-01), el umbral óptimo en validación sale demasiado alto. Con unos pocos tiles de validación con GT, el F1 sube a 0,63–0,85.
4. Un F1 = 0,86 de una sola ejecución (el que vimos en la primera prueba) **no es representativo**.

- **Para el paper:** es probablemente el resultado más interesante. *"Bajo supervisión débil, los modelos aprenden un buen ranking de probabilidad, pero la selección del punto de operación es el cuello de botella; una pequeña muestra etiquetada para fijar el umbral recupera la mayor parte de la diferencia con la supervisión completa."* Para afirmarlo bien, falta CV de 5 folds (A-03) y corregir A-01.

### A-09 — DeepLabV3+ inestable con pocos datos

- ResNet-34 entrenada desde cero con 23 patches de 128 × 128 converge en unas semillas y en otras no (F1 de 0,35 a 0,92 con GT), incluso tras F-10.
- **Opciones:** (a) usar un encoder preentrenado (ImageNet o un modelo preentrenado con SAR) adaptando el número de canales de entrada; (b) un encoder más pequeño (ResNet-18); (c) más patches con solapamiento y aumentos (rotaciones/volteos); (d) presentarlo como resultado: las arquitecturas grandes no compensan con un solo evento.

### Otros puntos abiertos

- **A-02 — Calibración de las CNN.** El ECE es de ~0,25–0,31 por `pos_weight = n_neg/n_pos` (de 7 a 26), que infla las probabilidades. Opciones: *temperature scaling* en el fold de validación o reportar el ECE solo para modelos calibrados.
- **A-03 — Una sola partición de test.** Las cifras actuales salen de un solo fold de test y una sola semilla. Para el paper hace falta validación cruzada por bloques (rotar el fold de test entre los 5) y media ± desviación; `spatial_split` ya admite `test_fold` / `val_fold`.
- **A-04 — Semántica de P(inundación | agua).** La cabeza de "inundación dado agua" se entrena de facto como clasificador directo de inundación sobre todos los píxeles, así que multiplicarla por P(agua) en `p_inundacion_final_*` no es la descomposición de probabilidad total. Pendiente: o se entrena solo sobre agua, o se presenta la descomposición como ablación frente a la segmentación directa.
- **A-05 — Málaga sin ground truth.** `data/labels/malaga_flood_mask.geojson` tiene 0 geometrías. No usar Málaga para métricas mientras no haya GT.
- **A-06 — Origen de EMSR773.** Hay que comprobar con qué imagen se delimitó (`dmg_src_id = 3` en el GeoJSON). Si fue con el S1 del 2024-11-01, la evaluación es parcialmente circular y hay que decirlo.
- **A-07 — Sin buffer entre bloques.** Los tiles de entrenamiento y test son contiguos. Con la autocorrelación espacial y el campo receptivo de las CNN, los píxeles de los bordes no son del todo independientes. Opción: excluir del entrenamiento una franja de N píxeles alrededor de los tiles de test.
- **Documentación.** El README decía "190 polígonos" (el fichero tiene 1117 geometrías) y atribuía los F1 bajos a "la dificultad real del evento"; ambas afirmaciones se corrigen en el README.

---

## Resultados por versión

> **Las cifras de la v0 no son válidas** (F-01 a F-06). Se conservan solo para documentar el cambio. Manifiesto: [`resultados/v0_valencia_manifest_compare.json`](resultados/v0_valencia_manifest_compare.json). Copia local completa de salidas y figuras: `outputs/valencia_quicktest_rtc_v0_antes_de_arreglos/` (no versionada).

### v0 — antes de la revisión (no válida)

Cabeza de inundación tal como aparecía en el README, evaluada sobre las 2 franjas de F-03 (≈ 740 píxeles inundados):

| Modelo | AUC-ROC | PR-AUC | F1@0,5 | IoU | ECE |
|---|---|---|---|---|---|
| LogReg | 0,571 | 0,003 | 0,000 | 0,000 | 0,028 |
| Random Forest | 0,828 | 0,011 | 0,028 | 0,014 | 0,056 |
| XGBoost | 0,815 | 0,009 | 0,037 | 0,019 | 0,050 |
| MLP | 0,788 | 0,008 | 0,018 | 0,009 | 0,038 |
| U-Net | 0,920 | 0,521 | 0,342 | 0,206 | 0,284 |
| Attention U-Net | 0,905 | 0,484 | 0,199 | 0,111 | 0,307 |
| DeepLabV3+ | 0,826 | 0,338 | 0,453 | 0,293 | 0,067 |

Cabeza de agua: AUC de 0,999–1,000 para todos los modelos entrenables. Era un artefacto de F-02.

### v1 — tras F-01 a F-11 (2026-10-28)

Protocolo: Valencia, una ventana de evento, partición espacial por bloques de 128 px (5 folds, test = fold 0, validación = fold 1, semilla 42), umbral óptimo en validación, evaluación contra EMSR773 sobre **los mismos 112 571 píxeles de test (18,0 % inundados)** para todos los modelos. **Una sola semilla y una sola partición**: las cifras de las CNN varían mucho entre semillas (ver A-08) y no deben citarse sin la tabla de semillas.

Manifiestos: [`v1_valencia_pseudo_manifest_compare.json`](resultados/v1_valencia_pseudo_manifest_compare.json), [`v1_valencia_gt_manifest_compare.json`](resultados/v1_valencia_gt_manifest_compare.json).

**Cabeza de inundación — supervisión débil (`train_on: pseudo`)**

| Modelo | AUC-ROC | PR-AUC | F1 | IoU | Umbral | F1@0,5 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| LogReg | 0,921 | 0,806 | 0,736 | 0,582 | 0,055 | 0,173 | 0,105 | 0,112 |
| Random Forest | 0,836 | 0,510 | 0,327 | 0,195 | 0,234 | 0,329 | 0,160 | 0,159 |
| XGBoost | 0,755 | 0,371 | 0,322 | 0,192 | 0,015 | 0,326 | 0,160 | 0,156 |
| MLP | 0,879 | 0,527 | 0,326 | 0,194 | 0,010 | 0,328 | 0,161 | 0,157 |
| U-Net | 0,933 | 0,743 | 0,247 | 0,141 | 0,858 | 0,728 | 0,132 | 0,233 |
| Attention U-Net | 0,913 | 0,647 | 0,193 | 0,107 | 0,842 | 0,682 | 0,147 | 0,238 |
| DeepLabV3+ ¹ | 0,946 | 0,831 | 0,763 | 0,617 | 0,701 | 0,672 | 0,131 | 0,229 |

Referencia: la propia regla de pseudo-etiquetado, en los píxeles que etiqueta, tiene F1 = 0,25 frente a EMSR en test.

**Cabeza de inundación — supervisión completa (`train_on: gt`)**

| Modelo | AUC-ROC | PR-AUC | F1 | IoU | Umbral | F1@0,5 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| LogReg | 0,950 | 0,908 | 0,861 | 0,757 | 0,309 | 0,875 | 0,037 | 0,012 |
| Random Forest | 0,973 | 0,945 | 0,886 | 0,795 | 0,268 | 0,902 | 0,029 | 0,017 |
| XGBoost | 0,973 | 0,945 | 0,891 | 0,803 | 0,134 | 0,904 | 0,028 | 0,007 |
| MLP | 0,973 | 0,945 | 0,885 | 0,794 | 0,140 | 0,905 | 0,028 | 0,008 |
| U-Net | 0,987 | 0,967 | 0,922 | 0,856 | 0,647 | 0,922 | 0,077 | 0,218 |
| Attention U-Net | 0,973 | 0,955 | 0,921 | 0,854 | 0,621 | 0,918 | 0,094 | 0,246 |
| DeepLabV3+ ¹ | 0,973 | 0,958 | 0,921 | 0,853 | 0,733 | 0,912 | 0,036 | 0,086 |

¹ DeepLabV3+ no es determinista (F-11) y es inestable entre semillas (A-09). Esta ejecución concreta salió bien; en el experimento de 5 semillas, su F1 con GT varió entre 0,35 y 0,92.

**Cabeza de agua** (referencia: `water_gt` = EMSR ∪ WorldCover; 37,3 % positivos en test). Con supervisión completa, los modelos por píxel quedan en AUC 0,96–0,98 y F1 0,91–0,93. Con pseudo-etiquetas, en AUC 0,77–0,94 y F1 0,57–0,76. El baseline de umbral fijo da AUC 0,774.

**Lecturas principales para la memoria:**
1. Los resultados de la v0 eran artefactos. Con un protocolo correcto, la supervisión completa da F1 ≈ 0,89 (por píxel) y ≈ 0,92 (U-Net / Attention U-Net, estables entre semillas) frente a EMSR773.
2. Con supervisión débil, el modelo más simple (LogReg, F1 = 0,736) es el más fiable. Los modelos de más capacidad ajustan el ruido de las pseudo-etiquetas o eligen mal el umbral, aunque las CNN mantienen un AUC alto (0,91–0,95).
3. Con pseudo-etiquetas, el F1@0,5 de las CNN (0,67–0,73) es mucho mayor que el F1 con el umbral de validación (0,19–0,25): elegir el umbral con pseudo-etiquetas es el cuello de botella (A-08).
4. La regla de pseudo-etiquetado está limitada sobre todo por la definición de agua permanente (A-01), no por la señal SAR: el cambio de backscatter sin más alcanza AUC 0,91 frente a EMSR.

### Próximos pasos que cambiarán estas cifras

A-01 (agua permanente con JRC GSW), A-03 (CV de 5 folds × semillas) y A-02 (calibración). Al hacerlos, añade una sección **v2** con el mismo formato y guarda los manifiestos en `docs/resultados/`.

---

## Plantilla para nuevas entradas

```markdown
### F-XX — Título corto

- **Síntoma:** qué se observaba (métrica rara, figura incoherente…).
- **Causa:** fichero y línea, y por qué ocurría.
- **Evidencia:** cifras medidas *antes* de corregir.
- **Impacto:** qué resultados quedaban invalidados o sesgados.
- **Corrección:** qué se cambió (ficheros, opciones de config).
- **Evidencia tras la corrección:** cifras después.
- **Para la memoria:** una o dos frases redactables.
```
