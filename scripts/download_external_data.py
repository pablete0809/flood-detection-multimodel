"""
Script de descarga de datos externos para el pipeline floodprob.
Funciona con CUALQUIER configuración — las rutas de salida se leen
directamente del YAML (dem.path y external_labels.*).

Descarga:
  1. DEM Copernicus GLO-30 (vía Planetary Computer STAC)
       → ruta definida en cfg.dem.path
  2. ESA WorldCover 2021 (clase 80 = agua permanente) como etiqueta de agua
       → ruta definida en cfg.external_labels.water
  3. Máscara de inundación — búsqueda automática multi-fuente con fallback:
       Prioridad 1: Copernicus EMS Rapid Mapping (WFS, búsqueda por AOI+fecha)
       Prioridad 2: Copernicus Global Flood Monitoring (GFM API)
       Prioridad 3: SNCZI MITECO (mapa estático de peligrosidad T=10)
       Fallback:    GeoJSON vacío + aviso (pipeline continúa con pseudo-etiquetas)
       → ruta definida en cfg.external_labels.flood

Uso:
    cd <proyecto>
    python3 scripts/download_external_data.py --config configs/mi_evento.yaml

    # Descargar solo un producto:
    python3 scripts/download_external_data.py --config configs/mi_evento.yaml --only dem
    python3 scripts/download_external_data.py --config configs/mi_evento.yaml --only water
    python3 scripts/download_external_data.py --config configs/mi_evento.yaml --only flood

    # Forzar re-descarga aunque el archivo ya exista:
    python3 scripts/download_external_data.py --config configs/mi_evento.yaml --force
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import timedelta
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

# ── Endpoints ──────────────────────────────────────────────────────────────────
_PC_STAC  = "https://planetarycomputer.microsoft.com/api/stac/v1"
_EMS_WFS  = (
    "https://emergency.copernicus.eu/mapping/wfs"
    "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
    "&TYPENAMES={layer}&SRSNAME=urn:ogc:def:crs:EPSG::4326&outputFormat=json"
)
_EMS_API  = "https://emergency.copernicus.eu/mapping/rest/api/v2/activations"
_GFM_BASE = "https://global-flood.emergency.copernicus.eu/api/v1"
_SNCZI_WFS = (
    "https://sig.mapama.gob.es/wfs/cliente.aspx"
    "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
    "&TYPENAMES={layer}"
    "&SRSNAME=urn:ogc:def:crs:EPSG::4326"
    "&BBOX={south},{west},{north},{east},urn:ogc:def:crs:EPSG::4326"
    "&outputFormat=application/json&count=500"
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers genéricos
# ══════════════════════════════════════════════════════════════════════════════

def _bbox_wgs84(cfg) -> tuple[float, float, float, float]:
    """
    Devuelve (west, south, east, north) en WGS-84 a partir del AOI del config.
    Los GeoJSON siempre están en WGS-84 (RFC 7946), no hace falta transformar.
    """
    from shapely.geometry import shape

    aoi_path = Path(cfg.aoi_geojson)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI GeoJSON no encontrado: {aoi_path}")
    with open(aoi_path) as f:
        geojson = json.load(f)

    if geojson.get("type") == "FeatureCollection":
        geom = shape(geojson["features"][0]["geometry"])
    elif geojson.get("type") == "Feature":
        geom = shape(geojson["geometry"])
    else:
        geom = shape(geojson)

    minx, miny, maxx, maxy = geom.bounds
    buf = 0.05  # ~5 km de margen
    return minx - buf, miny - buf, maxx + buf, maxy + buf


def _event_date_range(cfg, margin_days: int = 14) -> tuple[str, str]:
    """
    Devuelve (start, end) como strings ISO abarcando todos los targets
    del config ± margin_days (para ampliar la búsqueda de activaciones EMS).
    """
    from datetime import date as _date
    targets = cfg.time.targets
    start = _date.fromisoformat(str(targets[0][0]))  - timedelta(days=margin_days)
    end   = _date.fromisoformat(str(targets[-1][1])) + timedelta(days=margin_days)
    return str(start), str(end)


def _resolve_output_path(cfg_path_str: str, data_dir: Path, suffix: str) -> Path:
    """
    Resuelve la ruta de salida:
    - Si el YAML define una ruta (no vacía), la usa directamente.
    - Si está vacía, genera un nombre por defecto basado en run_name.
    """
    if cfg_path_str:
        p = Path(cfg_path_str)
        # Si es relativa, la hacemos relativa al data_dir del proyecto
        return p if p.is_absolute() else p
    # Fallback basado en run_name
    return data_dir / suffix


def _fetch_json(url: str, timeout: int = 30) -> dict | None:
    """GET a URL; devuelve el JSON parseado o None si falla."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("GET %s → %s", url, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 1. DEM Copernicus GLO-30
# ══════════════════════════════════════════════════════════════════════════════

def download_dem(cfg, out_path: Path) -> None:
    """Descarga y fusiona los tiles del DEM Copernicus GLO-30 que cubren el AOI."""
    if out_path.exists():
        LOGGER.info("DEM ya existe: %s — omitiendo.", out_path)
        return

    LOGGER.info("Descargando DEM Copernicus GLO-30 para el AOI…")
    try:
        import pystac_client
        import planetary_computer
        import rasterio
        from rasterio.merge import merge as rio_merge
        import rioxarray as rxr
    except ImportError as exc:
        LOGGER.error(
            "Dependencia faltante: %s\n"
            "Instala con: pip install pystac-client planetary-computer rasterio rioxarray",
            exc,
        )
        sys.exit(1)

    bbox = _bbox_wgs84(cfg)
    client = pystac_client.Client.open(_PC_STAC, modifier=planetary_computer.sign_inplace)
    items = list(client.search(collections=["cop-dem-glo-30"], bbox=bbox, limit=50).items())
    if not items:
        LOGGER.error("No se encontraron tiles DEM para el AOI.")
        sys.exit(1)

    LOGGER.info("  %d tile(s) encontrados. Fusionando…", len(items))
    src_files = [rasterio.open(planetary_computer.sign(it).assets["data"].href) for it in items]
    mosaic_arr, mosaic_transform = rio_merge(src_files, bounds=bbox, res=0.0003)  # ~30 m
    nodata_val = src_files[0].nodata
    for src in src_files:
        src.close()

    mosaic_2d = mosaic_arr[0].astype(np.float32)
    if nodata_val is not None:
        mosaic_2d[mosaic_2d == nodata_val] = np.nan

    tmp = Path(tempfile.mktemp(suffix=".tif"))
    with rasterio.open(tmp, "w", driver="GTiff", dtype="float32", count=1,
                       crs="EPSG:4326", transform=mosaic_transform,
                       height=mosaic_2d.shape[0], width=mosaic_2d.shape[1],
                       compress="deflate", tiled=True) as dst:
        dst.write(mosaic_2d, 1)

    da = rxr.open_rasterio(str(tmp), masked=True).squeeze(drop=True)
    da = da.rio.reproject(cfg.crs_salida)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(str(out_path), compress="deflate", tiled=True, dtype="float32")
    tmp.unlink(missing_ok=True)
    LOGGER.info("  ✓ DEM guardado → %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 2. ESA WorldCover 2021 — clase agua permanente
# ══════════════════════════════════════════════════════════════════════════════

def download_worldcover_water(cfg, out_path: Path) -> None:
    """Descarga ESA WorldCover 2021 y extrae la clase 80 (agua permanente)."""
    if out_path.exists():
        LOGGER.info("WorldCover agua ya existe: %s — omitiendo.", out_path)
        return

    LOGGER.info("Descargando ESA WorldCover 2021 (clase agua=80)…")
    try:
        import pystac_client
        import planetary_computer
        import rasterio
        from rasterio.merge import merge as rio_merge
        import rioxarray as rxr
    except ImportError as exc:
        LOGGER.error("Dependencia faltante: %s", exc)
        sys.exit(1)

    bbox = _bbox_wgs84(cfg)
    client = pystac_client.Client.open(_PC_STAC, modifier=planetary_computer.sign_inplace)

    items = list(client.search(
        collections=["esa-worldcover"], bbox=bbox,
        query={"worldcover:year": {"eq": 2021}}, limit=20,
    ).items())
    if not items:  # fallback sin filtro de año
        items = list(client.search(collections=["esa-worldcover"], bbox=bbox, limit=20).items())

    if not items:
        LOGGER.warning("No se encontraron tiles WorldCover. Etiqueta de agua omitida.")
        return

    LOGGER.info("  %d tile(s) encontrados. Fusionando…", len(items))
    src_files = [rasterio.open(planetary_computer.sign(it).assets["map"].href) for it in items]
    mosaic_arr, mosaic_transform = rio_merge(src_files, bounds=bbox, res=0.00009)  # ~10 m
    for src in src_files:
        src.close()

    water_mask = (mosaic_arr[0] == 80).astype(np.uint8)
    tmp = Path(tempfile.mktemp(suffix=".tif"))
    with rasterio.open(tmp, "w", driver="GTiff", dtype="uint8", count=1,
                       crs="EPSG:4326", transform=mosaic_transform,
                       height=water_mask.shape[0], width=water_mask.shape[1],
                       compress="deflate", tiled=True) as dst:
        dst.write(water_mask, 1)

    import rioxarray as rxr
    da = rxr.open_rasterio(str(tmp), masked=False).squeeze(drop=True)
    da = da.rio.reproject(cfg.crs_salida)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.astype("uint8").rio.to_raster(str(out_path), compress="deflate", tiled=True, dtype="uint8")
    tmp.unlink(missing_ok=True)
    LOGGER.info("  ✓ WorldCover agua guardado → %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Máscara de inundación — búsqueda multi-fuente
# ══════════════════════════════════════════════════════════════════════════════

def _is_flood_feature(feature: dict) -> bool:
    """Devuelve True si el feature GeoJSON corresponde a una zona inundada."""
    props = feature.get("properties", {}) or {}
    for key in ("obj_type", "class", "type", "flooded", "label", "event_type"):
        val = str(props.get(key, "")).lower()
        if any(kw in val for kw in ("flood", "water", "wet", "inun", "agua")):
            return True
    return True  # si no se puede determinar, incluir (mejor sobre-incluir)


# ── Fuente 1: Copernicus EMS Rapid Mapping ────────────────────────────────────

def _fetch_emsr_layers(emsr_id: str) -> list[dict]:
    """
    Descarga features de delineación de inundación para un código EMSR dado.
    Prueba todas las combinaciones habituales de AOI/producto/revisión.
    """
    features: list[dict] = []
    for aoi_n in range(1, 10):
        for prod in ("DEL", "GRA", "OBJ"):
            for rev in ("r1", "r2", "r3"):
                layer = f"{emsr_id}_AOI{aoi_n:02d}_{prod}_PRODUCT_{rev}_RTP01_v1"
                url   = _EMS_WFS.format(layer=layer)
                data  = _fetch_json(url, timeout=20)
                if data:
                    feats = data.get("features", [])
                    if feats:
                        LOGGER.info("    ✓ %s → %d features", layer, len(feats))
                        features.extend(feats)
                        if len(features) > 300:
                            return features
    return features


def _try_ems_rapid_mapping(bbox: tuple, dates: tuple[str, str]) -> list[dict] | None:
    """
    Busca activaciones EMSR en el AOI + ventana temporal y descarga las
    capas de delineación de inundación disponibles.
    """
    LOGGER.info("[Fuente 1] Copernicus EMS Rapid Mapping…")
    west, south, east, north = bbox
    start, end = dates

    # Intentar API REST del portal EMS para listar activaciones
    for url in [
        f"{_EMS_API}?bbox={west},{south},{east},{north}&dateFrom={start}&dateTo={end}&format=json&limit=20",
        f"{_EMS_API}?dateFrom={start}&dateTo={end}&type=flood&format=json&limit=40",
    ]:
        data = _fetch_json(url, timeout=30)
        if data:
            break

    emsr_ids: list[str] = []
    if data:
        items = data if isinstance(data, list) else data.get("activations", data.get("items", []))
        for act in items:
            code = act.get("activation_code") or act.get("code") or act.get("id", "")
            if str(code).upper().startswith("EMSR"):
                emsr_ids.append(str(code).upper())

    if emsr_ids:
        LOGGER.info("  Activaciones encontradas: %s", ", ".join(emsr_ids))
    else:
        LOGGER.info("  ✗ No se encontraron activaciones EMSR en el período.")
        return None

    all_features: list[dict] = []
    for emsr_id in emsr_ids:
        LOGGER.info("  Descargando capas de %s…", emsr_id)
        feats = _fetch_emsr_layers(emsr_id)
        all_features.extend(feats)

    if not all_features:
        LOGGER.info("  ✗ Ninguna capa de delineación disponible en EMS.")
        return None

    LOGGER.info("  ✓ EMS: %d features en total.", len(all_features))
    return all_features


# ── Fuente 2: Copernicus Global Flood Monitoring (GFM) ───────────────────────

def _try_gfm(bbox: tuple, dates: tuple[str, str]) -> list[dict] | None:
    """
    Consulta el API de Copernicus GFM para obtener polígonos de inundación
    dentro del AOI y el rango de fechas del evento.
    """
    LOGGER.info("[Fuente 2] Copernicus Global Flood Monitoring (GFM)…")
    west, south, east, north = bbox
    start, end = dates

    for url in [
        f"{_GFM_BASE}/flood-events?bbox={west},{south},{east},{north}&start_date={start}&end_date={end}&format=geojson",
        f"{_GFM_BASE}/products?bbox={west},{south},{east},{north}&date_from={start}&date_to={end}",
    ]:
        data = _fetch_json(url, timeout=30)
        if data:
            feats = data.get("features", [])
            if feats:
                LOGGER.info("  ✓ GFM: %d features.", len(feats))
                return feats

    LOGGER.info("  ✗ GFM: sin datos para este AOI/fecha.")
    return None


# ── Fuente 3: SNCZI MITECO (capa estática de peligrosidad T=10) ──────────────

def _try_snczi(bbox: tuple) -> list[dict] | None:
    """
    Descarga la capa de inundabilidad T=10 del SNCZI (MITECO) como proxy
    estático. Solo aplica si el AOI está en España. Último recurso.
    """
    LOGGER.info("[Fuente 3] SNCZI MITECO — peligrosidad T=10 (capa estática)…")
    west, south, east, north = bbox

    # España mainland: lon [-9.5, 4.4], lat [36, 44]
    if not (-10 < west and east < 5 and 35 < south and north < 45):
        LOGGER.info("  ✗ AOI fuera de España — SNCZI no aplicable.")
        return None

    candidate_layers = [
        "ows:LAMINAAGUA_TR10_CUENCAS_MEDITERRANEAS",
        "ows:LAMINAAGUA_TR10",
        "ows:LAMINAINUNDACION_T10",
    ]
    for layer in candidate_layers:
        url = _SNCZI_WFS.format(
            layer=layer, west=west, south=south, east=east, north=north,
        )
        data = _fetch_json(url, timeout=45)
        if data:
            feats = data.get("features", [])
            if feats:
                LOGGER.info("  ✓ SNCZI [%s]: %d features.", layer, len(feats))
                return feats

    LOGGER.info("  ✗ SNCZI WFS sin datos accesibles.")
    return None


# ── Orquestador ───────────────────────────────────────────────────────────────

def download_flood_mask(cfg, out_path: Path) -> None:
    """
    Busca y descarga automáticamente una máscara de inundación para el evento
    definido en el config. Las fuentes se prueban en orden de prioridad:

      1. Copernicus EMS Rapid Mapping  — activaciones EMSR por AOI+fecha
      2. Copernicus GFM                — producto SAR dinámico
      3. SNCZI MITECO                  — peligrosidad T=10 (solo España)
      4. Fallback                      — GeoJSON vacío + aviso detallado
    """
    if out_path.exists():
        LOGGER.info("Máscara de inundación ya existe: %s — omitiendo.", out_path)
        return

    LOGGER.info("=" * 62)
    LOGGER.info("Buscando máscara de inundación (multi-fuente)…")
    LOGGER.info("  run_name: %s", cfg.run_name)
    LOGGER.info("  AOI:      %s", cfg.aoi_geojson)
    LOGGER.info("=" * 62)

    bbox  = _bbox_wgs84(cfg)
    dates = _event_date_range(cfg, margin_days=14)

    LOGGER.info("  bbox:   W=%.3f S=%.3f E=%.3f N=%.3f", *bbox)
    LOGGER.info("  fechas: %s → %s (±14d del evento)", *dates)
    LOGGER.info("-" * 62)

    features: list[dict] | None = None
    source_name = ""

    for try_fn, name in [
        (lambda: _try_ems_rapid_mapping(bbox, dates), "Copernicus EMS Rapid Mapping"),
        (lambda: _try_gfm(bbox, dates),               "Copernicus GFM"),
        (lambda: _try_snczi(bbox),                    "SNCZI MITECO T=10"),
    ]:
        features = try_fn()
        if features:
            source_name = name
            break

    # ── Fallback ─────────────────────────────────────────────────────────────
    if not features:
        LOGGER.warning("=" * 62)
        LOGGER.warning("⚠  Ninguna fuente de máscara de inundación disponible.")
        LOGGER.warning("   El pipeline continuará con pseudo-etiquetas automáticas.")
        LOGGER.warning("")
        LOGGER.warning("   Puedes añadir etiquetas manualmente desde:")
        LOGGER.warning("   · EMS:   https://emergency.copernicus.eu/mapping/")
        LOGGER.warning("   · GFM:   https://global-flood.emergency.copernicus.eu/")
        LOGGER.warning("   · GloFAS:https://www.globalfloods.eu/")
        LOGGER.warning("   Coloca el GeoJSON en: %s", out_path)
        LOGGER.warning("=" * 62)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "type": "FeatureCollection", "features": [],
                "_note": (
                    f"Ninguna fuente de inundación disponible para run_name={cfg.run_name}. "
                    "Sustituye este archivo con la máscara real o elimina la ruta en el YAML."
                ),
            }, f, indent=2, ensure_ascii=False)
        return

    # ── Guardar ───────────────────────────────────────────────────────────────
    import geopandas as gpd
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out_path), driver="GeoJSON")
    LOGGER.info("=" * 62)
    LOGGER.info("✓ Máscara guardada  [fuente: %s]", source_name)
    LOGGER.info("  %d features → %s", len(gdf), out_path)
    LOGGER.info("=" * 62)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Descarga datos externos para el pipeline floodprob.\n"
            "Las rutas de salida se leen del YAML (dem.path, external_labels.*).\n\n"
            "Productos:\n"
            "  dem   — DEM Copernicus GLO-30\n"
            "  water — ESA WorldCover 2021 (agua permanente)\n"
            "  flood — Máscara de inundación (EMS → GFM → SNCZI → fallback)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Ruta al fichero YAML (p.ej. configs/malaga_dana_2025.yaml)",
    )
    parser.add_argument(
        "--only", choices=["dem", "water", "flood"], default=None,
        help="Descargar solo un producto (omitir = todos)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Forzar re-descarga aunque el archivo ya exista",
    )
    args = parser.parse_args()

    try:
        from floodprob.config import load_config
    except ImportError:
        LOGGER.error("floodprob no instalado. Ejecuta: pip install -e '.[compare]'")
        sys.exit(1)

    cfg = load_config(args.config)
    data_dir = Path(cfg.outputs.data_dir)

    # ── Rutas de salida: vienen del YAML; si están vacías, usa defaults por run_name
    dem_out   = _resolve_output_path(
        cfg.dem.path, data_dir, f"dem/{cfg.run_name}_dem.tif"
    )
    water_out = _resolve_output_path(
        cfg.external_labels.water, data_dir, f"labels/{cfg.run_name}_worldcover_water.tif"
    )
    flood_out = _resolve_output_path(
        cfg.external_labels.flood, data_dir, f"labels/{cfg.run_name}_flood_mask.geojson"
    )

    LOGGER.info("run_name : %s", cfg.run_name)
    LOGGER.info("DEM      : %s", dem_out)
    LOGGER.info("Agua     : %s", water_out)
    LOGGER.info("Inundac. : %s", flood_out)

    if args.force:
        for p in [dem_out, water_out, flood_out]:
            if p.exists():
                p.unlink()
                LOGGER.info("Eliminado para re-descarga: %s", p)

    only = args.only
    if only is None or only == "dem":
        download_dem(cfg, dem_out)
    if only is None or only == "water":
        download_worldcover_water(cfg, water_out)
    if only is None or only == "flood":
        download_flood_mask(cfg, flood_out)

    LOGGER.info("")
    LOGGER.info("── Resumen ──────────────────────────────────────────────")
    LOGGER.info("  DEM:         %s  [existe=%s]", dem_out,   dem_out.exists())
    LOGGER.info("  Agua (WC):   %s  [existe=%s]", water_out, water_out.exists())
    LOGGER.info("  Inundación:  %s  [existe=%s]", flood_out, flood_out.exists())
    if flood_out.exists():
        with open(flood_out) as f:
            n = len(json.load(f).get("features", []))
        if n == 0:
            LOGGER.warning("  ⚠ Máscara de inundación VACÍA — se usarán pseudo-etiquetas.")
        else:
            LOGGER.info("  ✓ Máscara de inundación: %d features.", n)


if __name__ == "__main__":
    main()
