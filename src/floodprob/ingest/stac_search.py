from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any

import pystac
from pystac_client import Client

from floodprob.config import FiltersConfig, STACConfig
from floodprob.utils.io import with_retries

LOGGER = logging.getLogger(__name__)


def create_stac_client(stac_cfg: STACConfig) -> Client:
    headers: dict[str, str] = {}
    if stac_cfg.auth.type in {"bearer", "oauth"}:
        if not stac_cfg.auth.token_env:
            raise ValueError("auth.token_env is required for bearer/oauth auth")
        token = os.getenv(stac_cfg.auth.token_env, "")
        if not token:
            raise ValueError(
                f"Missing token in env var {stac_cfg.auth.token_env} for STAC auth"
            )
        headers["Authorization"] = f"Bearer {token}"

    client = with_retries(
        Client.open,
        stac_cfg.api_url,
        headers=headers,
        retries=3,
        sleep_s=2.0,
        label="stac_client_open",
    )
    return client


def list_collection_ids(client: Client) -> list[str]:
    return sorted([c.id for c in client.get_collections()])


def collection_exists(client: Client, collection_id: str) -> bool:
    try:
        client.get_collection(collection_id)
        return True
    except Exception:  # noqa: BLE001
        return False


def _datetime_range(window: tuple[date, date]) -> str:
    start, end = window
    return f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z"


def _base_query(sensor: str, filters: FiltersConfig) -> dict[str, dict[str, Any]]:
    query: dict[str, dict[str, Any]] = {}
    if sensor == "s2":
        query["eo:cloud_cover"] = {"lte": float(filters.s2_max_cloud)}
    if sensor == "s1" and filters.s1_orbit_state != "both":
        query["sat:orbit_state"] = {"eq": filters.s1_orbit_state}
    return query


def _matches_s1_filters(item: pystac.Item, filters: FiltersConfig) -> bool:
    if filters.s1_orbit_state != "both":
        orbit_state = str(item.properties.get("sat:orbit_state", "")).lower()
        if orbit_state and orbit_state != filters.s1_orbit_state:
            return False

    requested = {p.upper() for p in filters.s1_polarizations}
    props_pols = item.properties.get("sar:polarizations", [])
    item_pols = {str(p).upper() for p in props_pols if p is not None}
    if item_pols and not requested.issubset(item_pols):
        return False
    return True


def _matches_s2_filters(item: pystac.Item, filters: FiltersConfig) -> bool:
    cloud_cover = item.properties.get("eo:cloud_cover")
    if cloud_cover is None:
        return True
    try:
        return float(cloud_cover) <= float(filters.s2_max_cloud)
    except Exception:  # noqa: BLE001
        return True


def _sign_items_if_planetary(items: list[pystac.Item], api_url: str) -> list[pystac.Item]:
    if "planetarycomputer.microsoft.com" not in api_url.lower():
        return items
    try:
        import planetary_computer as pc  # type: ignore

        return [pc.sign(item) for item in items]
    except Exception:  # noqa: BLE001
        LOGGER.info(
            "planetary_computer package not available or signing failed; using raw hrefs."
        )
        return items


def search_sensor_items(
    client: Client,
    stac_cfg: STACConfig,
    sensor: str,
    collection: str,
    aoi_geojson: dict[str, Any],
    window: tuple[date, date],
    filters: FiltersConfig,
) -> list[pystac.Item]:
    query = _base_query(sensor, filters)
    dt = _datetime_range(window)
    LOGGER.info(
        "Searching STAC | sensor=%s collection=%s datetime=%s", sensor, collection, dt
    )

    search = with_retries(
        client.search,
        collections=[collection],
        intersects=aoi_geojson,
        datetime=dt,
        query=query if query else None,
        max_items=stac_cfg.max_items,
        retries=3,
        sleep_s=2.0,
        label=f"stac_search_{sensor}",
    )
    items = list(search.items())
    if sensor == "s1":
        items = [it for it in items if _matches_s1_filters(it, filters)]
    if sensor == "s2":
        items = [it for it in items if _matches_s2_filters(it, filters)]

    items = sorted(
        items,
        key=lambda i: str(
            i.properties.get("datetime")
            or i.properties.get("start_datetime")
            or i.id
        ),
    )
    items = _sign_items_if_planetary(items, stac_cfg.api_url)
    LOGGER.info("Search complete | sensor=%s items=%s", sensor, len(items))
    return items


def sample_item_asset_keys(items: list[pystac.Item], n: int = 3) -> dict[str, list[str]]:
    if not items:
        return {}
    sample = items[: max(1, min(n, len(items)))]
    out: dict[str, list[str]] = {}
    for it in sample:
        out[it.id] = sorted(list(it.assets.keys()))
    return out
