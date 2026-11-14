"""Harvest European wildfire incidents from authoritative sources.

Sources:
  * EFFIS (JRC/Copernicus) burnt-area perimeters, >= ~30 ha, via the public WFS.
    Per-year layers for past years, the ``season`` layer for the current year.
  * Copernicus EMS Rapid Mapping wildfire activations (category ``fire``), using the
    AOI extents published on the EMS dashboard API.

Output: ``data/incidents/incidents.geojson`` (FeatureCollection, EPSG:4326) with
properties ``incident_id, source, name, country, start_date, end_date, area_ha``.

    python -m firms.incidents            # fetch / refresh all sources
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from shapely import wkt
from shapely.geometry import mapping, shape

from . import config as C

log = logging.getLogger("firms.incidents")

GEOJSON_OUT = C.INCIDENTS_DIR / "incidents.geojson"
RAW_EFFIS = C.INCIDENTS_DIR / "raw" / "effis"
RAW_EMS = C.INCIDENTS_DIR / "raw" / "ems"
PAGE = 1000
TIMEOUT = 120
UA = "firms-europe-map/1.0"


# ---------------------------------------------------------------- HTTP helpers

def _get_json(url: str, params: dict | None = None, retries: int = 3) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": UA})
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:  # network error or non-JSON body
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def _read_or_none(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


# ---------------------------------------------------------------- EFFIS

def _effis_layers(today: date) -> list[tuple[str, bool]]:
    """(layer name, always_refresh). Past years are immutable; the season layer keeps growing."""
    first = int(C.INCIDENTS_FROM[:4])
    layers = [(f"ms:modis.ba.poly.{y}", False) for y in range(first, today.year)]
    layers.append(("ms:modis.ba.poly.season", True))
    return layers


def _effis_layer_features(layer: str, refresh: bool) -> list[dict]:
    """Fetch every feature of one WFS layer, paging with startIndex. Cached per layer."""
    cache = RAW_EFFIS / f"{layer.split(':')[-1]}.json"
    if not refresh and (cached := _read_or_none(cache)) is not None:
        return cached["features"]
    feats: list[dict] = []
    start = 0
    while True:
        params = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeName": layer, "outputFormat": "application/json; subtype=geojson",
            "count": PAGE, "startIndex": start,
        }
        page = _get_json(C.EFFIS_WFS, params)
        batch = page.get("features", [])
        feats.extend(batch)
        log.info("EFFIS %s: %d features so far", layer, len(feats))
        if len(batch) < PAGE:
            break
        start += PAGE
    RAW_EFFIS.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"features": feats}))
    return feats


def _effis_date(value: str | None) -> str | None:
    return value[:10] if value else None


def _normalize_effis(feats: list[dict]) -> list[dict]:
    out: dict[str, dict] = {}
    for f in feats:
        p = f.get("properties") or {}
        country = (p.get("COUNTRY") or "").upper()
        if country not in C.EUROPE_COUNTRIES or f.get("geometry") is None:
            continue
        fid = f"EFFIS-{p.get('id')}"
        if fid in out:  # same fire can appear in a year layer and the season layer
            continue
        start = _effis_date(p.get("FIREDATE"))
        if not start:
            continue
        end = _effis_date(p.get("FINALDATE")) or start
        try:
            area = float(p.get("AREA_HA")) if p.get("AREA_HA") not in (None, "") else None
        except ValueError:
            area = None
        # EFFIS writes "N.A." for unknown places; drop those and fall back to the country.
        name_parts = [x for x in (p.get("COMMUNE"), p.get("PROVINCE")) if x and x.strip() != "N.A."]
        place = ", ".join(name_parts) or country
        out[fid] = {
            "type": "Feature",
            "properties": {
                "incident_id": fid,
                "source": "EFFIS",
                "name": f"Burnt area: {place}",
                "country": country,
                "start_date": start,
                "end_date": max(end, start),
                "area_ha": area,
            },
            "geometry": f["geometry"],
        }
    return list(out.values())


# ---------------------------------------------------------------- Copernicus EMS

def _ems_fire_codes(since: str) -> list[dict]:
    """Wildfire activations since `since`, from the public activations list (paginated)."""
    rows: list[dict] = []
    url: str | None = C.EMS_ACTIVATIONS
    params: dict | None = {"category": "fire", "limit": 200}
    while url:
        page = _get_json(url, params)
        params = None  # `next` already carries the query string
        rows.extend(page.get("results", []))
        url = page.get("next")
    return [r for r in rows if (r.get("activationTime") or "")[:10] >= since]


def _ems_detail(code: str) -> dict | None:
    cache = RAW_EMS / f"{code}.json"
    if (cached := _read_or_none(cache)) is not None:
        return cached or None
    try:
        data = _get_json(C.EMS_DASHBOARD, {"code": code})
    except RuntimeError as exc:
        # 403/404 are permanent for non-rapid (EMSN) activations: cache the refusal so reruns skip it.
        # Anything else (network trouble) is not cached and will be retried next run.
        log.warning("skipping EMS %s: %s", code, exc)
        if " 403 " in str(exc) or " 404 " in str(exc):
            RAW_EMS.mkdir(parents=True, exist_ok=True)
            cache.write_text("{}")
        return None
    results = data.get("results") or []
    detail = results[0] if results else {}
    RAW_EMS.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(detail))
    return detail or None


def _polygon_area_ha(geom) -> float:
    """Approximate area of a lon/lat polygon in hectares (local equirectangular scale)."""
    import math
    lat = geom.centroid.y
    km2_per_deg2 = 111.32 * 111.32 * math.cos(math.radians(lat))
    return round(geom.area * km2_per_deg2 * 100, 1)  # 1 km^2 = 100 ha


def _normalize_ems(codes: list[dict]) -> list[dict]:
    feats: list[dict] = []
    for act in codes:
        code = act.get("code")
        if not code:
            continue
        detail = _ems_detail(code)
        if not detail or detail.get("continent") != "Europe":
            continue
        start_src = detail.get("eventTime") or act.get("activationTime") or ""
        start = start_src[:10]
        end = max((act.get("activationTime") or start_src)[:10], start)
        country = ", ".join(c.get("name", "") for c in detail.get("countries", []) if c.get("name"))
        for aoi in detail.get("aois", []):
            extent = aoi.get("extent")
            if not extent:
                continue
            geom = wkt.loads(extent)
            minx, miny, maxx, maxy = geom.bounds
            lon_min, lat_min, lon_max, lat_max = C.EUROPE_BBOX
            if maxx < lon_min or minx > lon_max or maxy < lat_min or miny > lat_max:
                continue
            aoi_no = aoi.get("number", 1)
            feats.append({
                "type": "Feature",
                "properties": {
                    "incident_id": f"EMS-{code}-AOI{aoi_no}",
                    "source": "Copernicus EMS",
                    "name": f"{detail.get('name', code)} ({aoi.get('name') or f'AOI {aoi_no}'})",
                    "country": country,
                    "start_date": start,
                    "end_date": end,
                    "area_ha": _polygon_area_ha(geom),
                },
                "geometry": mapping(geom),
            })
    return feats


# ---------------------------------------------------------------- orchestration

def build_incidents(today: date | None = None) -> dict:
    """Fetch all sources, normalize, write GEOJSON_OUT and return the FeatureCollection."""
    today = today or date.today()
    effis_raw: list[dict] = []
    for layer, refresh in _effis_layers(today):
        effis_raw.extend(_effis_layer_features(layer, refresh))
    effis = _normalize_effis(effis_raw)
    log.info("EFFIS: %d European incidents", len(effis))

    since = C.INCIDENTS_FROM
    ems = _normalize_ems(_ems_fire_codes(since))
    log.info("Copernicus EMS: %d European AOI polygons", len(ems))

    fc = {
        "type": "FeatureCollection",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "features": effis + ems,
    }
    C.INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    GEOJSON_OUT.write_text(json.dumps(fc))
    return fc


def load_incidents() -> list[dict]:
    """Features from the last build; empty list when none have been fetched yet."""
    if not GEOJSON_OUT.exists():
        return []
    return json.loads(GEOJSON_OUT.read_text())["features"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fc = build_incidents()
    log.info("wrote %d incidents to %s", len(fc["features"]), GEOJSON_OUT)


if __name__ == "__main__":
    main()
