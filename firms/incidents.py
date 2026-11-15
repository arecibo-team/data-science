"""Independent European fire incidents, used to check NASA detections.

Sources (none derived from MODIS/VIIRS):
  * Copernicus EMS Rapid Mapping wildfire activations (category ``fire``). The AOI extents
    are event-level mapped regions from separate imagery, not burn perimeters.
  * Sentinel-2 burn scars computed by ``firms.sentinel2`` (dNBR), only for events whose
    scar was confirmed.

EFFIS burnt areas (``firms.effis``) are a third incident source for the forecast label: they are mapped from
MODIS and Sentinel-2 imagery, so they are not independent of NASA's sensors and must not be used to judge how
accurate the detections are, only to say which detections belong to a mapped wildfire.

Output: ``data/incidents/incidents.geojson`` (FeatureCollection, EPSG:4326) with
properties ``incident_id, source, name, country, start_date, end_date, area_ha``.
Sentinel-2 features also carry ``buffer_km`` and ``slack_days`` for ``firms.match``.

    python -m firms.incidents            # fetch EMS and collect Sentinel-2 confirmations
    python -m firms.sentinel2            # run the Sentinel-2 check first (optional, slow)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from shapely import wkt
from shapely.geometry import mapping

from . import config as C
from . import sentinel2

log = logging.getLogger("firms.incidents")

GEOJSON_OUT = C.INCIDENTS_DIR / "incidents.geojson"
RAW_EMS = C.INCIDENTS_DIR / "raw" / "ems"
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

def build_incidents() -> dict:
    """Fetch EMS, collect confirmed Sentinel-2 scars, write GEOJSON_OUT and return the collection."""
    ems = _normalize_ems(_ems_fire_codes(C.INCIDENTS_FROM))
    log.info("Copernicus EMS: %d European AOI polygons", len(ems))

    s2 = sentinel2.confirmed_features()
    log.info("Sentinel-2: %d confirmed burn scars", len(s2))

    fc = {
        "type": "FeatureCollection",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "features": ems + s2,
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
