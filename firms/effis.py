"""EFFIS burnt-area polygons (Copernicus Emergency Management Service, JRC), 2016 to today.

Fire perimeters (about 30 ha and larger, mapped from satellite imagery after the fire; fires on purely agricultural land
are not mapped) with a start and an end date. Used as a second incident source so a NASA detection inside one of them
counts as a real wildfire, which separates wildfires from crop burning and other small heat sources.

Not independent of NASA's sensors: the perimeters are mapped from MODIS (and since 2022 Sentinel-2) imagery, not from
the FIRMS hotspot list. Mapping coverage grows over the years (about 1,300 polygons in 2016, 20,000 in 2024), so a label
built from them is only consistent from about 2022. Source: https://maps.effis.emergency.copernicus.eu/effis (WFS,
layers ms:modis.ba.poly.<year> and ms:modis.ba.poly.season for the running year).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

import requests
from shapely.geometry import mapping, shape

from . import config as C

log = logging.getLogger("firms.effis")

URL = "https://maps.effis.emergency.copernicus.eu/effis"
RAW = C.INCIDENTS_DIR / "raw" / "effis"
FIRST_YEAR = 2016
BUFFER_KM = 0.5
SLACK_DAYS = 1
SIMPLIFY = 0.0003            # degrees, about 30 m: keeps matching exact enough and the DB small


def _download(layer: str, path) -> None:
    params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": f"ms:{layer}",
              "outputFormat": "application/json; subtype=geojson"}
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, timeout=600, headers={"User-Agent": "firms-europe-map/1.0"})
            r.raise_for_status()
            j = r.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(j))
            return
        except (requests.RequestException, ValueError) as exc:
            log.warning("EFFIS %s attempt %d failed: %s", layer, attempt + 1, exc)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"EFFIS layer {layer} could not be downloaded")


def fetch(today: dt.date | None = None) -> list[dict]:
    """Past years are downloaded once and cached, the running year is refreshed on every call."""
    today = today or dt.date.today()
    feats = []
    for y in range(FIRST_YEAR, today.year + 1):
        path = RAW / f"{y}.json"
        if y == today.year or not path.exists():
            _download("modis.ba.poly.season" if y == today.year else f"modis.ba.poly.{y}", path)
        feats.extend(json.loads(path.read_text())["features"])
    return feats


def normalize(raw: list[dict]) -> list[dict]:
    w, s, e, n = C.EUROPE_BBOX
    out = []
    for f in raw:
        p = f["properties"]
        try:
            geom = shape(f["geometry"])
            x0, y0, x1, y1 = geom.bounds
            if x1 < w or x0 > e or y1 < s or y0 > n:
                continue
            start, end = p["FIREDATE"][:10], (p.get("FINALDATE") or p["FIREDATE"])[:10]
            geom = geom.simplify(SIMPLIFY) if geom.is_valid else geom.buffer(0).simplify(SIMPLIFY)
            if geom.is_empty:
                continue
            out.append({"type": "Feature", "geometry": mapping(geom), "properties": {
                "incident_id": f"EFFIS-{p['id']}", "source": "EFFIS", "name": p.get("COMMUNE") or p.get("PROVINCE"),
                "country": p.get("COUNTRY"), "start_date": start, "end_date": max(start, end),
                "area_ha": float(p["AREA_HA"]) if p.get("AREA_HA") else None,
                "buffer_km": BUFFER_KM, "slack_days": SLACK_DAYS}})
        except (KeyError, ValueError, TypeError):
            continue
    return out
