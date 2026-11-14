"""Writing pipeline results into SQLite."""
from __future__ import annotations

import json
import logging
import sqlite3

import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from .. import db

log = logging.getLogger("server.store")

SIMPLIFY_DEG = 0.002  # ~200 m; plenty for country-scale map views and keeps responses small
CHUNK = 200_000


def _round_coords(geom: dict, nd: int = 4) -> dict:
    def walk(c):
        if isinstance(c[0], (int, float)):
            return [round(c[0], nd), round(c[1], nd)]
        return [walk(x) for x in c]
    return {"type": geom["type"], "coordinates": walk(geom["coordinates"])}


def simplify(geometry: dict) -> dict | None:
    """Valid, simplified polygon(s) for the map, or None when nothing drawable is left."""
    g = make_valid(shape(geometry))
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        parts = [p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
        g = unary_union(parts) if parts else None
    if g is None or g.is_empty:
        return None
    g = g.simplify(SIMPLIFY_DEG, preserve_topology=True)
    if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    return _round_coords(mapping(g))


def replace_incidents(con: sqlite3.Connection, source: str, features: list[dict]) -> int:
    """Replace all incidents of one source. Features that cannot be drawn are dropped."""
    rows = []
    for f in features:
        simple = simplify(f["geometry"])
        if simple is None:
            continue
        p = f["properties"]
        w, s, e, n = shape(simple).bounds
        rows.append((p["incident_id"], source, p.get("name"), p.get("country"), p["start_date"], p["end_date"],
                     p.get("area_ha"), p.get("buffer_km"), p.get("slack_days"),
                     w, s, e, n, json.dumps(f["geometry"], separators=(",", ":")),
                     json.dumps(simple, separators=(",", ":"))))
    con.execute("DELETE FROM incidents WHERE source=?", (source,))
    con.executemany("INSERT OR REPLACE INTO incidents(incident_id, source, name, country, start_date, end_date, "
                    "area_ha, buffer_km, slack_days, west, south, east, north, geometry, geometry_simple) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def load_incident_features(con: sqlite3.Connection) -> list[dict]:
    """Incidents as GeoJSON features, in the shape `firms.match.tag_incidents` expects."""
    feats = []
    for r in con.execute("SELECT * FROM incidents ORDER BY incident_id"):
        props = {"incident_id": r["incident_id"], "start_date": r["start_date"], "end_date": r["end_date"]}
        if r["buffer_km"] is not None:
            props["buffer_km"] = r["buffer_km"]
        if r["slack_days"] is not None:
            props["slack_days"] = r["slack_days"]
        feats.append({"type": "Feature", "properties": props, "geometry": json.loads(r["geometry"])})
    return feats


def _codes(s: pd.Series, values: list, what: str) -> np.ndarray:
    codes = pd.Categorical(s.astype(object), categories=values[1:]).codes.astype(np.int64)
    if ((codes < 0) & s.notna().to_numpy()).any():
        bad = sorted(set(s[(codes < 0) & s.notna()].tolist()))
        raise ValueError(f"unexpected {what} values: {bad}")
    return codes + 1


def _num(a: pd.Series) -> list:
    return [None if pd.isna(x) else float(x) for x in a]


def write_detections(h: pd.DataFrame) -> int:
    """Replace the detections table with `h` (the harmonized frame) without disturbing readers.

    The new table is filled on the side and swapped in inside one transaction, so the map keeps
    seeing the old data until the new data is complete. Incident tags start empty (see the tag step).
    """
    con = db.connect()
    try:
        con.execute("DROP TABLE IF EXISTS detections_build")
        con.execute(db.DETECTIONS_DDL.format(name="detections_build"))
        h = h.sort_values("acq_time_utc", kind="stable").reset_index(drop=True)
        ts = (h["acq_time_utc"].astype("datetime64[s, UTC]").astype("int64")).to_numpy()
        cols = {
            "sensor": _codes(h["sensor"], db.SENSORS, "sensor"),
            "platform": _codes(h["platform"], db.PLATFORMS, "platform"),
            "product": _codes(h["product"], db.PRODUCTS, "product"),
            "daynight": _codes(h["daynight"], db.DAYNIGHT, "daynight"),
            "confidence": _codes(h["confidence_class"], db.CONFIDENCE, "confidence"),
            "status": _codes(h["harmonized_status"], db.STATUS, "status"),
        }
        cluster = h["cluster_id"]
        for a in range(0, len(h), CHUNK):
            b = min(a + CHUNK, len(h))
            sl = slice(a, b)
            con.executemany(
                "INSERT INTO detections_build(ts, lat, lon, frp, bt, cluster_id, sensor, platform, product, "
                "daynight, confidence, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                zip(ts[sl].tolist(), h["lat"].iloc[sl].tolist(), h["lon"].iloc[sl].tolist(),
                    _num(h["frp_mw"].iloc[sl]), _num(h["bt_mid_k"].iloc[sl]),
                    [None if pd.isna(x) else int(x) for x in cluster.iloc[sl]],
                    cols["sensor"][sl].tolist(), cols["platform"][sl].tolist(), cols["product"][sl].tolist(),
                    cols["daynight"][sl].tolist(), cols["confidence"][sl].tolist(), cols["status"][sl].tolist()))
            con.commit()
        con.execute("BEGIN IMMEDIATE")
        con.execute("DROP TABLE IF EXISTS detections")
        con.execute("ALTER TABLE detections_build RENAME TO detections")
        for ddl in db.DETECTIONS_INDEXES:
            con.execute(ddl)
        con.commit()
        return len(h)
    finally:
        con.close()
