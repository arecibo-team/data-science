"""Export harmonized detections for the web map.

Output under web/public/data:
    index.json                  format version, enums, day and month index, grid settings, incident summary
    months/YYYY-MM.bin.gz       detections of one UTC month: gzipped binary columns (see _encode_month)
    grid/YYYY-MM.json.gz        per-day grid cell counts of one month, used for long date ranges
    incidents.json.gz           GeoJSON FeatureCollection of all European incidents

Field codes: each categorical column is stored as a small integer. Code 0 is always "missing",
and the value lists are written to index.json under "enums", so the web app never hardcodes them.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import struct
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from . import config as C
from .match import incident_detection_counts

log = logging.getLogger("firms.export")

FORMAT_VERSION = 2
GRID_DEG = 1.0  # grid cell size for long-range overview, in degrees

# Binary column order inside a month file. The web loader reads them in exactly this order.
COLUMNS = [
    ("lat", "<f4"), ("lon", "<f4"), ("frp", "<f4"), ("bt", "<f4"), ("cluster", "<i4"),
    ("minute", "<u2"), ("incident", "<u2"),
    ("day", "u1"), ("sensor", "u1"), ("platform", "u1"), ("product", "u1"),
    ("daynight", "u1"), ("confidence", "u1"), ("status", "u1"),
]

SIMPLIFY_DEG = 0.002  # ~200 m; plenty for country-scale map views and keeps the file small


def _enums(h: pd.DataFrame) -> dict[str, list]:
    """Value lists for categorical columns. Index 0 is always None (missing)."""
    return {
        "sensor": [None, "MODIS", "VIIRS"],
        "platform": [None, *sorted(h["platform"].dropna().unique().tolist())],
        "product": [None, *sorted(h["product"].dropna().unique().tolist())],
        "daynight": [None, "D", "N"],
        "confidence": [None, "low", "nominal", "high"],
        "status": [None, "both", "modis_only", "viirs_only"],
    }


def _code(s: pd.Series, values: list) -> np.ndarray:
    """Map values to codes: values[k] -> k, missing -> 0. Unknown values are an error."""
    codes = pd.Categorical(s, categories=values[1:]).codes.astype(np.int64)
    unknown = (codes < 0) & s.notna().to_numpy()
    if unknown.any():
        raise ValueError(f"unexpected {s.name} values: {sorted(set(s[unknown].tolist()))}")
    return (codes + 1).astype("u1")


def _encode_month(part: pd.DataFrame, enums: dict[str, list], incident_ids: list[str]) -> bytes:
    """One month as: uint32 header length, JSON header, then each column as raw little-endian bytes.

    The header holds the row count and this month's incident id table; incident codes are 1-based
    positions in that table, 0 meaning not confirmed.
    """
    t = part["acq_time_utc"]
    code_of = {v: i + 1 for i, v in enumerate(incident_ids)}
    cols = {
        "lat": part["lat"].to_numpy(dtype="<f4"),
        "lon": part["lon"].to_numpy(dtype="<f4"),
        "frp": part["frp_mw"].to_numpy(dtype="<f4", na_value=np.nan),
        "bt": part["bt_mid_k"].to_numpy(dtype="<f4", na_value=np.nan),
        "cluster": part["cluster_id"].fillna(-1).to_numpy(dtype="<i4"),
        "minute": (t.dt.hour * 60 + t.dt.minute).to_numpy(dtype="<u2"),
        "incident": part["incident_id"].map(code_of).fillna(0).to_numpy(dtype="<u2"),
        "day": t.dt.day.to_numpy(dtype="u1"),
        "sensor": _code(part["sensor"], enums["sensor"]),
        "platform": _code(part["platform"], enums["platform"]),
        "product": _code(part["product"], enums["product"]),
        "daynight": _code(part["daynight"], enums["daynight"]),
        "confidence": _code(part["confidence_class"], enums["confidence"]),
        "status": _code(part["harmonized_status"], enums["status"]),
    }
    header = json.dumps({"n": len(part), "incidents": incident_ids}, separators=(",", ":")).encode()
    body = b"".join(np.ascontiguousarray(cols[name], dtype=dt).tobytes() for name, dt in COLUMNS)
    return struct.pack("<I", len(header)) + header + body


def _grid_month(part: pd.DataFrame, cell: float) -> dict:
    """Per-day counts per grid cell: [row, col, modis, viirs, confirmed_modis, confirmed_viirs]."""
    is_modis = (part["sensor"] == "MODIS").astype(int)
    is_viirs = (part["sensor"] == "VIIRS").astype(int)
    confirmed = part["incident_id"].notna().astype(int)
    df = pd.DataFrame({
        "date": part["acq_time_utc"].dt.strftime("%Y-%m-%d"),
        "r": np.floor((part["lat"] + 90) / cell).astype(int),
        "c": np.floor((part["lon"] + 180) / cell).astype(int),
        "m": is_modis,
        "v": is_viirs,
        "cm": confirmed * is_modis,
        "cv": confirmed * is_viirs,
    })
    summed = df.groupby(["date", "r", "c"]).sum().reset_index()
    days = {
        date: sub[["r", "c", "m", "v", "cm", "cv"]].values.tolist()
        for date, sub in summed.groupby("date")
    }
    return {"cell_deg": cell, "days": days}


def _simplified_incidents(incidents: list[dict], counts: dict[str, int]) -> list[dict]:
    """Ready-to-serve GeoJSON features, one per incident."""
    feats = []
    for f in incidents:
        g = make_valid(shape(f["geometry"]))
        if g.geom_type not in ("Polygon", "MultiPolygon"):
            parts = [p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
            g = unary_union(parts) if parts else None
        if g is None or g.is_empty:
            continue
        g = g.simplify(SIMPLIFY_DEG, preserve_topology=True)
        if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        props = dict(f["properties"])
        props["detections"] = int(counts.get(props["incident_id"], 0))
        feats.append({"type": "Feature", "properties": props, "geometry": _round_coords(mapping(g))})
    return feats


def _round_coords(geom: dict, nd: int = 4) -> dict:
    def walk(c):
        if isinstance(c[0], (int, float)):
            return [round(c[0], nd), round(c[1], nd)]
        return [walk(x) for x in c]
    return {"type": geom["type"], "coordinates": walk(geom["coordinates"])}


def _write_gz(path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data, compresslevel=6))


def _clear(path) -> None:
    """Remove a generated directory so files from an older export cannot linger."""
    if path.is_dir():
        shutil.rmtree(path)


def export_web(h: pd.DataFrame, incidents: list[dict] | None = None, out_dir=C.WEB_DATA_DIR) -> dict:
    incidents = incidents or []
    if "incident_id" not in h:  # frames built without incident matching: all unconfirmed
        h = h.assign(incident_id=None)
    counts = incident_detection_counts(h["incident_id"])

    for old in ("days", "incident-days"):  # formats from earlier exports
        _clear(out_dir / old)
    _clear(out_dir / "months")
    _clear(out_dir / "grid")
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = _simplified_incidents(incidents, counts)
    _write_gz(out_dir / "incidents.json.gz", json.dumps(
        {"type": "FeatureCollection", "features": feats}, separators=(",", ":")).encode())

    index: dict = {
        "version": FORMAT_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "grid_cell_deg": GRID_DEG,
        "incidents": {
            "file": "incidents.json.gz",
            "count": len(feats),
            "confirmed": int(sum(counts.values())),
            "sources": sorted({f["properties"]["source"] for f in incidents}),
        },
        "enums": {},
        "months": {},
        "days": {},
    }
    if h.empty:
        index["enums"] = _enums(h.assign(platform=[], product=[]))
        (out_dir / "index.json").write_text(json.dumps(index, indent=1))
        return index

    enums = _enums(h)
    index["enums"] = enums
    h = h.assign(_month=h["acq_time_utc"].dt.strftime("%Y-%m"), _day=h["acq_time_utc"].dt.strftime("%Y-%m-%d"))

    for month, part in h.groupby("_month", sort=True):
        ids = sorted(part["incident_id"].dropna().unique().tolist())
        if len(ids) > 65535:
            raise ValueError(f"{month}: {len(ids)} incidents exceed the 16-bit code space")
        _write_gz(out_dir / "months" / f"{month}.bin.gz", _encode_month(part, enums, ids))
        _write_gz(out_dir / "grid" / f"{month}.json.gz",
                  json.dumps(_grid_month(part, GRID_DEG), separators=(",", ":")).encode())
        index["months"][month] = {
            "file": f"months/{month}.bin.gz",
            "grid": f"grid/{month}.json.gz",
            "rows": int(len(part)),
            "days": sorted(part["_day"].unique().tolist()),
        }

    for date, day in h.groupby("_day", sort=True):
        s = day["sensor"].value_counts().to_dict()
        index["days"][date] = {
            "total": int(len(day)),
            "MODIS": int(s.get("MODIS", 0)),
            "VIIRS": int(s.get("VIIRS", 0)),
            "matched": int((day["harmonized_status"] == "both").sum()),
            "confirmed": int(day["incident_id"].notna().sum()),
        }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    log.info("exported %d days in %d months to %s", len(index["days"]), len(index["months"]), out_dir)
    return index


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .incidents import load_incidents
    h = pd.read_parquet(C.ANALYSIS_DIR / "harmonized.parquet")
    export_web(h, load_incidents())


if __name__ == "__main__":
    main()
