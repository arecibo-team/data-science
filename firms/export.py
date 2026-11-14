"""Export harmonized detections to compact per-day JSON for the web map.

Output:
    web/public/data/index.json          dates, per-day counts by sensor, field list, incident summary
    web/public/data/days/YYYY-MM-DD.json  columnar rows: {"fields": [...], "rows": [[...], ...]}
    web/public/data/incident-days/YYYY-MM-DD.json  GeoJSON of European incidents active that UTC day
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import pandas as pd
from shapely import make_valid
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from . import config as C
from .match import incident_detection_counts

log = logging.getLogger("firms.export")

FIELDS = ["lat", "lon", "sensor", "platform", "product", "time_utc", "daynight",
          "confidence", "frp_mw", "bt_mid_k", "status", "cluster_id", "incident"]


def _day_frame(day: pd.DataFrame) -> dict:
    t = day["acq_time_utc"].dt.strftime("%H:%M")
    rows = pd.DataFrame({
        "lat": day["lat"].round(5),
        "lon": day["lon"].round(5),
        "sensor": day["sensor"],
        "platform": day["platform"],
        "product": day["product"],
        "time_utc": t,
        "daynight": day["daynight"],
        "confidence": day["confidence_class"],
        "frp_mw": day["frp_mw"].round(2),
        "bt_mid_k": day["bt_mid_k"].round(1),
        "status": day["harmonized_status"],
        "cluster_id": day["cluster_id"],
        "incident": day["incident_id"],
    })
    # NA -> null for JSON
    rows = rows.astype(object).where(rows.notna(), None)
    return {"fields": FIELDS, "rows": rows.values.tolist()}


SIMPLIFY_DEG = 0.002  # ~200 m; plenty for country-scale map views and keeps the file small


def _round_coords(geom: dict, nd: int = 4) -> dict:
    def walk(c):
        if isinstance(c[0], (int, float)):
            return [round(c[0], nd), round(c[1], nd)]
        return [walk(x) for x in c]
    return {"type": geom["type"], "coordinates": walk(geom["coordinates"])}


def _polygonal(geom: BaseGeometry) -> BaseGeometry | None:
    """Repair geometry and keep only its polygon parts (simplify can leave lines or points)."""
    g = make_valid(geom)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    parts = [p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else None


def _simplified_incidents(incidents: list[dict], counts: dict[str, int]) -> pd.DataFrame:
    """One row per incident: start/end dates plus its ready-to-serve GeoJSON feature."""
    rows = []
    for f in incidents:
        geom = _polygonal(shape(f["geometry"]))
        if geom is None or geom.is_empty:
            continue
        geom = geom.simplify(SIMPLIFY_DEG, preserve_topology=True)
        if geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        props = dict(f["properties"])
        props["detections"] = int(counts.get(props["incident_id"], 0))
        rows.append({
            "start": props["start_date"],
            "end": props["end_date"],
            "feature": {"type": "Feature", "properties": props,
                        "geometry": _round_coords(mapping(geom))},
        })
    return pd.DataFrame(rows, columns=["start", "end", "feature"])


def _write_incident_days(inc: pd.DataFrame, dates: list[str], out_dir) -> None:
    """Per UTC day, the incidents whose [start, end] window covers that day."""
    inc_dir = out_dir / "incident-days"
    inc_dir.mkdir(parents=True, exist_ok=True)
    for old in inc_dir.glob("*.json"):
        old.unlink()
    for date in dates:
        mask = (inc["start"] <= date) & (inc["end"] >= date)
        feats = inc.loc[mask, "feature"].tolist()
        (inc_dir / f"{date}.json").write_text(
            json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))


def export_web(h: pd.DataFrame, incidents: list[dict] | None = None, out_dir=C.WEB_DATA_DIR) -> dict:
    days_dir = out_dir / "days"
    days_dir.mkdir(parents=True, exist_ok=True)
    for old in days_dir.glob("*.json"):
        old.unlink()  # remove stale days from previous exports

    incidents = incidents or []
    if "incident_id" not in h:  # frames built without incident matching: all unconfirmed
        h = h.assign(incident_id=None)
    counts = incident_detection_counts(h["incident_id"])
    inc = _simplified_incidents(incidents, counts)
    _write_incident_days(inc, sorted(h["acq_time_utc"].dt.strftime("%Y-%m-%d").unique()) if not h.empty else [], out_dir)

    index: dict = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sensors": ["MODIS", "VIIRS"],
        "fields": FIELDS,
        "incidents": {
            "dir": "incident-days",
            "count": len(inc),
            "confirmed": int(sum(counts.values())),
            "sources": sorted({f["properties"]["source"] for f in incidents}),
        },
        "days": {},
    }
    if h.empty:
        (out_dir / "index.json").write_text(json.dumps(index, indent=1))
        return index

    dates = h["acq_time_utc"].dt.strftime("%Y-%m-%d")
    for date, day in h.groupby(dates, sort=True):
        payload = _day_frame(day)
        (days_dir / f"{date}.json").write_text(json.dumps(payload, separators=(",", ":")))
        counts = day["sensor"].value_counts().to_dict()
        index["days"][date] = {
            "total": int(len(day)),
            "MODIS": int(counts.get("MODIS", 0)),
            "VIIRS": int(counts.get("VIIRS", 0)),
            "matched": int((day["harmonized_status"] == "both").sum()),
            "confirmed": int(day["incident_id"].notna().sum()),
        }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    log.info("exported %d days to %s", len(index["days"]), out_dir)
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
