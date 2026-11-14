"""Export harmonized detections to compact per-day JSON for the web map.

Output:
    web/public/data/index.json          dates, per-day counts by sensor, field list
    web/public/data/days/YYYY-MM-DD.json  columnar rows: {"fields": [...], "rows": [[...], ...]}
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from . import config as C

log = logging.getLogger("firms.export")

FIELDS = ["lat", "lon", "sensor", "platform", "product", "time_utc", "daynight",
          "confidence", "frp_mw", "bt_mid_k", "status", "cluster_id"]


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
    })
    # NA -> null for JSON
    rows = rows.astype(object).where(rows.notna(), None)
    return {"fields": FIELDS, "rows": rows.values.tolist()}


def export_web(h: pd.DataFrame, out_dir=C.WEB_DATA_DIR) -> dict:
    days_dir = out_dir / "days"
    days_dir.mkdir(parents=True, exist_ok=True)
    for old in days_dir.glob("*.json"):
        old.unlink()  # remove stale days from previous exports

    index: dict = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sensors": ["MODIS", "VIIRS"],
        "fields": FIELDS,
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
        }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    log.info("exported %d days to %s", len(index["days"]), out_dir)
    return index


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    h = pd.read_parquet(C.ANALYSIS_DIR / "harmonized.parquet")
    export_web(h)


if __name__ == "__main__":
    main()
