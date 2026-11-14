"""Run the offline pipeline after fetching: normalize -> harmonize -> analyze -> export.

    python -m firms.build                 # default match settings (1 km, 180 min)
    python -m firms.build --radius-km 0.75 --window-min 120
"""
from __future__ import annotations

import argparse
import logging

from . import config as C
from .export import export_web
from .harmonize import MatchParams, cluster_table, daily_summary, match_cross_sensor
from .incidents import load_incidents
from .match import incident_detection_counts, tag_incidents
from .normalize import load_normalized, normalize_all

log = logging.getLogger("firms.build")


def build(params: MatchParams) -> None:
    normalize_all()
    df = load_normalized()
    log.info("loaded %d unique detections (NRT/SP deduplicated)", len(df))

    h = match_cross_sensor(df, params)
    incidents = load_incidents()
    if not incidents:
        log.warning("no incidents file found; run `python -m firms.incidents` to confirm detections")
    h["incident_id"] = tag_incidents(h, incidents)
    log.info("confirmed %d detections inside %d incidents",
             int(h["incident_id"].notna().sum()), len(incident_detection_counts(h["incident_id"])))

    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    h.to_parquet(C.ANALYSIS_DIR / "harmonized.parquet", index=False)
    cluster_table(h).to_csv(C.ANALYSIS_DIR / "clusters.csv", index=False)
    daily_summary(h).to_csv(C.ANALYSIS_DIR / "daily_summary.csv", index=False)

    export_web(h, incidents)
    n_both = int((h["harmonized_status"] == "both").sum())
    log.info("matched %d detections into clusters (%d MODIS-only/VIIRS-only left single)",
             n_both, len(h) - n_both)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--radius-km", type=float, default=1.0)
    p.add_argument("--window-min", type=int, default=180)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(MatchParams(args.radius_km, args.window_min))


if __name__ == "__main__":
    main()
