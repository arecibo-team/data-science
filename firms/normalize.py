"""Normalize raw FIRMS CSVs (MODIS and VIIRS) into one common schema.

Raw:        data/raw/<SOURCE>/<window>.csv
Normalized: data/normalized/<SOURCE>/<window>.parquet   (one file per raw file, idempotent)

Key differences handled here:
  * time: acq_date (YYYY-MM-DD) + acq_time (HHMM UTC, not zero-padded) -> acq_time_utc
  * confidence: MODIS 0-100 numeric -> low/nominal/high; VIIRS l/n/h -> low/nominal/high
  * brightness: MODIS brightness/bright_t31 and VIIRS bright_ti4/bright_ti5 are mapped to
    bt_mid_k / bt_secondary_k. These are the closest equivalents, not identical channels.
  * platform: Terra/Aqua for MODIS; SNPP/NOAA-20/NOAA-21 for VIIRS
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger("firms.normalize")

_VIIRS_CONF = {"l": "low", "n": "nominal", "h": "high"}


def confidence_class(raw: object) -> str | None:
    """Classify a raw confidence value. MODIS numeric: <30 low, 30-79 nominal, >=80 high."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().lower()
    if s in _VIIRS_CONF:
        return _VIIRS_CONF[s]
    try:
        v = float(s)
    except ValueError:
        return None
    if v < 30:
        return "low"
    if v < 80:
        return "nominal"
    return "high"


def _confidence_pct(raw: object) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return np.nan


def _parse_acq_time(date: pd.Series, hhmm: pd.Series) -> pd.Series:
    hhmm = hhmm.astype("string").str.strip().str.zfill(4)
    stamp = date.astype("string") + " " + hhmm.str[:2] + ":" + hhmm.str[2:]
    return pd.to_datetime(stamp, format="%Y-%m-%d %H:%M", utc=True)


def normalize_frame(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Turn one raw FIRMS frame (already read as CSV) into the common schema."""
    family, product = C.SOURCE_INFO[source]
    sensor = C.FAMILIES[family]["sensor"]

    out = pd.DataFrame(index=raw.index)
    out["sensor"] = sensor
    out["platform"] = raw["satellite"].astype("string").map(C.PLATFORM_NAMES).fillna(raw["satellite"].astype("string"))
    out["product"] = product
    out["lat"] = raw["latitude"].astype(float)
    out["lon"] = raw["longitude"].astype(float)
    out["acq_time_utc"] = _parse_acq_time(raw["acq_date"], raw["acq_time"])
    out["daynight"] = raw["daynight"].astype("string")

    conf = raw["confidence"]
    out["confidence_raw"] = conf.astype("string")
    out["confidence_class"] = conf.map(confidence_class).astype("string")
    out["confidence_pct"] = conf.map(_confidence_pct).astype(float)

    out["frp_mw"] = raw["frp"].astype(float)
    if sensor == "MODIS":
        out["bt_mid_k"] = raw["brightness"].astype(float)
        out["bt_secondary_k"] = raw["bright_t31"].astype(float)
    else:
        out["bt_mid_k"] = raw["bright_ti4"].astype(float)
        out["bt_secondary_k"] = raw["bright_ti5"].astype(float)

    out["scan_km"] = raw["scan"].astype(float)
    out["track_km"] = raw["track"].astype(float)
    out["pixel_area_km2"] = out["scan_km"] * out["track_km"]
    out["firms_version"] = raw["version"].astype("string")
    out["firms_type"] = raw["type"].astype("string") if "type" in raw.columns else pd.NA
    out["source"] = source

    # Stable natural key: the same physical detection appears in NRT and SP files.
    out["uid"] = (
        out["platform"].astype(str) + "|"
        + out["acq_time_utc"].dt.strftime("%Y%m%dT%H%M") + "|"
        + out["lat"].round(5).map("{:.5f}".format) + "|"
        + out["lon"].round(5).map("{:.5f}".format)
    )
    return out[C.SCHEMA].reset_index(drop=True)


_READ_DTYPES = {
    "acq_time": "string", "acq_date": "string", "confidence": "string",
    "satellite": "string", "daynight": "string", "version": "string", "type": "string",
}


def normalize_file(path: Path, source: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=_READ_DTYPES)
    if raw.empty:
        return pd.DataFrame(columns=C.SCHEMA)
    return normalize_frame(raw, source)


def normalize_all(force: bool = False) -> list[Path]:
    """Normalize every raw file. Skips files whose parquet output already exists."""
    written: list[Path] = []
    for raw_path in sorted(C.RAW_DIR.glob("*/*.csv")):
        source = raw_path.parent.name
        if source not in C.SOURCE_INFO:
            log.warning("skipping unknown source folder %s", raw_path.parent)
            continue
        out = C.NORM_DIR / source / (raw_path.stem + ".parquet")
        if out.exists() and not force:
            written.append(out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        df = normalize_file(raw_path, source)
        df.to_parquet(out, index=False)
        log.info("%-16s %-22s %d rows", source, raw_path.name, len(df))
        written.append(out)
    return written


def load_normalized(dedupe: bool = True) -> pd.DataFrame:
    """Load all normalized files into one frame, removing NRT/SP duplicates (SP wins)."""
    files = sorted(C.NORM_DIR.glob("*/*.parquet"))
    if not files:
        return pd.DataFrame(columns=C.SCHEMA)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if dedupe and not df.empty:
        df["_prio"] = (df["product"] != "SP").astype(int)  # 0 = SP preferred
        df = (df.sort_values(["_prio"], kind="stable")
                .drop_duplicates(subset="uid", keep="first")
                .drop(columns="_prio"))
    return df.sort_values("acq_time_utc").reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="re-normalize existing files")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    normalize_all(force=args.force)


if __name__ == "__main__":
    main()
