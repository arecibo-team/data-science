from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import NAMES
from .area import ISO
from .optimize import haversine_km

HERE = Path(__file__).resolve().parent
REGIONS = ["north", "east", "south", "west"]
GEO = json.loads((HERE / "regions_geo.json").read_text())
MIN_DETECTIONS = 20


def load_table() -> pd.DataFrame:
    df = pd.read_csv(HERE / "firefighters_regions.csv", dtype={"source_period": str}).set_index("iso2")
    for c in ["national_base", *REGIONS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["available"] = df[REGIONS].notna().all(axis=1) & df["national_base"].notna()
    raw_sum = df[REGIONS].sum(axis=1)
    df["rescaled"] = df["available"] & ((raw_sum - df["national_base"]).abs() > 0.05 * df["national_base"])
    for r in REGIONS:
        share = df[r] / raw_sum.replace(0, np.nan)
        df[f"{r}_share"] = share
        df[r] = np.where(df["rescaled"], (share * df["national_base"]).round(), df[r])
    return df


TABLE = load_table()


def _bearing_region(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    k = math.cos(math.radians(lat0))
    ang = np.degrees(np.arctan2(np.asarray(lat) - lat0, (np.asarray(lon) - lon0) * k)) % 360
    idx = np.select([(ang >= 45) & (ang < 135), (ang >= 135) & (ang < 225), (ang >= 225) & (ang < 315)], [0, 3, 2], 1)
    return np.array(REGIONS)[idx]


def region_of(iso2: str, lat: float, lon: float) -> str | None:
    g = GEO.get(iso2)
    if not g:
        return None
    return str(_bearing_region(g["centroid"][0], g["centroid"][1], np.array([lat]), np.array([lon]))[0])


def firefighters(iso2: str, region: str) -> float | None:
    if iso2 not in TABLE.index or not TABLE.loc[iso2, "available"]:
        return None
    return float(TABLE.loc[iso2, region])


def departure(iso2: str, lat: float, lon: float) -> tuple[str | None, float, float]:
    g = GEO.get(iso2)
    if not g:
        return None, lat, lon
    best = None
    for r, (alat, alon) in g["anchors"].items():
        share = TABLE.loc[iso2, f"{r}_share"] if iso2 in TABLE.index and TABLE.loc[iso2, "available"] else 0.25
        if share < 0.1:
            continue
        dist = float(haversine_km(alat, alon, lat, lon))
        if best is None or dist < best[0]:
            best = (dist, r, alat, alon)
    if best is None:
        return None, g["centroid"][0], g["centroid"][1]
    return best[1], best[2], best[3]


def _month_detections(ad, d: date) -> pd.DataFrame:
    i0, i1 = ad.idx(d.replace(day=1)), ad.idx(d)
    mask = (ad.d["day"] >= i0) & (ad.d["day"] <= i1)
    iso = np.array(ISO)[ad.d["country"][mask]]
    lat, lon = ad.d["lat"][mask].astype(float), ad.d["lon"][mask].astype(float)
    region = np.empty(len(iso), dtype=object)
    for code in np.unique(iso):
        g = GEO.get(code)
        sel = iso == code
        if g:
            region[sel] = _bearing_region(g["centroid"][0], g["centroid"][1], lat[sel], lon[sel])
    return pd.DataFrame({"iso2": iso, "region": region})



def regional_day(d: date, power: dict[str, float], fires: list[dict], ad) -> list[dict]:
    det = _month_detections(ad, d) if ad is not None else pd.DataFrame(columns=["iso2", "region"])
    det_counts = det.groupby(["iso2", "region"]).size()
    det_total = det.groupby("iso2").size()
    fire_rows = {}
    for f in fires:
        r = region_of(f["iso2"], f["lat"], f["lon"])
        key = (f["iso2"], r)
        cur = fire_rows.setdefault(key, {"active_fires": 0, "hotspots_24h": 0, "largest": None, "largest_n": -1})
        cur["active_fires"] += 1
        cur["hotspots_24h"] += int(f["hotspots_24h"])
        if f["detections"] > cur["largest_n"]:
            cur["largest_n"], cur["largest"] = f["detections"], f["name"]
    out = []
    for iso2 in sorted(power):
        row = TABLE.loc[iso2] if iso2 in TABLE.index else None
        available = bool(row is not None and row["available"])
        total = int(det_total.get(iso2, 0))
        for r in REGIONS:
            ff = float(row[r]) if available else None
            ff_share = float(row[f"{r}_share"]) if available else None
            n = int(det_counts.get((iso2, r), 0))
            fire_share = n / total if total else None
            if available and total >= MIN_DETECTIONS:
                factor = min(1.6, max(0.4, ff_share / max(fire_share, 0.02)))
                load = round(fire_share / ff_share, 2) if ff_share else None
            elif available:
                factor = min(1.3, max(0.7, math.sqrt(ff_share / 0.25)))
                load = None
            else:
                factor, load = 1.0, None
            fr = fire_rows.get((iso2, r), {})
            out.append({
                "iso2": iso2, "name": NAMES[iso2], "region": r, "available": available,
                "firefighters": ff, "share": round(ff_share, 4) if ff_share is not None else None,
                "detections_month": n, "fire_share": round(fire_share, 4) if fire_share is not None else None,
                "load_ratio": load, "active_fires": fr.get("active_fires", 0), "hotspots_24h": fr.get("hotspots_24h", 0),
                "largest_fire": fr.get("largest"),
                "firefighters_per_fire": round(ff / fr["active_fires"]) if ff and fr.get("active_fires") else None,
                "power": round(min(100.0, max(0.0, power[iso2] * factor)), 1),
                "anchor": GEO.get(iso2, {}).get("anchors", {}).get(r),
            })
    return out


def sources() -> list[dict]:
    text = lambda v: None if pd.isna(v) else str(v)
    return [{"iso2": i, "name": NAMES.get(i, r["country"]), "measure": text(r["measure"]) or "Not available", "period": text(r["source_period"]),
             "national_base": None if pd.isna(r["national_base"]) else int(r["national_base"]),
             "available": bool(r["available"]), "rescaled": bool(r["rescaled"])} for i, r in TABLE.iterrows()]
