from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import EU27, RAW_DIR

FIRMS_COUNTRY = "https://firms.modaps.eosdis.nasa.gov/data/country/{folder}/{year}/{folder}_{year}_{name}.csv"
FIRMS_NAMES = {"CZ": "Czech_Republic"}
SENSORS = {"modis": ("modis", range(2000, 2026)), "viirs": ("viirs-snpp", range(2012, 2026))}
FIRMS_DIR = RAW_DIR / "firms"


def firms_name(iso2: str) -> str:
    return FIRMS_NAMES.get(iso2, EU27[iso2][1].replace(" ", "_"))


def _download_one(task: tuple[str, str, int, Path]) -> tuple[Path, int]:
    iso2, folder, year, path = task
    if path.exists():
        return path, -1
    url = FIRMS_COUNTRY.format(folder=folder, year=year, name=firms_name(iso2))
    for attempt in range(4):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
            if r.status_code == 404:
                path.write_text("")
                return path, 404
            r.raise_for_status()
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(r.content)
            tmp.replace(path)
            return path, r.status_code
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return path, 0


def download_all(workers: int = 8, verbose: bool = True, years: dict | None = None) -> None:
    tasks = []
    for sensor, (folder, default_years) in SENSORS.items():
        for iso2 in EU27:
            for year in (years or {}).get(sensor, default_years):
                path = FIRMS_DIR / sensor / f"{iso2}_{year}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                tasks.append((iso2, folder, year, path))
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for path, status in pool.map(_download_one, tasks):
            done += 1
            if verbose and status not in (-1, 200) or verbose and done % 100 == 0:
                print(f"{done}/{len(tasks)} {path.name} {status}", flush=True)


GRID = 0.01
TILE = 0.5
AQUA_START = pd.Timestamp("2002-07-04")
VIIRS_START = pd.Timestamp("2012-01-20")
OVERLAP = (2013, 2022)
SHRINK = 50.0
ROW_START = pd.Timestamp("2000-11-01")
ROW_END = pd.Timestamp("2025-12-31")


def _read(path: Path, cols: list[str]) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path, usecols=lambda c: c in cols, dtype={"confidence": str, "satellite": str})
    for c in ("latitude", "longitude", "frp", "acq_time", "type"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _concat(frames: list, cols: list[str]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({c: pd.Series(dtype=float) for c in cols})


def load_detections(iso2: str) -> pd.DataFrame:
    cols = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "type"]
    v = _concat([_read(FIRMS_DIR / "viirs" / f"{iso2}_{y}.csv", cols) for y in SENSORS["viirs"][1]], cols)
    m = _concat([_read(FIRMS_DIR / "modis" / f"{iso2}_{y}.csv", cols) for y in SENSORS["modis"][1]], cols)
    v["confidence"] = v["confidence"].astype(str)
    m["satellite"] = m["satellite"].astype(str)
    static = v[v["type"].isin([2, 3])]
    mask = set(zip(np.floor(static.latitude / GRID).astype(int), np.floor(static.longitude / GRID).astype(int)))
    mask = {(a + i, b + j) for a, b in mask for i in (-1, 0, 1) for j in (-1, 0, 1)}
    v = v[(v["type"] == 0) & (v["confidence"].str[0].str.lower() != "l")].assign(sensor="V")
    m = m[(m["type"] == 0) & (pd.to_numeric(m["confidence"], errors="coerce") >= 30)]
    m = m.assign(sensor=np.where(m["satellite"].str.startswith("T"), "T", "A"))
    d = pd.concat([m, v], ignore_index=True)
    d = d.rename(columns={"latitude": "lat", "longitude": "lon", "acq_time": "time"})
    d["date"] = pd.to_datetime(d["acq_date"])
    d["cy"] = np.floor(d["lat"] / GRID).astype(np.int32)
    d["cx"] = np.floor(d["lon"] / GRID).astype(np.int32)
    keep = [(a, b) not in mask for a, b in zip(d["cy"], d["cx"])]
    d = d[keep]
    return d[["lat", "lon", "date", "time", "sensor", "frp", "cy", "cx"]].reset_index(drop=True)


def cell_days(d: pd.DataFrame) -> pd.DataFrame:
    g = d.assign(group=np.where(d["sensor"] == "V", "viirs", "modis"))
    out = g.drop_duplicates(["group", "cy", "cx", "date"])[["group", "cy", "cx", "date", "sensor"]]
    terra = g[g["sensor"] == "T"].drop_duplicates(["cy", "cx", "date"])[["cy", "cx", "date"]].assign(group="terra", sensor="T")
    return pd.concat([out, terra], ignore_index=True)


def monthly_counts(cd: pd.DataFrame) -> pd.DataFrame:
    t = cd.assign(year=cd["date"].dt.year, month=cd["date"].dt.month)
    return t.groupby(["group", "year", "month"]).size().unstack("group", fill_value=0).reindex(columns=["modis", "viirs", "terra"], fill_value=0)


def fit_ratios(monthly: dict[str, pd.DataFrame]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    a, b = OVERLAP
    sel = lambda m: m[(m.index.get_level_values("year") >= a) & (m.index.get_level_values("year") <= b)].groupby(level="month").sum().reindex(range(1, 13), fill_value=0)
    eu = sum(sel(m) for m in monthly.values())
    r_eu = (eu["viirs"] / eu["modis"].replace(0, np.nan)).fillna(1.0).to_numpy()
    pre = lambda m: m[(m.index.get_level_values("year") >= 2003) & (m.index.get_level_values("year") <= 2011)].groupby(level="month").sum().reindex(range(1, 13), fill_value=0)
    eu_pre = sum(pre(m) for m in monthly.values())
    terra = (eu_pre["modis"] / eu_pre["terra"].replace(0, np.nan)).fillna(1.0).to_numpy()
    ratios = {}
    for iso2, m in monthly.items():
        s = sel(m)
        ratios[iso2] = ((s["viirs"].to_numpy() + SHRINK * r_eu) / (s["modis"].to_numpy() + SHRINK))
    return ratios, r_eu, terra


def harmonize_daily(cd: pd.DataFrame, ratio: np.ndarray, terra: np.ndarray) -> pd.DataFrame:
    days = pd.date_range(ROW_START, ROW_END, freq="D")
    m = cd[cd["group"] == "modis"].groupby("date").size().reindex(days, fill_value=0)
    v = cd[cd["group"] == "viirs"].groupby("date").size().reindex(days, fill_value=0)
    mi = days.month.to_numpy() - 1
    factor = ratio[mi] * np.where(days < AQUA_START, terra[mi], 1.0)
    harm = np.where(days < VIIRS_START, m.to_numpy() * factor, v.to_numpy())
    return pd.DataFrame({"modis": m.to_numpy(), "viirs": v.to_numpy(), "harmonized": harm}, index=days)


def cluster_fires(d: pd.DataFrame) -> pd.Series:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nodes = d[["cy", "cx", "date"]].drop_duplicates().reset_index(drop=True)
    nodes["day"] = (nodes["date"] - pd.Timestamp("2000-01-01")).dt.days.astype(np.int32)
    nodes["node"] = np.arange(len(nodes))
    key = nodes[["cy", "cx", "day", "node"]]
    rows, cols = [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for dd in (0, 1):
                if dy == dx == dd == 0:
                    continue
                shifted = key.assign(cy=key.cy + dy, cx=key.cx + dx, day=key.day + dd)
                j = shifted.merge(key, on=["cy", "cx", "day"], suffixes=("", "_n"))
                rows.append(j["node"].to_numpy())
                cols.append(j["node_n"].to_numpy())
    r = np.concatenate(rows) if rows else np.array([], dtype=int)
    c = np.concatenate(cols) if cols else np.array([], dtype=int)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(nodes), len(nodes)))
    _, labels = connected_components(g, directed=False)
    nodes["fire"] = labels
    return d.merge(nodes[["cy", "cx", "date", "fire"]], on=["cy", "cx", "date"], how="left")["fire"]


def load_places() -> tuple:
    from scipy.spatial import cKDTree
    path = RAW_DIR / "geonames" / "cities500.txt"
    cols = [1, 4, 5, 7, 8, 14]
    p = pd.read_csv(path, sep="\t", header=None, usecols=cols, names=["name", "lat", "lon", "fcode", "cc", "pop"],
                    dtype={"cc": str}, keep_default_na=False, quoting=3)
    p = p[p["cc"].replace({"EL": "GR"}).isin(EU27)].reset_index(drop=True)
    xy = np.c_[p["lat"].to_numpy(), p["lon"].to_numpy() * np.cos(np.radians(p["lat"].to_numpy()))]
    return cKDTree(xy), p


def name_fires(tree_places: tuple, lat: np.ndarray, lon: np.ndarray) -> list[str]:
    tree, p = tree_places
    xy = np.c_[lat, lon * np.cos(np.radians(lat))]
    out = []
    for q in xy:
        idx = tree.query_ball_point(q, 0.12)
        if idx:
            best = max(idx, key=lambda i: (p.at[i, "pop"], -((p.at[i, "lat"] - q[0]) ** 2)))
        else:
            best = int(tree.query(q)[1])
        out.append(str(p.at[best, "name"]))
    return out


def tile_daily(cd: pd.DataFrame, ratio: np.ndarray, terra: np.ndarray) -> pd.DataFrame:
    t = cd[cd["group"].isin(["modis", "viirs"])]
    t = t.assign(ty=np.floor(t["cy"] * GRID / TILE).astype(np.int16), tx=np.floor(t["cx"] * GRID / TILE).astype(np.int16))
    g = t.groupby(["ty", "tx", "date", "group"]).size().unstack("group", fill_value=0).reindex(columns=["modis", "viirs"], fill_value=0)
    dates = g.index.get_level_values("date")
    mi = dates.month.to_numpy() - 1
    factor = ratio[mi] * np.where(dates < AQUA_START, terra[mi], 1.0)
    harm = np.where(dates < VIIRS_START, g["modis"].to_numpy() * factor, g["viirs"].to_numpy())
    out = pd.DataFrame({"ty": g.index.get_level_values("ty"), "tx": g.index.get_level_values("tx"),
                        "day": (dates - ROW_START).days.astype(np.int32), "h": harm.astype(np.float32)})
    return out[out["h"] > 0]


def extract_fires(iso2: str, d: pd.DataFrame, places: tuple, min_detections: int = 10) -> list[dict]:
    d = d.assign(fire=cluster_fires(d).to_numpy())
    d = d.assign(px=np.floor(d["lat"] / 0.0035).astype(np.int64) * 100000 + np.floor(d["lon"] / 0.0035).astype(np.int64),
                 km=d["cy"].astype(np.int64) * 100000 + d["cx"].astype(np.int64))
    size = d.groupby("fire").size()
    big = size[size >= min_detections].index
    d = d[d["fire"].isin(big)]
    if d.empty:
        return []
    agg = d.groupby("fire").agg(n=("lat", "size"), lat=("lat", "mean"), lon=("lon", "mean"), start=("date", "min"),
                                end=("date", "max"), frp=("frp", "sum"),
                                vpx=("px", lambda s: s[d.loc[s.index, "sensor"] == "V"].nunique()), kmc=("km", "nunique"))
    last = d.sort_values(["date", "time"]).groupby("fire").tail(1).set_index("fire")
    daily = d.groupby(["fire", "date"]).size()
    lastday = d.sort_values(["date", "time"]).groupby(["fire", "date"]).tail(1).set_index(["fire", "date"])
    names = name_fires(places, agg["lat"].to_numpy(), agg["lon"].to_numpy())
    out = []
    for (fid, r), name in zip(agg.iterrows(), names):
        lr = last.loc[fid]
        out.append({
            "id": f"{iso2}-{r.start:%Y%m%d}-{int(fid)}", "iso2": iso2, "name": name,
            "lat": round(float(r.lat), 4), "lon": round(float(r.lon), 4),
            "start": f"{r.start:%Y-%m-%d}", "end": f"{r.end:%Y-%m-%d}", "detections": int(r.n),
            "area_ha": int(round(r.vpx * 14.06 if r.vpx else r.kmc * 100)), "frp": round(float(r.frp), 1),
            "last_sensor": "VIIRS" if lr.sensor == "V" else "MODIS", "last_time": f"{int(lr.time):04d}",
            "daily": {f"{k:%Y-%m-%d}": [int(v), f"{int(lastday.loc[(fid, k), 'time']):04d}", "VIIRS" if lastday.loc[(fid, k), "sensor"] == "V" else "MODIS"]
                      for k, v in daily.loc[fid].items()},
        })
    return out


def build(out: Path | None = None, verbose: bool = True) -> dict:
    import json
    from . import ARTIFACTS_DIR
    from .pipeline import write_json_atomic
    out = out or ARTIFACTS_DIR / "fires"
    out.mkdir(parents=True, exist_ok=True)
    det, cds, monthly = {}, {}, {}
    for iso2 in EU27:
        d = load_detections(iso2)
        det[iso2] = d
        cds[iso2] = cell_days(d)
        monthly[iso2] = monthly_counts(cds[iso2])
        if verbose:
            print(f"{iso2}: {len(d):,} detections, {len(cds[iso2]):,} cell-days", flush=True)
    ratios, r_eu, terra = fit_ratios(monthly)
    places = load_places()
    days = pd.date_range(ROW_START, ROW_END, freq="D")
    daily, fires, validation, tiles = {}, [], {}, []
    for iso2 in EU27:
        h = harmonize_daily(cds[iso2], ratios[iso2], terra)
        daily[iso2] = h
        tiles.append(tile_daily(cds[iso2], ratios[iso2], terra))
        fires += extract_fires(iso2, det[iso2], places)
        yv = h["viirs"].groupby(h.index.year).sum()
        ym = (h["modis"] * ratios[iso2][h.index.month - 1]).groupby(h.index.year).sum()
        yrs = [y for y in range(OVERLAP[0], OVERLAP[1] + 1) if yv.get(y, 0) > 0]
        validation[iso2] = {"mean_abs_error_pct": round(float(np.mean([abs(ym[y] - yv[y]) / yv[y] for y in yrs]) * 100), 1) if yrs else None,
                            "viirs_cell_days_overlap": int(sum(yv.get(y, 0) for y in yrs))}
        if verbose:
            print(f"{iso2}: fires so far {len(fires):,}", flush=True)
    series = {iso2: {k: [round(float(x), 2) for x in daily[iso2][k].to_numpy()] for k in ("harmonized", "modis", "viirs")} for iso2 in EU27}
    write_json_atomic(out / "daily.json", {"start": f"{ROW_START:%Y-%m-%d}", "end": f"{ROW_END:%Y-%m-%d}", "series": series})
    write_json_atomic(out / "events.json", fires)
    t = pd.concat(tiles, ignore_index=True).groupby(["ty", "tx", "day"], as_index=False)["h"].sum()
    np.savez_compressed(out / "tiles.npz", ty=t["ty"].to_numpy(np.int16), tx=t["tx"].to_numpy(np.int16),
                        day=t["day"].to_numpy(np.int32), h=t["h"].to_numpy(np.float32))
    meta = {"grid_deg": GRID, "tile_deg": TILE, "start": f"{ROW_START:%Y-%m-%d}", "overlap": list(OVERLAP), "shrink": SHRINK, "ratio_eu": [round(float(x), 4) for x in r_eu],
            "terra_factor": [round(float(x), 4) for x in terra],
            "ratio": {k: [round(float(x), 4) for x in v] for k, v in ratios.items()}, "validation": validation,
            "fires": len(fires), "source": "NASA FIRMS country archives: MODIS C6.1 and VIIRS S-NPP 375 m; places: GeoNames"}
    write_json_atomic(out / "meta.json", meta)
    return meta
