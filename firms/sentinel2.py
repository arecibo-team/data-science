"""Independent burn-scar check for NASA fire events, from Sentinel-2 imagery.

Sentinel-2 is an optical ESA sensor. Its burn scars are computed here from raw bands and
share no inputs with the MODIS/VIIRS fire products, so a NASA detection that falls inside
a scar is checked against separate evidence.

Steps:
  1. Link harmonized NASA detections into fire events (connected components, within
     S2_LINK_KM and S2_LINK_DAYS).
  2. For each large-enough event, find a clear Sentinel-2 L2A scene before the first
     detection (pre) and after the last (post), on Microsoft Planetary Computer.
  3. Burn severity: dNBR = NBR(pre) - NBR(post), NBR = (B8A - B12) / (B8A + B12).
     Clear land pixels with dNBR >= S2_DNBR become burn scar polygons.
  4. Verdict per event:
       confirmed      scar found, large enough and within S2_NEAR_KM of a detection
       not_confirmed  clear pre and post scenes, no qualifying scar
       unverifiable   no usable clear pre/post scene
       too_large      event box exceeds S2_MAX_SPAN_KM (not processed)

Only confirmed events become incident features (see firms.incidents). Each event result
is cached under data/incidents/raw/s2/ so runs can resume; verdicts are exported to
data/incidents/s2_verdicts.csv.

    python -m firms.sentinel2 --max-events 300 --workers 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from rasterio.features import shapes as raster_shapes
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely import make_valid
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform, unary_union

from . import config as C

log = logging.getLogger("firms.sentinel2")

RAW_S2 = C.INCIDENTS_DIR / "raw" / "s2"
VERDICTS_CSV = C.INCIDENTS_DIR / "s2_verdicts.csv"
EARTH_RADIUS_KM = 6371.0088
WORK_CRS = "EPSG:3035"  # ETRS89 / LAEA Europe, metres: one grid for the whole continent
EPOCH = pd.Timestamp(0, tz="UTC")
# SCL classes that are not usable land: 0 no data, 1 saturated, 3 cloud shadow,
# 8/9 cloud, 10 cirrus, 11 snow/ice.
INVALID_SCL = [0, 1, 3, 8, 9, 10, 11]

_to_work = Transformer.from_crs("EPSG:4326", WORK_CRS, always_xy=True)
_to_wgs = Transformer.from_crs(WORK_CRS, "EPSG:4326", always_xy=True)


# ---------------------------------------------------------------- events

def find_events(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Group detections into fire events.

    Returns (events indexed by a stable event_id, per-detection label aligned with h).
    Links are made within S2_LINK_KM on the sphere and between days at most S2_LINK_DAYS
    apart, using the same unit-vector KD-tree approach as firms.harmonize.
    """
    if h.empty:
        return pd.DataFrame(), pd.Series(dtype="int64")
    order = np.argsort(h["acq_time_utc"].to_numpy(), kind="stable")
    df = h.iloc[order]
    n = len(df)
    day = ((df["acq_time_utc"].dt.floor("D") - EPOCH) // pd.Timedelta(days=1)).to_numpy(dtype="int64")
    lat = np.radians(df["lat"].to_numpy(dtype=float))
    lon = np.radians(df["lon"].to_numpy(dtype=float))
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    chord = 2 * np.sin((C.S2_LINK_KM / EARTH_RADIUS_KM) / 2)
    L = C.S2_LINK_DAYS

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for d in np.unique(day):
        q0, q1 = np.searchsorted(day, [d, d + 1])
        c0, c1 = np.searchsorted(day, [d - L, d + L + 1])
        if c1 - c0 < 2:
            continue
        tree = cKDTree(xyz[c0:c1])
        neigh = tree.query_ball_point(xyz[q0:q1], r=chord)
        lengths = np.fromiter((len(x) for x in neigh), dtype=np.int64, count=len(neigh))
        total = int(lengths.sum())
        if total == 0:
            continue
        src = np.repeat(np.arange(q0, q1, dtype=np.int64), lengths)
        dst = c0 + np.fromiter(itertools.chain.from_iterable(neigh), dtype=np.int64, count=total)
        keep = src != dst
        rows.append(src[keep])
        cols.append(dst[keep])

    r = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    c = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
    graph = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    _, label = connected_components(graph, directed=False)

    det = pd.DataFrame({
        "label": label,
        "lat": df["lat"].to_numpy(dtype=float),
        "lon": df["lon"].to_numpy(dtype=float),
        "day": df["acq_time_utc"].dt.floor("D").dt.date.to_numpy(),
        "frp": df["frp_mw"].to_numpy(dtype=float),
    }, index=df.index)

    g = det.groupby("label")
    events = pd.DataFrame({
        "n_det": g.size(),
        "frp_sum": g["frp"].sum(),
        "first_day": g["day"].min(),
        "last_day": g["day"].max(),
        "min_lat": g["lat"].min(), "max_lat": g["lat"].max(),
        "min_lon": g["lon"].min(), "max_lon": g["lon"].max(),
        "first_lat": g["lat"].first(), "first_lon": g["lon"].first(),
    })
    events = events[events["n_det"] >= C.S2_MIN_DETECTIONS].copy()
    # Stable id from the first detection's day and place, so reruns map to the same cache file.
    events["event_id"] = [
        f"{fd:%Y%m%d}_{la:.3f}_{lo:.3f}"
        for fd, la, lo in zip(events["first_day"], events["first_lat"], events["first_lon"])
    ]
    events = events.sort_values("frp_sum", ascending=False)
    # Per-detection component label, aligned with h; events are indexed by that label.
    return events, pd.Series(label, index=df.index, name="label")


def event_span_km(ev: pd.Series) -> float:
    mid_lat = (ev["min_lat"] + ev["max_lat"]) / 2
    dy = (ev["max_lat"] - ev["min_lat"]) * 111.0
    dx = (ev["max_lon"] - ev["min_lon"]) * 111.0 * math.cos(math.radians(mid_lat))
    return max(dx, dy)


# ---------------------------------------------------------------- Planetary Computer

def _client():
    import planetary_computer as pc
    from pystac_client import Client

    return Client.open(C.STAC_URL, modifier=pc.sign_inplace)


def _search(client, bbox_ll: tuple, start: date, end: date) -> list:
    items = client.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox_ll),
        datetime=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
        query={"eo:cloud_cover": {"lt": C.S2_CLOUD_MAX}},
        max_items=200,
    ).items()
    return sorted(items, key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)))


def _load_scene(item, bbox_ll: tuple) -> dict | None:
    """Load B8A, B12 and SCL for one scene over the box; return NBR and validity.

    `bbox_ll` is lon/lat (odc.stac reads `bbox` in EPSG:4326); the grid is WORK_CRS.
    """
    import odc.stac

    ds = odc.stac.load(
        [item],
        bands=["B8A", "B12", "SCL"],
        crs=WORK_CRS,
        resolution=C.S2_RESOLUTION,
        bbox=bbox_ll,
        chunks={},
    ).isel(time=0).compute()
    scl = ds["SCL"].values
    # odc.stac returns raw digital numbers. From processing baseline 04.00 on, L2A adds an
    # offset of -1000 DN to every band; without it the NBR ratio is biased.
    baseline = float(item.properties.get("s2:processing_baseline") or 0)
    offset = -1000.0 if baseline >= 4.0 else 0.0
    b8 = ds["B8A"].values.astype("float64") + offset
    b12 = ds["B12"].values.astype("float64") + offset
    valid = ~np.isin(scl, INVALID_SCL) & np.isfinite(b8) & np.isfinite(b12) & ((b8 + b12) > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        nbr = np.where(valid, (b8 - b12) / (b8 + b12), np.nan)
    return {
        "id": item.id,
        "date": item.datetime.date(),
        "clear": float(valid.mean()) if valid.size else 0.0,
        "nbr": nbr,
        "valid": valid,
        "transform": ds.odc.transform,
        "shape": nbr.shape,
    }


def _best_scene(items: list, bbox_ll: tuple, latest: bool) -> dict | None:
    """Usable scene closest to the fire on one side: latest for pre, earliest for post.

    Only the S2_MAX_CANDIDATES lowest-cloud items are loaded. Among usable ones, the
    date decides and clear share breaks ties.
    """
    usable = []
    for item in items[: C.S2_MAX_CANDIDATES]:
        scene = _load_scene(item, bbox_ll)
        if scene and scene["clear"] >= C.S2_MIN_CLEAR:
            usable.append(scene)
    if not usable:
        return None
    if latest:
        return max(usable, key=lambda s: (s["date"], s["clear"]))
    return min(usable, key=lambda s: (s["date"], -s["clear"]))


# ---------------------------------------------------------------- burn scar

def _scar_polygons(pre: dict, post: dict) -> list:
    """Burn scar polygons (work CRS) from dNBR on pixels clear in both scenes."""
    valid = pre["valid"] & post["valid"]
    with np.errstate(invalid="ignore"):
        dnbr = pre["nbr"] - post["nbr"]
    burn = (dnbr >= C.S2_DNBR) & valid
    if not burn.any():
        return []
    polys = []
    for geom, val in raster_shapes(burn.astype("uint8"), mask=burn, transform=post["transform"]):
        if val == 1:
            polys.append(make_valid(shape(geom)))
    return polys


def _scar_geometry(polys: list, det_xy: np.ndarray) -> tuple[object | None, float]:
    """Keep scar polygons of at least 1 ha that lie within S2_NEAR_KM of a detection."""
    det_pts = shapely.points(det_xy[:, 0], det_xy[:, 1]) if len(det_xy) else None
    kept = []
    for p in polys:
        area_ha = p.area / 1e4
        if area_ha < 1.0 or det_pts is None:
            continue
        if shapely.dwithin(p, det_pts, C.S2_NEAR_KM * 1000).any():
            kept.append(p)
    if not kept:
        return None, 0.0
    merged = unary_union(kept)
    return merged, round(merged.area / 1e4, 1)


# ---------------------------------------------------------------- per event

def _bbox_ll(ev: pd.Series) -> tuple[float, float, float, float]:
    pad_lat = C.S2_PAD_KM / 111.0
    mid_lat = (ev["min_lat"] + ev["max_lat"]) / 2
    pad_lon = C.S2_PAD_KM / (111.0 * max(math.cos(math.radians(mid_lat)), 0.1))
    return (ev["min_lon"] - pad_lon, ev["min_lat"] - pad_lat,
            ev["max_lon"] + pad_lon, ev["max_lat"] + pad_lat)


def _bbox_work(bbox_ll: tuple) -> tuple[float, float, float, float]:
    x0, y0 = _to_work.transform(bbox_ll[0], bbox_ll[1])
    x1, y1 = _to_work.transform(bbox_ll[2], bbox_ll[3])
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def process_event(event_id: str, ev: pd.Series, det_lat: np.ndarray, det_lon: np.ndarray) -> dict:
    """Verdict for one fire event. Cached per event; errors are not cached so they are retried."""
    cache = RAW_S2 / f"{event_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    base = {
        "event_id": event_id, "n_det": int(ev["n_det"]), "frp_sum": round(float(ev["frp_sum"]), 1),
        "first_day": str(ev["first_day"]), "last_day": str(ev["last_day"]),
        "lat": round(float(ev["first_lat"]), 4), "lon": round(float(ev["first_lon"]), 4),
    }
    too_long = (pd.Timestamp(ev["last_day"]) - pd.Timestamp(ev["first_day"])).days > C.S2_MAX_DAYS
    if event_span_km(ev) > C.S2_MAX_SPAN_KM or too_long:
        reason = "event box exceeds S2_MAX_SPAN_KM" if not too_long else "event lasts longer than S2_MAX_DAYS"
        rec = {**base, "verdict": "too_large", "reason": reason,
               "scar_ha": None, "pre": None, "post": None, "geometry": None}
    else:
        rec = _verdict(base, ev, det_lat, det_lon)

    RAW_S2.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec))
    return rec


def _verdict(base: dict, ev: pd.Series, det_lat: np.ndarray, det_lon: np.ndarray) -> dict:
    first, last = ev["first_day"], ev["last_day"]
    bbox_ll = _bbox_ll(ev)
    client = _client()
    pre_items = _search(client, bbox_ll, first - timedelta(days=C.S2_PRE_DAYS), first - timedelta(days=1))
    post_items = _search(client, bbox_ll, last + timedelta(days=C.S2_POST_OFFSET_DAYS),
                         last + timedelta(days=C.S2_POST_DAYS))
    fail = {**base, "scar_ha": None, "pre": None, "post": None, "geometry": None}
    if not pre_items or not post_items:
        return {**fail, "verdict": "unverifiable", "reason": "no pre or post scene under the cloud limit"}

    pre = _best_scene(pre_items, bbox_ll, latest=True)
    post = _best_scene(post_items, bbox_ll, latest=False)
    if pre is None or post is None:
        return {**fail, "verdict": "unverifiable", "reason": "no scene with enough clear land"}
    if pre["shape"] != post["shape"]:
        return {**fail, "verdict": "unverifiable", "reason": "pre and post grids differ"}

    pre_meta = {"date": str(pre["date"]), "item": pre["id"]}
    post_meta = {"date": str(post["date"]), "item": post["id"]}
    det_x, det_y = _to_work.transform(det_lon, det_lat)  # detections in the scene grid's CRS
    polys = _scar_polygons(pre, post)
    geom, scar_ha = _scar_geometry(polys, np.column_stack([det_x, det_y]))
    base = {**base, "pre": pre_meta, "post": post_meta}
    if geom is None or scar_ha < C.S2_MIN_SCAR_HA:
        return {**base, "verdict": "not_confirmed", "reason": "no qualifying burn scar near detections",
                "scar_ha": scar_ha, "geometry": None}
    geo_ll = shp_transform(_to_wgs.transform, geom)
    return {**base, "verdict": "confirmed", "reason": "dNBR scar near detections",
            "scar_ha": scar_ha, "geometry": mapping(geo_ll)}


# ---------------------------------------------------------------- outputs

def load_records() -> list[dict]:
    if not RAW_S2.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(RAW_S2.glob("*.json"))]


def confirmed_features() -> list[dict]:
    """Incident features for confirmed events, consumed by firms.incidents."""
    feats = []
    for rec in load_records():
        if rec.get("verdict") != "confirmed" or not rec.get("geometry"):
            continue
        feats.append({
            "type": "Feature",
            "properties": {
                "incident_id": f"S2-{rec['event_id']}",
                "source": "Sentinel-2",
                "name": f"Burn scar {rec['first_day']} to {rec['last_day']}",
                "country": "",
                "start_date": rec["pre"]["date"],
                "end_date": rec["post"]["date"],
                "area_ha": rec["scar_ha"],
                "buffer_km": C.S2_MATCH_BUFFER_KM,
                "slack_days": 0,
            },
            "geometry": rec["geometry"],
        })
    return feats


def write_verdicts(records: list[dict]) -> pd.DataFrame:
    cols = ["event_id", "verdict", "reason", "n_det", "frp_sum", "first_day", "last_day",
            "lat", "lon", "scar_ha", "pre", "post"]
    rows = []
    for r in records:
        row = {k: r.get(k) for k in cols}
        for k in ("pre", "post"):
            row[k] = (r.get(k) or {}).get("date")
        rows.append(row)
    df = pd.DataFrame(rows, columns=cols)
    C.INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(VERDICTS_CSV, index=False)
    return df


# ---------------------------------------------------------------- orchestration

def run(max_events: int, workers: int) -> pd.DataFrame:
    h = pd.read_parquet(C.ANALYSIS_DIR / "harmonized.parquet", columns=[
        "lat", "lon", "acq_time_utc", "frp_mw"])
    events, labels = find_events(h)
    lon_min, lat_min, lon_max, lat_max = C.EUROPE_BBOX
    in_europe = events["first_lon"].between(lon_min, lon_max) & events["first_lat"].between(lat_min, lat_max)
    events = events[in_europe]
    labels = labels[labels.isin(events.index)]
    log.info("%d European fire events with >= %d detections", len(events), C.S2_MIN_DETECTIONS)
    if events["event_id"].duplicated().any():
        log.warning("%d duplicate event ids; later events share a cache file", int(events["event_id"].duplicated().sum()))
    selected = events.head(max_events)

    # Detection coordinates for each selected event, keyed by its component label.
    chosen = labels[labels.isin(selected.index)]
    coords = h.loc[chosen.index, ["lat", "lon"]].assign(label=chosen.values)
    by_label = {lbl: (g["lat"].to_numpy(dtype=float), g["lon"].to_numpy(dtype=float))
                for lbl, g in coords.groupby("label")}

    log.info("processing %d events with %d workers", len(selected), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_event, ev["event_id"], ev, *by_label[lbl]): lbl
            for lbl, ev in selected.iterrows()
        }
        for fut in as_completed(futures):
            lbl = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # network or data problem: skip, retried on next run
                log.warning("event %s failed: %s", selected.loc[lbl, "event_id"], exc)
                continue
            log.info("%s %s", rec["event_id"], rec["verdict"])
    return write_verdicts(load_records())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-events", type=int, default=300, help="largest events by FRP to check")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = run(args.max_events, args.workers)
    log.info("verdicts: %s", df["verdict"].value_counts().to_dict() if len(df) else {})


if __name__ == "__main__":
    main()
