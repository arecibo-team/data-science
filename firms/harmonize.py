"""Cross-sensor harmonization: pair MODIS and VIIRS detections that describe the same fire.

A MODIS detection and a VIIRS detection are linked when
    great-circle distance <= radius_km   and   |time difference| <= window_min.
Only cross-sensor links are made, so each cluster contains at least one MODIS and one
VIIRS detection ("both"). Connected components of those links form the clusters. Unmatched
detections are labelled "modis_only" or "viirs_only".

Defaults: 1.0 km (about one MODIS pixel) and 180 min. Both are parameters, not constants.
Chaining is possible: A-B and B-C links join A, B and C into one cluster.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import config as C

log = logging.getLogger("firms.harmonize")
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class MatchParams:
    radius_km: float = 1.0
    window_min: int = 180

    def __post_init__(self):
        if self.radius_km <= 0:
            raise ValueError("radius_km must be > 0")
        if not 0 < self.window_min <= 24 * 60:
            raise ValueError("window_min must be in (0, 1440]")


def match_cross_sensor(df: pd.DataFrame, params: MatchParams = MatchParams()) -> pd.DataFrame:
    """Return df with added columns: cluster_id (Int64, NA if unmatched) and harmonized_status.

    Processing is chunked by UTC day. Each day's detections are matched against the
    previous, current and next day, so pairs that cross midnight are still found.
    """
    df = df.sort_values("acq_time_utc").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return df.assign(cluster_id=pd.Series(dtype="Int64"), harmonized_status=pd.Series(dtype="string"))

    epoch = pd.Timestamp(0, tz="UTC")
    # Unit-independent conversion (pandas may store datetimes in ns or us).
    t_s = ((df["acq_time_utc"] - epoch) // pd.Timedelta(seconds=1)).to_numpy(dtype="int64")
    sensor = df["sensor"].to_numpy()
    # Unit vectors on the sphere: Euclidean distance between them is the chord length,
    # so a KD-tree radius query on xyz matches great-circle distance exactly.
    lat = np.radians(df["lat"].to_numpy(dtype=float))
    lon = np.radians(df["lon"].to_numpy(dtype=float))
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    day = ((df["acq_time_utc"].dt.floor("D") - epoch) // pd.Timedelta(days=1)).to_numpy(dtype="int64")

    radius_rad = params.radius_km / EARTH_RADIUS_KM
    chord = 2 * np.sin(radius_rad / 2)
    window_s = params.window_min * 60
    rows_a: list[np.ndarray] = []
    rows_b: list[np.ndarray] = []

    # df is sorted by time, so `day` is non-decreasing and each day is a contiguous slice.
    unique_days = np.unique(day)
    for d in unique_days:
        q0, q1 = np.searchsorted(day, [d, d + 1])
        c0, c1 = np.searchsorted(day, [d - 1, d + 2])
        query_idx = np.arange(q0, q1)
        cand_idx = np.arange(c0, c1)
        if len(cand_idx) < 2:
            continue
        tree = cKDTree(xyz[cand_idx])
        neighbours = tree.query_ball_point(xyz[query_idx], r=chord)
        # Flatten all neighbour lists once, then filter in a single vectorized pass.
        counts = np.fromiter((len(nb) for nb in neighbours), dtype=np.int64, count=len(neighbours))
        if counts.sum() == 0:
            continue
        qi = np.repeat(query_idx, counts)
        js = cand_idx[np.concatenate([np.asarray(nb, dtype=np.int64) for nb in neighbours])]
        keep = (sensor[js] != sensor[qi]) & (np.abs(t_s[js] - t_s[qi]) <= window_s)
        rows_a.append(qi[keep])
        rows_b.append(js[keep])

    if rows_a:
        a = np.concatenate(rows_a)
        b = np.concatenate(rows_b)
        adj = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
        _, labels = connected_components(adj, directed=False)
    else:
        labels = np.arange(n)

    # A label is a real cluster only when it has more than one member.
    counts = np.bincount(labels, minlength=labels.max() + 1)
    in_cluster = counts[labels] > 1
    cluster_ids = pd.Series(pd.NA, index=df.index, dtype="Int64")
    cluster_ids[in_cluster] = pd.factorize(labels[in_cluster])[0]

    status = np.where(in_cluster, "both", np.where(sensor == "MODIS", "modis_only", "viirs_only"))
    out = df.copy()
    out["cluster_id"] = cluster_ids
    out["harmonized_status"] = pd.Series(status, index=df.index, dtype="string")
    return out


def cluster_table(h: pd.DataFrame) -> pd.DataFrame:
    """One row per matched cluster: members, sensors, time span, peak FRP and centroid."""
    c = h[h["cluster_id"].notna()]
    if c.empty:
        return pd.DataFrame(columns=["cluster_id", "n_detections", "n_modis", "n_viirs",
                                     "first_utc", "last_utc", "max_frp_mw", "lat", "lon"])
    c = c.assign(_is_modis=c["sensor"] == "MODIS", _is_viirs=c["sensor"] == "VIIRS")
    g = c.groupby("cluster_id")
    return pd.DataFrame({
        "n_detections": g.size(),
        "n_modis": g["_is_modis"].sum(),
        "n_viirs": g["_is_viirs"].sum(),
        "first_utc": g["acq_time_utc"].min(),
        "last_utc": g["acq_time_utc"].max(),
        "max_frp_mw": g["frp_mw"].max(),
        "lat": g["lat"].mean(),
        "lon": g["lon"].mean(),
    }).reset_index()


def daily_summary(h: pd.DataFrame) -> pd.DataFrame:
    """Per UTC day: counts, FRP and the share of each sensor's detections that were matched."""
    if h.empty:
        return pd.DataFrame()
    d = h.assign(
        date=h["acq_time_utc"].dt.strftime("%Y-%m-%d"),
        _matched=(h["harmonized_status"] == "both").astype(int),
    )
    g = d.groupby(["date", "sensor"])
    per_sensor = pd.DataFrame({
        "detections": g.size(),
        "frp_sum_mw": g["frp_mw"].sum(),
        "frp_median_mw": g["frp_mw"].median(),
        "matched": g["_matched"].sum(),
    })
    per_sensor["matched_share"] = per_sensor["matched"] / per_sensor["detections"]
    wide = per_sensor.unstack("sensor")
    wide.columns = [f"{metric}_{sens.lower()}" for metric, sens in wide.columns]
    clusters = (h[h["cluster_id"].notna()]
                .assign(date=lambda x: x["acq_time_utc"].dt.strftime("%Y-%m-%d"))
                .groupby("date")["cluster_id"].nunique().rename("matched_clusters"))
    return wide.join(clusters, how="outer").fillna(0).sort_index().reset_index()


if __name__ == "__main__":
    raise SystemExit("run: python -m firms.build")
