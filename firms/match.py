"""Tag NASA detections with the real European incident they fall inside.

A detection is "confirmed" when it lies within `buffer_km` of an incident polygon
(EFFIS perimeter or Copernicus EMS AOI) and its acquisition date is within the
incident's [start_date, end_date] window widened by `slack_days`. When several
incidents qualify, the one whose date window is closest wins.

The buffer is applied in degrees (1 deg ~ 111 km). Longitude is not scaled by cos(lat),
so the east-west tolerance shrinks to roughly 0.6 km at 55 N; this is a deliberate
simplification that keeps matching a pure shapely operation.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
import shapely
from shapely import make_valid
from shapely.geometry import shape
from shapely.strtree import STRtree

from . import config as C

log = logging.getLogger("firms.match")


def tag_incidents(
    h: pd.DataFrame,
    features: list[dict],
    buffer_km: float = C.MATCH_BUFFER_KM,
    slack_days: int = C.MATCH_DATE_SLACK_DAYS,
) -> pd.Series:
    """Return a Series (aligned with `h`) of incident_id strings, or None when unmatched."""
    result = pd.Series([None] * len(h), index=h.index, dtype=object)
    if not features or h.empty:
        return result

    geoms = []
    starts = np.empty(len(features), dtype="datetime64[D]")
    ends = np.empty(len(features), dtype="datetime64[D]")
    ids = []
    deg = buffer_km / 111.0
    for i, f in enumerate(features):
        p = f["properties"]
        # Some EFFIS perimeters are self-intersecting; repair before buffering.
        geoms.append(make_valid(shape(f["geometry"])).buffer(deg))
        starts[i] = np.datetime64(p["start_date"])
        ends[i] = np.datetime64(p["end_date"])
        ids.append(p["incident_id"])

    tree = STRtree(geoms)
    points = shapely.points(h["lon"].to_numpy(dtype=float), h["lat"].to_numpy(dtype=float))
    point_idx, geom_idx = tree.query(points, predicate="intersects")
    if len(point_idx) == 0:
        return result

    det_day = h["acq_time_utc"].dt.tz_localize(None).dt.normalize().to_numpy(dtype="datetime64[D]")
    d = det_day[point_idx]
    slack = np.timedelta64(slack_days, "D")
    s = starts[geom_idx]
    e = ends[geom_idx]
    # Days outside the incident window (0 when inside). Keep only detections within slack.
    gap = np.maximum(np.maximum((s - d) / np.timedelta64(1, "D"), (d - e) / np.timedelta64(1, "D")), 0)
    ok = (d >= s - slack) & (d <= e + slack)
    if not ok.any():
        return result

    cand = pd.DataFrame({
        "row": point_idx[ok],
        "geom": geom_idx[ok],
        "gap": gap[ok],
    }).sort_values(["row", "gap", "geom"]).drop_duplicates("row", keep="first")

    labels = np.array(ids, dtype=object)[cand["geom"].to_numpy()]
    result.iloc[cand["row"].to_numpy()] = labels
    log.info("tagged %d / %d detections with an incident", len(cand), len(h))
    return result


def incident_detection_counts(tags: pd.Series) -> dict[str, int]:
    """Number of detections per incident_id (unmatched rows excluded)."""
    return tags.dropna().value_counts().to_dict()
