"""Region filter: detections on the African continent are dropped.

`regions/africa.geojson` is the union of the African countries of Natural Earth (1:50m, public domain),
simplified to about 2 km. Spain (including the Canary Islands) and Turkey are not part of it. A bounding
box cannot do this job: the Algerian coast lies at the latitude of southern Spain.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape

AFRICA_FILE = Path(__file__).with_name("regions") / "africa.geojson"


@lru_cache(maxsize=1)
def _africa():
    geom = shape(json.loads(AFRICA_FILE.read_text())["geometry"])
    shapely.prepare(geom)
    return geom


def in_africa(lat, lon) -> np.ndarray:
    """Boolean array: True where the point lies on the African continent."""
    return shapely.contains_xy(_africa(), np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
