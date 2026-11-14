"""Runtime settings, read from the environment (and .env for the FIRMS key)."""
from __future__ import annotations

import datetime as dt
import os

from firms import config as C

DB_PATH = os.environ.get("FIRMS_DB", str(C.DATA_DIR / "firms.db"))

# Fetch window and area. The end date defaults to today, so every day brings one new fetch step.
FETCH_START = os.environ.get("FIRMS_START", C.INCIDENTS_FROM)
FETCH_BBOX = os.environ.get("FIRMS_BBOX", "-25,-20,45,72")
FETCH_FAMILIES = [f for f in os.environ.get("FIRMS_FAMILIES", ",".join(C.FAMILIES)).split(",") if f]

MATCH_RADIUS_KM = float(os.environ.get("FIRMS_RADIUS_KM", "1.0"))
MATCH_WINDOW_MIN = int(os.environ.get("FIRMS_WINDOW_MIN", "180"))
S2_MAX_EVENTS = int(os.environ.get("FIRMS_S2_MAX_EVENTS", "300"))
S2_WORKERS = int(os.environ.get("FIRMS_S2_WORKERS", "4"))

# Comma-separated step names that are never run on start (e.g. "fetch,sentinel2" when offline).
SKIP_STEPS = {s.strip() for s in os.environ.get("PIPELINE_SKIP", "").split(",") if s.strip()}
# Set PIPELINE_ON_START=0 to start the server without running the pipeline.
RUN_ON_START = os.environ.get("PIPELINE_ON_START", "1") != "0"

# Above this many detections in the view, the API answers with grid cells instead of points.
MAX_POINTS = int(os.environ.get("FIRMS_MAX_POINTS", "250000"))


def today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()
