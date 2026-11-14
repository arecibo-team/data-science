"""Runtime settings, read from the environment (and .env for the FIRMS key)."""
from __future__ import annotations

import datetime as dt
import os

from firms import config as C

DB_PATH = os.environ.get("FIRMS_DB", str(C.DATA_DIR / "firms.db"))

# Fetch window and area. The end date defaults to today, so every day brings one new fetch step.
FETCH_START = os.environ.get("FIRMS_START", C.INCIDENTS_FROM)
# Europe: the same box that rejects non-European incidents (firms.config.EUROPE_BBOX), as W,S,E,N.
FETCH_BBOX = os.environ.get("FIRMS_BBOX", ",".join(str(v) for v in (C.EUROPE_BBOX[0], C.EUROPE_BBOX[1], C.EUROPE_BBOX[2], C.EUROPE_BBOX[3])))
FETCH_FAMILIES = [f for f in os.environ.get("FIRMS_FAMILIES", ",".join(C.FAMILIES)).split(",") if f]

# Drop detections on the African continent (the fetch box and older downloads reach into North Africa).
EXCLUDE_AFRICA = os.environ.get("FIRMS_EXCLUDE_AFRICA", "1") != "0"

MATCH_RADIUS_KM = float(os.environ.get("FIRMS_RADIUS_KM", "1.0"))
MATCH_WINDOW_MIN = int(os.environ.get("FIRMS_WINDOW_MIN", "180"))
S2_MAX_EVENTS = int(os.environ.get("FIRMS_S2_MAX_EVENTS", "300"))
S2_WORKERS = int(os.environ.get("FIRMS_S2_WORKERS", "4"))

# Comma-separated step names that are never run on start (e.g. "fetch,sentinel2" when offline).
SKIP_STEPS = {s.strip() for s in os.environ.get("PIPELINE_SKIP", "").split(",") if s.strip()}
# Set PIPELINE_ON_START=0 to start the server without running the pipeline.
RUN_ON_START = os.environ.get("PIPELINE_ON_START", "1") != "0"

# Above this many detections in the view, the API answers with grid cells instead of points.
MAX_POINTS = int(os.environ.get("FIRMS_MAX_POINTS", "60000"))


def today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()

# Heat tile cache: rendered PNGs per data version, date range, layer mask and zoom level.
TILE_DIR = os.environ.get("FIRMS_TILE_DIR", str(C.DATA_DIR / "tiles"))
TILE_CACHE_MB = int(os.environ.get("FIRMS_TILE_CACHE_MB", "2000"))
# Background tile generation (server/tilegen.py): every zoom level up to this one is rendered ahead of time
# for the preset date ranges; deeper zooms are rendered when first requested. Cost grows about 3x per level.
TILE_BG = os.environ.get("FIRMS_TILE_BG", "1") != "0"
TILE_BG_MAX_ZOOM = int(os.environ.get("FIRMS_TILE_BG_MAX_ZOOM", "9"))

# Daily weather features for the forecast model (NASA POWER, slow on first run, resumable).
WEATHER = os.environ.get("FIRMS_WEATHER", "1") != "0"
# Forecast model defaults.
HORIZON = int(os.environ.get("FIRMS_HORIZON", "3"))
TRAIN_END = os.environ.get("FIRMS_TRAIN_END", "2025-12-31")
CHECKPOINT_EVERY = int(os.environ.get("FIRMS_CHECKPOINT_EVERY", "25"))
# Label of the model: "confirmed" = the UI's COMMON layer (detections inside an EU incident), "both" = seen by MODIS and VIIRS.
LABEL = os.environ.get("FIRMS_LABEL", "confirmed")
# Parallel NASA POWER requests of the weather step. Lower it if the service answers 429.
WEATHER_WORKERS = int(os.environ.get("FIRMS_WEATHER_WORKERS", "4"))
