"""SQLite schema and helpers.

Tables
    detections        harmonized MODIS/VIIRS detections (one row per physical detection)
    incidents         independent European incidents (EMS areas, Sentinel-2 burn scars)
    clusters          one row per cross-sensor cluster
    day_stats         per UTC day counts, served as the date index of the map
    bin_day           per day and 0.1 degree cell counts, the source of low-zoom heat tiles
    pipeline_steps    which pipeline steps ran, with what inputs, and how they ended
    weather_cache     cached NASA POWER / DWD ICON responses for /api/weather

Categorical detection columns are small integers. Code 0 means missing and the value lists
live in ENUMS, which the API also hands to the web app, so the browser never hardcodes them.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from . import settings

SENSORS = [None, "MODIS", "VIIRS"]
DAYNIGHT = [None, "D", "N"]
CONFIDENCE = [None, "low", "nominal", "high"]
STATUS = [None, "both", "modis_only", "viirs_only"]
PLATFORMS = [None, "Terra", "Aqua", "SNPP", "NOAA-20", "NOAA-21"]
PRODUCTS = [None, "SP", "NRT"]

ENUMS = {
    "sensor": SENSORS, "platform": PLATFORMS, "product": PRODUCTS,
    "daynight": DAYNIGHT, "confidence": CONFIDENCE, "status": STATUS,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_steps (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    step         TEXT NOT NULL,
    fingerprint  TEXT NOT NULL,
    params       TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL CHECK (status IN ('running', 'done', 'failed')),
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    duration_s   REAL,
    row_count    INTEGER,
    message      TEXT
);
CREATE INDEX IF NOT EXISTS ix_steps_step ON pipeline_steps(step, status);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id  TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    name         TEXT,
    country      TEXT,
    start_date   TEXT NOT NULL,
    end_date     TEXT NOT NULL,
    area_ha      REAL,
    buffer_km    REAL,
    slack_days   INTEGER,
    detections   INTEGER NOT NULL DEFAULT 0,
    west REAL, south REAL, east REAL, north REAL,
    geometry     TEXT NOT NULL,          -- full-resolution GeoJSON, used for matching
    geometry_simple TEXT NOT NULL        -- simplified GeoJSON, served to the map
);
CREATE INDEX IF NOT EXISTS ix_incidents_dates ON incidents(start_date, end_date);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY, n_detections INTEGER, n_modis INTEGER, n_viirs INTEGER,
    first_utc TEXT, last_utc TEXT, max_frp_mw REAL, lat REAL, lon REAL
);

-- Detections per day and 0.1 degree cell, for the low-zoom heat tiles (see server/tiles.py).
CREATE TABLE IF NOT EXISTS bin_day (
    r INTEGER NOT NULL, c INTEGER NOT NULL, date TEXT NOT NULL,
    m INTEGER NOT NULL, v INTEGER NOT NULL, cm INTEGER NOT NULL, cv INTEGER NOT NULL,
    b INTEGER NOT NULL,                 -- matched by both sensors
    lo INTEGER NOT NULL, hi INTEGER NOT NULL,   -- low / high confidence
    PRIMARY KEY (r, c, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS day_stats (
    date TEXT PRIMARY KEY, total INTEGER, modis INTEGER, viirs INTEGER, matched INTEGER, confirmed INTEGER
);

-- Raw provider responses of /api/weather (see server/fire_weather.py), keyed by sha256 of the request URL.
CREATE TABLE IF NOT EXISTS weather_cache (
    key        TEXT PRIMARY KEY,
    provider   TEXT NOT NULL,       -- nasa_power | dwd_icon
    url        TEXT NOT NULL,
    fetched_at REAL NOT NULL,       -- unix seconds
    expires_at REAL NOT NULL,
    body       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_weather_expires ON weather_cache(expires_at);
-- CORINE bare ground, embedded from context-data by the `context` step. Cell index = floor((coord + offset) / 0.25).
CREATE TABLE IF NOT EXISTS soil_cells (
    r4 INTEGER NOT NULL, c4 INTEGER NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL,
    sand REAL, rock REAL, sparse REAL, burnt REAL, glacier REAL, intertidal REAL,
    quarry REAL, dump REAL, construction REAL,
    PRIMARY KEY (r4, c4)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS soil_squares (    -- 1.6 km squares with at least 25% bare ground
    lat REAL NOT NULL, lon REAL NOT NULL, class INTEGER NOT NULL, class_pct REAL, all_bare_pct REAL
);
CREATE INDEX IF NOT EXISTS ix_soil_sq ON soil_squares(lat, lon);

-- Daily weather on a 1 degree grid (see server/weather_grid.py), the model's weather features.
CREATE TABLE IF NOT EXISTS weather_grid (
    r1 INTEGER NOT NULL, c1 INTEGER NOT NULL, date TEXT NOT NULL,
    t2m REAL, rh REAL, ws REAL, prec REAL, gwettop REAL, gwetroot REAL, gwetprof REAL,
    PRIMARY KEY (r1, c1, date)
) WITHOUT ROWID;

-- Forecast model runs and their weights (server/ml). One rolling checkpoint per run, plus the final model.
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('running','done','failed','stopped','interrupted')),
    params TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
    rounds_done INTEGER NOT NULL DEFAULT 0, rounds_total INTEGER NOT NULL,
    metrics TEXT, loss_history TEXT NOT NULL DEFAULT '[]', message TEXT
);
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, round INTEGER NOT NULL, created_at TEXT NOT NULL,
    train_loss REAL, val_loss REAL, is_final INTEGER NOT NULL DEFAULT 0, weights BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ckpt_run ON model_checkpoints(run_id, id);
CREATE TABLE IF NOT EXISTS forecast_cache (
    run_id INTEGER NOT NULL, date TEXT NOT NULL, r INTEGER NOT NULL, c INTEGER NOT NULL, p REAL NOT NULL,
    PRIMARY KEY (run_id, date, r, c)
) WITHOUT ROWID;
"""

BIN_DEG = 0.1  # cell size of bin_day, in degrees
BIN_SCHEMA = 2  # bump when bin_day's columns change; the stats step then rebuilds it

DETECTIONS_DDL = """
CREATE TABLE {name} (
    id         INTEGER PRIMARY KEY,
    ts         INTEGER NOT NULL,   -- acquisition time, unix seconds UTC
    lat        REAL NOT NULL,
    lon        REAL NOT NULL,
    frp        REAL,
    bt         REAL,
    cluster_id INTEGER,
    incident_id TEXT,
    sensor     INTEGER NOT NULL,
    platform   INTEGER,
    product    INTEGER,
    daynight   INTEGER,
    confidence INTEGER,
    status     INTEGER
)
"""
DETECTIONS_INDEXES = (
    "CREATE INDEX ix_det_ts ON detections(ts, lat, lon)",
    "CREATE INDEX ix_det_incident ON detections(incident_id) WHERE incident_id IS NOT NULL",
)


def connect(path: str | None = None, readonly: bool = False) -> sqlite3.Connection:
    path = path or settings.DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30, check_same_thread=False)
    else:
        con = sqlite3.connect(path, timeout=60, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA temp_store=MEMORY")
    return con


def init_db(path: str | None = None) -> None:
    con = connect(path)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(bin_day)")}
        if cols and "hi" not in cols:  # older layout: regenerable, rebuilt by the stats step
            con.execute("DROP TABLE bin_day")
        con.executescript(SCHEMA)
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name='detections'").fetchone():
            con.execute(DETECTIONS_DDL.format(name="detections"))
            for ddl in DETECTIONS_INDEXES:
                con.execute(ddl)
        con.commit()
    finally:
        con.close()


@contextmanager
def session(path: str | None = None):
    con = connect(path)
    try:
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()
