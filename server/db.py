"""SQLite schema and helpers.

Tables
    detections        harmonized MODIS/VIIRS detections (one row per physical detection)
    incidents         independent European incidents (EMS areas, Sentinel-2 burn scars)
    clusters          one row per cross-sensor cluster
    day_stats         per UTC day counts, served as the date index of the map
    pipeline_steps    which pipeline steps ran, with what inputs, and how they ended

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

CREATE TABLE IF NOT EXISTS day_stats (
    date TEXT PRIMARY KEY, total INTEGER, modis INTEGER, viirs INTEGER, matched INTEGER, confirmed INTEGER
);
"""

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
