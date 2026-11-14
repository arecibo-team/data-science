"""The pipeline steps, in run order.

    fetch        FIRMS CSVs into data/raw               (network, needs a MAP_KEY, once per day)
    normalize    raw CSVs into data/normalized parquet  (only files not yet converted)
    harmonize    drop Africa, match MODIS/VIIRS clusters -> detections table, clusters table
    ems          Copernicus EMS wildfire areas -> incidents table   (network, once per day)
    sentinel2    Sentinel-2 burn scars -> incidents table           (network, slow, resumable)
    tag          mark detections that fall inside an incident
    context      embed context-data (CORINE bare ground) into the soil tables
    stats        per-day counts, per-cell counts (bin_day) and per-incident detection counts

Every step reads its settings from `server.settings`, so a changed setting is a new fingerprint
and the step runs again.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging

import pandas as pd

from firms import config as C
from firms.harmonize import MatchParams, cluster_table, match_cross_sensor
from firms.match import tag_incidents
from firms.region import in_africa
from firms.normalize import load_normalized, normalize_all

from .. import context, db, settings, weather_grid
from . import store
from .runner import STEPS, Plan, Step

log = logging.getLogger("server.steps")


# ---------------------------------------------------------------- fetch

def _fetch_plan() -> Plan:
    try:
        C.load_map_keys()
    except RuntimeError:
        return Plan(skip="no MAP_KEY in .env")
    return Plan({"start": settings.FETCH_START, "end": settings.today(),
                 "bbox": settings.FETCH_BBOX, "families": settings.FETCH_FAMILIES})


def _fetch_run() -> int:
    from firms.fetch import FirmsClient, fetch_all, parse_bbox
    paths = fetch_all(dt.date.fromisoformat(settings.FETCH_START), dt.date.fromisoformat(settings.today()),
                      parse_bbox(settings.FETCH_BBOX), settings.FETCH_FAMILIES, FirmsClient(C.load_map_keys()))
    return len(paths)


# ---------------------------------------------------------------- normalize

def _raw_listing() -> list[tuple[str, int]]:
    return sorted((str(p.relative_to(C.RAW_DIR)), p.stat().st_size) for p in C.RAW_DIR.glob("*/*.csv"))


def _normalize_plan() -> Plan:
    files = _raw_listing()
    if not files:
        return Plan(skip="no raw files in data/raw (run fetch, or add CSVs)")
    digest = hashlib.sha256(repr(files).encode()).hexdigest()[:16]
    return Plan({"raw_files": len(files), "raw_digest": digest})


def _normalize_run() -> int:
    return len(normalize_all())


# ---------------------------------------------------------------- harmonize

def _harmonize_plan() -> Plan:
    return Plan({"radius_km": settings.MATCH_RADIUS_KM, "window_min": settings.MATCH_WINDOW_MIN,
                 "exclude_africa": settings.EXCLUDE_AFRICA})


def _harmonize_run() -> int:
    df = load_normalized()
    log.info("loaded %d unique detections (NRT/SP deduplicated)", len(df))
    if settings.EXCLUDE_AFRICA:
        africa = in_africa(df["lat"], df["lon"])
        df = df[~africa].reset_index(drop=True)
        log.info("dropped %d detections on the African continent", int(africa.sum()))
    h = match_cross_sensor(df, MatchParams(settings.MATCH_RADIUS_KM, settings.MATCH_WINDOW_MIN))
    n = store.write_detections(h)
    clusters = cluster_table(h)
    for col in ("first_utc", "last_utc"):
        clusters[col] = clusters[col].astype(str)
    with db.session() as con:
        con.execute("DELETE FROM clusters")
        con.executemany(
            "INSERT INTO clusters VALUES (?,?,?,?,?,?,?,?,?)",
            clusters[["cluster_id", "n_detections", "n_modis", "n_viirs", "first_utc", "last_utc",
                      "max_frp_mw", "lat", "lon"]].astype(object).where(clusters.notna(), None).itertuples(index=False, name=None))
    return n


# ---------------------------------------------------------------- incidents

def _ems_plan() -> Plan:
    return Plan({"since": C.INCIDENTS_FROM, "on": settings.today()})


def _ems_run() -> int:
    from firms import incidents
    feats = incidents._normalize_ems(incidents._ems_fire_codes(C.INCIDENTS_FROM))
    with db.session() as con:
        return store.replace_incidents(con, "Copernicus EMS", feats)


def _s2_plan() -> Plan:
    return Plan({"max_events": settings.S2_MAX_EVENTS})


def _s2_run() -> int:
    from firms import sentinel2
    with db.session() as con:
        h = pd.read_sql_query("SELECT ts, lat, lon, frp AS frp_mw FROM detections ORDER BY ts", con)
    h["acq_time_utc"] = pd.to_datetime(h.pop("ts"), unit="s", utc=True)
    sentinel2.run(settings.S2_MAX_EVENTS, settings.S2_WORKERS, h)
    with db.session() as con:
        return store.replace_incidents(con, "Sentinel-2", sentinel2.confirmed_features())


# ---------------------------------------------------------------- tag + stats

def _incident_digest() -> str:
    """Content hash of the incident set, so tagging reruns only when the incidents really changed."""
    con = db.connect()
    try:
        rows = con.execute("SELECT incident_id, start_date, end_date, buffer_km, slack_days, area_ha "
                           "FROM incidents ORDER BY incident_id").fetchall()
    finally:
        con.close()
    return hashlib.sha256(repr([tuple(r) for r in rows]).encode()).hexdigest()[:16]


def _tag_plan() -> Plan:
    return Plan({"buffer_km": C.MATCH_BUFFER_KM, "slack_days": C.MATCH_DATE_SLACK_DAYS,
                 "incidents": _incident_digest()})


def _tag_run() -> int:
    with db.session() as con:
        feats = store.load_incident_features(con)
        h = pd.read_sql_query("SELECT id, lat, lon, ts FROM detections", con)
    h["acq_time_utc"] = pd.to_datetime(h["ts"], unit="s", utc=True)
    tags = tag_incidents(h, feats) if feats else pd.Series([None] * len(h), index=h.index, dtype=object)
    hit = tags.notna()
    with db.session() as con:
        con.execute("UPDATE detections SET incident_id=NULL WHERE incident_id IS NOT NULL")
        con.executemany("UPDATE detections SET incident_id=? WHERE id=?",
                        zip(tags[hit].tolist(), h.loc[hit, "id"].astype(int).tolist()))
    return int(hit.sum())


def _stats_plan() -> Plan:
    return Plan({"bin_schema": db.BIN_SCHEMA})


def _stats_run() -> int:
    with db.session() as con:
        con.execute("DELETE FROM day_stats")
        con.execute("""
            INSERT INTO day_stats
            SELECT date(ts, 'unixepoch'), COUNT(*), SUM(sensor=1), SUM(sensor=2), SUM(status=1),
                   SUM(incident_id IS NOT NULL)
            FROM detections GROUP BY 1""")
        con.execute("DELETE FROM bin_day")
        con.execute("""
            INSERT INTO bin_day
            SELECT CAST((lat + 90) / :d AS INTEGER), CAST((lon + 180) / :d AS INTEGER), date(ts, 'unixepoch'),
                   SUM(sensor = 1), SUM(sensor = 2),
                   SUM(sensor = 1 AND incident_id IS NOT NULL), SUM(sensor = 2 AND incident_id IS NOT NULL),
                   SUM(status = 1), SUM(confidence = 1), SUM(confidence = 3)
            FROM detections GROUP BY 1, 2, 3""", {"d": db.BIN_DEG})
        con.execute("UPDATE incidents SET detections = COALESCE("
                    "(SELECT COUNT(*) FROM detections d WHERE d.incident_id = incidents.incident_id), 0)")
        return con.execute("SELECT COUNT(*) FROM day_stats").fetchone()[0]


# ---------------------------------------------------------------- context, weather

def _context_plan() -> Plan:
    d = context.digest()
    return Plan({"files": d}) if d else Plan(skip="context-data files missing")


def _context_run() -> int:
    return context.load()


def _weather_plan() -> Plan:
    if not settings.WEATHER:
        return Plan(skip="FIRMS_WEATHER=0")
    return Plan({"start": settings.FETCH_START, "end": weather_grid.end_date()})


def _weather_run() -> int:
    return weather_grid.run(settings.FETCH_START)


STEPS.extend([
    Step("fetch", "Download FIRMS CSVs", _fetch_plan, _fetch_run),
    Step("normalize", "Normalize raw CSVs", _normalize_plan, _normalize_run, soft=("fetch",)),
    Step("harmonize", "Match MODIS and VIIRS, load detections", _harmonize_plan, _harmonize_run, deps=("normalize",)),
    Step("ems", "Copernicus EMS incidents", _ems_plan, _ems_run),
    Step("sentinel2", "Sentinel-2 burn scars", _s2_plan, _s2_run, deps=("harmonize",)),
    Step("tag", "Tag detections with incidents", _tag_plan, _tag_run, deps=("harmonize",), soft=("ems", "sentinel2")),
    Step("stats", "Day, cell and incident counts", _stats_plan, _stats_run, deps=("tag",)),
    Step("context", "Embed context-data (soil)", _context_plan, _context_run),
    Step("weather", "Daily weather on a 1 degree grid", _weather_plan, _weather_run, soft=("stats",)),
])
