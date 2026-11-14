"""HTTP API for the map.

    GET /api/meta        enums, day index, incident summary, pipeline state
    GET /api/pipeline    the pipeline step ledger
    GET /api/points      detections in a date range and bounding box, as binary columns
    GET /api/tiles/z/x/y.png   heat tiles, for views with too many points (cached per zoom level)
    GET /api/tiles/status      progress of the background tile generation
    GET /api/counts      totals for a range and box, for the heat overview
    GET /api/incidents   incident polygons overlapping a date range and bounding box

The built web app (web/dist) is served from / when it exists. The pipeline runs in a background
thread on start; the API serves whatever is in the database meanwhile.
"""
from __future__ import annotations

import json
import logging
import math
import struct
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from . import db, pipeline, settings, tilegen, tiles

log = logging.getLogger("server")
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
FORMAT_VERSION = 4

# Binary column order of /api/points. The web loader reads them in exactly this order.
COLUMNS = [
    ("lat", "<f4"), ("lon", "<f4"), ("frp", "<f4"), ("bt", "<f4"), ("cluster", "<i4"),
    ("ts", "<u4"), ("incident", "<u2"),
    ("sensor", "u1"), ("platform", "u1"), ("product", "u1"),
    ("daynight", "u1"), ("confidence", "u1"), ("status", "u1"),
]

_pipeline_thread: threading.Thread | None = None


def _run_pipeline() -> None:
    try:
        out = pipeline.run_pending()
        log.info("pipeline finished: %s", out)
    except Exception:
        log.exception("pipeline crashed")


def start_pipeline() -> bool:
    """Run pending steps in a background thread. False when one is already running."""
    global _pipeline_thread
    if _pipeline_thread and _pipeline_thread.is_alive():
        return False
    _pipeline_thread = threading.Thread(target=_run_pipeline, name="pipeline", daemon=True)
    _pipeline_thread.start()
    return True


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    if settings.RUN_ON_START:
        start_pipeline()
    if settings.TILE_BG:
        tilegen.service.start()
    yield
    tilegen.service.stop()


app = FastAPI(title="FIRMS hotspots", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=4)


# ---------------------------------------------------------------- query parsing

class Query_:
    """Validated date range and bounding box shared by the data endpoints."""

    def __init__(self, date_from: str, date_to: str, west: float, south: float, east: float, north: float):
        try:
            d0, d1 = date.fromisoformat(date_from), date.fromisoformat(date_to)
        except ValueError:
            raise HTTPException(422, "from/to must be YYYY-MM-DD")
        if d1 < d0:
            raise HTTPException(422, "to is before from")
        if not (south <= north and west <= east):
            raise HTTPException(422, "bounding box must have west<=east and south<=north")
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        self.date_from, self.date_to = date_from, date_to
        self.t0 = int((datetime.combine(d0, datetime.min.time(), timezone.utc) - epoch).total_seconds())
        self.t1 = int((datetime.combine(d1 + timedelta(days=1), datetime.min.time(), timezone.utc) - epoch).total_seconds())
        self.west, self.east = max(west, -180.0), min(east, 180.0)
        self.south, self.north = max(south, -90.0), min(north, 90.0)

    @property
    def key(self) -> tuple:
        return (self.t0, self.t1, self.west, self.south, self.east, self.north)

    WHERE = "ts >= ? AND ts < ? AND lat >= ? AND lat <= ? AND lon >= ? AND lon <= ?"

    @property
    def args(self) -> tuple:
        return (self.t0, self.t1, self.south, self.north, self.west, self.east)


def _params(date_from: str = Query(..., alias="from"), date_to: str = Query(..., alias="to"),
            west: float = -180, south: float = -90, east: float = 180, north: float = 90) -> Query_:
    return Query_(date_from, date_to, west, south, east, north)


def _version(con) -> int:
    """Changes whenever a pipeline step finishes; keys the response cache."""
    return con.execute("SELECT COALESCE(MAX(id), 0) FROM pipeline_steps WHERE status='done'").fetchone()[0]


# ---------------------------------------------------------------- endpoints

@app.get("/api/pipeline")
def api_pipeline():
    return {"running": bool(_pipeline_thread and _pipeline_thread.is_alive()), "steps": pipeline.status()}


@app.get("/api/meta")
def api_meta():
    con = db.connect(readonly=True)
    try:
        days = {r["date"]: {"total": r["total"], "MODIS": r["modis"], "VIIRS": r["viirs"],
                            "matched": r["matched"], "confirmed": r["confirmed"]}
                for r in con.execute("SELECT * FROM day_stats ORDER BY date")}
        inc = con.execute("SELECT COUNT(*) n, COALESCE(SUM(detections), 0) c FROM incidents").fetchone()
        sources = [r[0] for r in con.execute("SELECT DISTINCT source FROM incidents ORDER BY 1")]
        last = con.execute("SELECT MAX(finished_at) FROM pipeline_steps WHERE status='done'").fetchone()[0]
    finally:
        con.close()
    return {
        "version": FORMAT_VERSION,
        "generated_utc": last,
        "enums": db.ENUMS,
        "incidents": {"count": inc["n"], "confirmed": inc["c"], "sources": sources},
        "days": days,
        "pipeline_running": bool(_pipeline_thread and _pipeline_thread.is_alive()),
    }


def _encode_points(df: pd.DataFrame, total: int, overflow: bool) -> bytes:
    """uint32 header length, JSON header, then each column as raw little-endian bytes."""
    if overflow:
        header = {"n": 0, "total": total, "overflow": True, "incidents": []}
        h = json.dumps(header, separators=(",", ":")).encode()
        return struct.pack("<I", len(h)) + h
    codes, uniques = pd.factorize(df["incident_id"].dropna())
    inc = np.zeros(len(df), dtype="<u2")
    if len(uniques) > 65534:
        raise HTTPException(500, "too many distinct incidents in one response")
    inc[df["incident_id"].notna().to_numpy()] = codes + 1
    nan = np.nan
    cols = {
        "lat": df["lat"].to_numpy("<f4"),
        "lon": df["lon"].to_numpy("<f4"),
        "frp": df["frp"].to_numpy("<f4", na_value=nan),
        "bt": df["bt"].to_numpy("<f4", na_value=nan),
        "cluster": df["cluster_id"].fillna(-1).to_numpy("<i4"),
        "ts": df["ts"].to_numpy("<u4"),
        "incident": inc,
    }
    for name in ("sensor", "platform", "product", "daynight", "confidence", "status"):
        cols[name] = df[name].fillna(0).to_numpy("u1")
    header = {"n": len(df), "total": total, "overflow": False, "incidents": uniques.tolist()}
    h = json.dumps(header, separators=(",", ":")).encode()
    body = b"".join(np.ascontiguousarray(cols[n], dtype=dt).tobytes() for n, dt in COLUMNS)
    return struct.pack("<I", len(h)) + h + body


@app.get("/api/points")
def api_points(q: Query_ = Depends(_params)):
    """Detections in the range and box. When there are more than MAX_POINTS, only the total is sent
    (`overflow: true`) and the client shows /api/tiles instead."""
    con = db.connect(readonly=True)
    try:
        total = con.execute(f"SELECT COUNT(*) FROM detections WHERE {q.WHERE}", q.args).fetchone()[0]
        if total > settings.MAX_POINTS:
            return Response(_encode_points(pd.DataFrame(), total, True), media_type="application/octet-stream")
        df = pd.read_sql_query(
            "SELECT lat, lon, frp, bt, cluster_id, ts, incident_id, sensor, platform, product, daynight, "
            f"confidence, status FROM detections WHERE {q.WHERE} ORDER BY ts", con, params=q.args)
    finally:
        con.close()
    return Response(_encode_points(df, total, False), media_type="application/octet-stream")


@app.get("/api/tiles/status")
def api_tiles_status():
    """Progress of the background tile generation. Tiles can be requested at any time regardless."""
    if not settings.TILE_BG:
        return {"enabled": False, "state": "off"}
    return tilegen.service.status()


@app.get("/api/tiles/{z}/{x}/{y}.png")
def api_tile(z: int, x: int, y: int, date_from: str = Query(..., alias="from"), date_to: str = Query(..., alias="to"),
             layers: int = 7):
    """Heat tile of the detections in a date range. `layers` is a bit mask: MODIS=1, VIIRS=2, COMMON=4."""
    if not (0 <= z <= tiles.MAX_ZOOM and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
        raise HTTPException(404, "no such tile")
    if not 0 <= layers <= 7:
        raise HTTPException(422, "layers must be 0..7")
    Query_(date_from, date_to, -180, -90, 180, 90)  # validates the dates
    con = db.connect(readonly=True)
    try:
        version = tiles.data_version(con)
        png = tiles.get_tile(con, z, x, y, date_from, date_to, layers)
    finally:
        con.close()
    return Response(png, media_type="image/png", headers={
        "Cache-Control": "public, max-age=3600", "ETag": f'"{version}"'})


@app.get("/api/counts")
def api_counts(q: Query_ = Depends(_params)):
    """Totals for the heat overview, where single detections are not sent. Summed from the per-day,
    per-0.1 degree cell table over the cells whose centre is inside the box, so they are exact for the
    date range and off by at most one cell width at the edge of the view."""
    deg = db.BIN_DEG
    con = db.connect(readonly=True)
    try:
        r = con.execute(
            "SELECT COALESCE(SUM(m), 0), COALESCE(SUM(v), 0), COALESCE(SUM(cm), 0), COALESCE(SUM(cv), 0), "
            "COALESCE(SUM(b), 0), COALESCE(SUM(lo), 0), COALESCE(SUM(hi), 0) FROM bin_day "
            "WHERE date >= ? AND date <= ? AND r BETWEEN ? AND ? AND c BETWEEN ? AND ?",
            (q.date_from, q.date_to,
             math.ceil((q.south + 90) / deg - 0.5), math.floor((q.north + 90) / deg - 0.5),
             math.ceil((q.west + 180) / deg - 0.5), math.floor((q.east + 180) / deg - 0.5))).fetchone()
    finally:
        con.close()
    m, v, cm, cv, both, low, high = r
    return {"MODIS": m, "VIIRS": v, "COMMON": cm + cv, "total": m + v, "matched": both,
            "confidence": {"low": low, "high": high, "nominal": m + v - low - high}}


@app.get("/api/incidents")
def api_incidents(q: Query_ = Depends(_params)):
    """GeoJSON of incidents whose dates overlap the range and whose bounds touch the box."""
    con = db.connect(readonly=True)
    try:
        rows = con.execute(
            "SELECT * FROM incidents WHERE start_date <= ? AND end_date >= ? "
            "AND east >= ? AND west <= ? AND north >= ? AND south <= ? ORDER BY incident_id",
            (q.date_to, q.date_from, q.west, q.east, q.south, q.north)).fetchall()
    finally:
        con.close()
    feats = ['{"type":"Feature","properties":%s,"geometry":%s}' % (json.dumps({
        "incident_id": r["incident_id"], "source": r["source"], "name": r["name"], "country": r["country"] or "",
        "start_date": r["start_date"], "end_date": r["end_date"], "area_ha": r["area_ha"],
        "detections": r["detections"]}, separators=(",", ":")), r["geometry_simple"]) for r in rows]
    return Response('{"type":"FeatureCollection","features":[%s]}' % ",".join(feats), media_type="application/json")


@app.get("/api/incidents/{incident_id}")
def api_incident(incident_id: str):
    con = db.connect(readonly=True)
    try:
        r = con.execute("SELECT incident_id, source, name, country, start_date, end_date, area_ha, detections "
                        "FROM incidents WHERE incident_id = ?", (incident_id,)).fetchone()
    finally:
        con.close()
    if not r:
        raise HTTPException(404, "unknown incident")
    return dict(r)


if WEB_DIST.is_dir():  # after the API routes so /api wins
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
