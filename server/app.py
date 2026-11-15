"""HTTP API for the map.

    GET /api/meta        enums, day index, incident summary, pipeline state
    GET /api/pipeline    the pipeline step ledger
    GET /api/points      detections in a date range and bounding box, as binary columns
    GET /api/tiles/z/x/y.png   heat tiles, for views with too many points (cached per zoom level)
    GET /api/tiles/status      progress of the background tile generation
    GET /api/counts      totals for a range and box, for the heat overview
    GET /api/incidents   incident polygons overlapping a date range and bounding box
    GET /api/weather     fire-weather features for a point and hour (NASA POWER / DWD ICON, cached in SQLite)

The built web app (web/dist) is served from / when it exists. The pipeline runs in a background
thread on start; the API serves whatever is in the database meanwhile.
"""
from __future__ import annotations

import json
import logging
import math
import queue
import struct
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, fire_weather, imports, pipeline, settings, soiltiles, tilegen, tiles
from .risk import serve as risk_serve
from .ml import jobs as ml_jobs, predict as ml_predict, regions as ml_regions, train as ml_train

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
    ml_train.mark_interrupted()
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


@app.get("/api/weather")
def api_weather(lon: float, lat: float, time: str = Query(..., description="ISO 8601 with timezone, e.g. 2024-07-01T12:00:00Z"),
                source: str = Query("auto", pattern="^(auto|nasa|dwd)$")):
    """Fire-weather features for the hour containing `time`. `source=auto` uses DWD ICON (Open-Meteo) for the
    last five days and up to six days ahead, NASA POWER for older times. Provider responses are cached in the
    weather_cache table (NASA 30 days, DWD 15 minutes)."""
    try:
        return fire_weather.get_fire_weather(lon, lat, time, source=source)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except fire_weather.WeatherDataError as exc:
        raise HTTPException(502, str(exc))


# ---------------------------------------------------------------- forecast model

class TrainRequest(BaseModel):
    rounds: int = Field(300, ge=1, le=5000)
    lr: float = Field(0.05, gt=0, le=1)
    horizon: int = Field(settings.HORIZON, ge=1, le=14)


@app.post("/api/model/train")
def api_model_train(req: TrainRequest):
    """Start a training run in the background. Progress is streamed from /api/model/stream."""
    run_id = ml_jobs.job.start(req.model_dump())
    if run_id is None:
        raise HTTPException(409, "a training run is already in progress")
    return {"run": run_id}


@app.post("/api/model/resume/{run_id}")
def api_model_resume(run_id: int):
    run = ml_train.get_run(run_id)
    if not run:
        raise HTTPException(404, "no such run")
    if run["status"] not in ("stopped", "interrupted", "failed"):
        raise HTTPException(409, f"run is {run['status']}")
    if ml_jobs.job.start(resume_id=run_id) is None:
        raise HTTPException(409, "a training run is already in progress")
    return {"run": run_id}


@app.post("/api/model/stop")
def api_model_stop():
    return {"stopping": ml_jobs.job.stop()}


@app.get("/api/model/runs")
def api_model_runs():
    active = ml_predict.latest_run()
    return {"running": ml_jobs.job.running(), "current": ml_jobs.job.run_id if ml_jobs.job.running() else None,
            "active": active["id"] if active else None, "runs": ml_train.list_runs()}


@app.get("/api/model/runs/{run_id}")
def api_model_run(run_id: int):
    run = ml_train.get_run(run_id)
    if not run:
        raise HTTPException(404, "no such run")
    return run


@app.get("/api/model/stream")
def api_model_stream():
    """Server-sent events: start, progress (round, train_loss, val_loss), checkpoint, done, stopped, error.
    On connect, the history of the current run is replayed first."""
    job = ml_jobs.job
    q = job.subscribe()
    replay = []
    if job.running() and job.run_id is not None:
        run = ml_train.get_run(job.run_id)
        if run:
            replay = [{"type": "progress", "round": h["round"], "total": run["rounds_total"],
                       "train_loss": h["train"], "val_loss": h["val"], "replay": True} for h in run["loss_history"]]
            replay.insert(0, {"type": "start", "run": run["id"], "total": run["rounds_total"], "replay": True})

    def gen():
        try:
            for ev in replay:
                yield f"data: {json.dumps(ev)}\n\n"
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    if not job.running() and q.empty():
                        return
                    continue
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            job.unsubscribe(q)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/forecast")
def api_forecast(date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$")):
    """Cells with the probability of a common fire in the next `horizon` days, for a forecast made on `date`
    (clamped to the last day with data). Binary: uint32 header length, JSON header, then lat, lon, p (f4) and
    actual (u1: a common fire really happened in the window)."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(422, "bad date")
    f = ml_predict.forecast(date)
    if f is None:
        raise HTTPException(404, "no completed model run yet")
    head = json.dumps({k: f[k] for k in ("run", "date", "feature_date", "horizon", "threshold", "complete")}
                      | {"n": len(f["p"])}).encode()
    body = struct.pack("<I", len(head)) + head + f["lat"].tobytes() + f["lon"].tobytes() + f["p"].tobytes() + f["actual"].tobytes()
    return Response(body, media_type="application/octet-stream")


@app.get("/api/risk")
def api_risk(date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$")):
    """Country level risk for a forecast made on `date`: expected number of cells with a confirmed fire in the
    next horizon days, and where it ranks against that country's own history (pct, 0..1)."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(422, "bad date")
    out = risk_serve.risk(date)          # the country risk model (server/risk), when it has been run
    if out is None:                      # otherwise the cell model's numbers summed per country
        run = ml_predict.latest_run()
        out = run and ml_regions.risk(run["id"], date, run["params"]["horizon"])
    if out is None:
        raise HTTPException(404, "no country risk for the newest run")
    return out


@app.get("/api/regions")
def api_regions():
    """Country outlines (Natural Earth, public domain), for the risk layer."""
    return Response(ml_regions.FILE.read_bytes(), media_type="application/geo+json",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/soil/tiles/{z}/{x}/{y}.png")
def api_soil_tile(z: int, x: int, y: int, cls: int | None = Query(None, ge=0, le=8)):
    """Bare-ground overlay tile from the embedded context data. cls limits it to one class (0..8)."""
    if not (0 <= z <= tiles.MAX_ZOOM and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
        raise HTTPException(404, "no such tile")
    con = db.connect(readonly=True)
    try:
        png = soiltiles.render(con, z, x, y, cls)
    finally:
        con.close()
    return Response(png or tiles.empty_tile(), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/soil/classes")
def api_soil_classes():
    from . import context
    return {"classes": context.SOIL_CLASSES, "colors": soiltiles.COLORS.astype(int).tolist()}


MAX_IMPORT_BYTES = 300 * 1024 * 1024
EXPORT_COLUMNS = ["latitude", "longitude", "acq_time_utc", "sensor", "platform", "product", "daynight", "confidence",
                  "frp_mw", "brightness_k", "harmonized_status", "cluster_id", "incident_id"]


@app.post("/api/import")
async def api_import(request: Request, name: str = Query("upload", max_length=120)):
    body = await request.body()
    if not body:
        raise HTTPException(422, "send the CSV as the request body")
    if len(body) > MAX_IMPORT_BYTES:
        raise HTTPException(413, "file too large, split it below 300 MB")
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, "the file is not UTF-8 text")
    try:
        saved = imports.save_import(text, name)
    except imports.ImportError_ as exc:
        raise HTTPException(422, str(exc))
    started = start_pipeline()
    return {"files": saved, "rows": sum(f["rows"] for f in saved), "pipeline_started": started,
            "pipeline_running": bool(_pipeline_thread and _pipeline_thread.is_alive())}


@app.get("/api/sources")
def api_sources():
    out = []
    for path in sorted(imports.C.RAW_DIR.glob("*/*.csv")):
        out.append({"source": path.parent.name, "file": path.name, "bytes": path.stat().st_size,
                    "imported": path.name.startswith("import_")})
    by = {}
    for f in out:
        s = by.setdefault(f["source"], {"source": f["source"], "files": 0, "bytes": 0})
        s["files"] += 1
        s["bytes"] += f["bytes"]
    return {"sources": list(by.values()), "files": len(out)}


def _export_frame(df: pd.DataFrame) -> pd.DataFrame:
    dec = lambda col, values: df[col].map(lambda v: values[int(v)] if v and 0 < int(v) < len(values) else "")
    return pd.DataFrame({
        "latitude": df["lat"].round(5), "longitude": df["lon"].round(5),
        "acq_time_utc": pd.to_datetime(df["ts"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%MZ"),
        "sensor": dec("sensor", db.SENSORS), "platform": dec("platform", db.PLATFORMS),
        "product": dec("product", db.PRODUCTS), "daynight": dec("daynight", db.DAYNIGHT),
        "confidence": dec("confidence", db.CONFIDENCE), "frp_mw": df["frp"].round(2), "brightness_k": df["bt"].round(2),
        "harmonized_status": dec("status", db.STATUS),
        "cluster_id": df["cluster_id"].astype("Int64"), "incident_id": df["incident_id"].fillna(""),
    })[EXPORT_COLUMNS]


@app.get("/api/export.csv")
def api_export(q: Query_ = Depends(_params), sensor: str = Query("all", pattern="^(all|MODIS|VIIRS)$")):
    where, args = q.WHERE, list(q.args)
    if sensor != "all":
        where += " AND sensor = ?"
        args.append(db.SENSORS.index(sensor))

    def rows():
        con = db.connect(readonly=True)
        try:
            yield ",".join(EXPORT_COLUMNS) + "\n"
            for chunk in pd.read_sql_query(
                    "SELECT lat, lon, ts, sensor, platform, product, daynight, confidence, frp, bt, status, cluster_id, "
                    f"incident_id FROM detections WHERE {where} ORDER BY ts", con, params=args, chunksize=50000):
                yield _export_frame(chunk).to_csv(index=False, header=False)
        finally:
            con.close()

    name = f"firms-harmonized-{q.date_from}-{q.date_to}.csv"
    return StreamingResponse(rows(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/summary")
def api_summary(q: Query_ = Depends(_params)):
    con = db.connect(readonly=True)
    try:
        rows = con.execute(
            "SELECT sensor, platform, product, status, confidence, COUNT(*) n, AVG(frp) frp, MAX(frp) max_frp, "
            f"SUM(incident_id IS NOT NULL) confirmed FROM detections WHERE {q.WHERE} "
            "GROUP BY sensor, platform, product, status, confidence", q.args).fetchall()
        days = con.execute(
            f"SELECT date(ts, 'unixepoch') d, SUM(sensor=1) m, SUM(sensor=2) v, SUM(status=1) b FROM detections "
            f"WHERE {q.WHERE} GROUP BY 1 ORDER BY 1", q.args).fetchall()
        span = con.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM detections").fetchone()
    finally:
        con.close()
    pick = lambda values, v: values[v] if v and v < len(values) else None
    total = sum(r["n"] for r in rows)
    by_platform, by_status, by_conf = {}, {}, {}
    for r in rows:
        key = f"{pick(db.SENSORS, r['sensor'])} {pick(db.PLATFORMS, r['platform'])}"
        by_platform[key] = by_platform.get(key, 0) + r["n"]
        st = pick(db.STATUS, r["status"]) or "unknown"
        by_status[st] = by_status.get(st, 0) + r["n"]
        cf = pick(db.CONFIDENCE, r["confidence"]) or "unknown"
        by_conf[cf] = by_conf.get(cf, 0) + r["n"]
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    as_day = lambda t: (epoch + timedelta(seconds=t)).date().isoformat() if t is not None else None
    return {
        "total": total, "confirmed": sum(r["confirmed"] or 0 for r in rows),
        "mean_frp": round(sum((r["frp"] or 0) * r["n"] for r in rows) / total, 2) if total else 0,
        "max_frp": max((r["max_frp"] or 0 for r in rows), default=0),
        "by_platform": by_platform, "by_status": by_status, "by_confidence": by_conf,
        "days": [{"date": d["d"], "modis": d["m"], "viirs": d["v"], "both": d["b"]} for d in days],
        "data": {"first": as_day(span[0]), "last": as_day(span[1]), "detections": span[2]},
    }


if WEB_DIST.is_dir():  # after the API routes so /api wins
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
