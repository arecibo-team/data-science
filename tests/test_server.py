"""Tests for the SQLite store, the step ledger and the HTTP API (temporary database, no network)."""
import struct

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import db, settings
from server.pipeline import runner, store
from server.pipeline.runner import Plan, Step


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "RUN_ON_START", False)
    db.init_db()
    return tmp_path


def _frame():
    df = pd.DataFrame({
        "sensor": ["MODIS", "VIIRS", "VIIRS"],
        "platform": ["Terra", "SNPP", "NOAA-21"],
        "product": ["NRT", "SP", "NRT"],
        "lat": [10.0, 10.0, 41.0],
        "lon": [20.0, 20.0, -4.0],
        "acq_time_utc": pd.to_datetime(["2025-08-01 10:00", "2025-08-01 10:10", "2025-08-02 01:00"], utc=True),
        "daynight": ["D", "D", "N"],
        "confidence_class": ["high", "nominal", "low"],
        "frp_mw": [12.5, 2.0, np.nan],
        "bt_mid_k": [330.0, 340.0, np.nan],
        "cluster_id": pd.array([0, 0, None], dtype="Int64"),
        "harmonized_status": ["both", "both", "viirs_only"],
    })
    return df


def _decode(content):
    (hlen,) = struct.unpack_from("<I", content, 0)
    import json
    from server.app import COLUMNS
    header = json.loads(content[4:4 + hlen])
    n, off, cols = header["n"], 4 + hlen, {}
    for name, dt in COLUMNS:
        size = np.dtype(dt).itemsize * n
        cols[name] = np.frombuffer(content[off:off + size], dtype=dt)
        off += size
    assert off == len(content)
    return header, cols


# ---------- step ledger ----------

def _steps(monkeypatch, calls, fail=()):
    def mk(name, deps=(), params=None):
        def run():
            calls.append(name)
            if name in fail:
                raise RuntimeError("boom")
            return 7
        return Step(name, name, lambda: Plan(dict(params or {})), run, deps=deps)
    monkeypatch.setattr(runner, "STEPS", [mk("a"), mk("b", ("a",)), mk("c", ("b",))])


def test_steps_run_once_then_are_up_to_date(tmp_db, monkeypatch):
    calls = []
    _steps(monkeypatch, calls)
    assert runner.run_pending() == {"a": "done", "b": "done", "c": "done"}
    assert runner.run_pending() == {"a": "up_to_date", "b": "up_to_date", "c": "up_to_date"}
    assert calls == ["a", "b", "c"]
    with db.session() as con:
        rows = con.execute("SELECT step, status, row_count FROM pipeline_steps ORDER BY id").fetchall()
    assert [(r["step"], r["status"], r["row_count"]) for r in rows] == [("a", "done", 7), ("b", "done", 7), ("c", "done", 7)]


def test_reset_reruns_the_step_and_everything_after_it(tmp_db, monkeypatch):
    calls = []
    _steps(monkeypatch, calls)
    runner.run_pending()
    assert runner.reset_step("b") == ["b", "c"]
    runner.run_pending()
    assert calls == ["a", "b", "c", "b", "c"]  # `a` was kept


def test_reset_only_still_reruns_dependents_through_fingerprint(tmp_db, monkeypatch):
    calls = []
    _steps(monkeypatch, calls)
    runner.run_pending()
    runner.reset_step("a", cascade=False)
    runner.run_pending()
    assert calls == ["a", "b", "c", "a", "b", "c"]  # new `a` row id changes b's and c's fingerprints


def test_failed_step_is_recorded_and_blocks_dependents(tmp_db, monkeypatch):
    calls = []
    _steps(monkeypatch, calls, fail={"b"})
    out = runner.run_pending()
    assert out["a"] == "done" and out["b"] == "failed" and out["c"].startswith("blocked")
    assert calls == ["a", "b"]
    state = {r["step"]: r["state"] for r in runner.status()}
    assert state["b"] == "failed" and state["c"] == "pending"


def test_changed_params_run_the_step_again(tmp_db, monkeypatch):
    calls = []
    steps = []
    for end in ("2025-01-01", "2025-01-02"):
        monkeypatch.setattr(runner, "STEPS", [Step("fetch", "f", lambda e=end: Plan({"end": e}),
                                                   lambda: calls.append("fetch") or 1)])
        runner.run_pending()
    assert calls == ["fetch", "fetch"]


def test_unknown_step_reset(tmp_db):
    with pytest.raises(KeyError):
        runner.reset_step("nope")


# ---------- API ----------

def _client_with_data():
    store.write_detections(_frame())
    with db.session() as con:
        store.replace_incidents(con, "Copernicus EMS", [{
            "type": "Feature",
            "properties": {"incident_id": "EMS-1", "name": "Fire", "country": "ES",
                           "start_date": "2025-08-01", "end_date": "2025-08-03", "area_ha": 10.0},
            "geometry": {"type": "Polygon", "coordinates": [[[-5, 40], [-3, 40], [-3, 42], [-5, 42], [-5, 40]]]},
        }])
        con.execute("UPDATE detections SET incident_id='EMS-1' WHERE lat = 41")
        con.execute("INSERT INTO day_stats VALUES ('2025-08-01', 2, 1, 1, 2, 0), ('2025-08-02', 1, 0, 1, 0, 1)")
    from server.app import app
    return TestClient(app)


def test_points_binary_columns_and_filters(tmp_db):
    c = _client_with_data()
    r = c.get("/api/points", params={"from": "2025-08-01", "to": "2025-08-02"})
    header, cols = _decode(r.content)
    enums = db.ENUMS
    assert header["n"] == 3 and header["total"] == 3 and not header["overflow"]
    assert [enums["sensor"][v] for v in cols["sensor"]] == ["MODIS", "VIIRS", "VIIRS"]
    assert [enums["platform"][v] for v in cols["platform"]] == ["Terra", "SNPP", "NOAA-21"]
    assert [enums["status"][v] for v in cols["status"]] == ["both", "both", "viirs_only"]
    assert cols["cluster"].tolist() == [0, 0, -1]
    assert np.isnan(cols["bt"][2]) and cols["bt"][0] == np.float32(330.0)
    assert header["incidents"] == ["EMS-1"] and cols["incident"].tolist() == [0, 0, 1]
    assert cols["ts"][0] == int(pd.Timestamp("2025-08-01 10:00", tz="UTC").timestamp())

    one_day = _decode(c.get("/api/points", params={"from": "2025-08-02", "to": "2025-08-02"}).content)[0]
    assert one_day["n"] == 1
    box = _decode(c.get("/api/points", params={"from": "2025-08-01", "to": "2025-08-02",
                                                "west": 15, "east": 25, "south": 5, "north": 15}).content)[0]
    assert box["n"] == 2  # only the two near (10, 20)


def test_points_overflow_sends_only_the_total(tmp_db, monkeypatch):
    c = _client_with_data()
    monkeypatch.setattr(settings, "MAX_POINTS", 2)
    header, cols = _decode(c.get("/api/points", params={"from": "2025-08-01", "to": "2025-08-02"}).content)
    assert header["overflow"] and header["total"] == 3 and header["n"] == 0
    assert all(len(v) == 0 for v in cols.values())


# ---------- heat tiles ----------

def _png_alpha(png):
    """Decode the RGBA PNG written by server.tiles.encode_png (filter 0 only) into an (h, w, 4) array."""
    import zlib
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    while pos < len(png):
        (n,) = struct.unpack_from(">I", png, pos)
        kind = png[pos + 4:pos + 8]
        if kind == b"IHDR":
            w, h = struct.unpack_from(">II", png, pos + 8)
        if kind == b"IDAT":
            idat += png[pos + 8:pos + 8 + n]
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * 4 + 1)
    assert (raw[:, 0] == 0).all()
    return raw[:, 1:].reshape(h, w, 4)


def _tile_xy(z, lat, lon):
    from server import tiles
    px, py = tiles.project(np.array([lat]), np.array([lon]), z)
    return int(px[0] // 256), int(py[0] // 256), int(px[0] % 256), int(py[0] % 256)


def _client_with_bins():
    c = _client_with_data()
    with db.session() as con:
        con.execute("DELETE FROM bin_day")
        con.execute("""INSERT INTO bin_day SELECT CAST((lat + 90) / 0.1 AS INTEGER), CAST((lon + 180) / 0.1 AS INTEGER),
                       date(ts, 'unixepoch'), SUM(sensor = 1), SUM(sensor = 2),
                       SUM(sensor = 1 AND incident_id IS NOT NULL), SUM(sensor = 2 AND incident_id IS NOT NULL),
                       SUM(status = 1), SUM(confidence = 1), SUM(confidence = 3)
                       FROM detections GROUP BY 1, 2, 3""")
        con.execute("INSERT INTO pipeline_steps(step, fingerprint, status, started_at) VALUES ('stats', 'x', 'done', 'now')")
    return c


def test_tile_draws_density_at_the_right_pixel_and_hue_follows_layers(tmp_db, monkeypatch):
    monkeypatch.setattr(settings, "TILE_DIR", str(tmp_db / "tiles"))
    c = _client_with_bins()
    for z in (4, 10):  # bins at z4, raw detections at z10
        x, y, px, py = _tile_xy(z, 10.0, 20.0)
        r = c.get(f"/api/tiles/{z}/{x}/{y}.png", params={"from": "2025-08-01", "to": "2025-08-02", "layers": 7})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        img = _png_alpha(r.content)
        assert img.shape == (256, 256, 4)
        assert img[py, px, 3] > 20  # coloured (faintly for 2 detections) at the detection pixel
        assert img[(py + 128) % 256, (px + 128) % 256, 3] == 0  # and clear far from it
        only_modis = _png_alpha(c.get(f"/api/tiles/{z}/{x}/{y}.png", params={
            "from": "2025-08-01", "to": "2025-08-02", "layers": 1}).content)
        only_viirs = _png_alpha(c.get(f"/api/tiles/{z}/{x}/{y}.png", params={
            "from": "2025-08-01", "to": "2025-08-02", "layers": 2}).content)
        assert only_modis[py, px, 0] > only_modis[py, px, 2]   # MODIS is orange (red channel high)
        assert only_viirs[py, px, 2] > only_viirs[py, px, 0]   # VIIRS is blue
        none = c.get(f"/api/tiles/{z}/{x}/{y}.png", params={"from": "2025-08-01", "to": "2025-08-02", "layers": 0})
        assert _png_alpha(none.content)[..., 3].max() == 0


def test_tile_is_cached_per_zoom_and_invalidated_by_a_new_data_version(tmp_db, monkeypatch):
    from server import tiles
    monkeypatch.setattr(settings, "TILE_DIR", str(tmp_db / "tiles"))
    c = _client_with_bins()
    x, y, _, _ = _tile_xy(4, 10.0, 20.0)
    q = {"from": "2025-08-01", "to": "2025-08-02"}
    first = c.get(f"/api/tiles/4/{x}/{y}.png", params=q).content
    files = list((tmp_db / "tiles").rglob("*.png"))
    assert len(files) == 1 and files[0].read_bytes() == first

    calls = []
    real = tiles.render_from_db
    monkeypatch.setattr(tiles, "render_from_db", lambda *a, **k: calls.append(a) or real(*a, **k))
    assert c.get(f"/api/tiles/4/{x}/{y}.png", params=q).content == first
    assert calls == []  # served from the cache, nothing rendered

    x2, y2, _, _ = _tile_xy(5, 10.0, 20.0)
    c.get(f"/api/tiles/5/{x2}/{y2}.png", params=q)
    assert len(calls) == 1 and len(list((tmp_db / "tiles").rglob("*.png"))) == 2  # each LOD cached on its own

    with db.session() as con:
        con.execute("INSERT INTO pipeline_steps(step, fingerprint, status, started_at) VALUES ('stats', 'y', 'done', 'now')")
    c.get(f"/api/tiles/4/{x}/{y}.png", params=q)
    assert len(calls) == 2  # new data version: rendered again
    tiles.clear_old_versions(tiles.data_version(db.connect()))
    assert len(list((tmp_db / "tiles").rglob("*.png"))) == 1


def test_blank_tile_is_remembered_and_bad_tiles_are_rejected(tmp_db, monkeypatch):
    monkeypatch.setattr(settings, "TILE_DIR", str(tmp_db / "tiles"))
    c = _client_with_bins()
    r = c.get("/api/tiles/4/0/0.png", params={"from": "2025-08-01", "to": "2025-08-02"})
    assert r.status_code == 200 and _png_alpha(r.content)[..., 3].max() == 0
    marker = list((tmp_db / "tiles").rglob("*.png"))
    assert len(marker) == 1 and marker[0].stat().st_size == 0  # blank is remembered, as a zero-byte file
    again = c.get("/api/tiles/4/0/0.png", params={"from": "2025-08-01", "to": "2025-08-02"})
    assert again.status_code == 200 and again.content == r.content
    assert c.get("/api/tiles/2/9/0.png", params={"from": "2025-08-01", "to": "2025-08-02"}).status_code == 404
    assert c.get("/api/tiles/2/0/0.png", params={"from": "2025-08-01", "to": "2025-08-02", "layers": 9}).status_code == 422


# ---------- background tile generation ----------

def test_plan_lists_only_tiles_with_data_lowest_zoom_first(tmp_db):
    from server import tilegen
    _client_with_bins()
    con = db.connect()
    jobs = tilegen.plan(con, 4)
    assert [j.z for j in jobs] == sorted(j.z for j in jobs)
    assert {j.z for j in jobs} == {0, 1, 2, 3, 4}
    z4 = next(j for j in jobs if j.z == 4)
    x, y, _, _ = _tile_xy(4, 10.0, 20.0)
    assert [x, y] in z4.xy.tolist()
    assert len(z4.xy) < 40  # two spots, not the 256 tiles of the zoom level
    assert (np.array([0, 0]) == z4.xy).all(axis=1).sum() == 0  # the empty corner of the world is not planned


def test_service_renders_every_level_in_the_background_and_skips_existing(tmp_db, monkeypatch):
    from server import tilegen, tiles
    monkeypatch.setattr(settings, "TILE_DIR", str(tmp_db / "tiles"))
    monkeypatch.setattr(tilegen, "PAUSE_PER_TILE_S", 0)
    _client_with_bins()
    con = db.connect()
    version = tiles.data_version(con)
    svc = tilegen.TileService(max_zoom=5)
    assert svc.run_once(con, version) is True
    st = svc.status()
    assert st["state"] == "idle" and st["done"] == st["total"] > 0 and st["max_zoom"] == 5
    files = list((tmp_db / "tiles").rglob("*.png"))
    assert {int(f.parent.parent.name) for f in files} == {0, 1, 2, 3, 4, 5}  # every zoom level, each cached

    # A second pass finds everything in place and renders nothing.
    calls = []
    real = tiles.render_from_db
    monkeypatch.setattr(tiles, "render_from_db", lambda *a, **k: calls.append(a) or real(*a, **k))
    assert svc.run_once(con, version) is True and calls == []


def test_a_request_is_served_while_generation_is_still_running(tmp_db, monkeypatch):
    import threading
    from server import tilegen, tiles
    monkeypatch.setattr(settings, "TILE_DIR", str(tmp_db / "tiles"))
    c = _client_with_bins()
    con = db.connect()
    version = tiles.data_version(con)

    gate, started = threading.Event(), threading.Event()
    real = tiles.render_from_db

    def slow(*a, **k):  # the background thread is held inside its first tile
        if threading.current_thread().name == "bg-test":
            started.set()
            gate.wait(10)
        return real(*a, **k)

    monkeypatch.setattr(tiles, "render_from_db", slow)
    monkeypatch.setattr(tilegen, "PAUSE_PER_TILE_S", 0)
    svc = tilegen.TileService(max_zoom=5)
    t = threading.Thread(target=lambda: svc.run_once(db.connect(), version), name="bg-test")
    t.start()
    assert started.wait(10)
    assert svc.status()["state"] == "building"

    x, y, px, py = _tile_xy(6, 10.0, 20.0)  # a level the service has not reached, requested right now
    r = c.get(f"/api/tiles/6/{x}/{y}.png", params={"from": "2025-08-01", "to": "2025-08-02"})
    assert r.status_code == 200 and _png_alpha(r.content)[py, px, 3] > 0
    assert svc.status()["state"] == "building"  # the request did not wait for, or finish, the generation
    gate.set()
    t.join(20)
    assert svc.status()["state"] == "idle"


def test_service_stops_when_new_data_arrives(tmp_db, monkeypatch):
    from server import tilegen, tiles
    monkeypatch.setattr(settings, "TILE_DIR", str(tmp_db / "tiles"))
    monkeypatch.setattr(tilegen, "PAUSE_PER_TILE_S", 0)
    monkeypatch.setattr(tilegen, "CHECK_EVERY", 1)
    _client_with_bins()
    con = db.connect()
    version = tiles.data_version(con)
    real = tiles.render_from_db

    def newer(*a, **k):  # a pipeline pass finishes while tiles are being drawn
        with db.session() as w:
            w.execute("INSERT INTO pipeline_steps(step, fingerprint, status, started_at) VALUES ('stats','z','done','now')")
        return real(*a, **k)

    monkeypatch.setattr(tiles, "render_from_db", newer)
    svc = tilegen.TileService(max_zoom=6)
    assert svc.run_once(con, version) is False
    assert svc.completed_version == 0


def test_service_stands_by_while_the_pipeline_rewrites_detections(tmp_db):
    from server import tilegen
    _client_with_bins()
    svc = tilegen.TileService(max_zoom=2)
    con = db.connect()
    assert svc._unstable(con) is False
    with db.session() as w:
        w.execute("INSERT INTO pipeline_steps(step, fingerprint, status, started_at) VALUES ('tag','q','running','now')")
    assert svc._unstable(con) is True


def test_status_endpoint_reports_the_service(tmp_db):
    c = _client_with_bins()
    body = c.get("/api/tiles/status").json()
    assert body["enabled"] is True and body["state"] in {"starting", "waiting", "paused", "building", "idle"}
    assert c.get("/api/tiles/status").status_code == 200  # not captured by the {z}/{x}/{y} route


def test_counts_for_the_overview_follow_range_and_box(tmp_db):
    c = _client_with_bins()
    r = c.get("/api/counts", params={"from": "2025-08-01", "to": "2025-08-02"}).json()
    assert r == {"MODIS": 1, "VIIRS": 2, "COMMON": 1, "total": 3, "matched": 2,
                 "confidence": {"low": 1, "nominal": 1, "high": 1}}
    one_day = c.get("/api/counts", params={"from": "2025-08-02", "to": "2025-08-02"}).json()
    assert (one_day["MODIS"], one_day["VIIRS"], one_day["COMMON"], one_day["matched"]) == (0, 1, 1, 0)
    near = c.get("/api/counts", params={"from": "2025-08-01", "to": "2025-08-02",
                                        "west": 15, "east": 25, "south": 5, "north": 15}).json()
    assert (near["MODIS"], near["VIIRS"], near["total"]) == (1, 1, 2)  # Spain's detection is outside the box
    empty = c.get("/api/counts", params={"from": "2030-01-01", "to": "2030-01-02"}).json()
    assert empty["total"] == 0 and empty["confidence"]["nominal"] == 0


def test_old_bin_day_layout_is_dropped_so_the_stats_step_can_rebuild_it(tmp_db):
    with db.session() as con:
        con.execute("DROP TABLE bin_day")
        con.execute("CREATE TABLE bin_day (r INTEGER, c INTEGER, date TEXT, m INTEGER, v INTEGER, cm INTEGER, cv INTEGER)")
    db.init_db()
    with db.session() as con:
        assert "hi" in {r[1] for r in con.execute("PRAGMA table_info(bin_day)")}


def test_incidents_and_meta(tmp_db):
    c = _client_with_data()
    fc = c.get("/api/incidents", params={"from": "2025-08-02", "to": "2025-08-02"}).json()
    assert [f["properties"]["incident_id"] for f in fc["features"]] == ["EMS-1"]
    assert c.get("/api/incidents", params={"from": "2025-09-01", "to": "2025-09-02"}).json()["features"] == []
    assert c.get("/api/incidents", params={"from": "2025-08-02", "to": "2025-08-02",
                                           "west": 10, "east": 30, "south": 0, "north": 20}).json()["features"] == []
    meta = c.get("/api/meta").json()
    assert meta["days"]["2025-08-02"] == {"total": 1, "MODIS": 0, "VIIRS": 1, "matched": 0, "confirmed": 1}
    assert meta["enums"]["sensor"] == [None, "MODIS", "VIIRS"]


def test_bad_query_is_rejected(tmp_db):
    c = _client_with_data()
    assert c.get("/api/points", params={"from": "2025-08-03", "to": "2025-08-01"}).status_code == 422
    assert c.get("/api/points", params={"from": "nope", "to": "2025-08-01"}).status_code == 422
    assert c.get("/api/points", params={"from": "2025-08-01", "to": "2025-08-02", "west": 5, "east": 1}).status_code == 422


def test_rewriting_detections_replaces_them(tmp_db):
    store.write_detections(_frame())
    store.write_detections(_frame().iloc[:1])
    with db.session() as con:
        assert con.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 1
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
    assert "detections_build" not in names and "ix_det_ts" in names
