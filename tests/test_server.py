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


def test_grid_counts_per_cell(tmp_db):
    c = _client_with_data()
    body = c.get("/api/grid", params={"from": "2025-08-01", "to": "2025-08-02", "cell": 1}).json()
    assert body["cell_deg"] == 1.0
    # (lat 10, lon 20) at 1 degree: row 100, col 200, one MODIS and one VIIRS, none confirmed.
    assert [100, 200, 1, 1, 0, 0] in body["cells"]
    assert [131, 176, 0, 1, 0, 1] in body["cells"]  # (41, -4): confirmed VIIRS
    assert c.get("/api/grid", params={"from": "2025-08-01", "to": "2025-08-02", "cell": 3}).status_code == 422


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
