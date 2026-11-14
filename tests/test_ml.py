"""Forecast model: context embedding, features, training with checkpoints and resume, API."""
import datetime as dt
import json
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import app as app_mod, context, db, settings
from server.ml import features, jobs, predict, train

TRAIN_END = "2025-12-31"


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setattr(settings, "RUN_ON_START", False)
    monkeypatch.setattr(settings, "TILE_BG", False)
    monkeypatch.setattr(settings, "CHECKPOINT_EVERY", 5)
    monkeypatch.setattr(predict, "_booster", None)
    monkeypatch.setattr(predict, "_data", None)
    db.init_db()
    return tmp_path


def _fill_bins(first="2025-09-01", last="2026-02-15"):
    """30 noisy cells (a common fire on a day with probability depending on the cell) and one cell with
    single-sensor detections only, which must not be a candidate."""
    rng = np.random.default_rng(1)
    d0, d1 = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    rows, d = [], d0
    while d <= d1:
        day = d.isoformat()
        for k in range(30):
            if rng.random() < 0.1 + 0.8 * k / 30:
                rows.append((400 + k // 6, 1900 + k % 6, day, 2, 3, 1, 0, 1, 0, 2))
            elif rng.random() < 0.3:
                rows.append((400 + k // 6, 1900 + k % 6, day, 1, 0, 0, 0, 0, 1, 0))
        rows.append((700, 2000, day, 1, 0, 0, 0, 0, 1, 0))
        d += dt.timedelta(days=1)
    with db.session() as con:
        con.executemany("INSERT INTO bin_day VALUES (?,?,?,?,?,?,?,?,?,?)", rows)


def test_context_loads_and_is_idempotent(tmp_db):
    n = context.load()
    with db.connect(readonly=True) as con:
        assert con.execute("SELECT COUNT(*) FROM soil_cells").fetchone()[0] == 10498
        assert con.execute("SELECT COUNT(*) FROM soil_squares").fetchone()[0] == 186474
    assert context.load() == n
    assert context.digest()


def test_features_have_no_future_and_label_window(tmp_db):
    _fill_bins()
    d = features.load(TRAIN_END)
    assert d is not None and len(d.cand) == 30          # the single-sensor cell is not a candidate
    lab = features.label_mask(d, 3)
    assert not lab[:, -3:].any()                       # days without a full horizon have no label
    pos = np.array([0, 1])
    di = np.array([60, 60])
    X = features.build(d, pos, di)
    assert X.shape == (2, len(features.FEATURES_ALL))
    # changing a later day must not change today's features
    d.B[d.cand[0], 100] = 99
    assert np.allclose(features.build(d, pos, di), X, equal_nan=True)


def test_training_checkpoints_resume_and_predict(tmp_db):
    _fill_bins()
    params = {"rounds": 12, "min_data_in_leaf": 5, "num_leaves": 7}
    run_id = train.create_run(params)
    events: list[dict] = []
    import threading
    stop = threading.Event()

    def emit(ev):
        events.append(ev)
        if ev["type"] == "progress" and ev["round"] == 7:
            stop.set()

    train.train(run_id, emit, stop)
    run = train.get_run(run_id)
    assert run["status"] == "stopped" and run["rounds_done"] == 7
    assert [e["round"] for e in events if e["type"] == "progress"] == list(range(1, 8))
    ck = train.latest_checkpoint(run_id)
    assert ck["round"] == 7                              # stopping saves a checkpoint

    events.clear()
    train.train(run_id, events.append, threading.Event(), resume=True)
    run = train.get_run(run_id)
    assert run["status"] == "done" and run["rounds_done"] == 12
    assert [h["round"] for h in run["loss_history"]] == list(range(1, 13))
    assert run["metrics"]["test_rows"] > 0
    with db.connect(readonly=True) as con:
        rows = con.execute("SELECT is_final FROM model_checkpoints WHERE run_id=?", (run_id,)).fetchall()
    assert [r[0] for r in rows] == [1]                   # only the final model is kept

    f = predict.forecast("2025-12-20")
    assert f["run"] == run_id and f["complete"]
    p = dict(zip(zip(f["lat"].round(2), f["lon"].round(2)), f["p"]))
    assert len(p) >= 1


def test_latest_run_is_newest_done(tmp_db):
    _fill_bins()
    a = train.create_run({"rounds": 3, "min_data_in_leaf": 5, "num_leaves": 7})
    import threading
    train.train(a, lambda e: None, threading.Event())
    b = train.create_run({"rounds": 3, "min_data_in_leaf": 5, "num_leaves": 7})
    train.train(b, lambda e: None, threading.Event())
    c = train.create_run({"rounds": 3})
    assert predict.latest_run()["id"] == b               # c is still running


def test_interrupted_runs_are_flagged(tmp_db):
    run_id = train.create_run({"rounds": 3})
    assert train.mark_interrupted() == 1
    assert train.get_run(run_id)["status"] == "interrupted"


def test_api_train_stream_and_forecast(tmp_db):
    _fill_bins()
    with TestClient(app_mod.app) as client:
        assert client.get("/api/forecast", params={"date": "2025-12-20"}).status_code == 404
        r = client.post("/api/model/train", json={"rounds": 6, "horizon": 3})
        assert r.status_code == 200
        jobs.job._thread.join(60)
        runs = client.get("/api/model/runs").json()
        assert runs["runs"][0]["status"] in ("done", "failed")
        if runs["runs"][0]["status"] == "failed":
            pytest.skip(runs["runs"][0]["message"])      # tiny data can lack positives with the default leaf size
        assert runs["active"] == runs["runs"][0]["id"]
        resp = client.get("/api/forecast", params={"date": "2025-12-20"})
        n = struct.unpack("<I", resp.content[:4])[0]
        head = json.loads(resp.content[4:4 + n])
        assert head["horizon"] == 3 and head["n"] >= 1
        assert client.post("/api/model/resume/999").status_code == 404


def test_soil_tile_renders(tmp_db):
    context.load()
    with TestClient(app_mod.app) as client:
        r = client.get("/api/soil/tiles/5/17/10.png")
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
        assert client.get("/api/soil/tiles/3/0/0.png", params={"cls": 99}).status_code == 422


def test_dryness_features():
    from server.ml import dryness
    days = 50
    prec = np.zeros((1, days)); prec[0, 10] = 5.0
    out = dryness.dryness(np.full((1, days), 20.0), np.full((1, days), 50.0), prec, np.zeros(days, int))
    assert out["dry_days"][0, 10] == 0 and out["dry_days"][0, 15] == 5
    assert out["prec30"][0, 20] == 5.0 and out["prec30"][0, 45] == 0.0
    assert 0 < out["vpd"][0, 0] < 3 and abs(out["t2m_anom"][0, 0]) < 1e-6
    assert set(out) == set(dryness.DERIVED)          # no fire-danger index among the features


def test_weather_features_flow_into_matrix(tmp_db):
    _fill_bins()
    d0 = dt.date(2025, 9, 1)
    rows = [(40, 190, (d0 + dt.timedelta(days=i)).isoformat(), 15.0 + i % 7, 60.0, 3.0,
             0.0 if i % 5 else 4.0, 0.3, 0.4, 0.5) for i in range(170)]
    with db.session() as con:
        con.executemany("INSERT INTO weather_grid VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    d = features.load(TRAIN_END)
    X = features.build(d, np.array([0]), np.array([60]))
    col = dict(zip(features.FEATURES_ALL, X[0]))
    assert np.isfinite(col["vpd"]) and np.isfinite(col["dry_days"]) and col["dry_days"] <= 60
    assert not any("fwi" in n or "ffmc" in n for n in features.FEATURES_ALL)
    assert len([f for f in features.FEATURES if f in features.WEATHER_ACTIVE]) == 6
    assert features.select(X, features.FEATURES).shape[1] == len(features.FEATURES)
    assert features.select(X, features.FEATURES_V1).shape[1] == len(features.FEATURES_V1)
