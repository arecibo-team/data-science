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
    assert d is not None and len(d.cand) == 31          # every cell with a detection is scored, the single-sensor one too
    lab = features.label_mask(d, 3)
    assert not lab[d.cand == len(d.keys) - 1].any()     # ...but it never counts as a fire with the default label
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
             0.0 if i % 5 else 4.0, 0.3, 0.4, 0.5, 22.0 + i % 7, 6.0 + i % 3) for i in range(170)]
    with db.session() as con:
        con.executemany("INSERT INTO weather_grid VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    d = features.load(TRAIN_END)
    X = features.build(d, np.array([0]), np.array([60]))
    col = dict(zip(features.FEATURES_ALL, X[0]))
    assert np.isfinite(col["vpd"]) and np.isfinite(col["dry_days"]) and col["dry_days"] <= 60
    assert all(np.isfinite(col[f]) for f in features.FORECAST)    # the weather of the next 3 days
    assert not any("fwi" in n or "ffmc" in n for n in features.FEATURES_ALL)
    assert len([f for f in features.FEATURES if f in features.WEATHER_ACTIVE]) == 6
    assert features.select(X, features.FEATURES).shape[1] == len(features.FEATURES)
    assert features.select(X, features.FEATURES_V1).shape[1] == len(features.FEATURES_V1)


def test_holdout_event_is_never_trained_on_and_is_replayed(tmp_db):
    _fill_bins()
    event = {"name": "synthetic", "first": "2025-11-01", "last": "2025-11-20", "bbox": [9.0, -51.0, 11.0, -48.0],
             "spike": ["2025-11-08", "2025-11-12"]}
    d = features.load(TRAIN_END, holdout=(event["first"], event["last"]))
    h0, h1 = d.hold
    for split in ("train", "val"):
        _, _, _, (pos, di) = features.training_rows(d, 3, split)
        assert not ((di >= h0 - 3) & (di <= h1)).any()      # nothing in or reaching into the window
    _, _, w, (pos, di) = features.training_rows(d, 3, "train")
    assert ((di < h0 - 3) | (di > h1)).all() and (di > h1).any() and (di < h0 - 3).any()

    run_id = train.create_run({"rounds": 8, "min_data_in_leaf": 5, "num_leaves": 7, "holdout_event": event})
    import threading
    train.train(run_id, lambda e: None, threading.Event())
    run = train.get_run(run_id)
    assert run["status"] == "done", run["message"]
    bm = run["metrics"]["benchmark"]
    assert bm["event"]["name"] == "synthetic" and len(bm["days"]) >= 15
    assert bm["europe"]["positives"] > 0 and set(bm["lead"]) == {"new", "burning"}
    assert all(0 <= v <= bm["lead"][g]["cells"] for g in bm["lead"] for v in bm["lead"][g]["flagged"].values())
    assert bm["lead_days"] == [1, 2, 3, 5, 7]                # leads that stay inside the held-out window


def test_new_runs_hold_out_the_configured_event(tmp_db):
    run_id = train.create_run({"rounds": 3})
    assert train.get_run(run_id)["params"]["holdout_event"]["first"] == settings.HOLDOUT_EVENT["first"]


def test_country_lookup_score_and_risk_endpoint(tmp_db):
    from server.ml import regions
    names, _ = regions.countries()
    ids = regions.country_ids(np.array([38.0, 48.8, 0.0]), np.array([23.7, 2.3, -30.0]))
    assert [names[i] for i in ids[:2]] == ["Greece", "France"] and ids[2] == -1       # the sea belongs to nobody
    assert regions.score(0.005, 0.96) < 0.01 and regions.score(3.0, 0.98) == 0.98 and regions.score(3.0, None) == 0.0
    run_id = train.create_run({"rounds": 3})
    with db.session() as con:
        con.execute("UPDATE model_runs SET status='done', metrics='{}' WHERE id=?", (run_id,))
        con.execute("INSERT INTO model_checkpoints(run_id,round,created_at,is_final,weights) VALUES (?,?,?,?,?)", (run_id, 1, "x", 1, b""))
        con.executemany("INSERT INTO country_risk VALUES (?,?,?,?,?,?,?)", [
            (run_id, "2025-08-08", "Greece", 2.9, 0.97, 109, 5), (run_id, "2025-08-20", "Greece", 0.0, 0.1, 109, 0)])
    with TestClient(app_mod.app) as client:
        r = client.get("/api/risk", params={"date": "2025-08-08"}).json()
        g = r["countries"][0]
        assert g["name"] == "Greece" and g["score"] > 0.9 and g["burned"] == 5 and r["complete"]
        far = client.get("/api/risk", params={"date": "2030-01-01"}).json()
        assert far["feature_date"] == "2025-08-20" and not far["complete"] and far["countries"][0]["burned"] is None
        assert client.get("/api/regions").json()["features"]


def test_sampled_negatives_are_weighted_back_to_every_cell_day(tmp_db):
    _fill_bins()
    d = features.load(TRAIN_END, label="both")
    tot_w, tot_pos = 0.0, 0
    for split in ("train", "val"):
        _, y, w, _ = features.training_rows(d, 3, split, 0.5, 0.5)
        assert w[y == 1].tolist() == [1.0] * int(y.sum())         # positives are never down-weighted
        tot_w += float(w[y == 0].sum()); tot_pos += int(y.sum())
    days = np.arange(d.n_days)
    n_days = int(((days >= features.MIN_DAY) & (days + 3 <= d.train_end) & (days + 3 < d.n_days)).sum())
    assert abs(tot_w - (len(d.cand) * n_days - tot_pos)) / (len(d.cand) * n_days) < 0.05


def test_forecast_weather_comes_from_the_forecast_table_after_the_reanalysis(tmp_db):
    _fill_bins()
    d0 = dt.date(2025, 9, 1)
    rows = [(40, 190, (d0 + dt.timedelta(days=i)).isoformat(), 15.0, 60.0, 3.0, 0.0, .3, .4, .5, 20.0, 5.0) for i in range(100)]
    fc = [(40, 190, (dt.date(2026, 2, 15) + dt.timedelta(days=i)).isoformat(), "x", 10.0, 30.0, 4.0, 25.0, 0.0, 40.0) for i in range(7)]
    with db.session() as con:
        con.executemany("INSERT INTO weather_grid VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.executemany("INSERT INTO weather_forecast VALUES (?,?,?,?,?,?,?,?,?,?)", fc)
    d = features.load(TRAIN_END)
    last = d.n_days - 1                                          # 2026-02-15: the day the forecast starts
    X = features.build(d, np.array([0]), np.array([last]))
    col = dict(zip(features.FEATURES_ALL, X[0]))
    assert col["fc_wsx"] == 25.0 and col["fc_prec"] == 0.0       # next days are the forecast, not a carried-over day


def test_terrain_cell_means_follow_the_raster_orientation():
    from server import terrain
    arr = np.zeros((20, 20), np.float32)
    arr[:10, :10], arr[10:, :10], arr[:10, 10:], arr[10:, 10:] = 1, 2, 3, 4      # the first raster row is the northern edge
    out = terrain._cell_means(arr, (0.0, 34.0, 0.2, 34.2))
    r0, c0 = 1240, 1800                                                          # (34 + 90) / 0.1 and 180 / 0.1
    assert out == {(r0 + 1, c0): 1.0, (r0, c0): 2.0, (r0 + 1, c0 + 1): 3.0, (r0, c0 + 1): 4.0}


def test_eobs_features_look_back_and_ahead():
    from server import eobs
    t = 60
    rr = np.zeros((1, t), np.float32); rr[0, 5] = 5.0
    tx, hu, fg = np.full((1, t), 20.0, np.float32), np.full((1, t), 50.0, np.float32), np.full((1, t), 3.0, np.float32)
    f = eobs._derive(rr, tx, hu, fg)
    assert f["e_dry_days"][0, 5] == 0 and f["e_dry_days"][0, 10] == 5          # days since rain over 1 mm
    assert f["e_prec30"][0, 10] == 5.0 and f["e_prec30"][0, 40] == 0.0          # rain of the last 30 days
    assert f["e_fc_prec"][0, 3] == 5.0 and f["e_fc_prec"][0, 6] == 0.0          # rain of D+1..D+3
    assert f["e_fc_fg"][0, 0] == 3.0 and np.isnan(f["e_fc_fg"][0, t - 1])        # no window after the last day
    assert set(f) == set(eobs.FEATURES)


def test_soil_and_terrain_fill_their_own_static_columns(tmp_db):
    _fill_bins()
    context.load()
    with db.session() as con:
        con.execute("INSERT INTO terrain_cells VALUES (400, 1900, 120.0, 15.0, 3.0, 0.6, 0.2, 0.3)")
    d = features.load(TRAIN_END)
    X = features.build(d, np.arange(len(d.cand)), np.full(len(d.cand), 60))
    col = {f: X[:, i] for i, f in enumerate(features.FEATURES_ALL)}
    assert np.isclose(col["elev"], 120.0).sum() == 1 and np.isnan(col["elev"]).sum() == len(d.cand) - 1
    assert np.isfinite(col["sand"]).all()
