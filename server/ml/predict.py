"""Predictions from the newest completed run."""
from __future__ import annotations

import threading

import lightgbm as lgb
import numpy as np

from .. import db
from . import features, train

_lock = threading.Lock()
_booster: tuple[int, lgb.Booster] | None = None
_data: tuple[tuple, features.Data | None] | None = None
_labels: tuple[int, int, np.ndarray] | None = None


def _label_mask(d: features.Data, horizon: int) -> np.ndarray:
    """label_mask is big (cells x days), so keep the last one."""
    global _labels
    if _labels and _labels[0] == id(d) and _labels[1] == horizon:
        return _labels[2]
    m = features.label_mask(d, horizon)
    _labels = (id(d), horizon, m)
    return m


def latest_run() -> dict | None:
    """The newest run with status done (the one used for predictions)."""
    with db.session() as con:
        r = con.execute("SELECT * FROM model_runs WHERE status='done' ORDER BY id DESC LIMIT 1").fetchone()
    return train._run_dict(r) if r else None


def _model(run: dict) -> lgb.Booster:
    global _booster
    if _booster and _booster[0] == run["id"]:
        return _booster[1]
    with db.session() as con:
        w = con.execute("SELECT weights FROM model_checkpoints WHERE run_id=? AND is_final=1", (run["id"],)).fetchone()
    b = lgb.Booster(model_str=w["weights"].decode())
    _booster = (run["id"], b)
    return b


def _features(train_end: str, label: str, event: dict | None) -> features.Data | None:
    global _data
    with db.session() as con:
        ver = (train_end, label, event and event["first"], con.execute("SELECT COALESCE(MAX(id), 0) FROM pipeline_steps WHERE status='done'").fetchone()[0],
               con.execute("SELECT COUNT(*) FROM weather_grid").fetchone()[0])
    if _data and _data[0] == ver:
        return _data[1]
    d = features.load(train_end, label=label, holdout=(event["first"], event["last"]) if event else None)
    _data = (ver, d)
    return d


def forecast(date: str) -> dict | None:
    """Cells for a forecast made on `date`: probability of a common fire in the next `horizon` days, and
    what actually happened when those days are in the data."""
    run = latest_run()
    if not run:
        return None
    p = run["params"]
    with _lock:
        d = _features(p["train_end"], p.get("label", "confirmed"), p.get("holdout_event"))
        if d is None:
            return None
        pos, di, X = features.predict_rows(d, date)
        X = features.select(X, p.get("features"))
        prob = train.calibrate(_model(run).predict(X), p["neg_keep"]) if len(pos) else np.zeros(0)
        day = int(di[0]) if len(di) else min(max(d.day_index(date), 0), d.n_days - 1)
        lab = _label_mask(d, p["horizon"])[:, day] if day + p["horizon"] < d.n_days else None
        known = day + p["horizon"] < d.n_days
        if not known:   # partial window: count what is in the data so far
            cum = d.Y[d.cand][:, day + 1:].sum(axis=1)
            lab = cum > 0
        floor = 0.5 * (run["metrics"] or {}).get("threshold", 0.04)   # show cells down to half the alert threshold
        sel = np.union1d(pos[prob >= floor], np.flatnonzero(lab)) if lab is not None else pos[prob >= floor]
        pmap = dict(zip(pos.tolist(), prob.tolist()))
        probs = np.array([pmap.get(i, 0.0) for i in sel], np.float32)
        actual = lab[sel].astype(np.uint8) if lab is not None else np.zeros(len(sel), np.uint8)
        cells = d.keys[d.cand[sel]]
        lat = (cells // features.KEY + 0.5) * db.BIN_DEG - 90
        lon = (cells % features.KEY + 0.5) * db.BIN_DEG - 180
    return {"run": run["id"], "date": date, "feature_date": d.date_of(day), "horizon": p["horizon"],
            "threshold": (run["metrics"] or {}).get("threshold", 0.5), "complete": bool(known),
            "lat": lat.astype(np.float32), "lon": lon.astype(np.float32), "p": probs, "actual": actual}
