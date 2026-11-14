"""LightGBM training with progress events and DB checkpoints.

A run is a row in `model_runs`. Every CHECKPOINT_EVERY rounds the booster is written to `model_checkpoints`
(one rolling checkpoint per run). If the backend stops, the run is marked `interrupted` and `resume` continues
from the newest checkpoint with `init_model`. A finished run keeps only its final model.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Callable

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score

from .. import db, settings
from . import features

log = logging.getLogger("server.ml")
NEG_KEEP = 0.1
DEFAULTS = {"rounds": 300, "lr": 0.05, "num_leaves": 63, "min_data_in_leaf": 50}


class Stopped(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def calibrate(p: np.ndarray, neg_keep: float = NEG_KEEP) -> np.ndarray:
    """Undo the negative subsampling of training (odds scale by neg_keep)."""
    return neg_keep * p / (neg_keep * p + (1 - p))


def create_run(params: dict) -> int:
    p = {**DEFAULTS, "horizon": settings.HORIZON, "train_end": settings.TRAIN_END, "neg_keep": NEG_KEEP, "label": settings.LABEL, **params}
    with db.session() as con:
        cur = con.execute("INSERT INTO model_runs (status, params, started_at, rounds_total) VALUES ('running', ?, ?, ?)",
                          (json.dumps(p), now(), int(p["rounds"])))
        return cur.lastrowid


def mark_interrupted() -> int:
    with db.session() as con:
        return con.execute("UPDATE model_runs SET status='interrupted', message='backend stopped' "
                           "WHERE status='running'").rowcount


def get_run(run_id: int) -> dict | None:
    with db.session() as con:
        r = con.execute("SELECT * FROM model_runs WHERE id=?", (run_id,)).fetchone()
    return _run_dict(r) if r else None


def _run_dict(r) -> dict:
    d = dict(r)
    d["params"] = json.loads(d["params"])
    d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else None
    d["loss_history"] = json.loads(d["loss_history"])
    return d


def list_runs() -> list[dict]:
    with db.session() as con:
        rows = con.execute("SELECT * FROM model_runs ORDER BY id DESC").fetchall()
        ck = {r["run_id"]: (r["n"], r["last"]) for r in con.execute(
            "SELECT run_id, COUNT(*) n, MAX(round) last FROM model_checkpoints GROUP BY run_id")}
    out = []
    for r in rows:
        d = _run_dict(r)
        d.pop("loss_history")
        d["checkpoints"], d["last_checkpoint_round"] = ck.get(d["id"], (0, None))
        out.append(d)
    return out


def latest_checkpoint(run_id: int):
    with db.session() as con:
        return con.execute("SELECT round, weights FROM model_checkpoints WHERE run_id=? ORDER BY id DESC LIMIT 1",
                           (run_id,)).fetchone()


def _save_checkpoint(run_id: int, rnd: int, booster, tr, va, final=False) -> None:
    with db.session() as con:
        if not final:
            con.execute("DELETE FROM model_checkpoints WHERE run_id=? AND is_final=0", (run_id,))
        con.execute("INSERT INTO model_checkpoints (run_id, round, created_at, train_loss, val_loss, is_final, weights) "
                    "VALUES (?,?,?,?,?,?,?)", (run_id, rnd, now(), tr, va, int(final), booster.model_to_string().encode()))


def train(run_id: int, emit: Callable[[dict], None], stop: threading.Event, resume: bool = False) -> None:
    run = get_run(run_id)
    p = run["params"]
    horizon, history = p["horizon"], run["loss_history"] if resume else []
    try:
        emit({"type": "start", "run": run_id, "total": run["rounds_total"], "resumed": resume})
        data = features.load(p["train_end"], label=p.get("label", "confirmed"))
        if data is None:
            raise RuntimeError("no common-fire data to train on; run the pipeline first")
        Xt, yt, _ = features.training_rows(data, horizon, "train", p["neg_keep"])
        Xv, yv, _ = features.training_rows(data, horizon, "val", p["neg_keep"])
        if len(yt) == 0 or yt.sum() == 0:
            raise RuntimeError("no training rows with positives")
        dt_ = lgb.Dataset(Xt, yt, feature_name=features.FEATURES, free_raw_data=False)
        dv = lgb.Dataset(Xv, yv, reference=dt_)
        init, done = None, 0
        if resume:
            ck = latest_checkpoint(run_id)
            if ck:
                init, done = lgb.Booster(model_str=ck["weights"].decode()), ck["round"]
                history = [h for h in history if h["round"] <= done]
        remaining = run["rounds_total"] - done
        with db.session() as con:
            con.execute("UPDATE model_runs SET status='running', message=NULL, finished_at=NULL, rounds_done=? WHERE id=?",
                        (done, run_id))

        def cb(env):
            rnd = env.iteration + 1
            vals = {f"{d}/{m}": v for d, m, v, _ in env.evaluation_result_list}
            tr, va = vals.get("train/binary_logloss"), vals.get("val/binary_logloss")
            history.append({"round": rnd, "train": tr, "val": va})
            with db.session() as con:
                con.execute("UPDATE model_runs SET rounds_done=?, loss_history=? WHERE id=?",
                            (rnd, json.dumps(history), run_id))
            emit({"type": "progress", "round": rnd, "total": run["rounds_total"], "train_loss": None if tr is None else float(tr), "val_loss": None if va is None else float(va)})
            stopping = stop.is_set()
            if rnd % settings.CHECKPOINT_EVERY == 0 or stopping:
                _save_checkpoint(run_id, rnd, env.model, tr, va)
                emit({"type": "checkpoint", "round": rnd})
            if stopping:
                raise Stopped

        params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": p["lr"],
                  "num_leaves": p["num_leaves"], "min_data_in_leaf": p["min_data_in_leaf"],
                  "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 0}
        try:
            booster = lgb.train(params, dt_, num_boost_round=remaining, valid_sets=[dt_, dv],
                                valid_names=["train", "val"], init_model=init, callbacks=[cb])
        except Stopped:
            with db.session() as con:
                con.execute("UPDATE model_runs SET status='stopped', message='stopped by user' WHERE id=?", (run_id,))
            emit({"type": "stopped", "run": run_id})
            return
        last = history[-1] if history else {}
        _save_checkpoint(run_id, booster.current_iteration(), booster, last.get("train"), last.get("val"), final=True)
        with db.session() as con:
            con.execute("DELETE FROM model_checkpoints WHERE run_id=? AND is_final=0", (run_id,))
        metrics = evaluate(booster, data, p, Xv, yv)
        with db.session() as con:
            con.execute("UPDATE model_runs SET status='done', finished_at=?, metrics=?, rounds_done=? WHERE id=?",
                        (now(), json.dumps(metrics), max(booster.current_iteration(), history[-1]["round"] if history else 0), run_id))
        emit({"type": "done", "run": run_id, "metrics": metrics})
    except Exception as exc:  # noqa: BLE001 - reported to the UI and the run row
        log.exception("training failed")
        with db.session() as con:
            con.execute("UPDATE model_runs SET status='failed', message=?, finished_at=? WHERE id=?",
                        (str(exc), now(), run_id))
        emit({"type": "error", "run": run_id, "message": str(exc)})


def evaluate(booster, data, p: dict, Xv, yv) -> dict:
    """Score on the held-out days (after train_end), at the threshold that maximises F1 on the validation rows."""
    pv = calibrate(booster.predict(Xv), p["neg_keep"])
    thr = best_threshold(pv, yv, base=None)
    out = {"threshold": float(thr), "horizon": p["horizon"], "label": p.get("label"), "train_end": p["train_end"]}
    Xs, ys, _ = features.training_rows(data, p["horizon"], "test")
    out["test_rows"] = int(len(ys))
    if len(ys) and ys.sum() > 0:
        ps = calibrate(booster.predict(Xs), p["neg_keep"])
        pred = ps >= thr
        tp = float((pred & (ys == 1)).sum())
        out.update(pr_auc=float(average_precision_score(ys, ps)), base_rate=float(ys.mean()),
                   precision=float(tp / max(pred.sum(), 1)), recall=float(tp / ys.sum()), test_positives=int(ys.sum()))
    return out


def best_threshold(p, y, base) -> float:
    if not len(y) or y.sum() == 0:
        return 0.5
    best, best_f = 0.5, -1.0
    for t in np.quantile(p, np.linspace(0.05, 0.99, 40)):
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        f = 2 * tp / max(pred.sum() + y.sum(), 1)
        if f > best_f:
            best, best_f = float(t), f
    return best
