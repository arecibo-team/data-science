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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .. import db, settings
from . import benchmark, features, regions

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
    p = {**DEFAULTS, "horizon": settings.HORIZON, "train_end": settings.TRAIN_END, "neg_keep": 1.0, "label": settings.LABEL, "early_stopping": 40,
         "keep_active": features.KEEP_ACTIVE, "keep_cold": features.KEEP_COLD, "features": features.FEATURES, "holdout_event": settings.HOLDOUT_EVENT, **params}
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
    p["features"] = p.get("features") or features.FEATURES_V1   # runs from before the derived weather features
    horizon, history = p["horizon"], run["loss_history"] if resume else []
    try:
        emit({"type": "start", "run": run_id, "total": run["rounds_total"], "resumed": resume})
        event = p.get("holdout_event")
        data = features.load(p["train_end"], label=p.get("label", "confirmed"),
                             holdout=(event["first"], event["last"]) if event else None)
        if data is None:
            raise RuntimeError("no common-fire data to train on; run the pipeline first")
        if "keep_cold" not in p:
            raise RuntimeError("this run used the older sampling of recently burned cells only; start a new run")
        Xt, yt, wt, _ = features.training_rows(data, horizon, "train", p["keep_active"], p["keep_cold"])
        Xv, yv, wv, _ = features.training_rows(data, horizon, "val", p["keep_active"], p["keep_cold"])
        Xt, Xv = features.select(Xt, p["features"]), features.select(Xv, p["features"])
        if len(yt) == 0 or yt.sum() == 0:
            raise RuntimeError("no training rows with positives")
        dt_ = lgb.Dataset(Xt, yt, weight=wt, feature_name=p["features"], free_raw_data=False)
        dv = lgb.Dataset(Xv, yv, weight=wv, reference=dt_)
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
                                valid_names=["train", "val"], init_model=init, callbacks=[cb, *([lgb.early_stopping(p["early_stopping"], verbose=False)] if p.get("early_stopping") else [])])
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
        if event:   # replay the held-out event with this very model
            try:
                metrics["benchmark"] = benchmark.replay(data, booster, metrics["threshold"], horizon, event, p["features"], 1.0, p["keep_cold"])
            except Exception:  # noqa: BLE001 - a failed replay must not lose the trained model
                log.exception("event replay failed")
        try:   # country level risk for the forecast map
            regions.store(run_id, data, regions.series(data, booster, p["features"], horizon, p["neg_keep"]), horizon)
        except Exception:  # noqa: BLE001
            log.exception("country risk failed")
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


def evaluate(booster, data, p: dict, Xv=None, yv=None) -> dict:
    """Score the days after train_end at the alert threshold that maximises F1 on the validation days.

    Every metric is weighted by the sampling weights, so it stands for all cell-days of Europe, not just the sampled
    ones. Xv and yv are unused; kept so callers need not change."""
    names = p.get("features")
    ka, kc = p.get("keep_active", 1.0), p.get("keep_cold", 1.0)
    Xf, yf, wf, _ = features.training_rows(data, p["horizon"], "val", ka, kc)
    pv = calibrate(booster.predict(features.select(Xf, names)), p["neg_keep"])
    thr = best_threshold(pv, yf, wf)
    out = {"threshold": float(thr), "horizon": p["horizon"], "label": p.get("label"), "train_end": p["train_end"]}
    Xs, ys, ws, _ = features.training_rows(data, p["horizon"], "test", ka, kc)
    out["test_rows"] = int(len(ys))
    if len(ys) and ys.sum() > 0:
        ps = calibrate(booster.predict(features.select(Xs, names)), p["neg_keep"])
        pred = ps >= thr
        tp = float(ws[pred & (ys == 1)].sum())
        out.update(pr_auc=float(average_precision_score(ys, ps, sample_weight=ws)),
                   base_rate=float((ys * ws).sum() / ws.sum()),
                   precision=float(tp / max(ws[pred].sum(), 1e-9)), recall=float(tp / (ys * ws).sum()),
                   roc_auc=float(roc_auc_score(ys, ps, sample_weight=ws)), test_positives=int(ys.sum()))
    return out


def best_threshold(p, y, w=None) -> float:
    """Probability threshold with the best F1 over every cut of the scores (weighted)."""
    if not len(y) or y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p, sample_weight=w)
    f = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f))])
