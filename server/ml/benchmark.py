"""Benchmark the model against one held-out event.

The event window is removed from training for all of Europe (see features.load(holdout=...)): not its fires, not
days whose label reaches into it, and cells that burned only inside it are not candidates. After a run, `replay`
scores the window with the finished model and answers: would it have seen the surge coming?

    uv run python -m server.ml.benchmark        # trains the default event's models without saving anything
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import db, settings
from . import features, train

LEADS = (1, 2, 3, 5, 7, 10)


def _box(d: features.Data, bbox) -> np.ndarray:
    lat, lon = d.static[:, 0], d.static[:, 1]
    return (lon >= bbox[0]) & (lat >= bbox[1]) & (lon <= bbox[2]) & (lat <= bbox[3])


def replay(d: features.Data, booster, thr: float, horizon: int, event: dict, names: list[str] | None = None,
           keep_active: float = 1.0, keep_cold: float = features.KEEP_COLD) -> dict:
    """Score the held-out window with a trained booster. Returns a JSON-ready summary. Rows are sampled as in
    training and weighted, so counts and metrics stand for every cell-day of the window."""
    names = names or features.FEATURES
    h0, h1 = d.hold
    s0, s1 = (d.day_index(x) for x in event["spike"])
    inbox = _box(d, event["bbox"])
    act, lab = features.active_mask(d), features.label_mask(d, horizon)
    score = lambda X: train.calibrate(booster.predict(features.select(X, names)), 1.0)

    days = np.arange(h0, h1 + 1)
    days = days[days + horizon < d.n_days]
    sel = np.zeros(d.n_days, bool)
    sel[days] = True
    pos, di, w, y = features.sample_rows(d, horizon, sel, keep_active, keep_cold, np.random.default_rng(0))
    X = features.build(d, pos, di)
    p, y = score(X), y.astype(bool)
    base = X[:, features.FEATURES_ALL.index("c7")]
    pred = p >= thr
    tp = float(w[pred & y].sum())
    npos = float(w[y].sum())
    europe = {"cell_days": int(w.sum()), "positives": int(y.sum()),
              "pr_auc": float(average_precision_score(y, p, sample_weight=w)) if y.any() else None,
              "baseline_pr_auc": float(average_precision_score(y, base, sample_weight=w)) if y.any() else None,
              "roc_auc": float(roc_auc_score(y, p, sample_weight=w)) if 0 < y.sum() < len(y) else None,
              "recall": tp / max(npos, 1e-9), "precision": tp / max(float(w[pred].sum()), 1e-9)}

    series = []
    for day in days:
        m = (di == day) & inbox[pos]
        series.append({"date": d.date_of(day), "burning": int(lab[inbox, day].sum()),
                       "alerts": int(round(w[m & pred].sum())), "real": int(round(w[m & pred & y].sum())),
                       "risk": float((p[m] * w[m]).sum())})

    # Cells of the region that burned in the spike, and how many the model could score at all.
    spike_hit = d.Y[d.cand][:, s0:s1 + 1].any(axis=1) & inbox
    cum = features._cum(d.MV[d.cand])
    before = (cum[:, s0] - cum[:, max(s0 - 10, 0)]) > 0
    n_all = 0
    for k in d.keys[np.flatnonzero(d.Y[:len(d.keys), s0:s1 + 1].any(axis=1))]:
        lo, la = (k % features.KEY + .5) * db.BIN_DEG - 180, (k // features.KEY + .5) * db.BIN_DEG - 90
        n_all += event["bbox"][0] <= lo <= event["bbox"][2] and event["bbox"][1] <= la <= event["bbox"][3]
    groups = {"new": spike_hit & ~before, "burning": spike_hit & before}
    lead_days = [k for k in LEADS if s0 - k >= h0]          # only days the model never trained on
    rows = [(q, s0 - k, k, g) for g, m in groups.items() for q in np.flatnonzero(m) for k in lead_days]
    lead = {g: {"cells": int(m.sum()), "flagged": {str(k): 0 for k in lead_days}} for g, m in groups.items()}
    if rows:
        qp, qd = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        pq = score(features.build(d, qp, qd))
        for (q, day, k, g), v in zip(rows, pq):
            lead[g]["flagged"][str(k)] += int(v >= thr)
    return {"event": event, "horizon": horizon, "threshold": float(thr), "europe": europe, "days": series,
            "spike_cells": {"in_region": int(n_all), "scoreable": int(spike_hit.sum()),
                            "active": int(act[spike_hit, s0 - 1].sum())},
            "lead": lead, "lead_days": lead_days}


def _fit(d: features.Data, horizon: int):
    """Train without saving anything, as the UI's training does."""
    names = features.FEATURES
    Xt, yt, wt, _ = features.training_rows(d, horizon, "train")
    Xv, yv, wv, _ = features.training_rows(d, horizon, "val")
    dt_ = lgb.Dataset(features.select(Xt, names), yt, weight=wt)
    dv = lgb.Dataset(features.select(Xv, names), yv, weight=wv, reference=dt_)
    b = lgb.train({"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "num_leaves": 63,
                   "min_data_in_leaf": 20, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
                   "verbose": -1, "seed": 0}, dt_, 600, valid_sets=[dv], valid_names=["val"],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    return b, train.best_threshold(b.predict(features.select(Xv, names)), yv, wv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="3,7")
    ev = settings.HOLDOUT_EVENT
    if not ev:
        raise SystemExit("no event configured (FIRMS_HOLDOUT=0)")
    d = features.load(settings.TRAIN_END, holdout=(ev["first"], ev["last"]))
    for h in (int(x) for x in ap.parse_args().horizons.split(",")):
        b, thr = _fit(d, h)
        r = replay(d, b, thr, h, ev)
        print(f"\nHorizon {h}: {ev['name']}, threshold {thr:.5f}, Europe {r['europe']}")
        for row in r["days"]:
            print("  ", row)
        print("  spike cells", r["spike_cells"], "lead", r["lead"])


if __name__ == "__main__":
    main()
