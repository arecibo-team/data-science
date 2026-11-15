"""Leave-one-season-out cross-validation.

One fold holds out the fire season (April to October) of a year: that window is removed from training for all of
Europe (the same mechanism as the Greece benchmark), a model is trained on every other day, and the window is scored.
Averaging folds gives hundreds of fire events instead of the single test year or single event the other benchmarks
rest on, so a change of features or data can be judged against the spread between seasons.

    uv run python -m server.ml.cv --year 2022 --tag base
    uv run python -m server.ml.cv --summary base

Results go to data/analysis/cv_<tag>_<year>.json. Features: --features all|v2|<comma list>.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from firms import config as C
from . import features, regions, train

OUT = C.ANALYSIS_DIR
SURGES = (5, 20)          # cells burning in the next 3 days that make a country-day a surge
ALERT_SHARE = 0.05        # a country is on alert on its 5% highest days of the window


def _features(name: str) -> list[str]:
    sets = {"default": features.FEATURES, "v2": features.FEATURES_V2,
            "terrain": features.FEATURES + features.TERRAIN_COLS,
            "eobs": features.FEATURES + features.EOBS,
            "both": features.FEATURES + features.TERRAIN_COLS + features.EOBS}
    return sets.get(name) or name.split(",")


def fold(year: int, names: list[str], rounds: int = 250, lr: float = 0.08, horizon: int = 3, label: str = "both",
         train_end: str = "2025-12-31", first: str | None = None, patience: int = 30, metric: str = "binary_logloss") -> dict:
    t0 = time.time()
    window = (f"{year}-04-01", f"{year}-10-31")
    d = features.load(train_end, label=label, holdout=window, first=first)
    h0, h1 = d.hold
    Xt, yt, wt, _ = features.training_rows(d, horizon, "train")
    Xv, yv, wv, _ = features.training_rows(d, horizon, "val")
    dt_ = lgb.Dataset(features.select(Xt, names), yt, weight=wt, feature_name=names)
    dv = lgb.Dataset(features.select(Xv, names), yv, weight=wv, reference=dt_)
    booster = lgb.train({"objective": "binary", "metric": metric, "learning_rate": lr, "num_leaves": 63,
                         "min_data_in_leaf": 50, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
                         "verbose": -1, "seed": 0}, dt_, rounds, valid_sets=[dv], valid_names=["val"],
                        callbacks=[lgb.early_stopping(patience, verbose=False)])
    thr = train.best_threshold(booster.predict(features.select(Xv, names)), yv, wv)
    del Xt, Xv

    days = np.arange(h0, min(h1, d.n_days - horizon - 1) + 1)
    sel = np.zeros(d.n_days, bool)
    sel[days] = True
    pos, di, w, y = features.sample_rows(d, horizon, sel, 1.0, features.KEEP_COLD, np.random.default_rng(0))
    X = features.build(d, pos, di)
    p = booster.predict(features.select(X, names))
    base = X[:, features.FEATURES_ALL.index("b7")]      # fires seen by both sensors in the last 7 days
    y = y.astype(bool)
    pred = p >= thr
    tp = float(w[pred & y].sum())
    cells = {"positives": int(y.sum()), "pr_auc": float(average_precision_score(y, p, sample_weight=w)),
             "baseline_pr_auc": float(average_precision_score(y, base, sample_weight=w)),
             "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
             "precision": tp / max(float(w[pred].sum()), 1e-9), "recall": tp / max(float(w[y].sum()), 1e-9),
             "base_rate": float((y * w).sum() / w.sum())}
    countries = country_metrics(d, booster, names, horizon, days)
    return {"year": year, "features": names, "rounds": booster.current_iteration(), "threshold": float(thr),
            "cells": cells, "countries": countries, "seconds": round(time.time() - t0)}


def country_metrics(d: features.Data, booster, names: list[str], horizon: int, days: np.ndarray) -> dict:
    """Country-day expected fire cells over the window against the cells that really burned in the next days."""
    cid = regions.country_ids(d.static[:, 0], d.static[:, 1])
    names_c, _ = regions.countries()
    n_c = len(names_c)
    ok = np.flatnonzero(cid >= 0)
    E = np.zeros((len(days), n_c))
    for a in range(0, len(days), features.BLOCK):
        part = days[a:a + features.BLOCK]
        pos, di = np.tile(ok, len(part)), np.repeat(part, len(ok))
        p = booster.predict(features.select(features.build(d, pos, di), names))
        E[a:a + len(part)] = np.bincount((di - part[0]) * n_c + cid[pos], weights=p, minlength=len(part) * n_c).reshape(len(part), n_c)
    lab = features.label_mask(d, horizon)
    burned = np.zeros((len(days), n_c))
    for c in range(n_c):
        m = cid == c
        if m.any():
            burned[:, c] = lab[m][:, days].sum(axis=0)
    out = {}
    for n in SURGES:
        y = burned >= n
        if not y.any():
            continue
        # onsets: first surge day after at least 7 quiet days, per country
        onsets = warned = 0
        alert = np.zeros_like(y)
        for c in range(n_c):
            if E[:, c].max() > 0:
                alert[:, c] = E[:, c] >= np.quantile(E[:, c], 1 - ALERT_SHARE)
        for c in range(n_c):
            quiet = 99
            for i in range(len(days)):
                if y[i, c] and quiet >= 7:
                    onsets += 1
                    warned += bool(alert[max(i - 5, 0):i + 1, c].any())
                quiet = 0 if y[i, c] else quiet + 1
        out[str(n)] = {"surge_days": int(y.sum()), "country_days": int(y.size),
                       "roc_auc": float(roc_auc_score(y.ravel(), E.ravel())),
                       "pr_auc": float(average_precision_score(y.ravel(), E.ravel())),
                       "onsets": onsets, "warned_5d": warned, "alert_share": ALERT_SHARE}
    return out


def summary(tag: str) -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob(f"cv_{tag}_*.json"))]
    if not rows:
        raise SystemExit(f"no results for tag {tag}")
    out = {"tag": tag, "years": [r["year"] for r in rows], "features": len(rows[0]["features"])}
    for k in ("pr_auc", "baseline_pr_auc", "roc_auc", "precision", "recall"):
        v = [r["cells"][k] for r in rows]
        out[k] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
    for n in map(str, SURGES):
        rs = [r["countries"][n] for r in rows if n in r["countries"]]
        if rs:
            out[f"surge{n}"] = {"roc_auc": float(np.mean([r["roc_auc"] for r in rs])),
                                "pr_auc": float(np.mean([r["pr_auc"] for r in rs])),
                                "onsets": sum(r["onsets"] for r in rs), "warned_5d": sum(r["warned_5d"] for r in rs)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--features", default="default")
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--label", default="both")
    ap.add_argument("--first")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--metric", default="binary_logloss")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.summary:
        print(json.dumps(summary(a.summary), indent=1))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    r = fold(a.year, _features(a.features), a.rounds, label=a.label, first=a.first, patience=a.patience, metric=a.metric)
    (OUT / f"cv_{a.tag}_{a.year}.json").write_text(json.dumps(r))
    print(json.dumps({k: r[k] for k in ("year", "rounds", "seconds", "cells", "countries")}, indent=1))


if __name__ == "__main__":
    main()
