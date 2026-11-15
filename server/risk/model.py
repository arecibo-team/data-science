"""Country fire risk, 1 to 7 days ahead.

One LightGBM model per lead k predicts, for every patch (the part of a 1 degree weather cell inside one country) and
issue day D, the number of 0.1 degree cells that burn on day D+k. The country number is the sum over its patches.

Honest by construction:
  * a model trained up to `train_end` uses no fire, weather or normal from a later day;
  * the number of trees is chosen on the last two training years, held out, and those held-out predictions give each
    country's alert thresholds, so nothing about the test season tunes anything;
  * training uses the weather that came (a perfect forecast); a replay can instead be fed the forecasts issued at the
    time (`forecasts.source`), which is what `mode="issued"` means.

    uv run python -m server.risk.model train            # fits, predicts, writes data/risk/
    uv run python -m server.risk.report                 # how it did for Greece 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import lightgbm as lgb
import numpy as np

from firms import config as C
from . import data, forecasts

log = logging.getLogger("server.risk")
DIR = C.DATA_DIR / "risk"
TRAIN_END = "2024-12-31"
VAL_YEARS = 2
KEEP_ZERO = 0.35          # share of the rows without a fire that training keeps (weighted back by 1 / share)
PARAMS = {"objective": "poisson", "metric": "poisson", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 0,
          "num_threads": 6}
SETS = {"full": data.FEATURES, "conditions": data.CONDITIONS}


def _rows(p: data.Panel, patches: np.ndarray, d0: int, d1: int, k: int, rng) -> tuple:
    """Sampled (pos, di, y, w) over issue days d0..d1 for lead k."""
    days = np.arange(d0, d1 + 1)
    y = p.B[np.ix_(patches, days + k)].astype(np.float32)
    keep = (y > 0) | (rng.random(y.shape) < KEEP_ZERO)
    pi, dj = np.nonzero(keep)
    yy = y[pi, dj]
    return patches[pi], days[dj], yy, np.where(yy > 0, 1.0, 1.0 / KEEP_ZERO).astype(np.float32)


def fit(p: data.Panel, train_end: str = TRAIN_END, sets=("full", "conditions")) -> dict:
    """Fits every lead and feature set. Returns the models, the patches used and the normals."""
    te = p.day_index(train_end)
    vs = p.day_index(f"{int(train_end[:4]) - VAL_YEARS}-12-31")        # last day of the first-stage training
    nm = data.normals(p, te)
    patches = np.flatnonzero(p.B[:, : te + 1].sum(axis=1) > 0)
    bs = data.base(p, nm)
    out = {"train_end": train_end, "val_start": p.date_of(vs + 1), "patches": patches, "normals": nm, "models": {}, "rounds": {}}
    for k, ol in data.outlooks(p, nm):
        rng = np.random.default_rng(k)
        tr = _rows(p, patches, data.MIN_DAY, vs - k, k, rng)
        va = _rows(p, patches, vs + 1, te - k, k, rng)
        Xall = {part: data.matrix(p, nm, bs, ol, k, r[0], r[1]) for part, r in (("tr", tr), ("va", va))}
        for s in sets:
            cols = [data.FEATURES.index(n) for n in SETS[s]]
            t0 = time.time()
            dtr = lgb.Dataset(Xall["tr"][:, cols], tr[2], weight=tr[3], feature_name=SETS[s])
            dva = lgb.Dataset(Xall["va"][:, cols], va[2], weight=va[3], reference=dtr)
            first = lgb.train(PARAMS, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            n = first.best_iteration
            X = np.concatenate([Xall["tr"][:, cols], Xall["va"][:, cols]])
            final = lgb.train(PARAMS, lgb.Dataset(X, np.concatenate([tr[2], va[2]]), weight=np.concatenate([tr[3], va[3]]),
                                                  feature_name=SETS[s]), int(n * 1.1))
            out["models"][(s, k)] = (first, final)
            out["rounds"][f"{s}{k}"] = n
            log.info("lead %d %s: %d rounds, val poisson %.4f, %ds", k, s, n, first.best_score["valid_0"]["poisson"], time.time() - t0)
    return out


def predict(p: data.Panel, fitted: dict, d0: int, src: dict | None = None, only: np.ndarray | None = None) -> dict:
    """Country numbers for issue days d0..: {set: (leads, countries, T) expected burning cells on D+k}, plus "fwi", the
    fire-weighted mean fire weather index of the target day (the classic danger rating, as a baseline). Days before
    the validation years are left at zero; validation days use the first-stage models, so they are out of sample."""
    nm, patches = fitted["normals"], fitted["patches"]
    if only is not None:
        patches = patches[np.isin(p.cid[patches], only)]
    bs = data.base(p, nm)
    T, nc = p.n_days, len(p.countries)
    vs = p.day_index(fitted["val_start"])
    te = p.day_index(fitted["train_end"])
    sets = sorted({s for s, _ in fitted["models"]})
    out = {s: np.zeros((len(data.LEADS), nc, T), np.float32) for s in sets + ["fwi"]}
    wsum = np.bincount(p.cid[patches], nm.fire_all[patches], nc)
    for k, ol in data.outlooks(p, nm, src):
        days = np.arange(max(d0, vs), T)
        pos, di = np.repeat(patches, len(days)), np.tile(days, len(patches))
        X = data.matrix(p, nm, bs, ol, k, pos, di)
        idx = p.cid[pos] * T + di
        f = X[:, data.FEATURES.index("fwi_t")] * nm.fire_all[pos]
        out["fwi"][k - 1] = np.bincount(idx, f, nc * T).reshape(nc, T) / np.maximum(wsum[:, None], 1e-9)
        late = di + k > te                                   # target day after the training period: the final model
        for s in sets:
            cols = [data.FEATURES.index(n) for n in SETS[s]]
            first, final = fitted["models"][(s, k)]
            y = np.where(late, final.predict(X[:, cols]), first.predict(X[:, cols]))
            out[s][k - 1] = np.bincount(idx, y, nc * T).reshape(nc, T)
    return out


def load_fitted(p: data.Panel, train_end: str = TRAIN_END) -> dict:
    """The models `run` saved, with the normals and patches rebuilt from the panel (they depend on train_end only)."""
    te = p.day_index(train_end)
    models = {(s, k): (lgb.Booster(model_file=str(DIR / f"{s}_{k}_first.txt")), lgb.Booster(model_file=str(DIR / f"{s}_{k}.txt")))
              for s in SETS for k in data.LEADS if (DIR / f"{s}_{k}.txt").exists()}
    if not models:
        raise FileNotFoundError("no saved risk models; run `python -m server.risk.model train` first")
    return {"train_end": train_end, "val_start": p.date_of(p.day_index(f"{int(train_end[:4]) - VAL_YEARS}-12-31") + 1),
            "patches": np.flatnonzero(p.B[:, : te + 1].sum(axis=1) > 0), "normals": data.normals(p, te), "models": models,
            "rounds": {f"{s}{k}": m[0].num_trees() for (s, k), m in models.items()}}


def run(train_end: str = TRAIN_END, issued_for: str = "Greece", reuse: bool = False) -> None:
    """Fit, predict with perfect and with issued forecasts, and store everything the report needs."""
    DIR.mkdir(parents=True, exist_ok=True)
    p = data.load()
    fitted = load_fitted(p, train_end) if reuse else fit(p, train_end)
    for (s, k), (first, final) in fitted["models"].items():
        final.save_model(str(DIR / f"{s}_{k}.txt"))
        first.save_model(str(DIR / f"{s}_{k}_first.txt"))
    d0 = p.day_index(fitted["val_start"])
    perfect = predict(p, fitted, d0)
    cid = p.countries.index(issued_for)
    src, have = forecasts.source(p, train_end)
    issued = predict(p, fitted, d0, src, only=np.array([cid]))
    # an issue day D is covered for lead k when every day D+1..D+k has a real forecast in every cell of the country
    cells = np.unique(p.cell[fitted["patches"][p.cid[fitted["patches"]] == cid]])
    cover = np.zeros((len(data.LEADS), p.n_days), bool)
    D = np.arange(p.n_days)
    ok = np.ones(p.n_days, bool)
    for k in data.LEADS:
        ok &= have[k - 1][cells][:, D + k].all(axis=0)
        cover[k - 1] = ok
    actual = {n: np.stack([np.bincount(p.cid, a[:, t], len(p.countries)) for t in range(p.n_days)], axis=1)
              for n, a in (("b", p.B), ("mv", p.MV), ("cc", p.CC), ("new", p.NEW))}
    clim = np.zeros((len(p.countries), 366), np.float32)
    np.add.at(clim, p.cid, fitted["normals"].fire)
    np.savez_compressed(DIR / "outlook.npz", **{f"perfect_{s}": v for s, v in perfect.items()},
                        **{f"issued_{s}": v[:, cid] for s, v in issued.items()}, cover=cover, clim=clim, doy=p.doy,
                        **{f"actual_{n}": v for n, v in actual.items()})
    (DIR / "meta.json").write_text(json.dumps({
        "train_end": train_end, "val_start": fitted["val_start"], "start": p.start.isoformat(), "n_days": p.n_days,
        "countries": p.countries, "issued_for": issued_for, "rounds": fitted["rounds"], "features": data.FEATURES,
        "importance": {f"{s}{k}": dict(sorted(zip(SETS[s], (m[1].feature_importance("gain") / max(m[1].feature_importance("gain").sum(), 1e-9)).round(4).tolist()),
                                              key=lambda kv: -kv[1])[:12]) for (s, k), m in fitted["models"].items()}}))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"], help="predict: reuse the saved models, redo the predictions")
    ap.add_argument("--train-end", default=TRAIN_END)
    ap.add_argument("--issued-for", default="Greece")
    a = ap.parse_args()
    run(a.train_end, a.issued_for, reuse=a.cmd == "predict")


if __name__ == "__main__":
    main()
