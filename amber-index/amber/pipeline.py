from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import ARTIFACTS_DIR, NAMES, SEED, data, scoring
from .features import PILLARS, build_panel, compute_rates
from .power import ALPHA, add_power


def load_sources(refresh: bool = False, first_year: int = 2014, stations_refresh: bool = False) -> dict:
    src = {
        "spending": data.load_spending(refresh), "population": data.load_population(refresh), "area": data.load_area(refresh),
        "burned": data.load_burned_area(refresh), "firefighters": data.load_firefighters(refresh),
        "stations": data.load_osm_stations(stations_refresh), "volunteers": data.load_volunteers(),
    }
    last = int(min(src["burned"].year.max(), src["population"].year.max()))
    src["years"] = list(range(first_year, last + 1))
    src["hazard_events"] = data.load_hazard_events(first_year, last, refresh)
    src["hazards"] = data.hazard_load(src["hazard_events"], src["years"])
    return src


def build(src: dict) -> pd.DataFrame:
    return build_panel(src["years"], src["spending"], src["population"], src["area"], src["burned"],
                       src["firefighters"], src["stations"], src["volunteers"], src["hazards"])


def transform(panel: pd.DataFrame, pre, feats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rates = compute_rates(panel)
    X = pre.transform(rates[feats])
    imputed = pre.missing_mask(rates)
    if "ff_per_100k" in imputed:
        imputed["ff_per_100k"] |= panel["firefighters_imputed"].to_numpy()
    if "spend_per_capita" in imputed:
        imputed["spend_per_capita"] |= panel["spending_meur_imputed"].to_numpy()
    panel["firefighters_est"] = panel["firefighters"].fillna(
        pd.Series(pre.inverse("ff_per_100k", X["ff_per_100k"]), index=X.index) * panel["population"] / 1e5)
    return rates, X, imputed


def fit_archetypes(scores: pd.DataFrame) -> dict:
    P = scores[[p for p in PILLARS if p in scores]]
    km = KMeans(4, n_init=10, random_state=SEED).fit(P)
    labels = scoring.label_archetypes(pd.DataFrame(km.cluster_centers_, columns=P.columns))
    return {"pillars": list(P.columns), "centers": km.cluster_centers_.round(6).tolist(), "labels": labels}


def assign_archetypes(scores: pd.DataFrame, arch: dict) -> pd.Series:
    P = scores[arch["pillars"]].to_numpy()
    C = np.asarray(arch["centers"])
    nearest = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
    return pd.Series([arch["labels"][i] for i in nearest], index=scores.index)


def score_all(panel: pd.DataFrame, X: pd.DataFrame, weights: dict, scale: dict, alpha: float = ALPHA,
              archetypes: dict | None = None, runs: int = 500) -> tuple[pd.DataFrame, dict]:
    meta = panel[["iso2", "year"]]
    scores = scoring.score_table(X, meta, weights, scale)
    scores = scores.join(scoring.monte_carlo_ranks(X, meta, weights, scale, runs=runs, jitter=0.2, seed=SEED))
    scores = add_power(scores, weights, alpha)
    archetypes = archetypes or fit_archetypes(scores)
    scores["archetype"] = assign_archetypes(scores, archetypes)
    return scores, archetypes


def records(panel: pd.DataFrame, rates: pd.DataFrame, X: pd.DataFrame, scores: pd.DataFrame,
            imputed: pd.DataFrame, feats: list[str]) -> list[dict]:
    out = []
    num = lambda v: None if pd.isna(v) else float(v)
    for idx, r in scores.iterrows():
        out.append({
            "iso2": r.iso2, "name": NAMES[r.iso2], "year": int(r.year),
            **{k: round(float(r[k]), 4) for k in ["danger", "hazards", "workforce", "capability", "readiness", "amber",
                                                  "hazard_pressure", "power"]},
            "power_rank": int(r.power_rank), "band": r.band, "rank": int(r["rank"]), "rank_lo": int(r.rank_lo),
            "rank_hi": int(r.rank_hi), "archetype": r.archetype,
            "hazard_events": {c.replace("events_", ""): int(panel.loc[idx, c]) for c in panel.columns if c.startswith("events_")},
            "features": {f: round(float(X.loc[idx, f]), 4) for f in feats},
            "imputed": {f: bool(imputed.loc[idx, f]) for f in feats},
            "firefighters_low_reliability": bool(panel.loc[idx, "firefighters_low_reliability"]),
            "raw": {f: num(rates.loc[idx, f]) for f in feats},
            "area_km2": float(panel.loc[idx, "area_km2"]), "population": float(panel.loc[idx, "population"]),
            "hazard_load": float(panel.loc[idx, "hazard_load"]), "major_load": float(panel.loc[idx, "major_load"]),
            "firefighters_est": float(panel.loc[idx, "firefighters_est"]),
        })
    return out


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def rescore(refresh: bool = True, stations_refresh: bool = False, out: Path = ARTIFACTS_DIR, verbose: bool = True) -> dict:
    from .model import load_artifacts
    art = load_artifacts(out)
    meta, w = art["metadata"], art["weights"]
    feats = meta["feature_order"]
    weights = {"within": {f: w[f]["w_final"] for f in feats}, "pillar": {p: w["_pillars"][p]["w_final"] for p in PILLARS}}
    src = load_sources(refresh, meta["training_years"][0], stations_refresh)
    panel = build(src)
    rates, X, imputed = transform(panel, art["preprocessor"], feats)
    scores, arch = score_all(panel, X, weights, meta["scale"], meta.get("alpha", ALPHA), meta.get("archetypes"))
    recs = records(panel, rates, X, scores, imputed, feats)
    meta.update({"latest_year": int(panel.year.max()), "archetypes": arch,
                 "refreshed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "data_end": {"burned_area": int(src["burned"].year.max()), "population": int(src["population"].year.max()),
                              "spending": int(src["spending"].year.max()), "firefighters": int(src["firefighters"].year.max()),
                              "hazards": str(src["hazard_events"]["end"].max().date()) if len(src["hazard_events"]) else None}})
    write_json_atomic(out / "scores.json", recs)
    write_json_atomic(out / "metadata.json", meta)
    if verbose:
        print(f"rescored {len(recs)} country-years, latest year {meta['latest_year']}, data end {meta['data_end']}")
    return meta
