"""The Amber Index: pillar scores, the 0-100 score, bands, ranks and explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES, PILLARS

BANDS = [(40, "Green"), (60, "Amber"), (101, "Red")]


def equal_weights(features: list[str]) -> dict:
    """Equal weights within each pillar and across the three pillars."""
    within = {}
    for p in PILLARS:
        fs = [f for f in features if FEATURES[f][0] == p]
        within.update({f: 1 / len(fs) for f in fs})
    return {"within": within, "pillar": {p: 1 / len(PILLARS) for p in PILLARS}}


def normalise(weights: dict, features: list[str]) -> dict:
    """Make within-pillar weights sum to 1 per pillar and pillar weights sum to 1."""
    within = {}
    for p in PILLARS:
        fs = [f for f in features if FEATURES[f][0] == p]
        tot = sum(weights["within"].get(f, 0) for f in fs) or 1
        within.update({f: weights["within"].get(f, 0) / tot for f in fs})
    ptot = sum(weights["pillar"].get(p, 0) for p in PILLARS) or 1
    return {"within": within, "pillar": {p: weights["pillar"].get(p, 0) / ptot for p in PILLARS}}


def pillar_scores(X: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Weighted mean of the scaled (0-1) features inside each pillar. Higher = more of that pillar."""
    out = pd.DataFrame(index=X.index)
    for p in PILLARS:
        fs = [f for f in X.columns if FEATURES[f][0] == p]
        w = np.array([weights["within"][f] for f in fs])
        out[p] = X[fs].to_numpy() @ w / w.sum()
    return out


def gap(P: pd.DataFrame, weights: dict) -> pd.Series:
    """Danger minus readiness: z = w_D*D - w_W*W - w_C*C (pillar weights sum to 1)."""
    w = weights["pillar"]
    return w["danger"] * P["danger"] - w["workforce"] * P["workforce"] - w["capability"] * P["capability"]


def fit_scale(z_train: pd.Series) -> dict:
    """Centre on the training median and pick k so the 5th-95th percentile maps to about 5-95."""
    c = float(np.median(z_train))
    spread = max(abs(np.percentile(z_train, 5) - c), abs(np.percentile(z_train, 95) - c)) or 1.0
    return {"k": float(np.log(19) / spread), "c": c}


def amber_score(z, scale: dict):
    return 100 / (1 + np.exp(-scale["k"] * (np.asarray(z) - scale["c"])))


def band(score: float) -> str:
    return next(name for cut, name in BANDS if score < cut)


def score_table(X: pd.DataFrame, meta: pd.DataFrame, weights: dict, scale: dict) -> pd.DataFrame:
    """X = scaled features, meta = [iso2, year] on the same index. Returns pillars, amber, band, rank."""
    P = pillar_scores(X, weights)
    df = pd.concat([meta, P], axis=1)
    df["readiness"] = (weights["pillar"]["workforce"] * P["workforce"] + weights["pillar"]["capability"] * P["capability"]) / (
        weights["pillar"]["workforce"] + weights["pillar"]["capability"])
    df["amber"] = amber_score(gap(P, weights), scale)
    df["band"] = df["amber"].map(band)
    df["rank"] = df.groupby("year")["amber"].rank(ascending=False, method="min").astype(int)
    return df


def monte_carlo_ranks(X: pd.DataFrame, meta: pd.DataFrame, weights: dict, scale: dict,
                      runs: int = 500, jitter: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Perturb every weight by up to +/-20% (uniform), renormalise, re-rank. Returns rank_lo / rank_hi (5th/95th pct)."""
    rng = np.random.default_rng(seed)
    feats = list(X.columns)
    ranks = []
    for _ in range(runs):
        w = {"within": {f: weights["within"][f] * rng.uniform(1 - jitter, 1 + jitter) for f in feats},
             "pillar": {p: weights["pillar"][p] * rng.uniform(1 - jitter, 1 + jitter) for p in PILLARS}}
        w = normalise(w, feats)
        z = gap(pillar_scores(X, w), w)
        s = pd.Series(amber_score(z, scale), index=X.index)
        ranks.append(s.groupby(meta["year"]).rank(ascending=False, method="min"))
    R = pd.concat(ranks, axis=1)
    return pd.DataFrame({"rank_lo": np.percentile(R, 5, axis=1).round().astype(int),
                         "rank_hi": np.percentile(R, 95, axis=1).round().astype(int)}, index=X.index)


def contributions(x: pd.Series, reference: pd.Series, weights: dict, scale: dict) -> pd.Series:
    """Points each feature adds to the score vs an average country-year.

    Inside the sigmoid the index is linear, so the exact Shapley value of feature i for z is
    sign_i * w_pillar * w_within * (x_i - ref_i). We scale them so they add up to score(x) - score(ref)."""
    phi = pd.Series({f: FEATURES[f][1] * weights["pillar"][FEATURES[f][0]] * weights["within"][f] * (x[f] - reference[f])
                     for f in x.index})
    z_x = float(phi.sum() + gap_of_row(reference, weights))
    diff = float(amber_score(z_x, scale) - amber_score(gap_of_row(reference, weights), scale))
    tot = phi.sum()
    return phi * (diff / tot) if abs(tot) > 1e-12 else phi * 0


def gap_of_row(x: pd.Series, weights: dict) -> float:
    P = pillar_scores(x.to_frame().T, weights).iloc[0]
    return float(gap(P.to_frame().T, weights).iloc[0])


REASON_PHRASES = {  # feature -> (phrase when the scaled value is high, phrase when low)
    "burned_share_pct": ("High burned area", "Low burned area"),
    "ff_per_100k": ("Many firefighters per inhabitant", "Few firefighters per inhabitant"),
    "volunteers_per_100k": ("Many volunteer firefighters", "Few volunteer firefighters"),
    "spend_per_capita": ("High fire-protection spending", "Low fire-protection spending"),
    "stations_per_1000km2": ("Dense fire-station network", "Sparse fire-station network"),
    "stations_per_100k": ("Many fire stations per inhabitant", "Few fire stations per inhabitant"),
}


def reason_text(feature: str, points: float, value_scaled: float) -> str:
    high, low = REASON_PHRASES[feature]
    direction = "up" if points > 0 else "down"
    return f"{high if value_scaled >= 0.5 else low} pushes the score {direction} {abs(points):.0f} pts"


def label_archetypes(centers: pd.DataFrame) -> list[str]:
    """Name KMeans(4) clusters from their pillar centres:
    highest danger -> Wildfire frontline; of the rest, highest readiness -> Well-resourced,
    lowest readiness -> Thin coverage, remaining -> Low exposure."""
    c = centers.assign(readiness=(centers["workforce"] + centers["capability"]) / 2)
    labels = {}
    front = c["danger"].idxmax(); labels[front] = "Wildfire frontline"
    rest = c.drop(index=front)
    well = rest["readiness"].idxmax(); labels[well] = "Well-resourced"
    rest = rest.drop(index=well)
    thin = rest["readiness"].idxmin(); labels[thin] = "Thin coverage"
    for i in rest.index.drop(thin):
        labels[i] = "Low exposure"
    return [labels[i] for i in centers.index]
