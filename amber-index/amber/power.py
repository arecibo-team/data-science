from __future__ import annotations

import pandas as pd

ALPHA = 0.5


def hazard_pressure(P: pd.DataFrame, weights: dict) -> pd.Series:
    w = weights["pillar"]
    wd, wh = w.get("danger", 0.0), w.get("hazards", 0.0)
    h = P["hazards"] if "hazards" in P else 0.0
    return (wd * P["danger"] + wh * h) / ((wd + wh) or 1.0)


def readiness(P: pd.DataFrame, weights: dict) -> pd.Series:
    w = weights["pillar"]
    return (w["workforce"] * P["workforce"] + w["capability"] * P["capability"]) / ((w["workforce"] + w["capability"]) or 1.0)


def power_index(P: pd.DataFrame, weights: dict, alpha: float = ALPHA) -> pd.Series:
    return 100 * readiness(P, weights) * (1 - alpha * hazard_pressure(P, weights))


def add_power(scores: pd.DataFrame, weights: dict, alpha: float = ALPHA) -> pd.DataFrame:
    out = scores.copy()
    out["hazard_pressure"] = hazard_pressure(out, weights)
    out["power"] = power_index(out, weights, alpha)
    out["power_rank"] = out.groupby("year")["power"].rank(ascending=False, method="min").astype(int)
    return out
