from __future__ import annotations

import pandas as pd

ALPHA = 0.5


PRESSURE_WEIGHTS = {"danger": 0.5, "hazards": 0.5}


def hazard_pressure(P: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    h = P["hazards"] if "hazards" in P else 0.0
    return PRESSURE_WEIGHTS["danger"] * P["danger"] + PRESSURE_WEIGHTS["hazards"] * h


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
