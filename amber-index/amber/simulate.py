from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ALERT_WEIGHT
from .power import ALPHA, add_power
from .scoring import amber_score, band, gap, pillar_scores

CREW_DEMAND = {"Green": 50, "Orange": 500, "Red": 2000}
MAX_COMMITTED = 0.95

PRESETS = {
    "major_flood": [("FL", "Red", 1)],
    "earthquake": [("EQ", "Orange", 2)],
    "wildfire_season": [("WF", "Orange", 3)],
    "drought": [("DR", "Orange", 1)],
    "multi_hazard": [("FL", "Red", 1), ("WF", "Red", 1), ("EQ", "Orange", 1)],
}


def shock_load(events: list[tuple[str, str, int]], major_only: bool = False) -> float:
    return float(sum(ALERT_WEIGHT.get(alert.capitalize(), 1.0) * count for _, alert, count in events
                     if not major_only or alert.capitalize() in ("Orange", "Red")))


def crew_demand(events: list[tuple[str, str, int]]) -> float:
    return float(sum(CREW_DEMAND.get(alert.capitalize(), 0) * count for _, alert, count in events))


def apply_shocks(raw: pd.DataFrame, shocks: dict[str, list[tuple[str, str, int]]]) -> pd.DataFrame:
    out = raw.copy()
    for iso2, events in shocks.items():
        if iso2 not in out.index:
            raise KeyError(iso2)
        out.loc[iso2, "hazard_load"] = np.nan_to_num(out.loc[iso2, "hazard_load"]) + shock_load(events)
        out.loc[iso2, "hazard_density"] = out.loc[iso2, "hazard_load"] / out.loc[iso2, "area_km2"] * 1e5
        out.loc[iso2, "major_load"] = np.nan_to_num(out.loc[iso2, "major_load"]) + shock_load(events, major_only=True)
        out.loc[iso2, "major_hazards"] = out.loc[iso2, "major_load"]
    return out


def evaluate(raw: pd.DataFrame, preprocessor, features: list[str], weights: dict, scale: dict,
             alpha: float = ALPHA, year: int = 0) -> pd.DataFrame:
    X = preprocessor.transform(raw[features])
    P = pillar_scores(X, weights)
    df = P.assign(iso2=raw.index, year=year)
    df["amber"] = amber_score(gap(P, weights), scale)
    df["band"] = df["amber"].map(band)
    df = add_power(df, weights, alpha)
    df["amber_rank"] = df["amber"].rank(ascending=False, method="min").astype(int)
    return df.set_index("iso2")


def run_scenario(raw: pd.DataFrame, shocks: dict, preprocessor, features: list[str], weights: dict, scale: dict,
                 alpha: float = ALPHA, year: int = 0) -> pd.DataFrame:
    before = evaluate(raw, preprocessor, features, weights, scale, alpha, year)
    after = evaluate(apply_shocks(raw, shocks), preprocessor, features, weights, scale, alpha, year)
    demand = pd.Series({iso2: crew_demand(shocks.get(iso2, [])) for iso2 in raw.index})
    crews = raw["firefighters_est"].astype(float)
    committed = (demand / crews).clip(upper=MAX_COMMITTED).fillna(0)
    after["power"] = after["power"] * (1 - committed)
    after["power_rank"] = after["power"].rank(ascending=False, method="min").astype(int)
    out = pd.DataFrame({
        "shocked": [iso2 in shocks for iso2 in before.index],
        "power_before": before["power"], "power_after": after["power"],
        "power_rank_before": before["power_rank"], "power_rank_after": after["power_rank"],
        "amber_before": before["amber"], "amber_after": after["amber"],
        "band_before": before["band"], "band_after": after["band"],
        "firefighters": crews, "crew_demand": demand, "committed_share": committed,
        "unmet_demand": (demand - crews * MAX_COMMITTED).clip(lower=0),
    }, index=before.index)
    out["power_change"] = out["power_after"] - out["power_before"]
    out["rank_change"] = out["power_rank_after"] - out["power_rank_before"]
    return out.sort_values("power_rank_after")
