from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass
class AssistanceParams:
    weight_location: float = 0.5
    deploy_share: float = 0.05
    max_share_per_donor: float = 0.4
    min_power: float = 20.0
    max_distance_km: float | None = None
    min_team: int = 20


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def donor_table(scenario: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({"power": scenario["power_after"], "firefighters": scenario["firefighters"],
                      "hazard_pressure": scenario["hazard_pressure_after"], "committed_share": scenario["committed_share"]},
                     index=scenario.index)
    return t.join(centroids[["lat", "lon"]])


def donor_capacity(table: pd.DataFrame, params: AssistanceParams) -> pd.Series:
    committed = table.get("committed_share", pd.Series(0.0, index=table.index)).fillna(0)
    cap = table["firefighters"] * params.deploy_share * (1 - table["hazard_pressure"].clip(0, 1)) * (1 - committed)
    return cap.where(table["power"] >= params.min_power, 0.0).clip(lower=0)


def _solve(cost: np.ndarray, upper: np.ndarray, target: int, min_team: int) -> tuple[np.ndarray, str]:
    n = len(cost)
    team = min(min_team, target)
    upper = np.where(upper >= team, np.floor(upper), 0.0)
    target = int(min(target, upper.sum()))
    if target <= 0:
        return np.zeros(n), "no capacity"
    c = np.concatenate([cost, np.zeros(n)])
    eye = np.eye(n)
    cons = [LinearConstraint(np.concatenate([np.ones(n), np.zeros(n)])[None, :], target, target),
            LinearConstraint(np.hstack([eye, -np.diag(upper)]), -np.inf, 0),
            LinearConstraint(np.hstack([eye, -team * eye]), 0, np.inf)]
    res = milp(c, constraints=cons, integrality=np.ones(2 * n), bounds=Bounds(np.zeros(2 * n), np.concatenate([upper, np.ones(n)])))
    if not res.success:
        return np.zeros(n), res.message
    return np.round(res.x[:n]), res.message


def optimize_assistance(table: pd.DataFrame, recipient: str, need: float, incident: tuple[float, float] | None = None,
                        params: AssistanceParams | None = None) -> tuple[pd.DataFrame, dict]:
    params = params or AssistanceParams()
    lat0, lon0 = incident if incident else (table.loc[recipient, "lat"], table.loc[recipient, "lon"])
    donors = table.drop(index=recipient).copy()
    donors["distance_km"] = haversine_km(donors["lat"], donors["lon"], lat0, lon0)
    donors["capacity"] = donor_capacity(donors, params)
    if params.max_distance_km is not None:
        donors.loc[donors["distance_km"] > params.max_distance_km, "capacity"] = 0.0
    donors["upper"] = np.minimum(donors["capacity"], params.max_share_per_donor * need)
    reach = donors["distance_km"].max() or 1.0
    w = params.weight_location
    donors["cost"] = w * donors["distance_km"] / reach + (1 - w) * (1 - donors["power"] / 100)
    x, status = _solve(donors["cost"].to_numpy(), donors["upper"].to_numpy(), int(np.floor(need)), params.min_team)
    donors["sent"] = x
    sent = donors[donors["sent"] > 0].copy()
    sent["share_of_need"] = sent["sent"] / need
    sent["share_of_own_crews"] = sent["sent"] / sent["firefighters"]
    sent["power_after"] = sent["power"] * (1 - sent["share_of_own_crews"])
    sent = sent.sort_values("sent", ascending=False)
    covered = int(sent["sent"].sum())
    summary = {
        "recipient": recipient, "need": float(need), "covered": covered, "shortfall": float(max(0.0, need - covered)),
        "donors": int(len(sent)), "incident": [float(lat0), float(lon0)],
        "mean_distance_km": float(np.average(sent["distance_km"], weights=sent["sent"])) if covered else None,
        "objective": float((sent["cost"] * sent["sent"]).sum()), "status": str(status),
        "available_capacity": float(donors["capacity"].sum()),
    }
    cols = ["sent", "share_of_need", "share_of_own_crews", "distance_km", "power", "power_after", "capacity", "cost"]
    return sent[cols], summary
