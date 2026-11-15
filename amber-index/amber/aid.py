from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import NAMES
from .optimize import _solve, haversine_km
from .regions import GEO, REGIONS, TABLE, departure, region_of
from .replay import FIRE_ALERT
from .simulate import CREW_DEMAND, MAX_COMMITTED

DEPLOY = 1 / 6
MIN_TEAM = 20
TEAM = 40
ROAD_KMH, ROAD_DETOUR, ROAD_MOBILISE_H = 65.0, 1.25, 2.0
AIR_KMH, AIR_MOBILISE_H, AIR_LAST_LEG_H = 750.0, 6.0, 2.0


def fire_need(detections: float, load: float = 1.0) -> tuple[str, float]:
    alert = next(a for cut, a in FIRE_ALERT if detections >= cut)
    return alert, CREW_DEMAND[alert] * load


def travel(km: float) -> tuple[float, str]:
    road = ROAD_MOBILISE_H + km * ROAD_DETOUR / ROAD_KMH
    air = AIR_MOBILISE_H + km / AIR_KMH + AIR_LAST_LEG_H
    return (road, "road") if road <= air else (air, "air")


def power(base: float, pressure: float) -> float:
    return base * (1 - min(MAX_COMMITTED, max(0.0, pressure)))


def region_force(iso2: str, force: float) -> dict[str, float]:
    if iso2 in TABLE.index and TABLE.loc[iso2, "available"]:
        return {r: float(TABLE.loc[iso2, f"{r}_share"]) * force for r in REGIONS}
    shares = GEO.get(iso2, {}).get("area_share") or {}
    total = sum(shares.get(r, 0) for r in REGIONS) or 1.0
    return {r: (shares.get(r, 0.25 * total) / total) * force for r in REGIONS}


def _anchor(iso2: str, region: str) -> tuple[float, float]:
    g = GEO.get(iso2, {})
    a = g.get("anchors", {}).get(region) or g.get("centroid") or [0.0, 0.0]
    return float(a[0]), float(a[1])


def plan(recipient: str, fires: list[dict], table: pd.DataFrame, floor: float | None = None, fastest: bool = True,
         load: float = 1.0, keep_half: bool = True) -> dict:
    force = table["firefighters"].astype(float)
    base = (table["power_after"] / (1 - table["committed_share"]).clip(lower=1e-6)).astype(float)
    needs: dict[str, float] = {i: 0.0 for i in table.index}
    mine = []
    for f in fires:
        alert, need = fire_need(f["detections"], load if f["iso2"] == recipient else 1.0)
        needs[f["iso2"]] = needs.get(f["iso2"], 0.0) + need
        if f["iso2"] == recipient:
            mine.append({"id": f["id"], "name": f["name"], "lat": f["lat"], "lon": f["lon"], "detections": int(f["detections"]),
                         "alert": alert, "need": need, "region": region_of(recipient, f["lat"], f["lon"]) or "north"})
    mine.sort(key=lambda f: -f["need"])
    F, D, B = float(force[recipient]), float(needs[recipient]), float(base[recipient])
    out = {"recipient": recipient, "name": NAMES[recipient], "firefighters": round(F), "need": round(D), "sustainable": round(F * DEPLOY),
           "deploy_share": DEPLOY, "load": load, "fires": mine}

    forces = region_force(recipient, F)
    reg = {r: {"region": r, "firefighters": forces[r], "need": sum(f["need"] for f in mine if f["region"] == r),
               "fires": sum(1 for f in mine if f["region"] == r), "from_country": 0.0, "from_abroad": 0.0, "sent": 0.0} for r in REGIONS}
    for r, x in reg.items():
        x["need"] = float(math.ceil(x["need"]))
        x["capacity"] = float(math.floor(x["firefighters"] * DEPLOY))
        x["local"] = min(x["need"], x["capacity"])
        x["gap"] = x["need"] - x["local"]
        x["spare"] = x["capacity"] - x["local"]
        top = next((f for f in mine if f["region"] == r), None)
        x["front"] = (top["lat"], top["lon"]) if top else _anchor(recipient, r)

    moves = []
    pairs = sorted(((float(haversine_km(*_anchor(recipient, s), *reg[t]["front"])), s, t)
                    for s in REGIONS for t in REGIONS if s != t and reg[s]["spare"] >= 1 and reg[t]["gap"] >= 1))
    for km, s, t in pairs:
        n = math.floor(min(reg[s]["spare"], reg[t]["gap"]))
        if n < 1:
            continue
        hours, mode = travel(km)
        reg[s]["spare"] -= n
        reg[s]["sent"] += n
        reg[t]["gap"] -= n
        reg[t]["from_country"] += n
        moves.append({"kind": "national", "from_iso2": recipient, "from_name": NAMES[recipient], "from_region": s, "to_region": t,
                      "firefighters": n, "eta_h": round(hours, 1), "mode": mode, "distance_km": round(km),
                      "from_point": list(_anchor(recipient, s)), "to_point": list(reg[t]["front"])})

    gap = int(round(sum(x["gap"] for x in reg.values())))
    target_power = power(B, DEPLOY) if gap > 0 else power(B, D / F if F else 0)
    worst = max(REGIONS, key=lambda r: (reg[r]["gap"], reg[r]["need"]))
    incident = reg[worst]["front"] if mine else _anchor(recipient, worst)

    donors, left_out = [], []
    rows = []
    for c in table.index:
        if c == recipient:
            continue
        Fc, Dc, Bc = float(force[c]), float(needs.get(c, 0.0)), float(base[c])
        p_before = power(Bc, Dc / Fc)
        spare = max(0.0, Fc * DEPLOY - Dc)
        usable = spare / 2 if keep_half else spare
        limit = Fc * (1 - floor / Bc) - Dc if floor and Bc > 0 else float("inf")
        equity = Fc * (1 - target_power / Bc) - Dc if Bc > 0 else 0.0
        reason = ""
        if p_before < target_power or equity < MIN_TEAM:
            reason = "weaker than recipient"
        elif floor and p_before < floor:
            reason = "below floor"
        elif usable < MIN_TEAM:
            reason = "own fires" if Dc > 0 else "small force"
        elif limit < MIN_TEAM:
            reason = "would drop below"
        region, lat, lon = departure(c, incident[0], incident[1])
        km = float(haversine_km(lat, lon, incident[0], incident[1]))
        hours, mode = travel(km)
        rows.append({"iso2": c, "force": Fc, "own_need": Dc, "base": Bc, "power_before": p_before, "spare": spare,
                     "upper": 0.0 if reason else math.floor(min(usable, limit, equity)), "reason": reason, "region": region,
                     "departure": [round(lat, 4), round(lon, 4)], "km": km, "eta_h": hours, "mode": mode})
    d = pd.DataFrame(rows).set_index("iso2")
    sent = pd.Series(0.0, index=d.index)
    if gap > 0 and len(d):
        w = 0.8 if fastest else 0.3
        cost = w * d["eta_h"] / (d["eta_h"].max() or 1.0) + (1 - w) * (1 - d["power_before"] / 100)
        x, _ = _solve(cost.to_numpy(), d["upper"].to_numpy(dtype=float), int(gap), MIN_TEAM)
        sent = pd.Series(x, index=d.index)
    covered_abroad = int(sent.sum())

    remaining = {r: reg[r]["gap"] for r in REGIONS}
    for c in sent[sent > 0].index[np.argsort(d.loc[sent > 0, "eta_h"].to_numpy())]:
        r = d.loc[c]
        n = int(sent[c])
        left = n
        target = None
        for g in sorted(REGIONS, key=lambda g: -remaining[g]):
            if left <= 0 or remaining[g] <= 0:
                continue
            take = min(left, math.ceil(remaining[g]))
            remaining[g] -= take
            reg[g]["from_abroad"] += take
            left -= take
            target = target or g
        target = target or worst
        if left > 0:
            reg[target]["from_abroad"] += left
        after = power(r["base"], (r["own_need"] + n) / r["force"])
        donors.append({"iso2": c, "name": NAMES[c], "region": r["region"], "departure": r["departure"], "to_region": target,
                       "to_point": list(reg[target]["front"]), "sent": n, "teams": max(1, math.ceil(n / TEAM)), "mode": r["mode"],
                       "eta_h": round(float(r["eta_h"]), 1), "distance_km": round(float(r["km"])),
                       "power_before": round(float(r["power_before"]), 1), "power_after": round(float(after), 1),
                       "spare": round(float(r["spare"]))})
        moves.append({"kind": "abroad", "from_iso2": c, "from_name": NAMES[c], "from_region": r["region"], "to_region": target,
                      "firefighters": n, "eta_h": round(float(r["eta_h"]), 1), "mode": r["mode"], "distance_km": round(float(r["km"])),
                      "from_point": r["departure"], "to_point": list(reg[target]["front"])})
    if gap > 0:
        cand = d[(sent == 0)].sort_values("eta_h")
        for c, r in cand.iterrows():
            reason = r["reason"] or ("not needed" if covered_abroad >= gap else "")
            if reason and reason != "not needed":
                left_out.append({"iso2": c, "name": NAMES[c], "reason": reason, "power": round(float(r["power_before"]), 1),
                                 "eta_h": round(float(r["eta_h"]), 1)})
        left_out = left_out[:6]

    regions = []
    for r in REGIONS:
        x = reg[r]
        Fg = max(x["firefighters"], 1.0)
        received = x["from_country"] + x["from_abroad"]
        before = x["need"] / Fg
        after = (x["need"] - received + x["sent"]) / Fg
        regions.append({"region": r, "firefighters": round(x["firefighters"]), "fires": x["fires"], "need": round(x["need"]),
                        "capacity": round(x["capacity"]), "local": round(x["local"]), "from_country": round(x["from_country"]),
                        "from_abroad": round(x["from_abroad"]), "sent": round(x["sent"]),
                        "uncovered": round(max(0.0, x["need"] - x["local"] - received)),
                        "pressure_before": round(before, 3), "pressure_after": round(after, 3),
                        "power_before": round(power(B, before), 1), "power_after": round(power(B, after), 1),
                        "anchor": list(_anchor(recipient, r)), "front": list(x["front"])})

    local = sum(x["local"] for x in reg.values())
    national = sum(m["firefighters"] for m in moves if m["kind"] == "national")
    shortfall = max(0, gap - covered_abroad)
    steps = [{"h": 0.0, "label": "Local crews", "kind": "local", "firefighters": round(local)}]
    steps += [{"h": m["eta_h"], "label": f"{m['from_region'].capitalize()} {m['from_name']}" if m["kind"] == "national" else m["from_name"],
               "kind": m["kind"], "firefighters": m["firefighters"], "mode": m["mode"]} for m in sorted(moves, key=lambda m: m["eta_h"])]
    run = 0.0
    for s in steps:
        run += s["firefighters"]
        s["cumulative"] = round(run)
        s["coverage"] = round(min(1.0, run / D), 3) if D else 1.0
    if not mine:
        status = "no_fires"
    elif shortfall > 0:
        status = "short"
    elif gap > 0:
        status = "european"
    elif national > 0:
        status = "national"
    else:
        status = "local"
    out.update({
        "status": status, "regions": regions, "moves": moves, "donors": donors, "left_out": left_out, "timeline": steps,
        "incident": [round(incident[0], 4), round(incident[1], 4)], "incident_region": worst,
        "local": round(local), "national": round(national), "abroad_needed": int(gap), "abroad": covered_abroad,
        "shortfall": int(shortfall), "coverage_before": round(min(1.0, (local + national) / D), 3) if D else 1.0,
        "coverage_after": round(min(1.0, (local + national + covered_abroad) / D), 3) if D else 1.0,
        "hours": max((m["eta_h"] for m in moves), default=0.0) if not shortfall else None,
        "power_before": round(power(B, D / F), 1), "power_after": round(power(B, (D - covered_abroad) / F), 1),
        "pressure_before": round(D / F, 3), "pressure_after": round((D - covered_abroad) / F, 3),
    })
    return out
