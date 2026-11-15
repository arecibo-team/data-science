from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import EU27, FIRE_NAMES
from .optimize import haversine_km

FIRE_ALERT = [(1000, "Red"), (200, "Orange"), (0, "Green")]
STATUS = [(4.0, "above"), (2.0, "watch"), (0.0, "normal")]
AIR_CAPABLE = {"FR", "IT", "ES", "GR", "HR", "PT", "SE"}
GROUND_TEAM = 40
BASE_YEARS = (2001, 2020)


class FireData:
    def __init__(self, folder: Path):
        d = json.loads((folder / "daily.json").read_text())
        self.start = date.fromisoformat(d["start"])
        self.series = {k: {kind: np.asarray(v[kind], dtype=float) for kind in v} for k, v in d["series"].items()}
        self.n = len(next(iter(self.series.values()))["harmonized"])
        self.end = self.start + timedelta(days=self.n - 1)
        self.events = pd.DataFrame(json.loads((folder / "events.json").read_text()))
        self.events["start_d"] = pd.to_datetime(self.events["start"]).dt.date
        self.events["end_d"] = pd.to_datetime(self.events["end"]).dt.date
        t = np.load(folder / "tiles.npz")
        self.tiles = {}
        df = pd.DataFrame({"ty": t["ty"], "tx": t["tx"], "day": t["day"], "h": t["h"]}).sort_values(["ty", "tx", "day"])
        for (ty, tx), g in df.groupby(["ty", "tx"]):
            self.tiles[(int(ty), int(tx))] = (g["day"].to_numpy(), np.concatenate([[0.0], np.cumsum(g["h"].to_numpy())]))
        self.meta = json.loads((folder / "meta.json").read_text())
        self.tile_deg = self.meta.get("tile_deg", 0.5)
        self.eu = sum(v["harmonized"] for k, v in self.series.items() if k in EU27)

    def idx(self, d: date) -> int:
        return (d - self.start).days

    def window(self, arr: np.ndarray, a: date, b: date) -> float:
        i, j = max(0, self.idx(a)), min(self.n - 1, self.idx(b))
        return float(arr[i:j + 1].sum()) if j >= i else 0.0

    def tile_window(self, lat: float, lon: float, a: date, b: date, radius: int = 2) -> float:
        ty, tx = math.floor(lat / self.tile_deg), math.floor(lon / self.tile_deg)
        i, j = self.idx(a), self.idx(b)
        total = 0.0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                t = self.tiles.get((ty + dy, tx + dx))
                if t is None:
                    continue
                days, cs = t
                total += cs[np.searchsorted(days, j, side="right")] - cs[np.searchsorted(days, i, side="left")]
        return float(total)

    def area_ratio(self, lat: float, lon: float, d: date) -> dict:
        a = d.replace(day=1)
        now = self.tile_window(lat, lon, a, d)
        base = [self.tile_window(lat, lon, _same(a, y), _month_end(_same(a, y))) for y in range(BASE_YEARS[0], BASE_YEARS[1] + 1)]
        med = float(np.median(base))
        return {"ratio": round(now / max(med, 1.0), 1), "cell_days": round(now), "median": round(med)}

    def active_fires(self, d: date, min_detections: int = 10, since: date | None = None, limit: int = 300) -> list[dict]:
        ev = self.events[(self.events["start_d"] <= d) & (self.events["end_d"] >= (since or d))]
        if since:
            ev = ev.nlargest(limit, "detections")
        out = []
        for e in ev.itertuples():
            upto = {k: v for k, v in e.daily.items() if k <= d.isoformat()}
            total = sum(v[0] for v in upto.values())
            if total < min_detections:
                continue
            last_day = max(upto)
            today = e.daily.get(d.isoformat(), [0])[0]
            ratio = self.area_ratio(e.lat, e.lon, d)
            status = next(s for cut, s in STATUS if ratio["ratio"] >= cut)
            out.append({"id": e.id, "iso2": e.iso2, "country": FIRE_NAMES[e.iso2], "name": e.name, "lat": e.lat, "lon": e.lon,
                        "start": e.start, "day": (d - e.start_d).days + 1, "detections": int(total), "hotspots_24h": int(today),
                        "last_sensor": upto[last_day][2], "last_time": f"{upto[last_day][1][:2]}:{upto[last_day][1][2:]}",
                        "last_date": last_day, "ratio": ratio["ratio"], "status": status,
                        "area_cell_days": ratio["cell_days"], "area_median": ratio["median"]})
        return sorted(out, key=lambda f: (-f["ratio"], -f["detections"]))

    def fire_shocks(self, fires: list[dict]) -> dict:
        shocks: dict[str, list] = {}
        for f in fires:
            if f["iso2"] not in EU27:
                continue
            alert = next(a for cut, a in FIRE_ALERT if f["detections"] >= cut)
            shocks.setdefault(f["iso2"], []).append(("WF", alert, 1))
        return shocks

    def normal(self, iso2: str, d: date) -> dict:
        arr = self.series[iso2]["harmonized"]
        a = d.replace(day=1)
        now = self.window(arr, a, d)
        years = [y for y in range(2001, d.year + 1)]
        whole = {y: self.window(arr, _same(a, y), _month_end(_same(a, y))) for y in years if y != d.year}
        full = [v for y, v in whole.items() if BASE_YEARS[0] <= y <= BASE_YEARS[1]]
        med = float(np.median(full)) if full else 0.0
        rank = 1 + sum(1 for v in whole.values() if v > now)
        return {"year": d.year, "month": d.month, "value": round(now), "median": round(med), "ratio": round(now / max(med, 1.0), 1),
                "rank": rank, "years": len(years), "since": years[0],
                "usual_low": round(float(np.percentile(full, 10))) if full else 0,
                "usual_high": round(float(np.percentile(full, 90))) if full else 0}

    def calendar(self, iso2: str, d: date) -> list[dict]:
        s = self.series[iso2]
        out = []
        for y in range(2001, d.year + 1):
            for m in range(1, 13):
                a = date(y, m, 1)
                if a > d:
                    out.append({"year": y, "month": m, "harmonized": None, "raw": None, "modelled": y < 2012, "partial": False})
                    continue
                b = min(_month_end(a), d)
                raw_kind = "modis" if y < 2012 else "viirs"
                out.append({"year": y, "month": m, "harmonized": round(self.window(s["harmonized"], a, b), 1),
                            "raw": round(self.window(s[raw_kind], a, b), 1), "modelled": y < 2012, "partial": b < _month_end(a)})
        return out

    def outlook(self, iso2: str, d: date, weeks: int = 6) -> dict:
        arr = self.series[iso2]["harmonized"]
        shares, weekly = [], []
        for y in range(2001, d.year):
            a = _same(d, y)
            total = self.window(arr, date(y, 1, 1), date(y, 12, 31))
            if total <= 0:
                continue
            shares.append(self.window(arr, a, a + timedelta(days=7 * weeks - 1)) / total)
            weekly += [self.window(arr, a + timedelta(days=7 * k), a + timedelta(days=7 * k + 6)) for k in range(weeks)]
        return {"weeks": weeks, "until": (d + timedelta(days=7 * weeks)).isoformat(),
                "share": round(float(np.median(shares)), 3) if shares else 0.0,
                "typical_week": round(float(np.median(weekly))) if weekly else 0,
                "bad_week": round(float(np.percentile(weekly, 90))) if weekly else 0}

    def largest(self, iso2: str, d: date, since: int = 2012, top: int = 4) -> list[dict]:
        ev = self.events[(self.events["iso2"] == iso2) & (self.events["start_d"] >= date(since, 1, 1)) & (self.events["start_d"] <= d)]
        ev = ev.sort_values("area_ha", ascending=False).head(top)
        return [{"name": e.name, "start": e.start, "area_ha": int(e.area_ha), "detections": int(e.detections)} for e in ev.itertuples()]

    def analog(self, iso2: str, d: date) -> dict | None:
        arr = self.series[iso2]["harmonized"]
        def vec(y):
            return np.log1p([self.window(arr, date(y, m, 1), min(_month_end(date(y, m, 1)), _same(d, y))) for m in range(1, d.month + 1)])
        cur = vec(d.year)
        best = min(((float(np.linalg.norm(vec(y) - cur)), y) for y in range(2001, d.year)), default=None)
        if best is None:
            return None
        y = best[1]
        ev = self.events[(self.events["iso2"] == iso2) & (self.events["start_d"] >= date(y, 1, 1)) & (self.events["start_d"] <= date(y, 12, 31))]
        top = ev.sort_values("area_ha", ascending=False).head(1)
        return {"year": y, "largest_fire": None if top.empty else top.iloc[0]["name"]}

    def eu_daily(self, a: date, b: date) -> list[dict]:
        out = []
        cur = a
        while cur <= b:
            i = self.idx(cur)
            out.append({"date": cur.isoformat(), "value": round(float(self.eu[i]), 1) if 0 <= i < self.n else None})
            cur += timedelta(days=1)
        return out


def _same(d: date, year: int) -> date:
    try:
        return d.replace(year=year)
    except ValueError:
        return d.replace(year=year, day=28)


def _month_end(d: date) -> date:
    nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return nxt - timedelta(days=1)


def eta_hours(iso2: str, distance_km: float) -> tuple[float, str]:
    if iso2 in AIR_CAPABLE and distance_km <= 3000:
        return 1.5 + distance_km / 330, "air"
    return 2.0 + distance_km * 1.25 / 80, "road"


def resources(iso2: str, sent: int, mode: str) -> dict:
    teams = max(1, math.ceil(sent / GROUND_TEAM))
    if mode == "air":
        bombers = 2
        return {"water_bombers": bombers, "ground_teams": teams, "text": f"{bombers} water bombers, {teams} ground team{'s' if teams > 1 else ''}"}
    return {"water_bombers": 0, "ground_teams": teams, "text": f"{teams} ground team{'s' if teams > 1 else ''} by road"}
