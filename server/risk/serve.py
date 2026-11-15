"""The risk model's country outlook in the database, and the answer of /api/risk built from it.

`store` writes one row per country, target day, lead and mode to `country_outlook`. `risk` sums the leads of
one issue day into the shape the map already draws (see server/ml/regions.py for the older, cell based version).
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np
import shapely

from .. import db
from ..ml import regions
from . import data

_cache: dict = {}
EARTH_KM = 6371.0
SEASON_DAYS = 30          # a day is ranked against stored days within this many days of the same time of year
MIN_SEASON = 30           # fewer than this many such days: rank against every stored day instead


def store(p: data.Panel, perfect: np.ndarray, issued: np.ndarray | None, issued_cid: int | None, cover: np.ndarray | None,
          actual: np.ndarray, val: tuple[int, int], first: int) -> int:
    """perfect: (leads, countries, T) by issue day; issued: (leads, T) for country issued_cid, valid where cover is
    True. actual: (countries, T) confirmed cells per day (the real data). pct ranks a value against the country's own
    held-out validation days (val = first and last day index) of the same lead; level is 1 at or above their 90th
    percentile within May to October."""
    T = p.n_days
    dates = [p.date_of(i) for i in range(T + max(data.LEADS))]
    season = np.array([5 <= int(d[5:7]) <= 10 for d in dates[:T]])
    rows = []
    for k in data.LEADS:
        D = np.arange(first, T)
        vd = np.arange(val[0], val[1] - k + 1)
        for c, name in enumerate(p.countries):
            ref = np.sort(perfect[k - 1, c, vd])
            if ref[-1] <= 0:
                continue
            alert = float(np.percentile(perfect[k - 1, c, vd[season[vd + k]]], 90))
            for mode, vals, ok in (("perfect", perfect[k - 1, c], None),
                                   ("issued", issued[k - 1] if issued is not None and c == issued_cid else None, cover[k - 1] if cover is not None else None)):
                if vals is None:
                    continue
                for d in (D if ok is None else D[ok[D]]):
                    v = float(vals[d])
                    t = d + k
                    rows.append((name, dates[t], k, mode, dates[d], v, float(np.searchsorted(ref, v, side="right") / len(ref)),
                                 int(v >= alert), int(actual[c, t]) if t < T else None))
    with db.session() as con:
        con.execute("DELETE FROM country_outlook")
        con.executemany("INSERT INTO country_outlook VALUES (?,?,?,?,?,?,?,?,?)", rows)
    _cache.clear()
    return len(rows)


@lru_cache(maxsize=1)
def _area_km2() -> dict:
    """Surface of every country in km2, from its outline projected to an equal-area approximation."""
    names, geoms = regions.countries()

    def to_km(a):
        lon, lat = np.radians(a[:, 0]), np.radians(a[:, 1])
        return np.column_stack([EARTH_KM * lon * np.cos(lat), EARTH_KM * lat])
    return {n: float(shapely.transform(g, to_km).area) for n, g in zip(names, geoms)}


def _history(con) -> tuple[dict, dict]:
    """Per country, for every stored issue day with all leads: (day of year, total) to rank a day against. The first
    dict is the model's total (expected cells), the second the confirmed cells that really burned."""
    ver = con.execute("SELECT COUNT(*) FROM country_outlook").fetchone()[0]
    if _cache.get("ver") != ver:
        def pack(sql: str) -> dict:
            hist: dict = {}
            for name, issued, tot in con.execute(sql, (len(data.LEADS),)):
                hist.setdefault(name, []).append((dt.date.fromisoformat(issued).timetuple().tm_yday, tot))
            return {k: (np.array([d for d, _ in v]), np.array([t for _, t in v])) for k, v in hist.items()}
        _cache.update(ver=ver,
                      hist=pack("SELECT country, issued, SUM(expected) FROM country_outlook WHERE mode='perfect' "
                                "GROUP BY country, issued HAVING COUNT(*)=?"),
                      real=pack("SELECT country, issued, SUM(actual) FROM country_outlook WHERE mode='perfect' "
                                "GROUP BY country, issued HAVING COUNT(actual)=?"))
    return _cache["hist"], _cache["real"]


def _pct(h, day: str, exp: float) -> float | None:
    """Share of the country's stored days in the same season that had a total at or below exp. The season is the days
    within SEASON_DAYS of the same day of the year (across all years), so a summer day is not ranked against winter.
    None for a country that never sees fire."""
    if h is None:
        return None
    doy, tot = h
    d0 = dt.date.fromisoformat(day).timetuple().tm_yday
    gap = np.abs(doy - d0)
    if tot.max() < 1.0:
        return None
    near = tot[np.minimum(gap, 366 - gap) <= SEASON_DAYS]
    if len(near) < MIN_SEASON or near.max() < 1.0:      # too few days, or a season without fire: use every stored day
        near = tot
    return float(np.searchsorted(np.sort(near), exp, side="right") / len(near))


def risk(date: str) -> dict | None:
    """Country risk for a forecast issued on `date` (clamped to the last stored issue day): expected burning cell-days
    over the next days (one per lead), ranked against the country's own stored days. None when nothing is stored."""
    with db.session() as con:
        last = con.execute("SELECT MAX(issued) FROM country_outlook WHERE mode='perfect'").fetchone()[0]
        if not last:
            return None
        day = min(date, last)
        rows = con.execute("SELECT country, mode, lead, expected, level, actual FROM country_outlook WHERE issued=? ORDER BY lead", (day,)).fetchall()
        hist, real_hist = _history(con)
        act = dict(con.execute("SELECT country, actual FROM country_outlook WHERE date=? AND mode='perfect' AND lead=1", (day,)).fetchall())
    by: dict = {}
    for name, mode, lead, exp, level, actual in rows:
        by.setdefault(name, {}).setdefault(mode, []).append((lead, exp, level, actual))
    out = []
    for name, modes in by.items():
        mode = "issued" if len(modes.get("issued", [])) == len(data.LEADS) else "perfect"
        ls = modes[mode]
        exp = float(sum(x[1] for x in ls))
        pct = _pct(hist.get(name), day, exp)
        area = _area_km2().get(name)
        # real: confirmed cells in the leads that are in the data so far (complete once all leads are)
        seen = [x[3] for x in ls if x[3] is not None]
        complete = len(seen) == len(data.LEADS)
        real = None
        if seen:
            rexp = float(sum(seen))
            rpct = _pct(real_hist.get(name), day, rexp)
            real = {"expected": rexp, "pct": rpct, "score": regions.score(rexp, rpct),
                    "density": rexp / area * 1000 if area else None, "complete": complete}
        out.append({"name": name, "expected": exp, "pct": pct, "score": regions.score(exp, pct), "active": int(act.get(name) or 0),
                    "area_km2": area, "density": exp / area * 1000 if area else None,
                    "burned": int(sum(seen)) if complete else None, "real": real, "weather": mode,
                    "alert_days": int(sum(x[2] for x in ls)), "by_lead": [round(float(x[1]), 2) for x in ls]})
    complete = dt.date.fromisoformat(day) + dt.timedelta(days=max(data.LEADS)) <= dt.date.fromisoformat(last)
    return {"model": "risk", "date": date, "feature_date": day, "horizon": max(data.LEADS), "complete": complete, "countries": out}
