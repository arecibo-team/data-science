"""The risk model's country outlook in the database, and the answer of /api/risk built from it.

`store` writes one row per country, target day, lead and mode to `country_outlook`. `risk` sums the leads of
one issue day into the shape the map already draws (see server/ml/regions.py for the older, cell based version).
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .. import db
from ..ml import regions
from . import data

_cache: dict = {}


def store(p: data.Panel, perfect: np.ndarray, issued: np.ndarray | None, issued_cid: int | None, cover: np.ndarray | None,
          actual: np.ndarray, val: tuple[int, int], first: int) -> int:
    """perfect: (leads, countries, T) by issue day; issued: (leads, T) for country issued_cid, valid where cover is
    True. pct ranks a value against the country's own held-out validation days (val = first and last day index) of
    the same lead; level is 1 at or above their 90th percentile within May to October."""
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


def _history(con) -> dict:
    """Per country: the sorted totals over the leads of every stored issue day (perfect mode), to rank a day against."""
    ver = con.execute("SELECT COUNT(*) FROM country_outlook").fetchone()[0]
    if _cache.get("ver") != ver:
        hist: dict = {}
        for name, tot in con.execute("SELECT country, SUM(expected) FROM country_outlook WHERE mode='perfect' "
                                     "GROUP BY country, issued HAVING COUNT(*)=?", (len(data.LEADS),)):
            hist.setdefault(name, []).append(tot)
        _cache.update(ver=ver, hist={k: np.sort(v) for k, v in hist.items()})
    return _cache["hist"]


def risk(date: str) -> dict | None:
    """Country risk for a forecast issued on `date` (clamped to the last stored issue day): expected burning cell-days
    over the next days (one per lead), ranked against the country's own stored days. None when nothing is stored."""
    with db.session() as con:
        last = con.execute("SELECT MAX(issued) FROM country_outlook WHERE mode='perfect'").fetchone()[0]
        if not last:
            return None
        day = min(date, last)
        rows = con.execute("SELECT country, mode, lead, expected, level, actual FROM country_outlook WHERE issued=? ORDER BY lead", (day,)).fetchall()
        hist = _history(con)
        act = dict(con.execute("SELECT country, actual FROM country_outlook WHERE date=? AND mode='perfect' AND lead=1", (day,)).fetchall())
    by: dict = {}
    for name, mode, lead, exp, level, actual in rows:
        by.setdefault(name, {}).setdefault(mode, []).append((lead, exp, level, actual))
    out = []
    for name, modes in by.items():
        mode = "issued" if len(modes.get("issued", [])) == len(data.LEADS) else "perfect"
        ls = modes[mode]
        exp = float(sum(x[1] for x in ls))
        h = hist.get(name)
        pct = float(np.searchsorted(h, exp, side="right") / len(h)) if h is not None and len(h) and h[-1] >= 1.0 else None
        known = all(x[3] is not None for x in ls) and len(ls) == len(data.LEADS)
        out.append({"name": name, "expected": exp, "pct": pct, "score": regions.score(exp, pct), "active": int(act.get(name) or 0),
                    "burned": int(sum(x[3] for x in ls)) if known else None, "weather": mode,
                    "alert_days": int(sum(x[2] for x in ls)), "by_lead": [round(float(x[1]), 2) for x in ls]})
    complete = dt.date.fromisoformat(day) + dt.timedelta(days=max(data.LEADS)) <= dt.date.fromisoformat(last)
    return {"model": "risk", "date": date, "feature_date": day, "horizon": max(data.LEADS), "complete": complete, "countries": out}
