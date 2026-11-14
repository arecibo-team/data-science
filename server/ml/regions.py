"""Country level risk: the cell model summed over each country.

Risk of a country on day D = expected number of 0.1 degree cells with a confirmed fire in D+1..D+horizon (the sum
of the calibrated cell probabilities, over cells with recent fire activity). Colours compare that number with the
country's own history, so a quiet country in a bad week turns red even if its absolute number is small.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape

from .. import db
from . import features, train

FILE = Path(__file__).resolve().parents[2] / "firms" / "regions" / "countries.geojson"


@lru_cache(maxsize=1)
def countries() -> tuple[list[str], list]:
    fs = json.loads(FILE.read_text())["features"]
    geoms = [shape(f["geometry"]) for f in fs]
    for g in geoms:
        shapely.prepare(g)
    return [f["properties"]["name"] for f in fs], geoms


def country_ids(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Index into countries()[0] per point, -1 for sea or unknown. Small countries win over big ones (Kosovo in Serbia)."""
    _, geoms = countries()
    out = np.full(len(lat), -1, np.int16)
    for i in sorted(range(len(geoms)), key=lambda i: -geoms[i].area):
        out[shapely.contains_xy(geoms[i], lon, lat)] = i
    return out


def series(d: features.Data, booster, names: list[str], horizon: int, neg_keep: float, chunk: int = 60) -> dict:
    """Per day and country: expected fire cells (sum p), cells at risk, and cells that really burned in the window."""
    lat = d.static[:, 0]
    lon = d.static[:, 1]
    cid = country_ids(lat, lon)
    names_c, _ = countries()
    n_c = len(names_c)
    act, lab = features.active_mask(d), features.label_mask(d, horizon)
    T = d.n_days
    expected = np.zeros((T, n_c), np.float32)
    active = np.zeros((T, n_c), np.int32)
    burned = np.zeros((T, n_c), np.int32)
    for a in range(features.MIN_DAY, T, chunk):
        days = np.arange(a, min(a + chunk, T))
        pos, ci = np.nonzero(act[:, days])
        if not len(pos):
            continue
        di = days[ci]
        p = train.calibrate(booster.predict(features.select(features.build(d, pos, di), names)), neg_keep)
        ok = cid[pos] >= 0
        np.add.at(expected, (di[ok], cid[pos][ok]), p[ok])
        np.add.at(active, (di[ok], cid[pos][ok]), 1)
    # real: every cell with a confirmed fire in D+1..D+horizon, scored or not
    kc = d.keys
    clat, clon = (kc // features.KEY + 0.5) * db.BIN_DEG - 90, (kc % features.KEY + 0.5) * db.BIN_DEG - 180
    call = country_ids(clat, clon)
    cum = features._cum(d.Y[:len(kc)])
    idx = np.arange(T)
    hi = np.minimum(idx + horizon + 1, T)
    w = (cum[:, hi] - cum[:, idx + 1]) > 0
    w[:, T - horizon:] = False
    for c in range(n_c):
        m = call == c
        if m.any():
            burned[:, c] = w[m].sum(axis=0)
    return {"countries": names_c, "expected": expected, "active": active, "burned": burned}


def store(run_id: int, d: features.Data, s: dict, horizon: int) -> int:
    """Write the series to country_risk. pct = share of the country's own days (with recent fires) that had a
    lower or equal expected count; None for countries that never reach one expected fire cell (nothing to rank)."""
    E, A, B = s["expected"], s["active"], s["burned"]
    rows = []
    for c, name in enumerate(s["countries"]):
        days = np.arange(features.MIN_DAY, d.n_days)
        days = days[A[days, c] > 0]
        if not len(days):
            continue
        hist = np.sort(E[days, c])
        ranked = E[days, c].max() >= 1.0
        for day in days:
            pct = float(np.searchsorted(hist, E[day, c], side="right") / len(hist)) if ranked else None
            rows.append((run_id, d.date_of(int(day)), name, float(E[day, c]), pct, int(A[day, c]), int(B[day, c])))
    with db.session() as con:
        con.execute("DELETE FROM country_risk WHERE run_id=?", (run_id,))
        con.executemany("INSERT INTO country_risk VALUES (?,?,?,?,?,?,?)", rows)
    return len(rows)


def score(expected: float, pct: float | None) -> float:
    """0 (green) to 1 (red): unusual for this country AND at least about one fire cell expected. The second factor
    stops a country whose numbers are all near zero from turning red on a rank alone."""
    return 0.0 if pct is None else float(pct * min(1.0, expected / 1.0))


def risk(run_id: int, date: str, horizon: int) -> dict | None:
    """Country risk for a forecast made on `date` (clamped to the last stored day)."""
    with db.session() as con:
        last = con.execute("SELECT MAX(date) FROM country_risk WHERE run_id=?", (run_id,)).fetchone()[0]
        if not last:
            return None
        day = min(date, last)
        rows = con.execute("SELECT country, expected, pct, active, burned FROM country_risk WHERE run_id=? AND date=?",
                           (run_id, day)).fetchall()
    import datetime as dt
    known = dt.date.fromisoformat(day) + dt.timedelta(days=horizon) <= dt.date.fromisoformat(last)
    return {"run": run_id, "date": date, "feature_date": day, "horizon": horizon, "complete": known,
            "countries": [{"name": r[0], "expected": r[1], "pct": r[2], "score": score(r[1], r[2]), "active": r[3],
                           "burned": r[4] if known else None}
                          for r in rows]}
