"""The panel behind the country risk model: fires and weather per patch and day.

A patch is the part of a 1 degree weather cell that lies in one country (a border cell gives one patch per country).
Per patch and day the panel holds the number of 0.1 degree cells with a fire, and per weather cell and day the NASA
POWER weather with the Canadian fire weather codes computed from it. Country numbers are sums over a country's patches.

Everything that is "normal" (the climatology of fires, the usual temperature, the usual fire weather) is computed from
days up to `clim_end` only, so a model trained up to that day never sees a later season.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shapely

from .. import db
from ..ml import regions
from . import fwi

LEADS = (1, 2, 3)        # days ahead; each lead is its own model
MIN_DAY = 30
NEW_GAP = 7               # a cell counts as newly burning when it had no fire in the 7 days before
NEAR_DEG = 0.25           # a cell whose centre is in the sea belongs to the nearest country within this distance
WEATHER = ["t2m", "t2mx", "rh", "ws", "wsx", "prec"]          # what a forecast must supply
SOIL = ["gwettop", "gwetroot"]


@dataclass
class Panel:
    start: dt.date
    n_days: int                      # days with fire data
    n_w: int                         # days in the weather arrays (n_days + the longest lead, so the last forecasts have a target day)
    cell: np.ndarray                 # (P,) weather row of each patch
    cid: np.ndarray                  # (P,) country index of each patch
    lat: np.ndarray                  # (P,) fire-weighted centre of the patch
    lon: np.ndarray
    ncell: np.ndarray                # (P,) number of 0.1 degree cells that ever burned in the patch
    wkeys: np.ndarray                # (C, 2) r1, c1 of the weather cells
    B: np.ndarray                    # (P, T) cells with a fire seen by both sensors
    MV: np.ndarray                   # (P, T) cells with any detection
    CC: np.ndarray                   # (P, T) cells with a detection inside a mapped incident
    NEW: np.ndarray                  # (P, T) cells newly burning (both sensors)
    nbr: np.ndarray                  # (C, 8) rows of the neighbouring weather cells, or C (a row of zeros)
    W: dict = field(default_factory=dict)        # (C, n_w) weather and fire weather codes
    countries: list = field(default_factory=list)
    month: np.ndarray | None = None  # (n_w,) month 0..11
    doy: np.ndarray | None = None    # (n_w,) day of year 0..365

    def day_index(self, date: str) -> int:
        return (dt.date.fromisoformat(date) - self.start).days

    def date_of(self, i: int) -> str:
        return (self.start + dt.timedelta(days=int(i))).isoformat()


def country_of(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Country index per point; a point in the sea gets the nearest country within NEAR_DEG, else -1."""
    cid = regions.country_ids(lat, lon)
    miss = np.flatnonzero(cid < 0)
    if len(miss):
        _, geoms = regions.countries()
        pts = shapely.points(lon[miss], lat[miss])
        dist = np.stack([shapely.distance(g, pts) for g in geoms])
        best = dist.argmin(axis=0)
        cid[miss] = np.where(dist.min(axis=0) <= NEAR_DEG, best, -1)
    return cid


def load(con=None) -> Panel:
    own = con is None
    con = con or db.connect(readonly=True)
    try:
        bd = pd.read_sql("SELECT r, c, date, m + v AS mv, b, cm + cv AS cc FROM bin_day ORDER BY r, c, date", con)
        wg = pd.read_sql("SELECT r1, c1, date, t2m, t2mx, rh, ws, wsx, prec, gwettop, gwetroot FROM weather_grid", con)
    finally:
        if own:
            con.close()
    day = pd.to_datetime(bd["date"])
    start = day.min()
    T = int((day.max() - start).days) + 1
    di = (day - start).dt.days.to_numpy()
    key = bd["r"].to_numpy(np.int64) * 100_000 + bd["c"].to_numpy(np.int64)
    keys, inv = np.unique(key, return_inverse=True)
    r, c = keys // 100_000, keys % 100_000
    cid_cell = country_of((r + 0.5) * db.BIN_DEG - 90, (c + 0.5) * db.BIN_DEG - 180)

    wk = wg["r1"].to_numpy(np.int64) * 1000 + wg["c1"].to_numpy(np.int64)
    wkeys, winv = np.unique(wk, return_inverse=True)
    wrow = np.searchsorted(wkeys, (r // 10) * 1000 + c // 10)
    ok = (cid_cell >= 0) & (wrow < len(wkeys)) & (wkeys[np.minimum(wrow, len(wkeys) - 1)] == (r // 10) * 1000 + c // 10)
    pkey = np.where(ok, wrow * 1000 + cid_cell, -1)
    pkeys, pinv = np.unique(pkey[ok], return_inverse=True)
    patch_of = np.full(len(keys), -1)
    patch_of[ok] = pinv
    P = len(pkeys)

    b = bd["b"].to_numpy() > 0
    # newly burning: a fire seen by both sensors with none in the same cell in the NEW_GAP days before
    prev = np.full(len(bd), -10_000)
    idx = np.flatnonzero(b)
    same = key[idx][1:] == key[idx][:-1]
    prev[idx[1:][same]] = di[idx[:-1][same]]
    new = b & (di - prev > NEW_GAP)
    p_row = patch_of[inv]
    keep = p_row >= 0

    def dense(mask):
        a = np.zeros((P, T), np.uint16)
        np.add.at(a, (p_row[keep & mask], di[keep & mask]), 1)
        return a
    B, MV, CC, NEW = dense(b), dense(bd["mv"].to_numpy() > 0), dense(bd["cc"].to_numpy() > 0), dense(new)

    burned = np.zeros(len(keys))
    np.add.at(burned, inv, b)
    w8 = burned[ok] + 0.01
    lat = np.bincount(pinv, ((r + 0.5) * db.BIN_DEG - 90)[ok] * w8, P) / np.bincount(pinv, w8, P)
    lon = np.bincount(pinv, ((c + 0.5) * db.BIN_DEG - 180)[ok] * w8, P) / np.bincount(pinv, w8, P)

    n_w = T + max(LEADS)
    wday = (pd.to_datetime(wg["date"]) - start).dt.days.to_numpy()
    wok = (wday >= 0) & (wday < n_w)
    W = {}
    for col in WEATHER + SOIL:
        a = np.full((len(wkeys), n_w), np.nan, np.float32)
        a[winv[wok], wday[wok]] = wg[col].to_numpy(np.float32)[wok]
        W[col] = pd.DataFrame(a.T).ffill().bfill().to_numpy().T.astype(np.float32)   # carry the last known day forward
    dates = start + pd.to_timedelta(np.arange(n_w), unit="D")
    wr, wc = wkeys // 1000, wkeys % 1000
    nbr = np.empty((len(wkeys), 8), np.int64)
    for j, (dr, dc) in enumerate((a, b_) for a in (-1, 0, 1) for b_ in (-1, 0, 1) if (a, b_) != (0, 0)):
        k2 = (wr + dr) * 1000 + wc + dc
        pos = np.minimum(np.searchsorted(wkeys, k2), len(wkeys) - 1)
        nbr[:, j] = np.where(wkeys[pos] == k2, pos, len(wkeys))
    names, _ = regions.countries()
    p = Panel(start.date(), T, n_w, pkeys // 1000, pkeys % 1000, lat.astype(np.float32), lon.astype(np.float32),
              np.bincount(pinv, minlength=P), np.column_stack([wr, wc]), B, MV, CC, NEW, nbr, W, list(names),
              dates.month.to_numpy() - 1, dates.day_of_year.to_numpy() - 1)
    codes(p)
    return p


def codes(p: Panel) -> None:
    """Fire weather codes of every weather cell and day from the reanalysis, in p.W."""
    C = len(p.wkeys)
    out = {k: np.empty((C, p.n_w), np.float32) for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")}
    f, m, d = (np.full(C, v) for v in fwi.START)
    t, h, w = fwi.noon(p.W["t2m"], p.W["t2mx"], p.W["rh"], p.W["ws"], p.W["wsx"])
    for i in range(p.n_w):
        f, m, d, isi, bui, s = fwi.step(f, m, d, t[:, i], h[:, i], w[:, i], p.W["prec"][:, i], int(p.month[i]))
        for k, v in zip(out, (f, m, d, isi, bui, s)):
            out[k][:, i] = v
    p.W.update(out)
    p.W["rhx"] = h.astype(np.float32)
    rain = p.W["prec"] > 1.0
    since = np.zeros_like(p.W["prec"])
    run = np.zeros(C, np.float32)
    for i in range(p.n_w):
        run = np.where(rain[:, i], 0, np.minimum(run + 1, 90))
        since[:, i] = run
    p.W["dry_days"] = since
    p.W["prec30"] = _wsum(p.W["prec"], 30)


def _wsum(a: np.ndarray, w: int) -> np.ndarray:
    """Sum of the last w days including the day, along the last axis."""
    cs = np.concatenate([np.zeros((a.shape[0], 1), np.float64), np.cumsum(a, axis=1, dtype=np.float64)], axis=1)
    lo = np.maximum(np.arange(1, a.shape[1] + 1) - w, 0)
    return (cs[:, 1:] - cs[:, lo]).astype(np.float32)


@dataclass
class Normals:
    """What is usual, from days up to `end` only."""
    end: int
    fire: np.ndarray        # (P, 366) usual number of burning cells of a patch on a day of the year
    fire_all: np.ndarray    # (P,) the same over the whole year
    tmax: np.ndarray        # (C, 366) usual maximum temperature
    fwi95: np.ndarray       # (C,) the cell's 95th percentile of the fire weather index


def _by_doy(a: np.ndarray, doy: np.ndarray, half: int = 15) -> np.ndarray:
    """Mean per day of the year, smoothed over +-half days (wrapping around the year)."""
    s = np.zeros((a.shape[0], 366))
    n = np.bincount(doy, minlength=366).astype(float)
    for d in np.unique(doy):
        s[:, d] = a[:, doy == d].sum(axis=1)
    k = np.arange(-half, half + 1)
    idx = (np.arange(366)[:, None] + k[None, :]) % 366
    return (s[:, idx].sum(axis=2) / np.maximum(n[idx].sum(axis=1), 1)).astype(np.float32)


def normals(p: Panel, end: int) -> Normals:
    end = min(end, p.n_days - 1)
    doy = p.doy[: end + 1]
    return Normals(end, _by_doy(p.B[:, : end + 1].astype(np.float32), doy), p.B[:, : end + 1].mean(axis=1).astype(np.float32),
                   _by_doy(p.W["t2mx"][:, : end + 1], doy), np.percentile(p.W["fwi"][:, : end + 1], 95, axis=1).astype(np.float32))


FIRE = ["b0", "b3", "b7", "b30", "mv0", "mv7", "new3", "nbr3", "ctry3"]
STATE = ["ffmc0", "dmc0", "dc0", "bui0", "fwi0", "prec30", "dry_days", "gwettop", "gwetroot", "tmaxa7"]
OUTLOOK = ["fwi_t", "isi_t", "ffmc_t", "wsx_t", "tmaxa_t", "rh_t", "prec_t", "prec_path", "fwi_max", "wsx_max", "fwi_rel"]
STATIC = ["lat", "lon", "ncell", "clim", "clim_all", "doy_sin", "doy_cos"]
FEATURES = FIRE + STATE + OUTLOOK + STATIC
CONDITIONS = STATE + OUTLOOK + STATIC          # no fire history: weather, season and place only


def outlooks(p: Panel, nm: Normals, src: dict | None = None):
    """For each lead k in LEADS: the weather a forecast made on day D gives for D+k, as (C, n_days) arrays indexed by D.

    src[k] holds the forecast weather of every valid day made k days earlier (the WEATHER columns, (C, n_w), indexed
    by valid day); None uses the reanalysis itself, a perfect forecast. The fire weather codes start from the
    reanalysis codes of day D and are carried through the forecast days, exactly as in live use."""
    T = p.n_days
    D = np.arange(T)
    f, m, d = p.W["ffmc"][:, :T], p.W["dmc"][:, :T], p.W["dc"][:, :T]
    prec_path = np.zeros((len(p.wkeys), T), np.float32)
    fwi_max = np.zeros_like(prec_path)
    wsx_max = np.zeros_like(prec_path)
    for k in LEADS:
        w = {c: (src[k][c] if src else p.W[c])[:, D + k] for c in WEATHER}
        t, h, wind = fwi.noon(w["t2m"], w["t2mx"], w["rh"], w["ws"], w["wsx"])
        mon = p.month[D + k]
        nf, nm_, nd = np.empty_like(f), np.empty_like(m), np.empty_like(d)
        isi, s = np.empty_like(f), np.empty_like(f)
        for mth in np.unique(mon):
            c = mon == mth
            nf[:, c], nm_[:, c], nd[:, c], isi[:, c], _, s[:, c] = fwi.step(f[:, c], m[:, c], d[:, c], t[:, c], h[:, c], wind[:, c], w["prec"][:, c], int(mth))
        f, m, d = nf, nm_, nd
        prec_path = prec_path + w["prec"]
        fwi_max = np.maximum(fwi_max, s)
        wsx_max = np.maximum(wsx_max, w["wsx"])
        yield k, {"fwi_t": s, "isi_t": isi, "ffmc_t": f, "wsx_t": w["wsx"], "tmaxa_t": w["t2mx"] - nm.tmax[:, p.doy[D + k]],
                  "rh_t": h.astype(np.float32), "prec_t": w["prec"], "prec_path": prec_path, "fwi_max": fwi_max, "wsx_max": wsx_max,
                  "fwi_rel": s / np.maximum(nm.fwi95[:, None], 1.0)}


def base(p: Panel, nm: Normals) -> dict:
    """Features known on day D that do not depend on the lead: (P, T) fire history, (C, T) dryness."""
    T = p.n_days
    Bf, MVf, NEWf = (x.astype(np.float32) for x in (p.B, p.MV, p.NEW))
    out = {"b0": Bf, "b3": _wsum(Bf, 3), "b7": _wsum(Bf, 7), "b30": _wsum(Bf, 30), "mv0": MVf, "mv7": _wsum(MVf, 7),
           "new3": _wsum(NEWf, 3)}
    C = len(p.wkeys)
    cell3 = np.zeros((C + 1, T), np.float32)
    np.add.at(cell3, p.cell, out["b3"])
    out["nbr3"] = cell3[p.nbr].sum(axis=1)[p.cell]
    ctry = np.zeros((len(p.countries), T), np.float32)
    np.add.at(ctry, p.cid, out["b3"])
    out["ctry3"] = ctry[p.cid]
    w = {"ffmc0": "ffmc", "dmc0": "dmc", "dc0": "dc", "bui0": "bui", "fwi0": "fwi", "prec30": "prec30", "dry_days": "dry_days",
         "gwettop": "gwettop", "gwetroot": "gwetroot"}
    cellf = {k: p.W[v][:, :T] for k, v in w.items()}
    cellf["tmaxa7"] = _wsum(p.W["t2mx"][:, :T] - nm.tmax[:, p.doy[:T]], 7) / 7
    return {"patch": out, "cell": cellf}


def matrix(p: Panel, nm: Normals, bs: dict, ol: dict, k: int, pos: np.ndarray, di: np.ndarray, names=FEATURES) -> np.ndarray:
    """Feature rows for patches `pos` on issue days `di`, for the target day di + k."""
    X = np.empty((len(pos), len(names)), np.float32)
    cell = p.cell[pos]
    tdoy = p.doy[di + k]
    ang = 2 * np.pi * tdoy / 365.25
    for j, n in enumerate(names):
        if n in bs["patch"]:
            X[:, j] = bs["patch"][n][pos, di]
        elif n in bs["cell"]:
            X[:, j] = bs["cell"][n][cell, di]
        elif n in ol:
            X[:, j] = ol[n][cell, di]
        else:
            X[:, j] = {"lat": lambda: p.lat[pos], "lon": lambda: p.lon[pos], "ncell": lambda: p.ncell[pos],
                       "clim": lambda: nm.fire[pos, tdoy], "clim_all": lambda: nm.fire_all[pos],
                       "doy_sin": lambda: np.sin(ang), "doy_cos": lambda: np.cos(ang)}[n]()
    return X
