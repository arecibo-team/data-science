"""Cell-day features for the forecast model.

Unit: a 0.1 degree `bin_day` cell on a day D. Label: a fire in the cell on any of D+1..D+HORIZON: label="both" (seen by
MODIS and VIIRS, available for every year since 2012) or label="confirmed" (the UI's COMMON layer: a detection inside an
EU incident, `cm + cv`, only as old as the incident data). Every feature uses days <= D only, except the `fc_*` weather
features, which look at the weather of D+1..D+3 (a forecast when the model is used, the reanalysis when it is trained).
Candidate cells are all cells with any detection up to the train end (minus a held-out window), so the test period
does not choose the cells. Every candidate cell is scored on every day: there is no "recently burned" filter.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import context, db
from . import dryness as dry

KEY = 100_000
MIN_DAY = 30          # first usable day index, so the 30 day windows are full
ACTIVE_DAYS = 90
SOIL_COLS = ["sand", "rock", "sparse", "burnt", "glacier", "intertidal", "quarry", "dump", "construction"]
WEATHER_COLS = ["t2m", "rh", "ws", "prec", "gwettop", "gwetroot", "gwetprof"]
WEATHER_EXTRA = ["t2mx", "wsx"]
FEATURES_V1 = (  # what runs trained before the lean weather set used
    ["b0", "hi0", "mv0", "b3", "b7", "b30", "c7", "c30", "mv7", "mv30", "hi7", "lo7", "nb7", "since_b",
     "doy_sin", "doy_cos", "lat", "lon"]
    + SOIL_COLS + WEATHER_COLS + ["prec7", "t2m7"])
# build() computes every column; a run trains on its own list. The active list has one weather feature per driver:
# 30-day rain, days since rain, vapor pressure deficit, root-zone soil wetness, temperature anomaly, wind.
WEATHER_ACTIVE = ["prec30", "dry_days", "vpd", "gwetroot", "t2m_anom", "ws"]
FORECAST = ["fc_wsx", "fc_tmaxa", "fc_prec", "fc_vpd"]   # weather of D+1..D+3: peak wind, peak temperature anomaly, rain, dryness
FEATURES_ALL = FEATURES_V1 + dry.DERIVED + FORECAST
FEATURES_V2 = [f for f in FEATURES_V1 if f not in WEATHER_COLS and f not in ("prec7", "t2m7")] + WEATHER_ACTIVE   # before forecast weather
FEATURES = FEATURES_V2 + FORECAST
FUTURE_DAYS = 7        # weather arrays reach this many days past the last fire day, for the forecast
# No fire history at all: season, place, soil and weather only. Tests whether conditions alone can foresee fires.
FEATURES_CONDITIONS = ["doy_sin", "doy_cos", "lat", "lon"] + SOIL_COLS + WEATHER_ACTIVE + FORECAST
NEIGHBOURS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


@dataclass
class Data:
    start: dt.date
    n_days: int
    keys: np.ndarray                 # sorted cell keys r * KEY + c, length n_all
    cand: np.ndarray                 # indices into keys of the candidate cells
    B: np.ndarray                    # (n_all + 1, T) uint8 common detections; last row is zeros
    MV: np.ndarray                   # MODIS + VIIRS detections
    HI: np.ndarray                   # high-confidence detections
    LO: np.ndarray
    C: np.ndarray                    # confirmed inside an EU incident (the UI's COMMON)
    Y: np.ndarray                    # the label channel: C or B
    nb: np.ndarray                   # (n_cand, 8) neighbour indices into B rows
    static: np.ndarray               # (n_cand, 2 + 9) lat, lon, soil
    wi: np.ndarray                   # (n_cand,) row in W or -1
    last: np.ndarray                 # (n_cand, T) int16 index of the latest common day <= day, or -1
    W: dict[str, np.ndarray] = field(default_factory=dict)
    train_end: int = 0               # day index of the last train day
    hold: tuple[int, int] | None = None   # day indices of a held-out window, never trained on (see load)

    def day_index(self, date: str) -> int:
        return (dt.date.fromisoformat(date) - self.start).days

    def date_of(self, i: int) -> str:
        return (self.start + dt.timedelta(days=int(i))).isoformat()


def load(train_end: str, con=None, label: str = "confirmed", holdout: tuple[str, str] | None = None) -> Data | None:
    """holdout=(first, last) dates: that window is removed from training, and from the choice of candidate cells, so a
    cell that only ever burned inside it is unknown to the model. Used to benchmark against one event."""
    own = con is None
    con = con or db.connect(readonly=True)
    try:
        bd = pd.read_sql("SELECT r, c, date, m + v AS mv, b, hi, lo, cm + cv AS cc FROM bin_day", con)
        soil = pd.read_sql("SELECT * FROM soil_cells", con)
        wg = pd.read_sql("SELECT * FROM weather_grid", con)
        wf = pd.read_sql("SELECT r1, c1, date, t2m, t2mx, ws, wsx, prec, rh FROM weather_forecast", con)
    finally:
        if own:
            con.close()
    if bd.empty:
        return None
    day = pd.to_datetime(bd["date"])
    start = day.min().date()
    t = int((day.max() - day.min()).days) + 1
    di = (day - day.min()).dt.days.to_numpy()
    key = bd["r"].to_numpy(np.int64) * KEY + bd["c"].to_numpy(np.int64)
    keys, inv = np.unique(key, return_inverse=True)
    n = len(keys)

    def dense(col):
        a = np.zeros((n + 1, t), np.uint8)
        a[inv, di] = np.minimum(bd[col].to_numpy(), 255)
        return a
    B, MV, HI, LO, C = dense("b"), dense("mv"), dense("hi"), dense("lo"), dense("cc")
    del bd, day, di, inv
    Y = C if label == "confirmed" else B
    te = (dt.date.fromisoformat(train_end) - start).days
    te = min(te, t - 1)
    hold = None
    seen = np.ones(te + 1, bool)
    if holdout:
        hold = tuple((dt.date.fromisoformat(x) - start).days for x in holdout)
        seen[max(hold[0], 0): hold[1] + 1] = False
    ever = MV[:n, : te + 1][:, seen].any(axis=1) | Y[:n, : te + 1][:, seen].any(axis=1)
    cand = np.flatnonzero(ever)
    if not len(cand):
        return None
    r, c = keys[cand] // KEY, keys[cand] % KEY
    nb = np.empty((len(cand), 8), np.int64)
    for j, (dr, dc) in enumerate(NEIGHBOURS):
        k2 = (r + dr) * KEY + (c + dc)
        pos = np.minimum(np.searchsorted(keys, k2), n - 1)
        nb[:, j] = np.where(keys[pos] == k2, pos, n)
    lat, lon = (r + 0.5) * db.BIN_DEG - 90, (c + 0.5) * db.BIN_DEG - 180
    static = np.zeros((len(cand), 2 + len(SOIL_COLS)), np.float32)
    static[:, 0], static[:, 1] = lat, lon
    if len(soil):
        sk = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(soil["r4"], soil["c4"]))}
        sv = soil[SOIL_COLS].to_numpy(np.float32)
        for i, (la, lo_) in enumerate(zip(lat, lon)):
            j = sk.get(context.cell_index(la, lo_))
            if j is not None:
                static[i, 2:] = sv[j]
    W, wi = _weather(start, t, wg, wf, r, c)
    last = np.empty((len(cand), t), np.int16)
    for a in range(0, len(cand), 20000):     # latest fire day <= each day, in blocks of cells
        blk = cand[a:a + 20000]
        idx = np.where(B[blk] > 0, np.arange(t, dtype=np.int16)[None, :], np.int16(-1))
        last[a:a + 20000] = np.maximum.accumulate(idx, axis=1)
    return Data(start, t, keys, cand, B, MV, HI, LO, C, Y, nb, static, wi, last, W, te, hold)


HORIZON_FC = 3


def _weather(start, t, wg, wf, r, c):
    """Weather arrays (1 degree cells, t + FUTURE_DAYS days) and, per candidate cell, its row (or -1). Days after the
    NASA reanalysis ends are taken from the forecast table, then carried forward."""
    W: dict[str, np.ndarray] = {}
    if wg.empty:
        return W, np.full(len(r), -1, np.int64)
    tw = t + FUTURE_DAYS
    wk = wg["r1"].to_numpy(np.int64) * KEY + wg["c1"].to_numpy(np.int64)
    wkeys, winv = np.unique(wk, return_inverse=True)
    wday = (pd.to_datetime(wg["date"]) - pd.Timestamp(start)).dt.days.to_numpy()
    ok = (wday >= 0) & (wday < tw)
    if len(wf):
        fkey = wf["r1"].to_numpy(np.int64) * KEY + wf["c1"].to_numpy(np.int64)
        fk = np.minimum(np.searchsorted(wkeys, fkey), len(wkeys) - 1)
        fday = (pd.to_datetime(wf["date"]) - pd.Timestamp(start)).dt.days.to_numpy()
        fok = (fday >= 0) & (fday < tw) & (wkeys[fk] == fkey)
    for col in WEATHER_COLS + WEATHER_EXTRA:
        a = np.full((len(wkeys), tw), np.nan, np.float32)
        if len(wf) and col in wf:
            a[fk[fok], fday[fok]] = wf[col].to_numpy(np.float32)[fok]       # forecast first, reanalysis overwrites it
        a[winv[ok], wday[ok]] = wg[col].to_numpy(np.float32)[ok]
        W[col] = pd.DataFrame(a.T).ffill().to_numpy().T.astype(np.float32)   # carry the last known day forward
    months = ((pd.Timestamp(start) + pd.to_timedelta(np.arange(tw), unit="D")).month.to_numpy() - 1)
    W.update(dry.dryness(W["t2m"], W["rh"], W["prec"], months))
    W.update(dry.lookahead(W, months, HORIZON_FC))
    ck = (r // 10) * KEY + (c // 10)
    pos = np.minimum(np.searchsorted(wkeys, ck), len(wkeys) - 1)
    return W, np.where(wkeys[pos] == ck, pos, -1)


def _cum(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], a.shape[1] + 1), np.int32)
    np.cumsum(a, axis=1, dtype=np.int32, out=out[:, 1:])
    return out


def active_mask(d: Data) -> np.ndarray:
    """(n_cand, T) bool: any detection in the previous ACTIVE_DAYS days (including the day). Used to stratify the
    sampling and for reports; the model scores every cell."""
    out = np.empty((len(d.cand), d.n_days), bool)
    lo = np.maximum(np.arange(1, d.n_days + 1) - ACTIVE_DAYS, 0)
    for a in range(0, len(d.cand), 20000):
        cum = _cum(d.MV[d.cand[a:a + 20000]])
        out[a:a + 20000] = (cum[:, 1:] - cum[:, lo]) > 0
    return out


def label_mask(d: Data, horizon: int) -> np.ndarray:
    """(n_cand, T) bool: a fire in D+1..D+horizon. Days without a full horizon are False."""
    out = np.empty((len(d.cand), d.n_days), bool)
    idx = np.arange(d.n_days)
    hi = np.minimum(idx + horizon + 1, d.n_days)
    for a in range(0, len(d.cand), 20000):
        cum = _cum(d.Y[d.cand[a:a + 20000]])
        out[a:a + 20000] = (cum[:, hi] - cum[:, idx + 1]) > 0
    out[:, d.n_days - horizon:] = False
    return out


BLOCK = 32


def build(d: Data, pos: np.ndarray, di: np.ndarray) -> np.ndarray:
    """Feature matrix (len(pos), len(FEATURES_ALL)) for candidate positions `pos` on day indices `di`; use select()
    for a run's own feature list. Rows are processed in blocks of days, so the window sums only need the cumulative
    counts of the 30 days before the block, never the whole history."""
    out = np.empty((len(pos), len(FEATURES_ALL)), np.float32)
    for a in (np.unique(di // BLOCK) if len(di) else []):
        sel = np.flatnonzero(di // BLOCK == a)
        lo = max(a * BLOCK - 30, 0)
        hi = min((a + 1) * BLOCK, d.n_days)
        cb, cm_, ch, cl, cc = (_cum(x[:, lo:hi]) for x in (d.B, d.MV, d.HI, d.LO, d.C))
        p, dd = pos[sel], di[sel]
        rows, dl = d.cand[p], dd - lo
        wsum = lambda cum, w: cum[rows, dl + 1] - cum[rows, np.maximum(dl + 1 - w, 0)]
        cols: dict[str, np.ndarray] = {}
        cols["b0"], cols["hi0"], cols["mv0"] = d.B[rows, dd], d.HI[rows, dd], d.MV[rows, dd]
        for w in (3, 7, 30):
            cols[f"b{w}"] = wsum(cb, w)
        cols["c7"], cols["c30"] = wsum(cc, 7), wsum(cc, 30)
        cols["mv7"], cols["mv30"] = wsum(cm_, 7), wsum(cm_, 30)
        cols["hi7"], cols["lo7"] = wsum(ch, 7), wsum(cl, 7)
        nbr = d.nb[p]
        lo7 = np.maximum(dl + 1 - 7, 0)[:, None]
        cols["nb7"] = (cb[nbr, dl[:, None] + 1] - cb[nbr, lo7]).sum(axis=1)
        last = d.last[p, dd].astype(np.int64)
        cols["since_b"] = np.where(last >= 0, np.minimum(dd - last, 90), 90)
        ang = 2 * np.pi * (((d.start.toordinal() + dd) - dt.date(2000, 1, 1).toordinal()) % 365.25) / 365.25
        cols["doy_sin"], cols["doy_cos"] = np.sin(ang), np.cos(ang)
        cols["lat"], cols["lon"] = d.static[p, 0], d.static[p, 1]
        for j, name in enumerate(SOIL_COLS):
            cols[name] = d.static[p, 2 + j]
        w_ok = d.wi[p] >= 0
        wrow = np.where(w_ok, d.wi[p], 0)
        for name in WEATHER_COLS + dry.DERIVED + FORECAST:
            arr = d.W.get(name)
            cols[name] = np.where(w_ok, arr[wrow, dd], np.nan) if arr is not None else np.full(len(p), np.nan)
        for name, src, agg in (("prec7", "prec", "sum"), ("t2m7", "t2m", "mean")):
            arr = d.W.get(src)
            if arr is None:
                cols[name] = np.full(len(p), np.nan)
                continue
            cs = np.cumsum(np.nan_to_num(arr), axis=1)
            cs = np.concatenate([np.zeros((cs.shape[0], 1), cs.dtype), cs], axis=1)
            l7 = np.maximum(dd + 1 - 7, 0)
            sm = cs[wrow, dd + 1] - cs[wrow, l7]
            cols[name] = np.where(w_ok, sm if agg == "sum" else sm / (dd + 1 - l7), np.nan)
        out[sel] = np.column_stack([np.asarray(cols[f], np.float32) for f in FEATURES_ALL])
    return out


def select(X: np.ndarray, names: list[str] | None) -> np.ndarray:
    """Columns of a build() matrix for a run's own feature list (older runs used fewer features)."""
    names = names or FEATURES_V1
    return X[:, [FEATURES_ALL.index(n) for n in names]]


KEEP_ACTIVE = 0.1      # share of the cell-days with recent fires that are kept as negatives in training
KEEP_COLD = 0.02       # same for cell-days without; every kept negative carries weight 1 / keep, so probabilities stay true


def sample_rows(d: Data, horizon: int, sel: np.ndarray, keep_active: float, keep_cold: float, rng):
    """Sampled cell-days on the days where `sel` is True: every positive, negatives at keep_active (cells with recent
    fires) or keep_cold (the rest), weighted by 1/keep. Returns pos, di, w, y."""
    act, lab = active_mask(d), label_mask(d, horizon)
    ppos, pdi = np.nonzero(lab & sel[None, :])
    npos = len(ppos)
    apos, adi = np.nonzero(act & ~lab & sel[None, :])
    if keep_active < 1:
        keep = rng.random(len(apos)) < keep_active
        apos, adi = apos[keep], adi[keep]
    n_cells, sdays = len(d.cand), np.flatnonzero(sel)     # the rest: uniform cell-days, minus active and positive ones
    if len(sdays) and keep_cold > 0:
        n_cold = int(n_cells * len(sdays) * keep_cold)
        cpos, cdi = rng.integers(0, n_cells, n_cold), sdays[rng.integers(0, len(sdays), n_cold)]
        ok = ~act[cpos, cdi] & ~lab[cpos, cdi]
        cpos, cdi = cpos[ok], cdi[ok]
    else:
        cpos = cdi = np.zeros(0, np.int64)
    pos = np.concatenate([ppos, apos, cpos])
    di = np.concatenate([pdi, adi, cdi])
    w = np.concatenate([np.ones(npos), np.full(len(apos), 1 / keep_active), np.full(len(cpos), 1 / keep_cold)]).astype(np.float32)
    y = np.concatenate([np.ones(npos), np.zeros(len(apos) + len(cpos))]).astype(np.float32)
    return pos, di, w, y


def training_rows(d: Data, horizon: int, split: str, keep_active: float = KEEP_ACTIVE, keep_cold: float = KEEP_COLD, seed: int = 0):
    """(X, y, w, (pos, di)) for split 'train' | 'val' | 'test'. train and val are random 80/20 parts of the train days.
    Negatives are sampled (see sample_rows), so a weighted loss or metric stands for every cell-day."""
    rng = np.random.default_rng(seed)
    days = np.arange(d.n_days)
    sel = (days > d.train_end) if split == "test" else ((days >= MIN_DAY) & (days + horizon <= d.train_end))
    sel &= (days >= MIN_DAY) & (days + horizon < d.n_days)
    if d.hold and split != "test":    # drop the window and every day whose label window reaches into it
        sel &= ~((days >= d.hold[0] - horizon) & (days <= d.hold[1]))
    pos, di, w, y = sample_rows(d, horizon, sel, keep_active, keep_cold, rng)
    if split != "test":
        part = rng.random(len(y)) < 0.8
        k = part if split == "train" else ~part
        pos, di, w, y = pos[k], di[k], w[k], y[k]
    return build(d, pos, di), y, w, (pos, di)


def predict_rows(d: Data, date: str):
    """Every candidate cell for a forecast made on `date` (clamped to the last day with data)."""
    di = min(max(d.day_index(date), 0), d.n_days - 1)
    pos = np.arange(len(d.cand))
    return pos, np.full(len(pos), di), build(d, pos, np.full(len(pos), di))
