"""Cell-day features for the forecast model.

Unit: a 0.1 degree `bin_day` cell on a day D. Label: a COMMON detection (the UI's COMMON layer: a detection inside
an EU incident, `cm + cv`) in the cell on any of D+1..D+HORIZON. label="both" uses detections seen by both sensors instead. Every feature uses days <= D only.
Candidate cells are those with a common or both-sensor detection up to the train end, so the 2026 test set does not pick the
cells. Rows are cell-days with any detection in the previous 90 days.
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
FEATURES_V1 = (  # what runs trained before the lean weather set used
    ["b0", "hi0", "mv0", "b3", "b7", "b30", "c7", "c30", "mv7", "mv30", "hi7", "lo7", "nb7", "since_b",
     "doy_sin", "doy_cos", "lat", "lon"]
    + SOIL_COLS + WEATHER_COLS + ["prec7", "t2m7"])
# build() computes every column; a run trains on its own list. The active list has one weather feature per driver:
# 30-day rain, days since rain, vapor pressure deficit, root-zone soil wetness, temperature anomaly, wind.
WEATHER_ACTIVE = ["prec30", "dry_days", "vpd", "gwetroot", "t2m_anom", "ws"]
FEATURES_ALL = FEATURES_V1 + dry.DERIVED
FEATURES = [f for f in FEATURES_V1 if f not in WEATHER_COLS and f not in ("prec7", "t2m7")] + WEATHER_ACTIVE
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
    Y = C if label == "confirmed" else B
    te = (dt.date.fromisoformat(train_end) - start).days
    te = min(te, t - 1)
    hold = None
    seen = np.ones(te + 1, bool)
    if holdout:
        hold = tuple((dt.date.fromisoformat(x) - start).days for x in holdout)
        seen[max(hold[0], 0): hold[1] + 1] = False
    ever = (B[:n, : te + 1][:, seen].any(axis=1)) | (C[:n, : te + 1][:, seen].any(axis=1))
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
    W, wi = {}, np.full(len(cand), -1, np.int64)
    if len(wg):
        wk = (wg["r1"].to_numpy(np.int64) * KEY + wg["c1"].to_numpy(np.int64))
        wkeys, winv = np.unique(wk, return_inverse=True)
        wday = (pd.to_datetime(wg["date"]) - pd.Timestamp(start)).dt.days.to_numpy()
        ok = (wday >= 0) & (wday < t)
        for col in WEATHER_COLS:
            a = np.full((len(wkeys), t), np.nan, np.float32)
            a[winv[ok], wday[ok]] = wg[col].to_numpy(np.float32)[ok]
            W[col] = pd.DataFrame(a.T).ffill().to_numpy().T.astype(np.float32)  # carry the last known day forward
        months = ((pd.Timestamp(start) + pd.to_timedelta(np.arange(t), unit="D")).month.to_numpy() - 1)
        W.update(dry.dryness(W["t2m"], W["rh"], W["prec"], months))
        ck = (r // 10) * KEY + (c // 10)
        pos = np.minimum(np.searchsorted(wkeys, ck), len(wkeys) - 1)
        wi = np.where(wkeys[pos] == ck, pos, -1)
    idx = np.where(B[cand] > 0, np.arange(t, dtype=np.int16)[None, :], np.int16(-1))
    last = np.maximum.accumulate(idx, axis=1)
    return Data(start, t, keys, cand, B, MV, HI, LO, C, Y, nb, static, wi, last, W, te, hold)


def _cum(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], a.shape[1] + 1), np.int32)
    np.cumsum(a, axis=1, dtype=np.int32, out=out[:, 1:])
    return out


def _wsum(cum, rows, di, w):
    lo = np.maximum(di + 1 - w, 0)
    return cum[rows, di + 1] - cum[rows, lo]


def active_mask(d: Data) -> np.ndarray:
    """(n_cand, T) bool: any detection in the previous ACTIVE_DAYS days (including the day)."""
    cum = _cum(d.MV[d.cand])
    lo = np.maximum(np.arange(1, d.n_days + 1) - ACTIVE_DAYS, 0)
    return (cum[:, 1:] - cum[:, lo]) > 0


def label_mask(d: Data, horizon: int) -> np.ndarray:
    """(n_cand, T) bool: a common detection in D+1..D+horizon. Days without a full horizon are False."""
    cum = _cum(d.Y[d.cand])
    idx = np.arange(d.n_days)
    hi = np.minimum(idx + horizon + 1, d.n_days)
    lab = (cum[:, hi] - cum[:, idx + 1]) > 0
    lab[:, d.n_days - horizon:] = False
    return lab


def build(d: Data, pos: np.ndarray, di: np.ndarray) -> np.ndarray:
    """Feature matrix (len(pos), len(FEATURES_ALL)); use select() for a run's own list for candidate positions `pos` on day indices `di`."""
    rows = d.cand[pos]
    cols: dict[str, np.ndarray] = {}
    cb, cm_, ch, cl, cc = _cum(d.B), _cum(d.MV), _cum(d.HI), _cum(d.LO), _cum(d.C)
    cols["b0"], cols["hi0"], cols["mv0"] = d.B[rows, di], d.HI[rows, di], d.MV[rows, di]
    for w in (3, 7, 30):
        cols[f"b{w}"] = _wsum(cb, rows, di, w)
    cols["c7"], cols["c30"] = _wsum(cc, rows, di, 7), _wsum(cc, rows, di, 30)
    cols["mv7"], cols["mv30"] = _wsum(cm_, rows, di, 7), _wsum(cm_, rows, di, 30)
    cols["hi7"], cols["lo7"] = _wsum(ch, rows, di, 7), _wsum(cl, rows, di, 7)
    nbr = d.nb[pos]
    lo7 = np.maximum(di + 1 - 7, 0)[:, None]
    cols["nb7"] = (cb[nbr, di[:, None] + 1] - cb[nbr, lo7]).sum(axis=1)
    last = d.last[pos, di].astype(np.int64)
    has = last >= 0
    cols["since_b"] = np.where(has, np.minimum(di - last, 90), 90)
    ang = 2 * np.pi * (((d.start.toordinal() + di) - dt.date(2000, 1, 1).toordinal()) % 365.25) / 365.25
    cols["doy_sin"], cols["doy_cos"] = np.sin(ang), np.cos(ang)
    cols["lat"], cols["lon"] = d.static[pos, 0], d.static[pos, 1]
    for j, name in enumerate(SOIL_COLS):
        cols[name] = d.static[pos, 2 + j]
    w_ok = d.wi[pos] >= 0
    wrow = np.where(w_ok, d.wi[pos], 0)
    for name in WEATHER_COLS + dry.DERIVED:
        a = d.W.get(name)
        cols[name] = np.where(w_ok, a[wrow, di], np.nan) if a is not None else np.full(len(pos), np.nan)
    for name, src, agg in (("prec7", "prec", "sum"), ("t2m7", "t2m", "mean")):
        a = d.W.get(src)
        if a is None:
            cols[name] = np.full(len(pos), np.nan)
            continue
        cs = np.cumsum(np.nan_to_num(a), axis=1)
        cs = np.concatenate([np.zeros((cs.shape[0], 1), cs.dtype), cs], axis=1)
        lo = np.maximum(di + 1 - 7, 0)
        s = cs[wrow, di + 1] - cs[wrow, lo]
        cols[name] = np.where(w_ok, s if agg == "sum" else s / (di + 1 - lo), np.nan)
    return np.column_stack([np.asarray(cols[f], np.float32) for f in FEATURES_ALL])


def select(X: np.ndarray, names: list[str] | None) -> np.ndarray:
    """Columns of a build() matrix for a run's own feature list (older runs used fewer features)."""
    names = names or FEATURES_V1
    return X[:, [FEATURES_ALL.index(n) for n in names]]


def training_rows(d: Data, horizon: int, split: str, neg_keep: float = 0.1, seed: int = 0):
    """(X, y) for split 'train' | 'val' | 'test'. train and val are random 80/20 parts of the train days."""
    rng = np.random.default_rng(seed)
    act, lab = active_mask(d), label_mask(d, horizon)
    days = np.arange(d.n_days)
    if split == "test":
        sel = (days > d.train_end)
    else:
        sel = (days >= MIN_DAY) & (days + horizon <= d.train_end)
    sel &= (days >= MIN_DAY) & (days + horizon < d.n_days)
    if d.hold and split != "test":    # drop the window and every day whose label window reaches into it
        sel &= ~((days >= d.hold[0] - horizon) & (days <= d.hold[1]))
    mask = act & sel[None, :]
    if split != "test":
        mask &= lab | (rng.random(mask.shape) < neg_keep)
        part = rng.random(mask.shape) < 0.8
        mask &= part if split == "train" else ~part
    pos, di = np.nonzero(mask)
    return build(d, pos, di), lab[pos, di].astype(np.float32), (pos, di)


def predict_rows(d: Data, date: str):
    """Active candidate cells for a forecast made on `date` (clamped to the last day with data)."""
    di = min(max(d.day_index(date), 0), d.n_days - 1)
    act = active_mask(d)[:, di]
    pos = np.flatnonzero(act)
    return pos, np.full(len(pos), di), build(d, pos, np.full(len(pos), di))
