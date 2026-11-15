"""Daily weather on the 0.1 degree grid from E-OBS (ECA&D, public NetCDF, Europe, 2011 to mid 2025).

E-OBS cells line up with the model's cells, so every cell gets its own weather instead of one value per 100 km. The
step turns the daily rain, maximum temperature, humidity and wind into the same kind of features the 1 degree NASA
POWER weather gives (rain of 30 days, days since rain, dryness of the air, temperature against normal, wind, and the
weather of the next 3 days), for every cell that has a detection in `bin_day`, and stores them as float16 memory maps
in data/eobs/ (one file per feature, rows follow eobs_cells.npy).

E-OBS ends months before today, so days after its last one are NaN. That is a limit of the source, not of the
features: use it for training and for judging it over past seasons, and use a fresher source for live days.
"""
from __future__ import annotations

import datetime as dt
import logging
import warnings
from pathlib import Path

import numpy as np

from firms import config as C
from . import db

log = logging.getLogger("server.eobs")

DIR = C.DATA_DIR / "eobs"
SOURCES = {"rr": "rr", "tx": "tx", "fg": "fg", "hu": "hu"}
START = dt.date(2011, 1, 1)
FEATURES = ["e_prec30", "e_dry_days", "e_vpd", "e_txa", "e_fg", "e_fc_fg", "e_fc_txa", "e_fc_prec", "e_fc_vpd"]
KEY = 100_000
LAT0, LON0 = 1150, 1550          # bin_day row and column of the first E-OBS cell (lat 25.0, lon -25.0)


def files_ok() -> bool:
    return all((DIR / f"{v}_0.1deg_2011-2025_v32.0e.nc").exists() for v in SOURCES)


def _vpd(t, rh):
    return np.maximum(0.0, 0.6108 * np.exp(17.27 * t / (t + 237.3)) * (1 - rh / 100))


def _lookahead(a: np.ndarray, how: str, horizon: int = 3) -> np.ndarray:
    """Per day D the max (or sum, mean) of D+1..D+horizon along the last axis; NaN where the window is not full."""
    n = a.shape[1]
    parts = [np.concatenate([a[:, k:], np.full((a.shape[0], k), np.nan, a.dtype)], axis=1) for k in range(1, horizon + 1)]
    st = np.stack(parts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # windows without data stay NaN
        return {"max": np.nanmax, "sum": np.nansum, "mean": np.nanmean}[how](st, axis=0) if n else a


def _derive(rr, tx, hu, fg) -> dict[str, np.ndarray]:
    """Feature series (cells, days) from daily series of the same shape."""
    n_c, t = rr.shape
    out = {}
    cs = np.concatenate([np.zeros((n_c, 1), np.float32), np.cumsum(np.nan_to_num(rr), axis=1, dtype=np.float32)], axis=1)
    lo = np.maximum(np.arange(1, t + 1) - 30, 0)
    out["e_prec30"] = cs[:, 1:] - cs[:, lo]
    since = np.zeros((n_c, t), np.float32)
    run = np.zeros(n_c, np.float32)
    rainy = np.nan_to_num(rr) > 1.0
    for d in range(t):
        run = np.where(rainy[:, d], 0, np.minimum(run + 1, 60))
        since[:, d] = run
    out["e_dry_days"] = since
    vpd = _vpd(tx, hu).astype(np.float32)
    out["e_vpd"] = vpd
    months = np.array([(START + dt.timedelta(days=i)).month - 1 for i in range(t)])
    anom = np.empty_like(tx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for m in range(12):
            cols = months == m
            anom[:, cols] = tx[:, cols] - np.nanmean(tx[:, cols], axis=1, keepdims=True)
    out["e_txa"] = anom
    out["e_fg"] = fg
    out["e_fc_fg"] = _lookahead(fg, "max")
    out["e_fc_txa"] = _lookahead(anom, "max")
    out["e_fc_prec"] = _lookahead(rr, "sum")
    out["e_fc_vpd"] = _lookahead(vpd, "mean")
    return out


def build(block_rows: int = 30) -> int:
    """Write the feature memory maps for every cell of bin_day. Returns cells written."""
    import xarray as xr
    DIR.mkdir(parents=True, exist_ok=True)
    with db.session() as con:
        cells = np.array(con.execute("SELECT DISTINCT r, c FROM bin_day ORDER BY r, c").fetchall(), np.int64)
    keys = cells[:, 0] * KEY + cells[:, 1]
    np.save(DIR / "eobs_cells.npy", keys)
    dsets = {v: xr.open_dataset(DIR / f"{v}_0.1deg_2011-2025_v32.0e.nc")[v] for v in SOURCES}
    t = dsets["rr"].sizes["time"]
    nlat = dsets["rr"].sizes["latitude"]
    maps = {f: np.lib.format.open_memmap(DIR / f"{f}.npy", mode="w+", dtype=np.float16, shape=(len(keys), t)) for f in FEATURES}
    ri, ci = cells[:, 0] - LAT0, cells[:, 1] - LON0
    inside = (ri >= 0) & (ri < nlat) & (ci >= 0) & (ci < dsets["rr"].sizes["longitude"])
    for f in FEATURES:
        maps[f][:] = np.nan
    for a in range(0, nlat, block_rows):
        sel = np.flatnonzero(inside & (ri >= a) & (ri < a + block_rows))
        if not len(sel):
            continue
        arrs = {}
        for v, ds in dsets.items():
            blk = ds.isel(latitude=slice(a, a + block_rows)).values.astype(np.float32)      # (time, lat, lon)
            arrs[v] = blk[:, ri[sel] - a, ci[sel]].T                                          # (cells, time)
        feats = _derive(arrs["rr"], arrs["tx"], arrs["hu"], arrs["fg"])
        for f in FEATURES:
            maps[f][sel] = feats[f].astype(np.float16)
        log.info("eobs rows %d-%d: %d cells", a, a + block_rows, len(sel))
    for m in maps.values():
        m.flush()
    return len(keys)


def load():
    """(cell keys, {feature: memmap (cells, days)}, first date) or None when the step has not run."""
    if not (DIR / "eobs_cells.npy").exists() or not all((DIR / f"{f}.npy").exists() for f in FEATURES):
        return None
    return np.load(DIR / "eobs_cells.npy"), {f: np.load(DIR / f"{f}.npy", mmap_mode="r") for f in FEATURES}, START
