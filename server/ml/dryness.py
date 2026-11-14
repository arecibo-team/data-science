"""Physical dryness features per weather cell and day, from daily NASA POWER values.

Deliberately no fire-danger index (such as the Canadian FWI): those are hand-built formulas that encode prior fire
knowledge, and the model should learn the link between weather and fire itself from plain physical quantities.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)   # all-NaN cells: no weather yet

DRY_COLS = ["vpd", "dry_days", "prec30", "t2m_anom"]
DERIVED = DRY_COLS


def vpd(t: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Vapor pressure deficit in kPa."""
    return np.maximum(0.0, 0.6108 * np.exp(17.27 * t / (t + 237.3)) * (1 - rh / 100))


def dryness(t2m, rh, prec, month):
    """Dryness features per weather cell and day. All inputs (cells, days)."""
    days = t2m.shape[1]
    out = {"vpd": vpd(t2m, rh).astype(np.float32)}
    rain = np.nan_to_num(prec) > 1.0
    since = np.zeros_like(prec, dtype=np.float32)
    run = np.zeros(prec.shape[0], np.float32)
    for t in range(days):
        run = np.where(rain[:, t], 0, np.minimum(run + 1, 60))
        since[:, t] = run
    out["dry_days"] = since
    cs = np.concatenate([np.zeros((prec.shape[0], 1)), np.cumsum(np.nan_to_num(prec), axis=1)], axis=1)
    lo = np.maximum(np.arange(1, days + 1) - 30, 0)
    out["prec30"] = (cs[:, 1:] - cs[:, lo]).astype(np.float32)
    # temperature against the cell's own mean for the same calendar month across all years
    anom = np.empty_like(t2m, dtype=np.float32)
    for mth in range(12):
        cols = month == mth
        if cols.any():
            anom[:, cols] = t2m[:, cols] - np.nanmean(t2m[:, cols], axis=1, keepdims=True)
    out["t2m_anom"] = anom
    return out


FORECAST = ["fc_wsx", "fc_tmaxa", "fc_prec", "fc_vpd"]


def _anomaly(x, month):
    out = np.empty_like(x, dtype=np.float32)
    for mth in range(12):
        cols = month == mth
        if cols.any():
            out[:, cols] = x[:, cols] - np.nanmean(x[:, cols], axis=1, keepdims=True)
    return out


def lookahead(W: dict, month, horizon: int) -> dict:
    """Weather of the next `horizon` days (D+1..D+horizon) as features of day D: peak wind, peak temperature against
    the cell's normal for the month, rain, and dryness of the air. Trained on reanalysis, used with the forecast."""
    tmaxa = _anomaly(W["t2mx"], month)
    vp = vpd(W["t2m"], W["rh"])
    T = W["t2m"].shape[1]
    out = {k: np.full(W["t2m"].shape, np.nan, np.float32) for k in FORECAST}
    for d in range(T - 1):
        sl = slice(d + 1, min(d + 1 + horizon, T))
        out["fc_wsx"][:, d] = np.nanmax(W["wsx"][:, sl], axis=1)
        out["fc_tmaxa"][:, d] = np.nanmax(tmaxa[:, sl], axis=1)
        out["fc_prec"][:, d] = np.nansum(W["prec"][:, sl], axis=1)
        out["fc_vpd"][:, d] = np.nanmean(vp[:, sl], axis=1)
    return out
