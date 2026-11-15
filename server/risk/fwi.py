"""Canadian Forest Fire Weather Index System (Van Wagner 1987), vectorised over cells, one day per step.

Inputs per day: temperature (C) and relative humidity (%) around midday, wind (km/h), rain of the last 24 h (mm).
The three moisture codes carry memory from day to day (FFMC about a day, DMC about two weeks, DC about two months),
so `step` takes yesterday's codes and returns today's. The same step runs on reanalysis days and on forecast days,
which is how a forecast made on day D is carried to D+1..D+7.
"""
from __future__ import annotations

import numpy as np

START = (85.0, 6.0, 15.0)      # standard start values of FFMC, DMC, DC
# Day length factors by month for latitudes north of 30 N (DMC) and 20 N (DC).
LE = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])
LF = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])


def es(t):
    """Saturation vapour pressure in kPa."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def noon(t2m, t2mx, rh, ws, wsx):
    """Midday temperature, humidity and wind (km/h) from daily values: the daily maximum temperature, the humidity
    the day's mean vapour pressure gives at that temperature, and the middle of mean and maximum wind."""
    h = np.clip(rh * es(t2m) / es(t2mx), 1.0, 100.0)
    return t2mx, h, 3.6 * 0.5 * (ws + wsx)


def step(ffmc, dmc, dc, t, h, w, rain, month: int):
    """One day. month is 0..11. Returns (ffmc, dmc, dc, isi, bui, fwi), each float32 of the input shape."""
    t, h, w, rain = (np.asarray(x, np.float64) for x in (t, h, w, rain))
    h = np.clip(h, 0.0, 100.0)
    with np.errstate(all="ignore"):
        # fine fuel moisture code
        mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        rf = np.maximum(rain - 0.5, 1e-9)
        mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
        mr = np.where(mo > 150.0, mr + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf), mr)
        mo = np.where(rain > 0.5, np.minimum(mr, 250.0), mo)
        ed = 0.942 * h ** 0.679 + 11.0 * np.exp((h - 100.0) / 10.0) + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h))
        ew = 0.618 * h ** 0.753 + 10.0 * np.exp((h - 100.0) / 10.0) + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h))
        kd = (0.424 * (1.0 - (h / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - (h / 100.0) ** 8)) * 0.581 * np.exp(0.0365 * t)
        kw = (0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - ((100.0 - h) / 100.0) ** 8)) * 0.581 * np.exp(0.0365 * t)
        m = np.where(mo > ed, ed + (mo - ed) * 10.0 ** (-kd), np.where(mo < ew, ew - (ew - mo) * 10.0 ** (-kw), mo))
        f = np.clip(59.5 * (250.0 - m) / (147.2 + m), 0.0, 101.0)

        # duff moisture code
        re = 0.92 * rain - 1.27
        mo_d = 20.0 + np.exp(5.6348 - dmc / 43.43)
        b = np.where(dmc <= 33.0, 100.0 / (0.5 + 0.3 * dmc),
                     np.where(dmc <= 65.0, 14.0 - 1.3 * np.log(np.maximum(dmc, 1e-9)), 6.2 * np.log(np.maximum(dmc, 1e-9)) - 17.2))
        mr_d = mo_d + 1000.0 * re / (48.77 + b * re)
        pr = np.maximum(244.72 - 43.43 * np.log(np.maximum(mr_d - 20.0, 1e-9)), 0.0)
        p0 = np.where(rain > 1.5, pr, dmc)
        p = p0 + 100.0 * 1.894 * (np.maximum(t, -1.1) + 1.1) * (100.0 - h) * LE[month] * 1e-6

        # drought code
        qr = 800.0 * np.exp(-dc / 400.0) + 3.937 * (0.83 * rain - 1.27)
        dr = np.maximum(400.0 * np.log(800.0 / np.maximum(qr, 1e-9)), 0.0)
        d0 = np.where(rain > 2.8, dr, dc)
        d = d0 + 0.5 * np.maximum(0.36 * (np.maximum(t, -2.8) + 2.8) + LF[month], 0.0)

        # initial spread, build-up, fire weather index
        mf = 147.2 * (101.0 - f) / (59.5 + f)
        isi = 0.208 * np.exp(0.05039 * w) * 91.9 * np.exp(-0.1386 * mf) * (1.0 + mf ** 5.31 / 4.93e7)
        u = np.where(p <= 0.4 * d, 0.8 * p * d / np.maximum(p + 0.4 * d, 1e-9),
                     p - (1.0 - 0.8 * d / np.maximum(p + 0.4 * d, 1e-9)) * (0.92 + (0.0114 * p) ** 1.7))
        u = np.maximum(u, 0.0)
        fd = np.where(u <= 80.0, 0.626 * u ** 0.809 + 2.0, 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * u)))
        bb = 0.1 * isi * fd
        s = np.where(bb > 1.0, np.exp(2.72 * (0.434 * np.log(np.maximum(bb, 1e-9))) ** 0.647), bb)
    return tuple(np.asarray(x, np.float32) for x in (f, p, d, isi, u, s))
