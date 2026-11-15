"""Weather forecasts as they were issued, for an honest replay of past days.

Open-Meteo's Previous Runs API keeps, for every hour, what the ECMWF IFS run made 1..7 days earlier said about it
(from March 2024). `fetch` stores those as daily values per weather cell in `weather_issued`; `source` turns them
into the arrays `data.outlooks` reads, corrected for the steady difference between the forecast model and the NASA
POWER reanalysis the risk model was trained on. The correction is fitted on days up to `fit_end` only.

    uv run python -m server.risk.forecasts --country Greece --from 2025-04-01 --to 2025-10-31
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time

import numpy as np
import pandas as pd
import requests

from .. import db
from . import data

log = logging.getLogger("server.risk.forecasts")

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs025"
VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
CHUNK_DAYS = 92
PAUSE = 2.0                 # seconds between requests: the free service counts long requests as many calls


def _get(params: dict) -> dict:
    for attempt in range(12):
        try:
            r = requests.get(URL, params=params, timeout=120)
        except requests.RequestException as exc:
            log.warning("open-meteo: %s; waiting 30s", str(exc)[:100])
            time.sleep(30)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            wait = 70 if "inutely" in r.text else (900 if "ourly" in r.text else 30 * (attempt + 1))
            log.warning("open-meteo %s: %s; waiting %ss", r.status_code, r.text[:100], wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError("open-meteo kept refusing the request")


def fetch_cell(r1: int, c1: int, lat: float, lon: float, start: str, end: str) -> list[tuple]:
    hourly = [f"{v}_previous_day{k}" for v in VARS for k in data.LEADS]
    j = _get({"latitude": round(lat, 3), "longitude": round(lon, 3), "hourly": ",".join(hourly), "start_date": start,
              "end_date": end, "models": MODEL, "wind_speed_unit": "ms", "timezone": "UTC"})
    h = pd.DataFrame(j["hourly"])
    h["date"] = h["time"].str[:10]
    g = h.groupby("date")
    rows = []
    for k in data.LEADS:
        t, rh, ws, pr = (f"{v}_previous_day{k}" for v in VARS)
        agg = pd.DataFrame({"t2m": g[t].mean(), "t2mx": g[t].max(), "rh": g[rh].mean(), "ws": g[ws].mean(), "wsx": g[ws].max(),
                            "prec": g[pr].sum(min_count=20), "n": g[t].count()})
        agg = agg[agg["n"] >= 20].drop(columns="n").dropna()
        rows += [(r1, c1, d, k, *map(float, v)) for d, v in zip(agg.index, agg.to_numpy())]
    return rows


def fetch(p: data.Panel, country: str, start: str, end: str) -> int:
    """Issued forecasts for every weather cell of `country`, at the fire-weighted centre of the country's part of the
    cell. Chunks already stored are skipped, so the fetch resumes."""
    db.init_db()
    cid = p.countries.index(country)
    total = 0
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    for i in np.flatnonzero(p.cid == cid):
        r1, c1 = (int(x) for x in p.wkeys[p.cell[i]])
        a = d0
        while a <= d1:
            b = min(a + dt.timedelta(days=CHUNK_DAYS - 1), d1)
            with db.session() as con:
                have = con.execute("SELECT COUNT(DISTINCT date) FROM weather_issued WHERE r1=? AND c1=? AND date BETWEEN ? AND ? AND lead=7",
                                   (r1, c1, a.isoformat(), b.isoformat())).fetchone()[0]
            if have < 0.9 * ((b - a).days + 1):
                rows = fetch_cell(r1, c1, float(p.lat[i]), float(p.lon[i]), a.isoformat(), b.isoformat())
                with db.session() as con:
                    con.executemany("INSERT OR REPLACE INTO weather_issued VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
                total += len(rows)
                log.info("cell %s,%s %s..%s: %d rows", r1, c1, a, b, len(rows))
                time.sleep(PAUSE)
            a = b + dt.timedelta(days=1)
    return total


def source(p: data.Panel, fit_end: str, con=None) -> tuple[dict, np.ndarray]:
    """(src, have): src[k][col] is (C, n_w) weather by valid day as forecast k days earlier, the reanalysis where no
    forecast is stored; have[k - 1] is (C, n_w) bool, True where the value is a real forecast.

    Each cell, lead and column is mapped onto the reanalysis with a line fitted on the days up to `fit_end` (rain: a
    factor), so a forecast and the training weather mean the same thing."""
    own = con is None
    con = con or db.connect(readonly=True)
    try:
        f = pd.read_sql("SELECT * FROM weather_issued", con)
    finally:
        if own:
            con.close()
    C = len(p.wkeys)
    src = {k: {c: p.W[c].copy() for c in data.WEATHER} for k in data.LEADS}
    have = np.zeros((len(data.LEADS), C, p.n_w), bool)
    if f.empty:
        return src, have
    wk = p.wkeys[:, 0].astype(np.int64) * 1000 + p.wkeys[:, 1]
    row = np.searchsorted(wk, f["r1"].to_numpy(np.int64) * 1000 + f["c1"].to_numpy(np.int64))
    day = (pd.to_datetime(f["date"]) - pd.Timestamp(p.start)).dt.days.to_numpy()
    fe = p.day_index(fit_end)
    ok = (day >= 0) & (day < p.n_w)
    lead = f["lead"].to_numpy()
    for k in data.LEADS:
        m = ok & (lead == k)
        have[k - 1][row[m], day[m]] = True
        for col in data.WEATHER:
            raw = np.full((C, p.n_w), np.nan, np.float32)
            raw[row[m], day[m]] = f[col].to_numpy(np.float32)[m]
            for ci in np.unique(row[m]):
                x, y = raw[ci, : fe + 1], p.W[col][ci, : fe + 1]
                fit = np.isfinite(x)
                if fit.sum() < 60:          # too few days to fit: leave this cell on the reanalysis
                    have[k - 1][ci] = False
                    continue
                if col == "prec":
                    a, b = 0.0, float(np.clip(y[fit].sum() / max(x[fit].sum(), 1e-6), 0.3, 3.0))
                else:
                    b, a = np.polyfit(x[fit], y[fit], 1)
                v = a + b * raw[ci]
                if col == "rh":
                    v = np.clip(v, 1, 100)
                elif col != "t2m" and col != "t2mx":
                    v = np.maximum(v, 0)
                src[k][col][ci] = np.where(np.isfinite(v), v, src[k][col][ci])
        hot = src[k]["t2mx"] < src[k]["t2m"]
        src[k]["t2mx"][hot] = src[k]["t2m"][hot]
        gust = src[k]["wsx"] < src[k]["ws"]
        src[k]["wsx"][gust] = src[k]["ws"][gust]
    return src, have


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="Greece")
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    a = ap.parse_args()
    print("rows stored:", fetch(data.load(), a.country, a.start, a.end))


if __name__ == "__main__":
    main()
