"""Daily weather on a 1 degree grid from NASA POWER, stored in `weather_grid` (the model's weather features).

One request per grid cell returns the whole date range. Cells already complete are skipped, so the step
resumes after a stop. Days newer than the NASA lag are filled forward when features are built.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from urllib.parse import urlencode

from . import db, fire_weather

log = logging.getLogger("server.weather_grid")

PARAMS = {"T2M": "t2m", "RH2M": "rh", "WS10M": "ws", "PRECTOTCORR": "prec",
          "GWETTOP": "gwettop", "GWETROOT": "gwetroot", "GWETPROF": "gwetprof"}
LAG_DAYS = 5
PAUSE_S = 0.4


def end_date() -> str:
    return (dt.date.today() - dt.timedelta(days=LAG_DAYS)).isoformat()


def cells_needed(con) -> list[tuple[int, int]]:
    """1 degree cells that ever held a detection."""
    return [(r, c) for r, c in con.execute(
        "SELECT DISTINCT r / 10, c / 10 FROM bin_day")]


def fetch_cell(r1: int, c1: int, start: str, end: str, timeout: float = 60) -> list[tuple]:
    lat, lon = r1 - 90 + 0.5, c1 - 180 + 0.5
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?" + urlencode({
        "parameters": ",".join(PARAMS), "community": "AG", "longitude": lon, "latitude": lat,
        "start": start.replace("-", ""), "end": end.replace("-", ""), "format": "JSON"})
    data, _, _ = fire_weather._fetch(url, "nasa_power", 0, timeout)  # ttl 0: parsed rows are stored below
    par = data["properties"]["parameter"]
    fill = data.get("header", {}).get("fill_value", -999)
    days = sorted(next(iter(par.values())))
    rows = []
    for day in days:
        vals = [par.get(code, {}).get(day) for code in PARAMS]
        vals = [None if v is None or v == fill else v for v in vals]
        rows.append((r1, c1, f"{day[:4]}-{day[4:6]}-{day[6:]}", *vals))
    return rows


def run(start: str, should_stop=lambda: False) -> int:
    end = end_date()
    done = 0
    with db.session() as con:
        cells = cells_needed(con)
        have = {(r, c) for r, c in con.execute(
            "SELECT r1, c1 FROM weather_grid GROUP BY r1, c1 HAVING MAX(date) >= ?", (end,))}
    for r1, c1 in cells:
        if (r1, c1) in have:
            done += 1
            continue
        if should_stop():
            break
        try:
            rows = fetch_cell(r1, c1, start, end)
        except (fire_weather.WeatherDataError, KeyError, StopIteration) as exc:
            log.warning("weather cell %s,%s skipped: %s", r1, c1, exc)
            continue
        with db.session() as con:
            con.executemany("INSERT OR REPLACE INTO weather_grid VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        done += 1
        time.sleep(PAUSE_S)
    return done
