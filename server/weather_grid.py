"""Daily weather on a 1 degree grid from NASA POWER, stored in `weather_grid` (the model's weather features).

One request per grid cell returns the whole date range. Cells already complete are skipped, so the step
resumes after a stop. Days newer than the NASA lag are filled forward when features are built.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

from . import db, fire_weather, settings

log = logging.getLogger("server.weather_grid")

PARAMS = {"T2M": "t2m", "RH2M": "rh", "WS10M": "ws", "PRECTOTCORR": "prec",
          "GWETTOP": "gwettop", "GWETROOT": "gwetroot", "GWETPROF": "gwetprof", "T2M_MAX": "t2mx", "WS10M_MAX": "wsx"}
COLS = list(PARAMS.values())
INSERT = f"INSERT OR REPLACE INTO weather_grid (r1, c1, date, {', '.join(COLS)}) VALUES ({','.join('?' * (3 + len(COLS)))})"
LAG_DAYS = 5


def end_date() -> str:
    return (dt.date.today() - dt.timedelta(days=LAG_DAYS)).isoformat()


def cells_needed(con) -> list[tuple[int, int]]:
    """1 degree cells the model reads weather for: every cell that touches a European country, plus any cell
    holding a detection (a coast or a border can put a fire just outside the outlines)."""
    from .ml import regions
    import shapely
    from shapely.geometry import box
    cells = {(r, c) for r, c in con.execute("SELECT DISTINCT r / 10, c / 10 FROM bin_day")}
    _, geoms = regions.countries()
    for r1 in range(34 + 90, 72 + 90):
        for c1 in range(-25 + 180, 45 + 180):
            b = box(c1 - 180, r1 - 90, c1 - 179, r1 - 89)
            if any(g.intersects(b) for g in geoms):
                cells.add((r1, c1))
    return sorted(cells)


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


def run(start: str, should_stop=lambda: False, workers: int | None = None, reverse: bool = False) -> int:
    """Fetch every needed cell not yet complete, `workers` requests at a time. Returns cells complete."""
    end = end_date()
    workers = workers or settings.WEATHER_WORKERS
    with db.session() as con:
        cells = cells_needed(con)
        have = {(r, c) for r, c in con.execute(
            "SELECT r1, c1 FROM weather_grid GROUP BY r1, c1 HAVING MAX(date) >= ? AND MIN(date) <= ? AND COUNT(wsx) > 0",
            (end, start))}
    todo = [c for c in cells if c not in have]
    done = len(cells) - len(todo)
    if reverse:
        todo.reverse()

    def one(cell):
        if should_stop():
            return cell, None
        for attempt in range(6):
            try:
                return cell, fetch_cell(*cell, start, end)
            except fire_weather.WeatherDataError as exc:
                if "429" in str(exc) and attempt < 5 and not should_stop():   # too many requests: back off, then retry
                    time.sleep(20 * (attempt + 1))
                    continue
                log.warning("weather cell %s skipped: %s", cell, str(exc)[:120])
                return cell, None
            except (KeyError, StopIteration) as exc:
                log.warning("weather cell %s skipped: %r", cell, exc)
                return cell, None
        return cell, None

    with ThreadPoolExecutor(workers) as pool:
        for cell, rows in pool.map(one, todo):
            if rows:
                with db.session() as con:
                    con.executemany(INSERT, rows)
                done += 1
    return done


FORECAST_DAYS = 7
FORECAST_URL = "https://api.open-meteo.com/v1/dwd-icon"
FORECAST_VARS = {"temperature_2m_mean": "t2m", "temperature_2m_max": "t2mx", "wind_speed_10m_mean": "ws",
                 "wind_speed_10m_max": "wsx", "precipitation_sum": "prec", "relative_humidity_2m_mean": "rh"}


def fetch_forecast(batch: int = 50, timeout: float = 60) -> int:
    """Next FORECAST_DAYS days of weather (DWD ICON via Open-Meteo, up to `batch` cells per request) for every cell
    of weather_grid's list, stored in weather_forecast. Returns rows written. Cells are the centres of the 1 degree
    boxes, so units and place match the NASA POWER rows the model trains on."""
    import requests
    with db.session() as con:
        cells = cells_needed(con)
    issued = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    total = 0
    for i in range(0, len(cells), batch):
        part = cells[i:i + batch]
        params = {"latitude": ",".join(f"{r - 90 + 0.5:.2f}" for r, _ in part),
                  "longitude": ",".join(f"{c - 180 + 0.5:.2f}" for _, c in part),
                  "daily": ",".join(FORECAST_VARS), "forecast_days": FORECAST_DAYS, "timezone": "UTC",
                  "wind_speed_unit": "ms"}
        for attempt in range(8):
            resp = requests.get(FORECAST_URL, params=params, timeout=timeout)
            if resp.status_code == 200:
                break
            if resp.status_code not in (429, 500, 502, 503, 504) or attempt == 7:
                resp.raise_for_status()
            time.sleep(65 if resp.status_code == 429 else 5 * (attempt + 1))   # 429: the per-minute quota resets
        out = resp.json()
        out = out if isinstance(out, list) else [out]
        rows = []
        for (r1, c1), loc in zip(part, out):
            d = loc["daily"]
            for k, day in enumerate(d["time"]):
                rows.append((r1, c1, day, issued, *(d[v][k] for v in FORECAST_VARS)))
        with db.session() as con:
            con.executemany("INSERT OR REPLACE INTO weather_forecast (r1, c1, date, issued, t2m, t2mx, ws, wsx, prec, rh) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        total += len(rows)
    return total
