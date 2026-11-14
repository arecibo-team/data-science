"""Fire-weather features for a point and hour: NASA POWER (historical) and DWD ICON via Open-Meteo
(recent days and forecast).

Provider responses are cached in the `weather_cache` table of the server database, keyed by request URL.
NASA requests fetch a whole UTC day, so every hour of that day is answered from one cached response.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from . import db

log = logging.getLogger("server.fire_weather")

NASA_TTL = 86400 * 30   # reanalysis values for past days do not change
DWD_TTL = 900           # forecasts are refreshed several times a day
NASA_MIN_AGE_DAYS = 5   # auto mode uses NASA only for times older than this

HOURLY = {
    "T2M": "temperature", "RH2M": "air_relative_humidity",
    "WS10M": "wind_speed", "WD10M": "wind_direction",
    "T2MDEW": "dew_point", "PRECTOTCORR": "precipitation",
    "PS": "surface_pressure", "ALLSKY_SFC_SW_DWN": "solar_irradiance",
}
DAILY = {"GWETTOP": "surface_soil_wetness",
         "GWETROOT": "root_zone_soil_wetness", "GWETPROF": "profile_soil_wetness"}
DWD = {
    "temperature_2m": "temperature", "relative_humidity_2m": "air_relative_humidity",
    "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_direction",
    "wind_gusts_10m": "wind_gusts", "dew_point_2m": "dew_point",
    "precipitation": "precipitation", "surface_pressure": "surface_pressure",
    "shortwave_radiation": "solar_irradiance", "cloud_cover": "cloud_cover",
    "soil_moisture_0_to_1cm": "soil_moisture_0_to_1cm",
    "soil_moisture_1_to_3cm": "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm": "soil_moisture_3_to_9cm",
    "soil_moisture_9_to_27cm": "soil_moisture_9_to_27cm",
    "soil_moisture_27_to_81cm": "soil_moisture_27_to_81cm",
}
WIND_CONVENTION = "Degrees clockwise from north, direction wind comes FROM"


class WeatherDataError(RuntimeError):
    """The public source could not supply the requested data."""


# ---------------------------------------------------------------- cache

def _cache_get(key: str) -> tuple[dict, float] | None:
    con = db.connect(readonly=True)
    try:
        row = con.execute("SELECT fetched_at, body FROM weather_cache WHERE key = ? AND expires_at > ?",
                          (key, time.time())).fetchone()
    finally:
        con.close()
    return (json.loads(row["body"]), row["fetched_at"]) if row else None


def _cache_put(key: str, provider: str, url: str, body: str, fetched: float, ttl: float) -> None:
    """Best effort: a pipeline step holding the write lock must not fail or stall the request."""
    try:
        con = db.connect()
        try:
            con.execute("PRAGMA busy_timeout = 5000")
            con.execute("DELETE FROM weather_cache WHERE expires_at <= ?", (fetched,))
            con.execute("INSERT OR REPLACE INTO weather_cache (key, provider, url, fetched_at, expires_at, body) "
                        "VALUES (?, ?, ?, ?, ?, ?)", (key, provider, url, fetched, fetched + ttl, body))
            con.commit()
        finally:
            con.close()
    except sqlite3.OperationalError as exc:
        log.warning("weather cache write skipped: %s", exc)


def _has_values(data: dict) -> bool:
    fill = data.get("header", {}).get("fill_value", -999)
    series = (data.get("properties", {}).get("parameter") or data.get("hourly") or {}).values()
    return any(isinstance(v, (int, float)) and math.isfinite(v) and v != fill
               for s in series for v in (s.values() if isinstance(s, dict) else s))


def _fetch(url: str, provider: str, ttl: float, timeout: float) -> tuple[dict, bool, float]:
    key = hashlib.sha256(url.encode()).hexdigest()
    if ttl > 0 and (hit := _cache_get(key)):
        return hit[0], True, hit[1]
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "FireWeatherPython/1.0", "Accept": "application/json"})
            with urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data.get("properties", {}).get("parameter"), dict) and not isinstance(data.get("hourly"), dict):
                raise WeatherDataError("Unexpected provider response: " + body[:500])
            break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise WeatherDataError(f"Provider HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            if attempt == 2:
                raise WeatherDataError(f"Provider connection failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise WeatherDataError("Provider returned invalid JSON") from exc
        time.sleep(2 ** attempt)
    fetched = time.time()
    if ttl > 0 and _has_values(data):  # never keep responses made only of missing values
        _cache_put(key, provider, url, body, fetched, ttl)
    return data, False, fetched


# ---------------------------------------------------------------- features

def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _vpd(features: dict, hour: datetime, derived_from: list[str]) -> dict:
    t, rh = features["temperature"]["value"], features["air_relative_humidity"]["value"]
    value = max(0.0, 0.6108 * math.exp(17.27 * t / (t + 237.3)) * (1 - rh / 100)) if t is not None and rh is not None else None
    return {"value": value, "unit": "kPa", "resolution": "hourly", "time": hour.isoformat(), "derived_from": derived_from}


def _status(features: dict) -> str:
    return "partial" if any(f["value"] is None for f in features.values()) else "complete"


def parse_time(value: str | datetime) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("time must be ISO 8601, e.g. 2024-07-01T12:00:00Z") from exc
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("time must include a timezone, e.g. 2024-07-01T12:00:00Z")
    return value.astimezone(timezone.utc)


def get_fire_weather(longitude: float, latitude: float, query_time: str | datetime, *,
                     source: str = "auto", cache_ttl_seconds: float | None = None, timeout: float = 45) -> dict:
    """Gridded fire-weather features for the hour containing query_time (minutes are floored).

    source: auto picks DWD for recent/future times and NASA for times over five days old; nasa or dwd forces one.
    Missing values are None. Raises ValueError for bad input, WeatherDataError when the provider fails.
    """
    longitude, latitude = float(longitude), float(latitude)
    if not (math.isfinite(longitude) and -180 <= longitude <= 180 and math.isfinite(latitude) and -90 <= latitude <= 90):
        raise ValueError("Longitude must be -180..180 and latitude -90..90")
    if source not in ("auto", "nasa", "dwd"):
        raise ValueError("source must be auto, nasa, or dwd")
    if cache_ttl_seconds is not None and (not math.isfinite(cache_ttl_seconds) or cache_ttl_seconds < 0):
        raise ValueError("Cache TTL must be nonnegative")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    requested = parse_time(query_time)
    now = datetime.now(timezone.utc)
    if source == "dwd" or (source == "auto" and requested >= now - timedelta(days=NASA_MIN_AGE_DAYS)):
        ttl = DWD_TTL if cache_ttl_seconds is None else min(cache_ttl_seconds, DWD_TTL)
        return _dwd(longitude, latitude, requested, now, ttl, timeout)
    if requested > now:
        raise ValueError("NASA POWER is historical data, not a forecast service")
    return _nasa(longitude, latitude, requested, NASA_TTL if cache_ttl_seconds is None else cache_ttl_seconds, timeout)


def _nasa(longitude, latitude, requested, ttl, timeout) -> dict:
    hour = requested.replace(minute=0, second=0, microsecond=0)
    day = hour.strftime("%Y%m%d")
    features, sources = {}, []
    warnings = ["Gridded model/reanalysis estimates; local conditions can differ.",
                "Soil wetness is a daily dimensionless 0-1 index, not volumetric water content.",
                "Daily soil values include the whole day; avoid future information leakage in intraday training."]
    for resolution, mapping, stamp in (("hourly", HOURLY, hour.strftime("%Y%m%d%H")), ("daily", DAILY, day)):
        url = f"https://power.larc.nasa.gov/api/temporal/{resolution}/point?" + urlencode({
            "parameters": ",".join(mapping), "community": "AG", "longitude": longitude,
            "latitude": latitude, "start": day, "end": day, "format": "JSON", "time-standard": "UTC"})
        data, hit, fetched = _fetch(url, "nasa_power", ttl, timeout)
        sources.append({"provider": "NASA POWER", "resolution": resolution, "url": url, "cache_hit": hit,
                        "fetched_at_utc": _iso(fetched), "header": data.get("header", {}), "geometry": data.get("geometry")})
        warnings.extend(data.get("messages", []))
        fill = data.get("header", {}).get("fill_value", -999)
        for code, name in mapping.items():
            value = data["properties"]["parameter"].get(code, {}).get(stamp)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value == fill:
                value = None
                warnings.append(f"{name} unavailable for requested {resolution} period")
            features[name] = {"value": value, "unit": data.get("parameters", {}).get(code, {}).get("units"),
                              "parameter": code, "resolution": resolution,
                              "time": hour.isoformat() if resolution == "hourly" else hour.date().isoformat()}
    features["vapor_pressure_deficit"] = _vpd(features, hour, ["T2M", "RH2M"])
    return {"longitude": longitude, "latitude": latitude, "requested_time_utc": requested.isoformat(),
            "weather_time_utc": hour.isoformat(), "soil_date_utc": hour.date().isoformat(),
            "data_kind": "historical_reanalysis", "status": _status(features), "features": features,
            "warnings": warnings, "wind_direction_convention": WIND_CONVENTION, "sources": sources}


def _dwd(longitude, latitude, requested, now, ttl, timeout) -> dict:
    hour = requested.replace(minute=0, second=0, microsecond=0)
    if not now.date() - timedelta(days=5) <= hour.date() <= now.date() + timedelta(days=6):
        raise ValueError("DWD queries support the previous 5 days through 6 days ahead; individual fields may have shorter coverage")
    url = "https://api.open-meteo.com/v1/dwd-icon?" + urlencode({
        "longitude": longitude, "latitude": latitude, "hourly": ",".join(DWD),
        "start_date": hour.date().isoformat(), "end_date": hour.date().isoformat(),
        "timezone": "UTC", "wind_speed_unit": "ms", "temperature_unit": "celsius", "precipitation_unit": "mm"})
    data, hit, fetched = _fetch(url, "dwd_icon", ttl, timeout)
    try:
        index = data["hourly"]["time"].index(hour.strftime("%Y-%m-%dT%H:00"))
    except (KeyError, ValueError) as exc:
        raise WeatherDataError("DWD response does not cover requested hour") from exc
    features = {}
    warnings = ["DWD model estimates delivered by Open-Meteo, not on-site observations.",
                "Retrieval time is not the model initialization time; this endpoint does not expose a run timestamp.",
                "Recent past values are archived forecasts, not verified observations.",
                "Soil moisture is volumetric water content; it is not interchangeable with NASA soil wetness."]
    for code, name in DWD.items():
        series = data["hourly"].get(code, [])
        value = series[index] if index < len(series) else None
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            value = None
            warnings.append(f"{name} unavailable for requested hour")
        features[name] = {"value": value, "unit": data.get("hourly_units", {}).get(code),
                          "parameter": code, "resolution": "hourly", "time": hour.isoformat()}
    features["vapor_pressure_deficit"] = _vpd(features, hour, ["temperature_2m", "relative_humidity_2m"])
    current = now.replace(minute=0, second=0, microsecond=0)
    return {"longitude": longitude, "latitude": latitude, "requested_time_utc": requested.isoformat(),
            "weather_time_utc": hour.isoformat(), "data_kind": "forecast" if hour >= current else "archived_forecast",
            "status": _status(features), "features": features, "warnings": warnings,
            "wind_direction_convention": WIND_CONVENTION,
            "sources": [{"provider": "DWD ICON via Open-Meteo", "url": url, "cache_hit": hit,
                         "fetched_at_utc": _iso(fetched), "model_run_time_utc": None,
                         "grid_latitude": data.get("latitude"), "grid_longitude": data.get("longitude"),
                         "elevation_m": data.get("elevation")}]}
