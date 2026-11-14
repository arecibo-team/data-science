"""/api/weather and its SQLite response cache (provider calls are faked, no network)."""
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from server import db, fire_weather, settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "RUN_ON_START", False)
    monkeypatch.setattr(settings, "TILE_BG", False)
    from server.app import app
    with TestClient(app) as c:
        yield c


class _Resp:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


def _fake(monkeypatch, calls):
    def urlopen(req, timeout):
        url = req.full_url
        calls.append(url)
        if "open-meteo" in url:
            day = url.split("start_date=")[1][:10]
            hours = [f"{day}T{h:02d}:00" for h in range(24)]
            hourly = {"time": hours, **{code: [20.0] * 24 for code in fire_weather.DWD}}
            hourly["relative_humidity_2m"] = [40.0] * 24
            return _Resp({"latitude": 45.0, "longitude": 10.0, "elevation": 100, "hourly": hourly, "hourly_units": {}})
        day = url.split("start=")[1][:8]
        if "/hourly/" in url:
            param = {code: {f"{day}{h:02d}": 25.0 for h in range(24)} for code in fire_weather.HOURLY}
            param["RH2M"] = {f"{day}{h:02d}": 30.0 for h in range(24)}
        else:
            param = {code: {day: 0.4} for code in fire_weather.DAILY}
            param["GWETPROF"] = {day: -999.0}
        return _Resp({"header": {"fill_value": -999.0}, "properties": {"parameter": param}, "parameters": {}})
    monkeypatch.setattr(fire_weather, "urlopen", urlopen)


def test_nasa_is_cached_per_day(client, monkeypatch):
    calls = []
    _fake(monkeypatch, calls)
    r = client.get("/api/weather", params={"lon": 10, "lat": 45, "time": "2024-07-01T12:34:00Z"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data_kind"] == "historical_reanalysis" and body["weather_time_utc"] == "2024-07-01T12:00:00+00:00"
    assert body["features"]["temperature"]["value"] == 25.0
    assert body["features"]["profile_soil_wetness"]["value"] is None and body["status"] == "partial"
    assert body["features"]["vapor_pressure_deficit"]["value"] == pytest.approx(2.2174, abs=1e-3)
    assert len(calls) == 2 and not any(s["cache_hit"] for s in body["sources"])

    r = client.get("/api/weather", params={"lon": 10, "lat": 45, "time": "2024-07-01T18:00:00+02:00"})
    assert all(s["cache_hit"] for s in r.json()["sources"]) and len(calls) == 2
    con = db.connect(readonly=True)
    try:
        assert {row[0] for row in con.execute("SELECT provider FROM weather_cache")} == {"nasa_power"}
    finally:
        con.close()


def test_recent_times_use_dwd(client, monkeypatch):
    calls = []
    _fake(monkeypatch, calls)
    t = (datetime.now(timezone.utc) + timedelta(days=1)).replace(hour=6).isoformat()
    body = client.get("/api/weather", params={"lon": 10, "lat": 45, "time": t}).json()
    assert body["data_kind"] == "forecast" and body["status"] == "complete"
    assert body["features"]["air_relative_humidity"]["value"] == 40.0
    assert client.get("/api/weather", params={"lon": 10, "lat": 45, "time": t}).json()["sources"][0]["cache_hit"]
    assert len(calls) == 1


def test_expired_rows_are_not_served(client, monkeypatch):
    calls = []
    _fake(monkeypatch, calls)
    params = {"lon": 10, "lat": 45, "time": "2024-07-01T12:00:00Z"}
    client.get("/api/weather", params=params)
    with db.session() as con:
        con.execute("UPDATE weather_cache SET expires_at = 0")
    assert not any(s["cache_hit"] for s in client.get("/api/weather", params=params).json()["sources"])
    assert len(calls) == 4


@pytest.mark.parametrize("params", [
    {"lon": 200, "lat": 45, "time": "2024-07-01T12:00:00Z"},
    {"lon": 10, "lat": 45, "time": "2024-07-01T12:00:00"},
    {"lon": 10, "lat": 45, "time": "yesterday"},
    {"lon": 10, "lat": 45, "time": "2099-01-01T00:00:00Z"},
])
def test_bad_input_is_422(client, params):
    assert client.get("/api/weather", params=params).status_code == 422


def test_provider_failure_is_502(client, monkeypatch):
    def urlopen(req, timeout):
        return _Resp({"error": True})
    monkeypatch.setattr(fire_weather, "urlopen", urlopen)
    r = client.get("/api/weather", params={"lon": 10, "lat": 45, "time": "2024-07-01T12:00:00Z"})
    assert r.status_code == 502
