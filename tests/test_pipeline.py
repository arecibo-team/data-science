"""Tests for planning, normalization, dedupe, cross-sensor matching and export.

Fixtures reproduce the real FIRMS CSV column layouts seen in the API.
"""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from firms import config as C
from firms import fetch, normalize, harmonize
from firms.export import export_web
from firms.harmonize import MatchParams, match_cross_sensor

MODIS_HEADER = "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight"
VIIRS_HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight"
VIIRS_SP_HEADER = VIIRS_HEADER + ",type"


def modis_csv(rows):
    return MODIS_HEADER + "\n" + "\n".join(rows) + "\n"


def viirs_csv(rows, header=VIIRS_HEADER):
    return header + "\n" + "\n".join(rows) + "\n"


# ---------- planning ----------

def test_plan_uses_sp_then_nrt_in_5_day_chunks():
    avail = {
        "MODIS_SP": (dt.date(2000, 1, 1), dt.date(2026, 8, 31)),
        "MODIS_NRT": (dt.date(2024, 1, 1), dt.date(2026, 11, 14)),
    }
    plan = fetch.plan_requests(dt.date(2026, 8, 25), dt.date(2026, 9, 5), ["MODIS"], avail)
    assert [(r.source, r.start, r.days) for r in plan] == [
        ("MODIS_SP", dt.date(2026, 8, 25), 5),
        ("MODIS_SP", dt.date(2026, 8, 30), 2),
        ("MODIS_NRT", dt.date(2026, 9, 1), 5),
    ]


def test_plan_skips_days_with_no_coverage_and_respects_max_range():
    avail = {"VIIRS_NOAA21_NRT": (dt.date(2024, 1, 17), dt.date(2026, 11, 14))}
    plan = fetch.plan_requests(dt.date(2024, 1, 1), dt.date(2024, 1, 25), ["VIIRS_NOAA21"], avail)
    assert [(r.source, r.start, r.days) for r in plan] == [
        ("VIIRS_NOAA21_NRT", dt.date(2024, 1, 17), 5),
        ("VIIRS_NOAA21_NRT", dt.date(2024, 1, 22), 4),
    ]
    long = fetch.plan_requests(dt.date(2025, 1, 1), dt.date(2025, 1, 25), ["VIIRS_NOAA21"], avail)
    assert [r.days for r in long] == [5, 5, 5, 5, 5]


def test_parse_bbox_validation():
    assert fetch.parse_bbox("world") == "world"
    assert fetch.parse_bbox("1,2,3,4") == "1.0,2.0,3.0,4.0"
    with pytest.raises(Exception):
        fetch.parse_bbox("3,2,1,4")  # W >= E


# ---------- normalization ----------

def test_modis_and_viirs_share_one_schema(tmp_path):
    m = normalize.normalize_frame(
        pd.read_csv(_write(tmp_path / "m.csv", modis_csv([
            "-10.1,20.2,330.5,1.2,1.1,2025-08-01,938,Terra,MODIS,85,6.1NRT,300.0,12.5,D",
        ])), dtype=normalize._READ_DTYPES),
        "MODIS_NRT",
    )
    v = normalize.normalize_frame(
        pd.read_csv(_write(tmp_path / "v.csv", viirs_csv([
            "-10.1,20.2,340.0,0.5,0.4,2025-08-01,151,N21,VIIRS,n,2.0NRT,290.0,2.0,N",
        ])), dtype=normalize._READ_DTYPES),
        "VIIRS_NOAA21_NRT",
    )
    assert list(m.columns) == C.SCHEMA == list(v.columns)
    assert m.iloc[0]["sensor"] == "MODIS" and m.iloc[0]["platform"] == "Terra"
    assert v.iloc[0]["sensor"] == "VIIRS" and v.iloc[0]["platform"] == "NOAA-21"
    assert m.iloc[0]["acq_time_utc"] == pd.Timestamp("2025-08-01 09:38", tz="UTC")
    assert v.iloc[0]["acq_time_utc"] == pd.Timestamp("2025-08-01 01:51", tz="UTC")
    assert m.iloc[0]["confidence_class"] == "high"
    assert v.iloc[0]["confidence_class"] == "nominal"
    assert m.iloc[0]["bt_mid_k"] == 330.5 and v.iloc[0]["bt_mid_k"] == 340.0
    assert m.iloc[0]["pixel_area_km2"] == pytest.approx(1.2 * 1.1)


@pytest.mark.parametrize("raw,expected", [
    ("0", "low"), ("29", "low"), ("30", "nominal"), ("79", "nominal"), ("80", "high"), ("100", "high"),
    ("l", "low"), ("n", "nominal"), ("h", "high"), ("x", None),
])
def test_confidence_classes(raw, expected):
    assert normalize.confidence_class(raw) == expected


def test_nrt_sp_duplicates_resolve_to_sp(tmp_path, monkeypatch):
    row = "-10.1,20.2,330.5,1.2,1.1,2025-08-01,938,Terra,MODIS,85,6.1NRT,300.0,12.5,D"
    monkeypatch.setattr(C, "NORM_DIR", tmp_path / "norm")
    for src, ver in [("MODIS_NRT", "6.1NRT"), ("MODIS_SP", "6.1SP")]:
        out = tmp_path / "norm" / src
        out.mkdir(parents=True)
        df = normalize.normalize_frame(
            pd.read_csv(_write(tmp_path / f"{src}.csv", modis_csv([row.replace("6.1NRT", ver)])),
                        dtype=normalize._READ_DTYPES), src)
        df.to_parquet(out / "w.parquet", index=False)
    loaded = normalize.load_normalized()
    assert len(loaded) == 1
    assert loaded.iloc[0]["product"] == "SP"


def test_viirs_sp_keeps_firms_type(tmp_path):
    line = "-10.1,20.2,340.0,0.5,0.4,2025-08-01,151,N,VIIRS,h,2.0,290.0,2.0,N,0"
    df = normalize.normalize_frame(
        pd.read_csv(_write(tmp_path / "s.csv", viirs_csv([line], VIIRS_SP_HEADER)),
                    dtype=normalize._READ_DTYPES), "VIIRS_SNPP_SP")
    assert df.iloc[0]["firms_type"] == "0"
    assert df.iloc[0]["platform"] == "SNPP" and df.iloc[0]["product"] == "SP"


# ---------- cross-sensor matching ----------

def _det(sensor, lat, lon, when, frp=5.0):
    return {"sensor": sensor, "platform": "x", "product": "SP", "lat": lat, "lon": lon,
            "acq_time_utc": pd.Timestamp(when, tz="UTC"), "frp_mw": frp, "daynight": "D",
            "confidence_class": "nominal"}


def test_matching_links_only_close_cross_sensor_pairs():
    df = pd.DataFrame([
        _det("MODIS", 10.0, 20.0, "2025-08-01 10:00"),
        _det("VIIRS", 10.005, 20.003, "2025-08-01 10:20"),   # ~0.6 km, 20 min: match
        _det("VIIRS", 10.5, 20.0, "2025-08-01 10:05"),       # ~55 km: no match
        _det("MODIS", 10.0, 20.001, "2025-08-01 16:00"),     # 6 h later: no match to first MODIS
        _det("VIIRS", 10.0, 20.001, "2025-08-01 10:30"),     # within window of MODIS #1 (joins that cluster)
    ])
    h = match_cross_sensor(df, MatchParams(radius_km=1.0, window_min=180))

    def row(sensor, lat, when):
        r = h[(h["sensor"] == sensor) & (h["lat"] == lat) & (h["acq_time_utc"] == pd.Timestamp(when, tz="UTC"))]
        assert len(r) == 1
        return r.iloc[0]

    modis_1 = row("MODIS", 10.0, "2025-08-01 10:00")
    viirs_close = row("VIIRS", 10.005, "2025-08-01 10:20")
    assert modis_1["harmonized_status"] == "both"
    assert viirs_close["harmonized_status"] == "both"
    assert viirs_close["cluster_id"] == modis_1["cluster_id"]
    assert row("VIIRS", 10.5, "2025-08-01 10:05")["harmonized_status"] == "viirs_only"
    assert row("MODIS", 10.0, "2025-08-01 16:00")["harmonized_status"] == "modis_only"


def test_matching_works_across_midnight():
    df = pd.DataFrame([
        _det("MODIS", 10.0, 20.0, "2025-08-01 23:50"),
        _det("VIIRS", 10.0, 20.0, "2025-08-02 00:10"),
    ])
    h = match_cross_sensor(df, MatchParams(radius_km=1.0, window_min=60))
    assert (h["harmonized_status"] == "both").all()


def test_matching_empty_frame():
    h = match_cross_sensor(pd.DataFrame(columns=C.SCHEMA).astype({
        "acq_time_utc": "datetime64[ns, UTC]", "lat": float, "lon": float, "sensor": "string",
        "frp_mw": float}))
    assert h.empty


def test_match_params_validation():
    with pytest.raises(ValueError):
        MatchParams(radius_km=0)
    with pytest.raises(ValueError):
        MatchParams(window_min=0)


# ---------- export ----------

def test_export_writes_index_and_day_files(tmp_path):
    df = pd.DataFrame([
        _det("MODIS", 10.0, 20.0, "2025-08-01 10:00"),
        _det("VIIRS", 10.0, 20.0, "2025-08-01 10:10"),
        _det("VIIRS", 11.0, 21.0, "2025-08-02 01:00"),
    ])
    df["platform"] = ["Terra", "SNPP", "NOAA-21"]
    df["bt_mid_k"] = [330.0, 340.0, np.nan]
    df["confidence_class"] = ["high", "nominal", "low"]
    df["daynight"] = "D"
    df["cluster_id"] = pd.array([0, 0, None], dtype="Int64")
    df["harmonized_status"] = ["both", "both", "viirs_only"]
    df["uid"] = ["a", "b", "c"]
    df["firms_version"] = "x"
    df["source"] = "s"
    index = export_web(df, out_dir=tmp_path)
    assert index["days"]["2025-08-01"] == {"total": 2, "MODIS": 1, "VIIRS": 1, "matched": 2}
    day2 = json.loads((tmp_path / "days" / "2025-08-02.json").read_text())
    assert "bt_mid_k" in day2["fields"] and len(day2["rows"][0]) == len(day2["fields"])
    assert day2["rows"][0][-1] is None  # unmatched cluster_id -> null
    assert day2["rows"][0][-2] == "viirs_only"


def _write(path, text):
    path.write_text(text)
    return path


# ---------- API key load balancing ----------

class _Resp:
    def __init__(self, status, text):
        self.status_code = status
        self.text = text


class _FakeSession:
    """Returns scripted responses per key, so the test sees which key was used."""

    def __init__(self, script):
        self.script = script  # key -> list of (status, text) responses
        self.calls = []

    def get(self, url, timeout=None):
        key = url.split("/api/")[1].split("/")[2]  # .../api/<path>/<key>/...
        self.calls.append(key)
        status, text = self.script[key].pop(0)
        return _Resp(status, text)


CSV_OK = "latitude,longitude\n1,2\n"


def test_round_robin_spreads_requests_across_keys(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    session = _FakeSession({
        "k1": [(200, CSV_OK)], "k2": [(200, CSV_OK)], "k3": [(200, CSV_OK)],
    })
    client = fetch.FirmsClient(["k1", "k2", "k3"], session=session)
    for _ in range(3):
        client.area_csv("MODIS_NRT", "world", dt.date(2024, 1, 1), 5)
    assert session.calls == ["k1", "k2", "k3"]


def test_limit_error_cools_key_down_and_moves_to_next(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    session = _FakeSession({
        "k1": [(400, "Exceeding allowed transaction limit.")],
        "k2": [(200, CSV_OK)],
    })
    client = fetch.FirmsClient(["k1", "k2"], session=session)
    text = client.area_csv("MODIS_NRT", "world", dt.date(2024, 1, 1), 5)
    assert text == CSV_OK
    assert session.calls == ["k1", "k2"]
    assert "k1" in client.pool.cooldown_until and "k2" not in client.pool.cooldown_until


def test_invalid_key_is_removed_from_rotation(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    session = _FakeSession({
        "bad": [(400, "Invalid MAP_KEY.")],
        "good": [(200, CSV_OK), (200, CSV_OK)],
    })
    client = fetch.FirmsClient(["bad", "good"], session=session)
    client.area_csv("MODIS_NRT", "world", dt.date(2024, 1, 1), 5)
    client.area_csv("MODIS_NRT", "world", dt.date(2024, 1, 6), 5)
    assert "bad" in client.pool.dead
    assert session.calls == ["bad", "good", "good"]


def test_bad_day_range_fails_fast_without_rotating(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    session = _FakeSession({"k1": [(400, "Invalid day range. Expects [1..5].")]})
    client = fetch.FirmsClient(["k1"], session=session)
    with pytest.raises(fetch.FetchError, match="day range"):
        client.area_csv("MODIS_NRT", "world", dt.date(2024, 1, 1), 10)
    assert session.calls == ["k1"]
