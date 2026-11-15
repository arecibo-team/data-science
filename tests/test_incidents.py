"""Tests for incident normalization, Sentinel-2 burn-scar logic and NASA-detection matching (synthetic fixtures)."""
import json

import numpy as np
import pandas as pd
from rasterio.transform import from_origin
from shapely.geometry import mapping, box

from firms import incidents as I
from firms import sentinel2 as S
from firms import config as C
from firms.match import incident_detection_counts, tag_incidents


def _feature(incident_id, poly, start, end, **props):
    return {
        "type": "Feature",
        "properties": {
            "incident_id": incident_id, "source": "Copernicus EMS", "name": incident_id,
            "country": "ES", "start_date": start, "end_date": end, "area_ha": 1.0, **props,
        },
        "geometry": mapping(poly),
    }


def _detections(rows):
    """rows: (lat, lon, 'YYYY-MM-DD HH:MM') -> frame shaped like the harmonized table."""
    return pd.DataFrame({
        "lat": [r[0] for r in rows],
        "lon": [r[1] for r in rows],
        "acq_time_utc": pd.to_datetime([r[2] for r in rows], utc=True),
        "frp_mw": [1.0] * len(rows),
    })


# A 0.1 x 0.1 degree fire in Spain, active 2025-08-10 to 2025-08-12.
FIRE = _feature("EMS-1", box(-5.0, 40.0, -4.9, 40.1), "2025-08-10", "2025-08-12")


# ---------------------------------------------------------------- matching

def test_detection_inside_polygon_and_window_is_tagged():
    h = _detections([(40.05, -4.95, "2025-08-11 13:00")])
    tags = tag_incidents(h, [FIRE], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] == "EMS-1"


def test_detection_outside_date_window_is_not_tagged():
    h = _detections([(40.05, -4.95, "2025-08-17 13:00")])  # 5 days after end, slack 2
    tags = tag_incidents(h, [FIRE], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] is None


def test_slack_lets_detection_just_outside_window_match():
    h = _detections([(40.05, -4.95, "2025-08-14 09:00")])  # 2 days after end
    tags = tag_incidents(h, [FIRE], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] == "EMS-1"


def test_detection_beyond_buffer_is_not_tagged():
    # ~5 km east of the polygon edge (0.05 deg lon at 40N is ~4.3 km): beyond the 1 km buffer.
    h = _detections([(40.05, -4.85, "2025-08-11 13:00")])
    tags = tag_incidents(h, [FIRE], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] is None


def test_detection_just_inside_buffer_is_tagged():
    # 0.005 deg east of the edge is ~0.4-0.55 km, inside the 1 km buffer.
    h = _detections([(40.05, -4.895, "2025-08-11 13:00")])
    tags = tag_incidents(h, [FIRE], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] == "EMS-1"


def test_feature_buffer_overrides_default():
    # Same detection as the "beyond buffer" case, but the feature asks for a 10 km buffer.
    wide = _feature("EMS-wide", box(-5.0, 40.0, -4.9, 40.1), "2025-08-10", "2025-08-12", buffer_km=10.0)
    h = _detections([(40.05, -4.85, "2025-08-11 13:00")])
    tags = tag_incidents(h, [wide], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] == "EMS-wide"


def test_feature_slack_zero_is_strict_window():
    s2 = _feature("S2-1", box(-5.0, 40.0, -4.9, 40.1), "2025-08-10", "2025-08-12",
                  buffer_km=1.0, slack_days=0)
    h = _detections([(40.05, -4.95, "2025-08-14 09:00")])  # 2 days after end, slack 0
    tags = tag_incidents(h, [s2], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] is None


def test_overlapping_incidents_prefer_closest_date_window():
    early = _feature("EMS-early", box(-5.0, 40.0, -4.9, 40.1), "2025-08-01", "2025-08-05")
    late = _feature("EMS-late", box(-5.0, 40.0, -4.9, 40.1), "2025-08-10", "2025-08-12")
    h = _detections([(40.05, -4.95, "2025-08-11 13:00")])
    tags = tag_incidents(h, [early, late], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] == "EMS-late"


def test_no_incidents_leaves_everything_unmatched():
    h = _detections([(40.05, -4.95, "2025-08-11 13:00")])
    tags = tag_incidents(h, [], buffer_km=1.0, slack_days=2)
    assert tags.isna().all()


def test_detection_counts_exclude_unmatched():
    h = _detections([
        (40.05, -4.95, "2025-08-11 13:00"),
        (40.06, -4.94, "2025-08-11 14:00"),
        (10.0, 10.0, "2025-08-11 14:00"),
    ])
    tags = tag_incidents(h, [FIRE], buffer_km=1.0, slack_days=2)
    assert incident_detection_counts(tags) == {"EMS-1": 2}


# ---------------------------------------------------------------- EMS normalization

def test_polygon_area_ha_is_plausible():
    # 0.1 x 0.1 deg at 40N is about 8.5 km x 11.1 km = ~94 km^2 = ~9400 ha.
    ha = I._polygon_area_ha(box(-5.0, 40.0, -4.9, 40.1))
    assert 8500 < ha < 10500


def test_ems_normalization_uses_aoi_extent_and_skips_non_europe(monkeypatch):
    details = {
        "EMSR1": {
            "name": "Wildfire in Testland", "continent": "Europe",
            "eventTime": "2025-08-10T12:00:00", "countries": [{"name": "Spain"}],
            "aois": [{"number": 1, "name": "North", "extent": "POLYGON ((-5 40, -4.9 40, -4.9 40.1, -5 40.1, -5 40))"}],
        },
        "EMSR2": {"name": "Fire in Chile", "continent": "South America", "aois": []},
    }
    monkeypatch.setattr(I, "_ems_detail", lambda code: details.get(code))
    act = [{"code": "EMSR1", "activationTime": "2025-08-11T08:00:00"},
           {"code": "EMSR2", "activationTime": "2025-08-11T08:00:00"}]
    out = I._normalize_ems(act)
    assert len(out) == 1
    p = out[0]["properties"]
    assert p["incident_id"] == "EMS-EMSR1-AOI1"
    assert p["start_date"] == "2025-08-10" and p["end_date"] == "2025-08-11"
    assert p["country"] == "Spain"


def test_effis_is_no_longer_a_source():
    assert not hasattr(I, "_normalize_effis")
    assert not hasattr(C, "EFFIS_WFS")


# ---------------------------------------------------------------- Sentinel-2 burn scars

def _scene(nbr, valid=None):
    nbr = np.asarray(nbr, dtype="float64")
    valid = np.ones(nbr.shape, bool) if valid is None else valid
    return {"nbr": nbr, "valid": valid, "transform": from_origin(4_000_000, 2_000_000, 20, 20),
            "shape": nbr.shape}


def test_dnbr_scar_is_burnt_region_only():
    # Pre: healthy vegetation everywhere (NBR 0.5). Post: a 10x10 block burnt (NBR -0.1).
    pre = _scene(np.full((50, 50), 0.5))
    post_arr = np.full((50, 50), 0.5)
    post_arr[10:20, 10:20] = -0.1
    polys = S._scar_polygons(pre, _scene(post_arr))
    assert polys, "burnt block should produce a polygon"
    total = sum(p.area for p in polys)
    assert abs(total - 100 * 20 * 20) < 1e-6  # 100 px at 20 m


def test_cloud_pixels_are_not_burn_scar():
    pre = _scene(np.full((20, 20), 0.5))
    post_arr = np.full((20, 20), -0.1)  # looks burnt everywhere...
    valid = np.zeros((20, 20), bool)    # ...but everything is cloud in the post scene
    assert S._scar_polygons(pre, _scene(post_arr, valid)) == []


def test_scar_far_from_detections_is_dropped():
    pre = _scene(np.full((50, 50), 0.5))
    post_arr = np.full((50, 50), 0.5)
    post_arr[10:20, 10:20] = -0.1
    polys = S._scar_polygons(pre, _scene(post_arr))
    # Detections 50 km away from the scar (EPSG:3035 metres).
    far = np.array([[4_050_000.0, 2_000_000.0]])
    geom, ha = S._scar_geometry(polys, far)
    assert geom is None and ha == 0.0


def test_scar_near_detection_is_kept_with_area():
    pre = _scene(np.full((50, 50), 0.5))
    post_arr = np.full((50, 50), 0.5)
    post_arr[10:20, 10:20] = -0.1
    polys = S._scar_polygons(pre, _scene(post_arr))
    near = np.array([[4_000_000 + 15 * 20, 2_000_000 - 15 * 20]])  # inside the scar
    geom, ha = S._scar_geometry(polys, near)
    assert geom is not None
    assert abs(ha - 4.0) < 0.01  # 100 px x 20 m x 20 m; the 5 ha verdict threshold is applied later


def test_event_linking_groups_nearby_detections_and_drops_small_ones():
    base_t = "2025-08-10 12:00"
    rows = [(40.0 + 0.001 * i, -5.0, base_t) for i in range(12)]  # one compact fire
    rows += [(42.0 + 0.001 * i, 3.0, "2025-08-10 12:00") for i in range(3)]  # too small
    h = _detections(rows)
    events, labels = S.find_events(h)
    assert len(events) == 1
    assert events.iloc[0]["n_det"] == 12
    assert events.iloc[0]["event_id"].startswith("20250810_40.000")


def test_event_linking_respects_day_gap():
    # Same place, but 5 days apart: two separate events of 10 detections each, not one of 20.
    rows = [(40.0, -5.0 + 0.0005 * i, "2025-08-01 12:00") for i in range(10)]
    rows += [(40.0, -5.0 + 0.0005 * i, "2025-08-06 12:00") for i in range(10)]
    events, _ = S.find_events(_detections(rows))
    assert len(events) == 2
    assert sorted(events["n_det"]) == [10, 10]


def test_confirmed_features_only_for_confirmed_records(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RAW_S2", tmp_path)
    geo = mapping(box(-5.0, 40.0, -4.9, 40.1))
    confirmed = {
        "event_id": "E1", "verdict": "confirmed", "first_day": "2025-08-10", "last_day": "2025-08-12",
        "scar_ha": 950.0, "pre": {"date": "2025-08-02", "item": "a"},
        "post": {"date": "2025-08-20", "item": "b"}, "geometry": geo,
    }
    other = {**confirmed, "event_id": "E2", "verdict": "not_confirmed", "geometry": None}
    (tmp_path / "E1.json").write_text(json.dumps(confirmed))
    (tmp_path / "E2.json").write_text(json.dumps(other))
    feats = S.confirmed_features()
    assert [f["properties"]["incident_id"] for f in feats] == ["S2-E1"]
    p = feats[0]["properties"]
    assert p["source"] == "Sentinel-2"
    assert p["start_date"] == "2025-08-02" and p["end_date"] == "2025-08-20"
    assert p["slack_days"] == 0 and p["buffer_km"] == C.S2_MATCH_BUFFER_KM
