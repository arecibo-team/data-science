"""Tests for incident normalization and NASA-detection matching (synthetic fixtures)."""
import datetime as dt

import pandas as pd
from shapely.geometry import mapping, box

from firms import incidents as I
from firms.match import incident_detection_counts, tag_incidents


def _feature(incident_id, poly, start, end):
    return {
        "type": "Feature",
        "properties": {
            "incident_id": incident_id, "source": "EFFIS", "name": incident_id,
            "country": "ES", "start_date": start, "end_date": end, "area_ha": 1.0,
        },
        "geometry": mapping(poly),
    }


def _detections(rows):
    """rows: (lat, lon, 'YYYY-MM-DD HH:MM') -> frame shaped like the harmonized table."""
    return pd.DataFrame({
        "lat": [r[0] for r in rows],
        "lon": [r[1] for r in rows],
        "acq_time_utc": pd.to_datetime([r[2] for r in rows], utc=True),
    })


# A 0.1 x 0.1 degree fire in Spain, active 2025-08-10 to 2025-08-12.
FIRE = _feature("EFFIS-1", box(-5.0, 40.0, -4.9, 40.1), "2025-08-10", "2025-08-12")


def test_detection_inside_polygon_and_window_is_tagged():
    h = _detections([(40.05, -4.95, "2025-08-11 13:00")])
    tags = tag_incidents(h, [FIRE], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] == "EFFIS-1"


def test_detection_outside_date_window_is_not_tagged():
    # 5 days after the fire ends, slack is 2 days.
    h = _detections([(40.05, -4.95, "2025-08-17 13:00")])
    tags = tag_incidents(h, [FIRE], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] is None


def test_slack_lets_detection_just_outside_window_match():
    h = _detections([(40.05, -4.95, "2025-08-14 09:00")])  # 2 days after end
    tags = tag_incidents(h, [FIRE], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] == "EFFIS-1"


def test_detection_beyond_buffer_is_not_tagged():
    # ~5 km east of the polygon edge (0.05 deg lon at 40N is ~4.3 km): beyond the 1 km buffer.
    h = _detections([(40.05, -4.85, "2025-08-11 13:00")])
    tags = tag_incidents(h, [FIRE], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] is None


def test_detection_just_inside_buffer_is_tagged():
    # 0.005 deg east of the edge is ~0.4-0.55 km, inside the 1 km buffer.
    h = _detections([(40.05, -4.895, "2025-08-11 13:00")])
    tags = tag_incidents(h, [FIRE], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] == "EFFIS-1"


def test_overlapping_incidents_prefer_closest_date_window():
    early = _feature("EFFIS-early", box(-5.0, 40.0, -4.9, 40.1), "2025-08-01", "2025-08-05")
    late = _feature("EFFIS-late", box(-5.0, 40.0, -4.9, 40.1), "2025-08-10", "2025-08-12")
    h = _detections([(40.05, -4.95, "2025-08-11 13:00")])
    tags = tag_incidents(h, [early, late], buffer_km=1.0, slack_days=2)
    assert tags.iloc[0] == "EFFIS-late"


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
    assert incident_detection_counts(tags) == {"EFFIS-1": 2}


def test_effis_normalization_filters_non_european_and_dedupes():
    feats = [
        {"type": "Feature", "geometry": mapping(box(0, 0, 1, 1)),
         "properties": {"id": "1", "COUNTRY": "PT", "FIREDATE": "2025-07-01 10:00:00",
                        "FINALDATE": "2025-07-03 10:00:00", "AREA_HA": "40",
                        "COMMUNE": "Sare", "PROVINCE": "Porto"}},
        # Same fire repeated in the season layer: must be deduplicated by id.
        {"type": "Feature", "geometry": mapping(box(0, 0, 1, 1)),
         "properties": {"id": "1", "COUNTRY": "PT", "FIREDATE": "2025-07-01 10:00:00"}},
        # North African fire: excluded.
        {"type": "Feature", "geometry": mapping(box(0, 0, 1, 1)),
         "properties": {"id": "2", "COUNTRY": "DZ", "FIREDATE": "2025-07-01 10:00:00"}},
    ]
    out = I._normalize_effis(feats)
    assert [f["properties"]["incident_id"] for f in out] == ["EFFIS-1"]
    p = out[0]["properties"]
    assert p["country"] == "PT"
    assert p["start_date"] == "2025-07-01" and p["end_date"] == "2025-07-03"
    assert p["area_ha"] == 40.0


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
