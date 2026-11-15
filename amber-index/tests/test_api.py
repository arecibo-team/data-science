"""API tests. Require artifacts/ (run the notebook first)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["latest_year"] >= 2024
    assert {m["model"] for m in body["cv_metrics"]} >= {"ElasticNet", "LightGBM"}


def test_countries(client):
    r = client.get("/countries")
    assert r.status_code == 200
    codes = {c["iso2"] for c in r.json()}
    assert len(codes) == 27 and "GR" in codes and "EL" not in codes


def test_scores_latest_sorted(client):
    rows = client.get("/scores").json()
    assert len(rows) == 27
    ambers = [r["amber"] for r in rows]
    assert ambers == sorted(ambers, reverse=True)
    assert all(0 <= a <= 100 for a in ambers)
    assert {r["band"] for r in rows} <= {"Green", "Amber", "Red"}
    assert all(r["rank_lo"] <= r["rank_hi"] for r in rows)
    assert all(0 <= r["power"] <= 100 for r in rows)
    assert sorted(r["power_rank"] for r in rows)[0] == 1


def test_scores_for_year(client):
    rows = client.get("/scores", params={"year": 2020}).json()
    assert len(rows) == 27 and all(r["year"] == 2020 for r in rows)


def test_scores_unknown_year(client):
    assert client.get("/scores", params={"year": 1990}).status_code == 404


def test_score_one_with_reasons(client):
    r = client.get("/scores/GR")
    assert r.status_code == 200
    body = r.json()
    assert body["iso2"] == "GR" and body["name"] == "Greece"
    assert len(body["reasons"]) == 3
    assert all("pts" in x["text"] for x in body["reasons"])


def test_score_one_accepts_lowercase_and_el(client):
    assert client.get("/scores/el").json()["iso2"] == "GR"


def test_score_one_unknown_country(client):
    r = client.get("/scores/XX")
    assert r.status_code == 404


def test_predict_with_nulls(client):
    body = {"iso2": "RO", "burned_ha": 25000, "area_km2": 238398, "population": 19000000,
            "firefighters": None, "volunteers": None, "spending_meur": 900, "stations": None}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["predicted_burned_share_pct_next_year"] >= 0
    assert 0 <= out["amber"] <= 100 and out["band"] in {"Green", "Amber", "Red"}
    assert "ff_per_100k" in out["imputed_features"]
    assert 0 <= out["power"] <= 100
    assert len(out["shap"]) == len(client.get("/health").json()["features"])


def test_predict_all_null(client):
    r = client.post("/predict", json={})
    assert r.status_code == 200


def test_predict_rejects_negative(client):
    assert client.post("/predict", json={"burned_ha": -5}).status_code == 422


def test_custom_weights(client):
    r = client.post("/score/custom-weights", json={"danger": 2, "workforce": 1, "capability": 1})
    assert r.status_code == 200
    body = r.json()
    assert abs(sum(body["weights"].values()) - 1) < 1e-3
    assert len(body["scores"]) == 27 and body["scores"][0]["rank"] == 1


def test_custom_weights_change_scores(client):
    a = client.post("/score/custom-weights", json={"danger": 1, "workforce": 0, "capability": 0}).json()
    b = client.post("/score/custom-weights", json={"danger": 0, "workforce": 1, "capability": 1}).json()
    assert [s["iso2"] for s in a["scores"]] != [s["iso2"] for s in b["scores"]]


def test_custom_weights_all_zero(client):
    assert client.post("/score/custom-weights", json={"danger": 0, "workforce": 0, "capability": 0}).status_code == 422


def test_more_hazards_lower_power(client):
    base = {"burned_ha": 20000, "area_km2": 130000, "population": 10400000, "firefighters": 17000,
            "spending_meur": 900, "stations": 500, "hazard_load": 0, "major_load": 0}
    calm = client.post("/predict", json=base).json()
    shocked = client.post("/predict", json={**base, "hazard_load": 300, "major_load": 27}).json()
    assert shocked["power"] < calm["power"]
    assert shocked["amber"] > calm["amber"]


def test_custom_weights_with_hazards(client):
    r = client.post("/score/custom-weights", json={"danger": 1, "workforce": 1, "capability": 1, "hazards": 1})
    assert r.status_code == 200 and abs(r.json()["weights"]["hazards"] - 0.25) < 1e-6


def test_simulate_presets(client):
    r = client.get("/simulate/presets")
    assert r.status_code == 200 and "multi_hazard" in r.json()


def test_simulate_shock_lowers_power_of_strongest(client):
    strongest = min(client.get("/scores").json(), key=lambda x: x["power_rank"])["iso2"]
    r = client.post("/simulate", json={"shocks": [{"iso2": strongest, "preset": "multi_hazard"}]})
    assert r.status_code == 200
    row = next(x for x in r.json()["rows"] if x["iso2"] == strongest)
    assert row["shocked"] and row["power_after"] < row["power_before"] and row["rank_change"] >= 0


def test_simulate_custom_events_on_weak_country(client):
    weakest = max(client.get("/scores").json(), key=lambda x: x["power_rank"])["iso2"]
    body = {"shocks": [{"iso2": weakest, "events": [{"type": "FL", "alert": "Red", "count": 2}]}]}
    row = next(x for x in client.post("/simulate", json=body).json()["rows"] if x["iso2"] == weakest)
    assert row["power_after"] <= row["power_before"] and row["amber_after"] >= row["amber_before"]


def test_simulate_unknown_country(client):
    assert client.post("/simulate", json={"shocks": [{"iso2": "XX", "preset": "drought"}]}).status_code == 404


def test_simulate_needs_events(client):
    assert client.post("/simulate", json={"shocks": [{"iso2": "RO"}]}).status_code == 422


def test_optimize_with_explicit_need(client):
    r = client.post("/optimize/assistance", json={"recipient": "PT", "need": 1500})
    assert r.status_code == 200
    body = r.json()
    assert body["covered"] + body["shortfall"] == pytest.approx(1500, abs=1)
    assert all(d["iso2"] != "PT" for d in body["donors"])
    assert all(d["share_of_need"] <= 0.4 + 1e-6 for d in body["donors"])
    assert sum(d["sent"] for d in body["donors"]) == body["covered"]
    assert all(d["sent"] >= 20 for d in body["donors"])


def test_optimize_need_from_shock(client):
    body = {"recipient": "CY", "shocks": [{"iso2": "CY", "preset": "multi_hazard"}]}
    r = client.post("/optimize/assistance", json=body)
    assert r.status_code == 200 and r.json()["need"] > 0 and r.json()["covered"] > 0


def test_optimize_location_weight_moves_donors_closer(client):
    far = client.post("/optimize/assistance", json={"recipient": "GR", "need": 2000, "weight_location": 0.0}).json()
    near = client.post("/optimize/assistance", json={"recipient": "GR", "need": 2000, "weight_location": 1.0}).json()
    assert near["mean_distance_km"] <= far["mean_distance_km"]


def test_optimize_shocked_donor_gives_less(client):
    calm = client.post("/optimize/assistance", json={"recipient": "GR", "need": 3000}).json()
    top = calm["donors"][0]["iso2"]
    shocked = client.post("/optimize/assistance", json={"recipient": "GR", "need": 3000,
                          "shocks": [{"iso2": top, "preset": "multi_hazard"}]}).json()
    before = calm["donors"][0]["sent"]
    after = next((d["sent"] for d in shocked["donors"] if d["iso2"] == top), 0)
    assert after <= before


def test_optimize_validation(client):
    assert client.post("/optimize/assistance", json={"recipient": "PT"}).status_code == 422
    assert client.post("/optimize/assistance", json={"recipient": "XX", "need": 10}).status_code == 404


def test_model_info(client):
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "LGBMRegressor"
    assert {f["pillar"] for f in body["features"]} >= {"danger", "hazards", "workforce", "capability"}
    assert abs(sum(p["w_final"] for p in body["pillars"].values()) - 1) < 1e-6


def _fake_events():
    import pandas as pd
    now = pd.Timestamp.now(tz="UTC")
    return pd.DataFrame([
        {"iso2": "CY", "eventid": 1, "eventtype": "WF", "alertlevel": "Red", "name": "Forest fires in Cyprus", "start": now, "end": now, "url": None},
        {"iso2": "CY", "eventid": 2, "eventtype": "FL", "alertlevel": "Red", "name": "Flood in Cyprus", "start": now, "end": now, "url": None},
        {"iso2": "DE", "eventid": 3, "eventtype": "DR", "alertlevel": "Orange", "name": "Drought", "start": now, "end": now, "url": None},
    ])


def test_live_uses_current_events(client, monkeypatch):
    import api.main as m
    monkeypatch.setattr(m.live, "fetch_active_events", lambda days=14: _fake_events())
    r = client.get("/live", params={"refresh": True})
    assert r.status_code == 200
    body = r.json()
    assert len(body["countries"]) == 27 and len(body["events"]) == 3
    assert "CY" in body["stretched"]
    de = next(c for c in body["countries"] if c["iso2"] == "DE")
    assert de["crew_demand"] == 50 and de["unmet_demand"] == 0
    assert client.get("/health").json()["live_events"] == 3


def test_live_assistance(client, monkeypatch):
    import api.main as m
    monkeypatch.setattr(m.live, "fetch_active_events", lambda days=14: _fake_events())
    client.get("/live", params={"refresh": True})
    r = client.get("/live/assistance", params={"recipient": "CY"})
    assert r.status_code == 200 and r.json()["covered"] > 0
    assert all(d["iso2"] != "CY" for d in r.json()["donors"])
    assert client.get("/live/assistance", params={"recipient": "FI"}).status_code == 422


def test_hot_reload_when_artifacts_change(client):
    import os, time
    import api.main as m
    before = client.get("/health").json()["loaded_at"]
    meta = m._artifacts_path() / "metadata.json"
    time.sleep(1.1)
    os.utime(meta, None)
    after = client.get("/health").json()["loaded_at"]
    assert after != before
    assert client.post("/admin/reload").status_code == 200


def test_rescore_reproduces_scores(tmp_path):
    import json, shutil
    from amber import ARTIFACTS_DIR
    from amber.pipeline import rescore
    for f in ARTIFACTS_DIR.iterdir():
        if f.is_file():
            shutil.copy(f, tmp_path / f.name)
    rescore(refresh=False, out=tmp_path, verbose=False)
    old = {(r["iso2"], r["year"]): r["amber"] for r in json.loads((ARTIFACTS_DIR / "scores.json").read_text())}
    new = {(r["iso2"], r["year"]): r["amber"] for r in json.loads((tmp_path / "scores.json").read_text())}
    assert old == new


def test_replay_day(client):
    r = client.get("/replay", params={"date": "2021-08-08"})
    assert r.status_code == 200
    body = r.json()
    assert body["reference_date"] == "2021-08-01" and body["counts"]["all"] == len(body["fires"]) > 0
    assert len(body["countries"]) == 27 and len(body["eu_daily"]) == 62
    gr = next(c for c in body["countries"] if c["iso2"] == "GR")
    assert gr["power"] <= gr["power_ref"] and gr["normal"]["rank"] <= 3
    assert all(f["status"] in {"above", "watch", "normal"} for f in body["fires"])


def test_replay_out_of_range(client):
    assert client.get("/replay", params={"date": "2005-08-01"}).status_code == 404


def test_history(client):
    r = client.get("/history/GR", params={"date": "2021-08-08"})
    assert r.status_code == 200
    body = r.json()
    months = body["months"]
    assert months[0]["year"] == 2001 and months[-1]["year"] == 2021
    assert all(m["modelled"] == (m["year"] < 2012) for m in months)
    assert next(m for m in months if m["year"] == 2021 and m["month"] == 9)["harmonized"] is None
    assert body["largest_fires"] and body["outlook"]["bad_week"] >= body["outlook"]["typical_week"]


def test_history_csv(client):
    r = client.get("/history/GR.csv", params={"date": "2021-08-08"})
    assert r.status_code == 200 and r.text.startswith("country,year,month,harmonized_cell_days")


def test_plan(client):
    r = client.post("/plan", json={"recipient": "GR", "date": "2021-08-08", "scenario": "wildfire_season"})
    assert r.status_code == 200
    body = r.json()
    rec = body["recipient"]
    assert rec["power_after"] <= rec["power_before"] and rec["power_with_help"] >= rec["power_after"]
    assert all(d["power_after"] >= 50 - 0.5 for d in body["donors"])
    etas = [d["eta_h"] for d in body["donors"]]
    assert etas == sorted(etas)
    assert all(lo["iso2"] not in {d["iso2"] for d in body["donors"]} for lo in body["left_out"])


def test_area_box_matches_range(client):
    r = client.get("/area", params={"bbox": "22.5,37.8,24.5,39.3", "start": "2021-07-25", "end": "2021-08-15"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "box" and body["series"]["unit"] == "day" and len(body["series"]["values"]) == 22
    s = body["summary"]
    assert s["detections"] == sum(s["detections_by_sensor"].values())
    assert abs(sum(v["harmonized"] for v in body["series"]["values"]) - s["harmonized_cell_days"]) < 1
    assert body["range_normal"]["rank"] >= 1


def test_area_country_matches_history(client):
    a = client.get("/area", params={"country": "GR", "end": "2021-08-08"}).json()
    h = client.get("/history/GR", params={"date": "2021-08-08"}).json()
    for x, y in zip(a["months"], h["months"]):
        if x["harmonized"] is not None:
            assert abs(x["harmonized"] - y["harmonized"]) < 0.5
    assert a["normal"]["rank"] == h["normal"]["rank"]


def test_area_long_range_is_monthly(client):
    body = client.get("/area", params={"country": "PT", "start": "2015-01-01", "end": "2020-12-31"}).json()
    assert body["series"]["unit"] == "month" and len(body["series"]["values"]) == 72
    assert body["range_normal"] is None


def test_area_csv_exports(client):
    params = {"bbox": "22.5,37.8,24.5,39.3", "start": "2021-08-01", "end": "2021-08-10"}
    hot = client.get("/area/hotspots.csv", params=params)
    assert hot.status_code == 200 and hot.text.startswith("latitude,longitude,acq_date")
    assert int(hot.headers["x-rows"]) == len(hot.text.strip().splitlines()) - 1
    daily = client.get("/area/daily.csv", params=params).text.strip().splitlines()
    assert len(daily) == 11
    cal = client.get("/area/calendar.csv", params=params).text
    assert "2021,8," in cal


def test_area_rejects_bad_input(client):
    assert client.get("/area", params={"bbox": "1,2,3"}).status_code == 422
    assert client.get("/area", params={"country": "GR", "bbox": "1,2,3,4"}).status_code == 422
    assert client.get("/area", params={"country": "GR", "start": "2021-08-10", "end": "2021-08-01"}).status_code == 422
    assert client.get("/area", params={"country": "XX"}).status_code == 404


def test_area_asof_keeps_calendar(client):
    body = client.get("/area", params={"country": "GR", "start": "2007-08-01", "end": "2007-08-31", "asof": "2021-08-08"}).json()
    assert body["date"] == "2021-08-08" and body["months"][-1]["year"] == 2021
    assert body["summary"]["days"] == 31 and body["range_normal"]["rank"] <= 3


def test_simulate_on_replay_day(client):
    body = {"date": "2021-08-08", "shocks": [{"iso2": "IT", "events": [{"type": "FL", "alert": "Red", "count": 1}]},
                                             {"iso2": "DE", "preset": "major_flood"}]}
    r = client.post("/simulate", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["date"] == "2021-08-08" and len(out["rows"]) == 27
    rows = {x["iso2"]: x for x in out["rows"]}
    assert rows["IT"]["shocked"] and rows["IT"]["power_after"] < rows["IT"]["power_before"]
    assert not rows["SE"]["shocked"] and abs(rows["SE"]["power_change"]) < 1e-6
    replay = client.get("/replay", params={"date": "2021-08-08"}).json()
    gr = next(c for c in replay["countries"] if c["iso2"] == "GR")
    assert abs(rows["GR"]["power_before"] - gr["power"]) < 0.06


def test_regions_split_national_firefighters(client):
    out = client.get("/regions", params={"date": "2021-08-08"}).json()
    rows = [r for r in out["regions"] if r["iso2"] == "GR"]
    assert [r["region"] for r in rows] == ["north", "east", "south", "west"]
    assert sum(r["firefighters"] for r in rows) == 20500
    assert abs(sum(r["fire_share"] for r in rows) - 1) < 1e-3
    east = next(r for r in rows if r["region"] == "east")
    assert east["largest_fire"] == "Roviés" and east["load_ratio"] > 2
    mt = [r for r in out["regions"] if r["iso2"] == "MT"]
    assert all(not r["available"] and r["firefighters"] is None for r in mt)
    at = next(s for s in out["sources"] if s["iso2"] == "AT")
    assert at["rescaled"] and sum(r["firefighters"] for r in out["regions"] if r["iso2"] == "AT") == 5200


def test_plan_reports_incident_region_and_departures(client):
    p = client.post("/plan", json={"recipient": "GR", "date": "2021-08-08"}).json()
    assert p["recipient"]["region"] in {"north", "east", "south", "west"}
    assert p["recipient"]["region_firefighters"] > 0
    assert all(d["region"] and len(d["departure"]) == 2 for d in p["donors"])


def test_score_history(client):
    rows = client.get("/scores/GR/history").json()
    assert len(rows) >= 10 and rows == sorted(rows, key=lambda r: r["year"])
    assert client.get("/scores/XX/history").status_code == 404
