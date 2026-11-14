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
        shutil.copy(f, tmp_path / f.name)
    rescore(refresh=False, out=tmp_path, verbose=False)
    old = {(r["iso2"], r["year"]): r["amber"] for r in json.loads((ARTIFACTS_DIR / "scores.json").read_text())}
    new = {(r["iso2"], r["year"]): r["amber"] for r in json.loads((tmp_path / "scores.json").read_text())}
    assert old == new
