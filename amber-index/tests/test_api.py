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
            "spending_meur": 900, "stations": 500, "hazard_load": 0}
    calm = client.post("/predict", json=base).json()
    shocked = client.post("/predict", json={**base, "hazard_load": 300}).json()
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
