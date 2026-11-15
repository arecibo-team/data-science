from fastapi.testclient import TestClient

from app import app, dryness, dryness_label

c = TestClient(app)


def test_dryness_combines_deficits_and_heat():
    assert dryness(-2.0, -1.0, 1.5) == 1.5
    assert dryness(None, None, None) is None
    assert dryness_label(1.6) == "Much drier than usual" and dryness_label(0.0) == "Close to normal"


def test_countries_ranked_for_a_month():
    rows = c.get("/countries", params={"month": "2025-08"}).json()
    assert len(rows) == 27 and all(r["eu"] for r in rows)
    ranked = [r for r in rows if r["rank"]]
    assert [r["rank"] for r in ranked] == list(range(1, len(ranked) + 1))
    assert all(ranked[i]["dryness_z"] >= ranked[i + 1]["dryness_z"] for i in range(len(ranked) - 1))


def test_country_detail_accepts_iso2_and_iso3():
    a = c.get("/countries/GR").json()
    b = c.get("/countries/GRC").json()
    assert a["iso3"] == b["iso3"] == "GRC" and len(a["series"]) == 34
    assert c.get("/countries/XX").status_code == 404


def test_point_and_grids():
    p = c.get("/point", params={"lat": 38.9, "lon": 23.3, "month": "2025-08"}).json()
    assert {"land", "terrain", "soil", "vegetation"} <= set(p)
    for layer in ("dryness", "soil", "ndvi", "lst", "fuel", "steep"):
        g = c.get(f"/grid/{layer}", params={"month": "2025-08", "west": 19, "south": 34, "east": 29, "north": 42}).json()
        assert g["cells"] and all(19 <= x[1] <= 29 for x in g["cells"])


def test_months_before_the_data_are_rejected():
    assert c.get("/countries", params={"month": "2021-08"}).status_code == 404
    p = c.get("/point", params={"lat": 38.9, "lon": 23.3, "month": "2021-08"}).json()
    assert p["month"] is None and "land" in p and "terrain" in p and "soil" not in p
