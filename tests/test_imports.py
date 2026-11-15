import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import app as app_module
from server import db, imports, settings
from server.pipeline import store

MODIS = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type
38.1,23.8,330.1,1.0,1.0,2021-08-05,934,Terra,MODIS,80,6.1,300.2,40.5,D,0
38.2,23.9,331.0,1.0,1.0,2021-08-05,1205,Aqua,MODIS,55,6.1NRT,301.0,22.0,D,0
"""
VIIRS = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
38.10,23.80,340.0,0.4,0.4,2021-08-05,1101,N,VIIRS,n,2,290.0,10.0,D
38.11,23.81,345.0,0.4,0.4,2021-08-05,1150,N20,VIIRS,h,2.0NRT,291.0,12.0,D
38.12,23.82,350.0,0.4,0.4,2021-08-06,0110,N21,VIIRS,l,2.0NRT,289.0,3.0,N
"""


def test_split_routes_rows_to_sources():
    m = imports.split_firms_csv(MODIS)
    assert {k: len(v) for k, v in m.items()} == {"MODIS_SP": 1, "MODIS_NRT": 1}
    v = imports.split_firms_csv(VIIRS)
    assert {k: len(v) for k, v in v.items()} == {"VIIRS_SNPP_SP": 1, "VIIRS_NOAA20_NRT": 1, "VIIRS_NOAA21_NRT": 1}


def test_split_rejects_other_files():
    with pytest.raises(imports.ImportError_):
        imports.split_firms_csv("a,b\n1,2\n")
    with pytest.raises(imports.ImportError_):
        imports.split_firms_csv("latitude,longitude,acq_date,acq_time,satellite,confidence,frp,daynight\n1,2,2021-01-01,1,N,n,1,D\n")


def test_save_import_is_idempotent(tmp_path):
    a = imports.save_import(VIIRS, "fires.csv", tmp_path)
    b = imports.save_import(VIIRS, "fires.csv", tmp_path)
    assert [x["file"] for x in a] == [x["file"] for x in b]
    assert all(x["replaced"] for x in b)
    assert sum(1 for _ in tmp_path.glob("*/*.csv")) == 3


def test_imported_files_normalize(tmp_path):
    from firms.normalize import normalize_file
    for item in imports.save_import(MODIS + "", "m", tmp_path) + imports.save_import(VIIRS, "v", tmp_path):
        df = normalize_file(tmp_path / item["file"], item["source"])
        assert len(df) == item["rows"]
        assert df["acq_time_utc"].notna().all()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "RUN_ON_START", False)
    monkeypatch.setattr(settings, "TILE_BG", False)
    monkeypatch.setattr(imports.C, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(app_module, "start_pipeline", lambda: False)
    db.init_db()
    df = pd.DataFrame({
        "sensor": ["MODIS", "VIIRS", "VIIRS"], "platform": ["Terra", "SNPP", "NOAA-20"], "product": ["SP", "SP", "NRT"],
        "lat": [38.1, 38.1, 45.0], "lon": [23.8, 23.8, 10.0],
        "acq_time_utc": pd.to_datetime(["2021-08-05 09:34", "2021-08-05 11:01", "2021-08-06 01:00"], utc=True),
        "daynight": ["D", "D", "N"], "confidence_class": ["high", "nominal", "low"], "frp_mw": [40.5, 10.0, 3.0],
        "bt_mid_k": [330.0, 340.0, 300.0], "cluster_id": pd.array([0, 0, None], dtype="Int64"),
        "harmonized_status": ["both", "both", "viirs_only"],
    })
    store.write_detections(df)
    with TestClient(app_module.app) as c:
        yield c


def test_import_endpoint_saves_files(client):
    r = client.post("/api/import?name=evia.csv", content=VIIRS, headers={"Content-Type": "text/csv"})
    assert r.status_code == 200
    assert r.json()["rows"] == 3 and len(r.json()["files"]) == 3
    assert client.post("/api/import", content="nope\n1\n").status_code == 422
    assert client.get("/api/sources").json()["files"] == 3


def test_export_and_summary(client):
    q = "from=2021-08-01&to=2021-08-31&west=20&south=35&east=30&north=42"
    r = client.get(f"/api/export.csv?{q}")
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("latitude,longitude,acq_time_utc,sensor,platform")
    assert len(lines) == 3 and "both" in lines[1]
    assert len(client.get(f"/api/export.csv?{q}&sensor=MODIS").text.strip().splitlines()) == 2
    s = client.get(f"/api/summary?{q}").json()
    assert s["total"] == 2 and s["by_status"] == {"both": 2}
    assert s["by_platform"] == {"MODIS Terra": 1, "VIIRS SNPP": 1}
    assert s["data"]["detections"] == 3
