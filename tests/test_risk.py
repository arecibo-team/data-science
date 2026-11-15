"""Country risk model: fire weather codes, the panel, no look-ahead, the outlook endpoint."""
import datetime as dt

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import app as app_mod, db, settings
from server.risk import data, fwi, serve


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setattr(settings, "RUN_ON_START", False)
    monkeypatch.setattr(settings, "TILE_BG", False)
    serve._cache.clear()
    db.init_db()
    return tmp_path


def test_fire_weather_codes_match_the_published_example():
    """The worked example of Van Wagner and Pickett (1985), three April days."""
    days = [(17.0, 42.0, 25.0, 0.0), (20.0, 21.0, 25.0, 2.4), (8.5, 40.0, 17.0, 0.0)]
    want = [(87.7, 8.5, 19.0, 10.9, 8.5, 10.1), (86.2, 10.4, 23.6, 8.8, 10.4, 9.3), (87.0, 11.8, 26.1, 6.5, 11.7, 7.6)]
    f, m, d = fwi.START
    for (t, h, w, r), exp in zip(days, want):
        f, m, d, isi, bui, s = fwi.step(f, m, d, t, h, w, r, 3)
        assert np.allclose([f, m, d, isi, bui, s], exp, atol=0.15)


def _fill(n_days=120):
    """Two weather cells in Greece and one in France; fires in the first Greek cell on hot days."""
    rng = np.random.default_rng(0)
    d0 = dt.date(2024, 5, 1)
    bins, wx = [], []
    for i in range(n_days):
        day = (d0 + dt.timedelta(days=i)).isoformat()
        hot = i % 10 >= 7
        for r1, c1 in ((128, 203), (129, 202), (136, 182)):          # 38.5N 23.5E, 39.5N 22.5E, 46.5N 2.5E
            wx.append((r1, c1, day, 30.0 if hot else 22.0, 40.0 if hot else 28.0, 20.0 if hot else 60.0, 6.0 if hot else 2.0,
                       9.0 if hot else 4.0, 0.0 if hot else float(rng.random() < 0.2) * 5, 0.3, 0.4))
        for j in range(4):
            if hot and rng.random() < 0.8:
                bins.append((1283 + j, 2032, day, 1, 2, 0, 0, 1, 0, 1))
        bins.append((1295, 2025, day, 0, 1, 0, 0, 0, 1, 0))             # single sensor only: a patch, never a fire
    with db.session() as con:
        con.executemany("INSERT INTO bin_day VALUES (?,?,?,?,?,?,?,?,?,?)", bins)
        con.executemany("INSERT INTO weather_grid (r1, c1, date, t2m, t2mx, rh, ws, wsx, prec, gwettop, gwetroot) VALUES (?,?,?,?,?,?,?,?,?,?,?)", wx)


def test_panel_and_features_never_look_past_the_issue_day_or_the_forecast(tmp_db):
    _fill()
    p = data.load()
    g = p.countries.index("Greece")
    assert (p.cid == g).all() and len(p.cell) == 2 and p.B.sum() > 0
    assert p.B[p.B.sum(axis=1) == 0].shape[0] == 1                       # the single-sensor patch has no fire
    nm = data.normals(p, 89)
    bs = data.base(p, nm)
    ols = dict(data.outlooks(p, nm))
    pos, di = np.array([0, 1]), np.array([60, 60])
    X3 = data.matrix(p, nm, bs, ols[3], 3, pos, di)
    assert X3.shape == (2, len(data.FEATURES)) and np.isfinite(X3).all()
    # a perfect forecast carried forward from day D gives the reanalysis codes of the target day
    assert np.allclose(ols[3]["fwi_t"][:, 60], p.W["fwi"][:, 63], rtol=1e-4)
    # the normals stop at their end day; later fires and later weather change nothing before
    p.B[:, 95:] += 5
    p.W["t2mx"][:, 95:] += 10
    nm2 = data.normals(p, 89)
    assert np.array_equal(nm.fire, nm2.fire) and np.array_equal(nm.tmax, nm2.tmax)
    # weather after the target day does not reach a lead-3 row; fires after the issue day do not either
    p.W["t2mx"][:, 64:] += 10
    p.B[:, 61:] += 3
    X3b = data.matrix(p, nm2, data.base(p, nm2), dict(data.outlooks(p, nm2))[3], 3, pos, di)
    assert np.allclose(X3, X3b)
    # a forecast source replaces the reanalysis of the forecast days only
    src = {k: {c: p.W[c].copy() for c in data.WEATHER} for k in data.LEADS}
    src[1]["wsx"][:, 61] = 25.0
    o1 = dict(data.outlooks(p, nm2, src))[1]
    assert (o1["wsx_t"][:, 60] == 25.0).all() and np.allclose(o1["wsx_t"][:, 59], p.W["wsx"][:, 60])


def test_a_day_is_ranked_against_its_own_season_only():
    """Two years of stored days: summer totals are large, winter totals are zero. A summer day's percentile must not
    be pushed up by the winter days, and a winter day is ranked against winter alone."""
    days = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(730)]
    doy = np.array([d.timetuple().tm_yday for d in days])
    tot = np.where((doy > 150) & (doy < 260), np.linspace(1.0, 10.0, len(days)), 0.0)
    h = (doy, tot)
    summer = serve._pct(h, "2024-07-01", 3.0)
    assert summer is not None and 0.0 < summer < 1.0
    # a winter with no fire at all falls back to every stored day, so an unusual winter number is still ranked
    assert serve._pct(h, "2024-02-01", 1.0) == pytest.approx(float(np.mean(tot <= 1.0)))
    assert serve._pct((doy, np.zeros(len(days))), "2024-07-01", 1.0) is None   # a country that never sees fire
    assert serve._pct(None, "2024-07-01", 1.0) is None


def test_outlook_is_stored_and_served_as_country_risk(tmp_db):
    _fill()
    p = data.load()
    g = p.countries.index("Greece")
    T, nc = p.n_days, len(p.countries)
    perfect = np.zeros((len(data.LEADS), nc, T), np.float32)
    perfect[:, g] = np.linspace(0.1, 3.0, T)[None, :]
    issued = perfect[:, g] * 2
    cover = np.zeros((len(data.LEADS), T), bool)
    cover[:, 100:110] = True
    actual = np.zeros((nc, T), int)
    actual[g] = 2
    assert serve.store(p, perfect, issued, g, cover, actual, (30, 89), 30) > 0
    with TestClient(app_mod.app) as client:
        day = p.date_of(105)
        r = client.get("/api/risk", params={"date": day}).json()
        c = r["countries"][0]
        assert r["model"] == "risk" and r["horizon"] == 3 and c["name"] == "Greece" and c["weather"] == "issued"
        assert c["burned"] == 6 and c["expected"] == pytest.approx(float(issued[:, 105].sum()), rel=1e-4) and c["score"] > 0.5
        assert c["area_km2"] > 0 and c["density"] == pytest.approx(c["expected"] / c["area_km2"] * 1000, rel=1e-6)
        assert c["real"]["expected"] == 6 and c["real"]["complete"] and c["real"]["density"] == pytest.approx(6 / c["area_km2"] * 1000)
        assert 0.0 <= c["real"]["score"] <= 1.0
        early = client.get("/api/risk", params={"date": p.date_of(50)}).json()["countries"][0]
        assert early["weather"] == "perfect" and early["expected"] < c["expected"]
        late = client.get("/api/risk", params={"date": "2030-01-01"}).json()
        assert late["feature_date"] == p.date_of(T - 1) and not late["complete"] and late["countries"][0]["burned"] is None
        assert late["countries"][0]["real"] is None      # no confirmed cells known for any lead of that day
        last_days = client.get("/api/risk", params={"date": p.date_of(T - 2)}).json()["countries"][0]["real"]
        assert last_days is not None and not last_days["complete"] and last_days["expected"] == 2     # one lead known
