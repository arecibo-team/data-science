"""Pipeline runner: time slices, paused steps, restart recovery; and terrain resuming from its tile cache."""
import numpy as np
import pytest

from server import db, settings, terrain
from server.pipeline import runner
from server.pipeline.runner import Plan, Step


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "t.db"))
    db.init_db()
    return tmp_path


def test_a_paused_step_lets_later_steps_run_and_is_resumed_next_pass(tmp_db, monkeypatch):
    calls = {"weather": 0, "after": 0}

    def weather(should_stop):
        calls["weather"] += 1
        if calls["weather"] == 1:
            should_stop()           # the time slice is already used up: the step pauses
        return 1

    def after():
        calls["after"] += 1
        return 1

    monkeypatch.setattr(runner, "STEPS", [
        Step("weather", "w", Plan, weather, resumable=True),
        Step("after", "a", Plan, after),
    ])
    monkeypatch.setattr(settings, "PIPELINE_SLICE_S", 0.0)
    assert runner.run_pending() == {"weather": "done", "after": "up_to_date"}
    assert calls == {"weather": 2, "after": 1}
    with db.session() as con:
        statuses = [r[0] for r in con.execute("SELECT status FROM pipeline_steps WHERE step='weather' ORDER BY id")]
    assert statuses == ["paused", "done"]


def test_a_restart_pauses_resumable_steps_and_fails_the_rest(tmp_db, monkeypatch):
    monkeypatch.setattr(runner, "STEPS", [
        Step("weather", "w", Plan, lambda should_stop: 0, resumable=True),
        Step("fetch", "f", Plan, lambda: 0),
    ])
    with db.session() as con:
        for name in ("weather", "fetch"):
            con.execute("INSERT INTO pipeline_steps(step, fingerprint, params, status, started_at) "
                        "VALUES (?,'fp','{}','running','2026-11-15T00:00:00Z')", (name,))
    runner._recover_interrupted()
    with db.session() as con:
        got = {r["step"]: r["status"] for r in con.execute("SELECT step, status FROM pipeline_steps")}
    assert got == {"weather": "paused", "fetch": "failed"}


def test_terrain_resumes_from_cached_tiles(tmp_db, monkeypatch, tmp_path):
    monkeypatch.setattr(terrain, "CACHE_DIR", tmp_path / "terrain")
    monkeypatch.setattr(terrain, "_tiles", lambda bbox: [(0.0, 40.0, 0.1, 40.1), (0.1, 40.0, 0.2, 40.1)])
    monkeypatch.setattr(terrain, "credentials", lambda: ("id", "secret"))
    calls = []

    def raster(self, bbox, dataset, script):
        calls.append(bbox)
        return np.full((2, 10, 10), 2.0, np.float32)

    monkeypatch.setattr(terrain._Client, "raster", raster)

    # first pass: the slice ends once the first tile is done (three layers)
    assert terrain.run(lambda: len(calls) >= 3) == 0
    assert len(list((tmp_path / "terrain").glob("*.npz"))) == 1
    with db.session() as con:
        assert con.execute("SELECT COUNT(*) FROM terrain_cells").fetchone()[0] == 0

    # second pass: only the second tile is fetched, and the table is written once for both
    before = len(calls)
    assert terrain.run() == 2
    assert len(calls) - before == 3
    with db.session() as con:
        assert con.execute("SELECT COUNT(*) FROM terrain_cells").fetchone()[0] == 2
