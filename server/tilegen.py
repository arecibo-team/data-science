"""Background tile generation: renders every zoom level ahead of time, without making anyone wait.

A daemon thread plans the tiles that contain data for the preset date ranges (all data, last year, last 30
days; all layers on), from zoom 0 up to `FIRMS_TILE_BG_MAX_ZOOM`, and renders the missing ones in order of
zoom, so every level is usable as early as possible. It works at low priority (a short pause after each tile).

Requests never wait for it. `/api/tiles/...` renders a missing tile itself, right away, and caches it; the
service skips tiles that already exist, so both write into the same cache without clashing. Zoom levels above
the limit, other date ranges and other layer masks are rendered on first request only.

The service starts with the server and follows the data: when a pipeline pass produces a new data version it
drops the old cache and starts over. It stands by while a pass is rewriting detections (harmonize, tag, stats).
`status()` reports progress; the web app shows it.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import db, settings, tiles

log = logging.getLogger("server.tilegen")

IDLE_POLL_S = 10.0
BUSY_POLL_S = 3.0
PAUSE_PER_TILE_S = 0.004   # keeps the thread polite so requests and the pipeline are not starved
CHECK_EVERY = 25           # tiles between checks for a new data version


@dataclass(frozen=True)
class Job:
    """All tiles of one zoom level and date range that contain data."""
    d0: str
    d1: str
    z: int
    xy: np.ndarray  # (n, 2) int array of tile x, y


def plan(con, max_zoom: int) -> list[Job]:
    """Tiles with data per zoom and preset range, lowest zoom first.

    A tile is included when a bin cell lies inside it or within PAD px of it, because the blur of
    a spot reaches over the border into the next tile.
    """
    days = [r[0] for r in con.execute("SELECT date FROM day_stats ORDER BY date")]
    if not days:
        return []
    ranges = tiles.common_ranges(days[0], days[-1])
    cells = {rng: tiles.load_cells(con, *rng) for rng in ranges}
    jobs: list[Job] = []
    for z in range(0, max_zoom + 1):
        n = 2 ** z
        for rng in ranges:
            lat, lon = cells[rng][0], cells[rng][1]
            if len(lat) == 0:
                continue
            px, py = tiles.project(lat, lon, z)
            # Per cell, the tiles its padded footprint touches: all four corner combinations of x and y.
            cx = np.stack([np.floor((px - tiles.PAD) / tiles.TILE), np.floor((px + tiles.PAD) / tiles.TILE)]).astype(np.int64)
            cy = np.stack([np.floor((py - tiles.PAD) / tiles.TILE), np.floor((py + tiles.PAD) / tiles.TILE)]).astype(np.int64)
            pairs = np.unique(np.concatenate([np.stack([cx[i], cy[j]], axis=1) for i in (0, 1) for j in (0, 1)]), axis=0)
            pairs = pairs[(pairs[:, 0] >= 0) & (pairs[:, 0] < n) & (pairs[:, 1] >= 0) & (pairs[:, 1] < n)]
            jobs.append(Job(rng[0], rng[1], z, pairs))
    return jobs


class TileService:
    def __init__(self, max_zoom: int | None = None):
        self.max_zoom = settings.TILE_BG_MAX_ZOOM if max_zoom is None else max_zoom
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = {"state": "starting", "version": 0, "done": 0, "total": 0, "zoom": None}
        self.completed_version = 0

    # ---- control
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="tilegen", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            s = dict(self._state)
        s["max_zoom"] = self.max_zoom
        s["enabled"] = True
        return s

    def _set(self, **kw) -> None:
        with self._lock:
            self._state.update(kw)

    # ---- work
    def _unstable(self, con) -> bool:
        """True while a pipeline pass is rewriting the data tiles are drawn from."""
        return con.execute("SELECT 1 FROM pipeline_steps WHERE status='running' "
                           "AND step IN ('harmonize', 'tag', 'stats')").fetchone() is not None

    def _loop(self) -> None:
        wait = BUSY_POLL_S
        while not self._stop.wait(wait):
            wait = IDLE_POLL_S
            try:
                con = db.connect(readonly=True)
            except Exception:  # database not created yet
                wait = BUSY_POLL_S
                continue
            try:
                version = tiles.data_version(con)
                if version == 0:
                    self._set(state="waiting", version=0)
                    wait = BUSY_POLL_S
                elif version == self.completed_version:
                    self._set(state="idle")
                elif self._unstable(con):
                    self._set(state="paused")
                    wait = BUSY_POLL_S
                else:
                    if self.run_once(con, version):
                        self.completed_version = version
                    wait = BUSY_POLL_S
            except Exception:
                log.exception("tile generation failed; retrying")
                self._set(state="error")
            finally:
                con.close()

    def run_once(self, con, version: int) -> bool:
        """Render the missing tiles of one data version. False when interrupted (stop, or newer data)."""
        tiles.clear_old_versions(version)
        jobs = plan(con, self.max_zoom)
        total = sum(len(j.xy) for j in jobs)
        done = 0
        self._set(state="building", version=version, done=0, total=total, zoom=0)
        t0, written = time.monotonic(), 0
        for job in jobs:
            self._set(zoom=job.z)
            for x, y in job.xy.tolist():
                if self._stop.is_set():
                    return False
                if done % CHECK_EVERY == 0 and (tiles.data_version(con) != version or self._unstable(con)):
                    log.info("tile generation interrupted by new data")
                    return False
                path = tiles.tile_path(version, job.d0, job.d1, tiles.MASK_ALL, job.z, x, y)
                if not path.exists():  # a request may have rendered it meanwhile
                    png = tiles.render_from_db(con, job.z, x, y, job.d0, job.d1, tiles.MASK_ALL)
                    tiles.write_tile(path, png)  # a blank tile is stored too (zero bytes), so it is not redone
                    written += png is not None
                    time.sleep(PAUSE_PER_TILE_S)
                done += 1
                if done % 50 == 0:
                    self._set(done=done)
        self._set(state="idle", done=total)
        log.info("tiles: %d of %d tiles rendered for z0-%d in %.0fs", written, total, self.max_zoom, time.monotonic() - t0)
        return True


service = TileService()
