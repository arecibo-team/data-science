"""Step runner. Every step is recorded in the `pipeline_steps` table.

A step is up to date when a `done` row exists for its name with the same *fingerprint*. The
fingerprint covers the step's parameters and inputs, plus the id of the latest `done` row of every
step it depends on. So:

  * new inputs (new raw files, a new day for the fetch window, other match settings) give a new
    fingerprint, and only that step and its dependents run again;
  * deleting a step's rows (`reset_step`) makes it run again on the next pass, and because the
    dependents' fingerprints contain the upstream row id, they run again too.

Dependencies are `deps` (must succeed, and part of the fingerprint) or `soft` (run order and reset
cascade only). A step whose inputs are content-hashed in its own params, such as `tag`, lists the
steps that produce those inputs as `soft`, so a failed network step does not block it.

A `resumable` step takes a `should_stop` callable and works in chunks it keeps across runs. Each pass gives it
`PIPELINE_SLICE_S` seconds; when the slice ends it is recorded as `paused`, the later steps run, and another pass
continues it. A restart that finds a resumable step `running` records it as `paused` too, not `failed`.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .. import db, settings

log = logging.getLogger("server.pipeline")


@dataclass
class Plan:
    """What a step would do now: its inputs, or a reason it cannot run."""
    params: dict = field(default_factory=dict)
    skip: str | None = None


@dataclass(frozen=True)
class Step:
    name: str
    title: str
    plan: Callable[[], Plan]
    run: Callable[..., int | None]      # called as run(should_stop) when resumable, else run()
    deps: tuple[str, ...] = ()
    soft: tuple[str, ...] = ()
    resumable: bool = False


STEPS: list[Step] = []


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _latest_done(con, name: str) -> int | None:
    r = con.execute("SELECT id FROM pipeline_steps WHERE step=? AND status='done' ORDER BY id DESC LIMIT 1",
                    (name,)).fetchone()
    return r["id"] if r else None


def _fingerprint(con, s: Step, params: dict) -> str:
    ups = {d: _latest_done(con, d) for d in s.deps}
    blob = json.dumps({"params": params, "upstream": ups}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _by_name(name: str) -> Step:
    for s in STEPS:
        if s.name == name:
            return s
    raise KeyError(f"unknown step {name!r}; known: {', '.join(x.name for x in STEPS)}")


def _dependents(name: str) -> list[str]:
    out, frontier = [], {name}
    while frontier:
        nxt = {s.name for s in STEPS if (set(s.deps) | set(s.soft)) & frontier} - set(out)
        out.extend(sorted(nxt, key=[x.name for x in STEPS].index))
        frontier = nxt
    return out


def status() -> list[dict]:
    """One entry per step: `up_to_date`, `pending`, `running`, `failed` or `skipped`, plus its last row."""
    con = db.connect()
    try:
        rows = []
        for s in STEPS:
            last = con.execute("SELECT * FROM pipeline_steps WHERE step=? ORDER BY id DESC LIMIT 1", (s.name,)).fetchone()
            p = s.plan()
            if s.name in settings.SKIP_STEPS:
                state = "skipped"
            elif p.skip:
                state = "skipped"
            elif last and last["status"] == "running":
                state = "running"
            else:
                fp = _fingerprint(con, s, p.params)
                done = con.execute("SELECT 1 FROM pipeline_steps WHERE step=? AND fingerprint=? AND status='done'",
                                   (s.name, fp)).fetchone()
                state = "up_to_date" if done else ("failed" if last and last["status"] == "failed" and last["fingerprint"] == fp else "pending")
            paused = last is not None and last["status"] == "paused" and state == "pending"
            rows.append({
                "step": s.name, "title": s.title, "state": state,
                "reason": p.skip or (f"skipped by PIPELINE_SKIP" if s.name in settings.SKIP_STEPS else
                                     "paused; continues on the next pass" if paused else None),
                "last": dict(last) if last else None,
            })
        return rows
    finally:
        con.close()


def reset_step(name: str, cascade: bool = True) -> list[str]:
    """Forget a step (and by default everything downstream) so the next pass runs it again."""
    _by_name(name)
    names = [name, *(_dependents(name) if cascade else [])]
    with db.session() as con:
        for n in names:
            con.execute("DELETE FROM pipeline_steps WHERE step=?", (n,))
    log.info("reset %s", ", ".join(names))
    return names


def _recover_interrupted() -> None:
    """Rows left `running` by a process that died: resumable steps are paused, the rest failed."""
    resumable = [s.name for s in STEPS if s.resumable]
    with db.session() as con:
        if resumable:
            con.execute(f"UPDATE pipeline_steps SET status='paused', finished_at=?, message=? "
                        f"WHERE status='running' AND step IN ({','.join('?' * len(resumable))})",
                        (_now(), "interrupted by restart; continues on the next pass", *resumable))
        con.execute("UPDATE pipeline_steps SET status='failed', finished_at=?, message='interrupted' "
                    "WHERE status='running'", (_now(),))


def run_pending(only: list[str] | None = None) -> dict[str, str]:
    """Run every step that is not up to date, in dependency order. Returns step -> outcome of the last pass.

    While a resumable step is paused, another pass follows, so the work finishes without a restart.
    """
    # One pass at a time across processes (the server and `python -m server run` share the database).
    lock = open(settings.DB_PATH + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        log.info("pipeline already running in another process")
        return {}
    try:
        db.init_db()
        _recover_interrupted()
        while True:
            outcome = _pass(only)
            if "paused" not in outcome.values():
                return outcome
            log.info("pass ended with a paused step; continuing")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _pass(only: list[str] | None) -> dict[str, str]:
    outcome: dict[str, str] = {}
    for s in STEPS:
        if only and s.name not in only:
            continue
        if s.name in settings.SKIP_STEPS:
            outcome[s.name] = "skipped (PIPELINE_SKIP)"
            continue
        blocked = [d for d in s.deps if outcome.get(d, "").split(" ")[0] in ("failed", "blocked", "skipped", "paused")
                   or _latest_done_safe(d) is None]
        if blocked:
            outcome[s.name] = f"blocked by {', '.join(blocked)}"
            log.warning("%s: blocked by %s", s.name, ", ".join(blocked))
            continue
        p = s.plan()
        if p.skip:
            outcome[s.name] = f"skipped ({p.skip})"
            log.info("%s: skipped (%s)", s.name, p.skip)
            continue
        outcome[s.name] = _run_one(s, p)
    return outcome


def _latest_done_safe(name: str) -> int | None:
    con = db.connect()
    try:
        return _latest_done(con, name)
    finally:
        con.close()


def _run_one(s: Step, p: Plan) -> str:
    con = db.connect()
    try:
        fp = _fingerprint(con, s, p.params)
        if con.execute("SELECT 1 FROM pipeline_steps WHERE step=? AND fingerprint=? AND status='done'",
                       (s.name, fp)).fetchone():
            log.info("%s: up to date", s.name)
            return "up_to_date"
        cur = con.execute("INSERT INTO pipeline_steps(step, fingerprint, params, status, started_at) "
                          "VALUES (?,?,?,'running',?)", (s.name, fp, json.dumps(p.params, default=str), _now()))
        con.commit()
        row_id, t0 = cur.lastrowid, time.monotonic()
        log.info("%s: running", s.name)
        hit = [False]

        def should_stop() -> bool:
            if time.monotonic() >= t0 + settings.PIPELINE_SLICE_S:
                hit[0] = True
            return hit[0]

        try:
            rows = s.run(should_stop) if s.resumable else s.run()
            status_, msg = ("paused", "time slice used; continues on the next pass") if hit[0] else ("done", None)
        except Exception as exc:  # recorded, and the pass continues with independent steps
            log.exception("%s failed", s.name)
            rows, status_, msg = None, "failed", f"{type(exc).__name__}: {exc}"[:500]
        con.execute("UPDATE pipeline_steps SET status=?, finished_at=?, duration_s=?, row_count=?, message=? WHERE id=?",
                    (status_, _now(), round(time.monotonic() - t0, 2), rows, msg, row_id))
        con.commit()
        log.info("%s: %s in %.1fs", s.name, status_, time.monotonic() - t0)
        return status_
    finally:
        con.close()
