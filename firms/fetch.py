"""Download FIRMS hotspot CSVs (MODIS + VIIRS) for a bbox and date range.

Each request covers <=5 days for one FIRMS source. Output is one raw CSV per
(source, window) under data/raw/<SOURCE>/<YYYYMMDD>_<days>.csv. Existing files are
skipped, so runs are resumable.

Example:
    python -m firms.fetch --start 2025-08-01 --end 2025-08-31 --bbox 15,-20,35,-5

MAP_KEYs (KEY in .env, comma-separated) are used round-robin. A key that hits its
transaction limit is cooled down for 10 minutes while the others keep working.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from . import config as C

log = logging.getLogger("firms.fetch")


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Request:
    source: str
    start: dt.date
    days: int

    @property
    def path(self) -> Path:
        return C.RAW_DIR / self.source / f"{self.start:%Y%m%d}_{self.days}.csv"


def parse_bbox(text: str) -> str:
    """Validate 'W,S,E,N' or 'world' and return the normalized string for the URL."""
    if text.lower() == "world":
        return "world"
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be W,S,E,N or 'world'")
    w, s, e, n = map(float, parts)
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise argparse.ArgumentTypeError("bbox out of range or not W<E and S<N")
    return f"{w},{s},{e},{n}"


def _mask(text: str, keys: list[str]) -> str:
    for k in keys:
        text = text.replace(k, "***")
    return text[:200]


COOLDOWN_S = 600  # FIRMS limits transactions per key per 10-minute window


class KeyPool:
    """Round-robin over MAP_KEYs. A key that hits its transaction limit cools down for
    COOLDOWN_S; a key FIRMS rejects as invalid is removed. Keys are never logged."""

    def __init__(self, keys: list[str]):
        self.keys = list(keys)
        self.cooldown_until: dict[str, float] = {}
        self.dead: set[str] = set()
        self._next = 0
        self.uses: dict[str, int] = {k: 0 for k in self.keys}

    def _usable(self) -> list[str]:
        return [k for k in self.keys if k not in self.dead]

    def acquire(self) -> str:
        usable = self._usable()
        if not usable:
            raise FetchError("all MAP_KEYs were rejected as invalid")
        while True:
            now = time.monotonic()
            n = len(usable)
            for i in range(n):
                key = usable[(self._next + i) % n]
                if self.cooldown_until.get(key, 0.0) <= now:
                    self._next = (self._next + i + 1) % n
                    self.uses[key] += 1
                    return key
            wait = min(self.cooldown_until[k] for k in usable) - now
            log.info("all keys at transaction limit; waiting %.0fs", wait)
            time.sleep(max(1.0, wait))

    def cool_down(self, key: str) -> None:
        self.cooldown_until[key] = time.monotonic() + COOLDOWN_S
        log.warning("key #%d hit transaction limit; cooling down %ds", self.keys.index(key) + 1, COOLDOWN_S)

    def kill(self, key: str) -> None:
        self.dead.add(key)
        log.error("key #%d rejected as invalid; removed from rotation", self.keys.index(key) + 1)


class FirmsClient:
    """HTTP client that load-balances requests across MAP_KEYs."""

    def __init__(self, keys: list[str], session: requests.Session | None = None):
        self.pool = KeyPool(keys)
        self.session = session or requests.Session()

    def _get(self, path_after_key: str, max_attempts: int = 30) -> tuple[int, str]:
        last = ""
        for attempt in range(max_attempts):
            key = self.pool.acquire()
            url = f"{C.API_BASE}/{path_after_key.format(key=key)}"
            try:
                r = self.session.get(url, timeout=120)
            except requests.RequestException as exc:
                log.warning("network error (%s); retrying", type(exc).__name__)
                time.sleep(min(60, 2 ** min(attempt, 6)))
                continue
            text = r.text
            low = text.lower()
            if r.status_code == 200 and not low.startswith(("invalid", "transaction", "unknown", "error", "exceeding")):
                return r.status_code, text
            last = f"HTTP {r.status_code}: {_mask(text, self.pool.keys)[:120]}"
            if "transaction limit" in low or "exceeding allowed" in low or r.status_code == 429:
                self.pool.cool_down(key)
                continue
            if "invalid" in low and "key" in low:
                self.pool.kill(key)
                continue
            if 400 <= r.status_code < 500:
                # Bad day range, bad source, etc. Retrying will not help.
                raise FetchError(f"FIRMS rejected request ({r.status_code}): {_mask(text, self.pool.keys)[:200]}")
            log.warning("FIRMS server error, retrying: %s", last)
            time.sleep(min(60, 2 ** min(attempt, 6)))
        raise FetchError(f"FIRMS request failed after {max_attempts} attempts: {last}")

    def availability(self, source: str) -> tuple[dt.date, dt.date] | None:
        """Return (min_date, max_date) for a source, or None if the source is not offered."""
        try:
            _, text = self._get("data_availability/csv/{key}/" + source)
        except FetchError:
            return None
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            return None
        return dt.date.fromisoformat(rows[0]["min_date"]), dt.date.fromisoformat(rows[0]["max_date"])

    def area_csv(self, source: str, area: str, start: dt.date, days: int) -> str:
        _, text = self._get(f"area/csv/{{key}}/{source}/{area}/{days}/{start.isoformat()}")
        if not text.startswith("latitude"):
            raise FetchError(f"unexpected response for {source} {start}: {text[:120]!r}")
        return text


def plan_requests(
    start: dt.date,
    end: dt.date,
    families: list[str],
    availability: dict[str, tuple[dt.date, dt.date] | None],
) -> list[Request]:
    """Pick SP for days inside an SP archive and NRT otherwise, then chunk into <=10-day requests.

    `availability` maps FIRMS source id -> (min, max) or None. Days with no coverage are skipped.
    """
    requests_out: list[Request] = []
    for fam_name in families:
        fam = C.FAMILIES[fam_name]
        sp_range = availability.get(fam["sp"]) if fam["sp"] else None
        nrt_range = availability.get(fam["nrt"])

        # Assign each day to a source, then merge consecutive days into chunks.
        run_source: str | None = None
        run_start: dt.date | None = None
        run_len = 0
        day = start
        while day <= end + dt.timedelta(days=1):  # sentinel day flushes the last run
            src = None
            if day <= end:
                if sp_range and sp_range[0] <= day <= sp_range[1]:
                    src = fam["sp"]
                elif nrt_range and nrt_range[0] <= day <= nrt_range[1]:
                    src = fam["nrt"]
            if src != run_source or run_len == C.MAX_DAY_RANGE:
                if run_source and run_start:
                    requests_out.append(Request(run_source, run_start, run_len))
                run_source, run_start, run_len = src, day, 0
            if src is not None:
                run_len += 1
            day += dt.timedelta(days=1)
    return requests_out


def fetch_all(
    start: dt.date,
    end: dt.date,
    bbox: str,
    families: list[str],
    client: FirmsClient,
) -> list[Path]:
    avail = {}
    for fam_name in families:
        fam = C.FAMILIES[fam_name]
        for src in filter(None, [fam["sp"], fam["nrt"]]):
            avail[src] = client.availability(src)
            log.info("availability %-16s %s", src, avail[src])

    plan = plan_requests(start, end, families, avail)
    log.info("%d requests planned", len(plan))
    written: list[Path] = []
    for req in plan:
        if req.path.exists():
            written.append(req.path)
            continue
        req.path.parent.mkdir(parents=True, exist_ok=True)
        text = client.area_csv(req.source, bbox, req.start, req.days)
        req.path.write_text(text, encoding="utf-8")
        rows = max(0, text.count("\n") - 1)
        log.info("%-16s %s +%dd -> %d rows", req.source, req.start, req.days, rows)
        written.append(req.path)
    log.info("requests per key: %s", {f"#{i+1}": client.pool.uses[k] for i, k in enumerate(client.pool.keys)})
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=dt.date.fromisoformat, required=True)
    p.add_argument("--end", type=dt.date.fromisoformat, required=True)
    p.add_argument("--bbox", type=parse_bbox, required=True, help="W,S,E,N or 'world'")
    p.add_argument("--families", default=",".join(C.FAMILIES),
                   help=f"comma-separated subset of {','.join(C.FAMILIES)} (default: all)")
    args = p.parse_args(argv)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = set(families) - set(C.FAMILIES)
    if unknown:
        p.error(f"unknown families: {sorted(unknown)}")
    if args.end < args.start:
        p.error("--end must be on or after --start")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = FirmsClient(C.load_map_keys())
    fetch_all(args.start, args.end, args.bbox, families, client)


if __name__ == "__main__":
    main()
