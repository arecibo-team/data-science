from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from . import EU27
from .data import GDACS_NAMES, _affected, _gdacs_pages


def fetch_active_events(days: int = 14, now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now(timezone.utc)
    start = (now - timedelta(days=days)).date().isoformat()
    end = now.date().isoformat()
    rows, seen = [], set()
    for iso2, (iso3, name) in EU27.items():
        for q in dict.fromkeys([GDACS_NAMES.get(iso2, name), iso3]):
            for e in _gdacs_pages(q, start, end):
                key = (iso2, e.get("eventtype"), e.get("eventid"))
                if iso3 not in _affected(e) or key in seen:
                    continue
                seen.add(key)
                rows.append({"iso2": iso2, "eventid": e.get("eventid"), "eventtype": e.get("eventtype"),
                             "alertlevel": str(e.get("alertlevel", "Green")).capitalize(), "name": e.get("name"),
                             "start": e.get("fromdate"), "end": e.get("todate"),
                             "url": (e.get("url") or {}).get("report") if isinstance(e.get("url"), dict) else None})
    df = pd.DataFrame(rows, columns=["iso2", "eventid", "eventtype", "alertlevel", "name", "start", "end", "url"])
    if df.empty:
        return df
    df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    df["end"] = pd.to_datetime(df["end"], errors="coerce", utc=True).fillna(df["start"])
    return df[df["end"] >= pd.Timestamp(now) - pd.Timedelta(days=days)].reset_index(drop=True)


def active_shocks(events: pd.DataFrame) -> dict[str, list[tuple[str, str, int]]]:
    shocks: dict[str, list[tuple[str, str, int]]] = {}
    for e in events.itertuples():
        shocks.setdefault(e.iso2, []).append((e.eventtype, e.alertlevel, 1))
    return shocks
