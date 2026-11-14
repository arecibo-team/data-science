"""Fetchers for every data source, each cached under data/raw/ and never re-downloaded unless refresh=True."""
from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import EU27, ISO3_TO_ISO2, MANUAL_DIR, NAME_TO_ISO2, RAW_DIR, SOURCES

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
GENERIC_UA = "amber-index/0.1 (hackathon research script)"


def _get(url: str, *, ua: str = BROWSER_UA, tries: int = 4, timeout: int = 120, **kw) -> requests.Response:
    """GET/POST with exponential backoff on network errors and 429/5xx."""
    method = kw.pop("method", "GET")
    wait = 2.0
    for attempt in range(tries):
        try:
            r = requests.request(method, url, headers={"User-Agent": ua}, timeout=timeout, **kw)
            if r.status_code < 400:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
        except requests.RequestException:
            if attempt == tries - 1:
                raise
        time.sleep(wait)
        wait *= 2
    raise RuntimeError(f"giving up on {url}")


def _cached_bytes(path: Path, url: str, refresh: bool, **kw) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return path.read_bytes()
    data = _get(url, **kw).content
    path.write_bytes(data)
    return data


# ---------------------------------------------------------------- Eurostat
def decode_jsonstat(js: dict) -> pd.DataFrame:
    """Flatten a JSON-stat 2.0 dataset into rows [geo, year, value].

    Values are stored by flat position; position = sum(index_d * stride_d) over dimensions in js["id"] order.
    All dimensions except geo and time must have a single selected category.
    """
    ids, sizes = js["id"], js["size"]
    strides = np.cumprod([1] + sizes[::-1])[:-1][::-1]
    cats = {d: js["dimension"][d]["category"]["index"] for d in ids}
    geo_pos = {v: k for k, v in cats["geo"].items()}
    time_pos = {v: k for k, v in cats["time"].items()}
    gi, ti = ids.index("geo"), ids.index("time")
    rows = []
    for flat, value in js["value"].items():
        flat = int(flat)
        idx = [(flat // strides[i]) % sizes[i] for i in range(len(ids))]
        rows.append((geo_pos[idx[gi]], int(time_pos[idx[ti]]), float(value)))
    df = pd.DataFrame(rows, columns=["geo", "year", "value"])
    df["geo"] = df["geo"].replace({"EL": "GR"})
    return df


def fetch_eurostat(dataset: str, params: str, refresh: bool = False) -> pd.DataFrame:
    """Download one Eurostat dataset (JSON-stat) and return [geo, year, value], EU-27 countries only."""
    url = SOURCES["eurostat_api"] + dataset + ("?" + params if params else "")
    raw = _cached_bytes(RAW_DIR / f"eurostat_{dataset}.json", url, refresh)
    df = decode_jsonstat(json.loads(raw))
    return df[df["geo"].isin(EU27)].sort_values(["geo", "year"]).reset_index(drop=True)


def _split(source_key: str) -> tuple[str, str]:
    ds, _, params = SOURCES[source_key].partition("?")
    return ds, params


def load_spending(refresh: bool = False) -> pd.DataFrame:
    return fetch_eurostat(*_split("spending"), refresh=refresh).rename(columns={"value": "spending_meur"})


def load_population(refresh: bool = False) -> pd.DataFrame:
    return fetch_eurostat(*_split("population"), refresh=refresh).rename(columns={"value": "population"})


def load_area(refresh: bool = False) -> pd.DataFrame:
    return fetch_eurostat(*_split("area"), refresh=refresh).rename(columns={"value": "area_km2"})


# ---------------------------------------------------------------- firefighters (LFS ad hoc extractions)
def firefighter_file_urls(refresh: bool = False) -> list[str]:
    """The 2024 file plus every .xlsx linked from the Eurostat news pages."""
    urls = [SOURCES["firefighters_2024"]]
    for i, page in enumerate(SOURCES["firefighters_news"]):
        html = _cached_bytes(RAW_DIR / f"lfs_news_{i}.html", page, refresh).decode("utf-8", "replace")
        urls += sorted(set(re.findall(r'href="([^"]+\.xlsx[^"]*)"', html)))
    return list(dict.fromkeys(urls))


def _parse_firefighter_xlsx(content: bytes, url: str) -> pd.DataFrame:
    """Two layouts exist: a 'Share by country' table with country names (2023, 2024 files) and a long
    table with COUNTRY/YEAR/ISCO08_4D/THS_POP columns (2022 data). Values are thousands of persons."""
    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    for name, sh in sheets.items():
        for hdr in range(min(5, len(sh))):
            cols = [str(c).strip() for c in sh.iloc[hdr].tolist()]
            if "COUNTRY" in cols and "ISCO08_4D" in cols:  # long layout
                df = sh.iloc[hdr + 1:].set_axis(cols, axis=1)
                df = df[df["ISCO08_4D"].astype(str) == "5411"]
                out = pd.DataFrame({"iso2": df["COUNTRY"].replace({"EL": "GR"}), "year": df["YEAR"].astype(int),
                                    "ths": pd.to_numeric(df["THS_POP"], errors="coerce"), "flag": df["FLAG_RELIAB"]})
                return out
            if "Country" in cols and any("Number of firefighters" in c for c in cols):  # wide layout
                df = sh.iloc[hdr + 1:].set_axis(cols, axis=1)
                num = next(c for c in cols if "Number of firefighters" in c)
                m = re.search(r"(20\d\d)", url.split("?")[0].rsplit("/", 2)[-2])
                year = int(m.group(1)) if m else None
                out = pd.DataFrame({"iso2": df["Country"].map(NAME_TO_ISO2), "year": year,
                                    "ths": pd.to_numeric(df[num], errors="coerce"), "flag": df.get("FLAG_RELIAB")})
                return out
    raise ValueError(f"no firefighter table found in {url}")


def load_firefighters(refresh: bool = False) -> pd.DataFrame:
    """Observed firefighters per country-year: [iso2, year, firefighters, low_reliability].
    'x' (unpublishable) -> missing; 'u' (low reliability) -> kept and flagged."""
    frames = []
    for i, url in enumerate(firefighter_file_urls(refresh)):
        content = _cached_bytes(RAW_DIR / f"lfs_firefighters_{i}.xlsx", url, refresh)
        frames.append(_parse_firefighter_xlsx(content, url))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["iso2"].isin(EU27)].copy()
    df["flag"] = df["flag"].astype("string").str.strip().str.lower()
    df.loc[df["flag"] == "x", "ths"] = np.nan
    df["firefighters"] = df["ths"] * 1000
    df["low_reliability"] = (df["flag"] == "u").fillna(False).astype(bool)
    df = df.dropna(subset=["firefighters"])
    return (df.sort_values("year").drop_duplicates(["iso2", "year"], keep="last")
              [["iso2", "year", "firefighters", "low_reliability"]].reset_index(drop=True))


# ---------------------------------------------------------------- burned area (OWID / GWIS)
def load_burned_area(refresh: bool = False) -> pd.DataFrame:
    raw = _cached_bytes(RAW_DIR / "owid_burned_area_gwis.csv", SOURCES["burned_area"], refresh)
    df = pd.read_csv(io.BytesIO(raw))
    df = df[df["code"].isin(ISO3_TO_ISO2)]
    return pd.DataFrame({"iso2": df["code"].map(ISO3_TO_ISO2), "year": df["year"].astype(int),
                         "burned_ha": df["all"].astype(float)}).reset_index(drop=True)


# ---------------------------------------------------------------- fire stations (OpenStreetMap / Overpass)
def _overpass_count(iso2: str) -> int:
    q = (f'[out:json][timeout:240];area["ISO3166-1"="{iso2}"][admin_level=2]->.a;'
         '(node["amenity"="fire_station"](area.a);way["amenity"="fire_station"](area.a);'
         'relation["amenity"="fire_station"](area.a););out count;')
    last = None
    for i, mirror in enumerate(SOURCES["overpass"]):
        try:  # the main server gets patience, mirrors get one quick try
            r = _get(mirror, ua=GENERIC_UA, method="POST", data={"data": q},
                     tries=3 if i == 0 else 1, timeout=260 if i == 0 else 90)
            return int(r.json()["elements"][0]["tags"]["total"])
        except Exception as e:
            last = e
    raise RuntimeError(f"Overpass failed for {iso2}: {last}")


def load_osm_stations(refresh: bool = False, verbose: bool = True, budget_s: float | None = None,
                      retry_missing: bool = False) -> pd.DataFrame:
    """Current snapshot of fire stations per country, cached to data/raw/osm_stations.csv.
    Progress is saved per country (osm_stations.partial.csv) so an interrupted run resumes.
    budget_s caps the total time; countries not reached or failed stay missing (NaN) and are imputed later.
    retry_missing=True re-queries only the countries that are still missing in the cache."""
    path, partial = RAW_DIR / "osm_stations.csv", RAW_DIR / "osm_stations.partial.csv"
    if path.exists() and not refresh and not retry_missing:
        return pd.read_csv(path)
    if retry_missing and path.exists():
        pd.read_csv(path).dropna(subset=["stations"]).to_csv(partial, index=False)
    start = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(partial) if partial.exists() and not refresh else pd.DataFrame(columns=["iso2", "stations", "fetched"])
    for iso2 in EU27:
        if iso2 in set(done["iso2"]) and pd.notna(done.loc[done.iso2 == iso2, "stations"]).any():
            continue
        if budget_s is not None and time.monotonic() - start > budget_s:
            if verbose:
                print(f"  {iso2}: skipped (time budget used up)", flush=True)
            continue
        try:
            n = _overpass_count(iso2)
        except Exception as e:
            n = np.nan
            if verbose:
                print(f"  {iso2}: failed ({str(e)[:80]})", flush=True)
        if verbose and pd.notna(n):
            print(f"  {iso2}: {n}", flush=True)
        done = pd.concat([done[done.iso2 != iso2], pd.DataFrame([{"iso2": iso2, "stations": n,
                          "fetched": pd.Timestamp.now(tz="UTC").date().isoformat()}])], ignore_index=True)
        done.to_csv(partial, index=False)
        time.sleep(2.0)
    done = done.set_index("iso2").reindex(list(EU27)).reset_index()
    done.to_csv(path, index=False)
    return done


ALERT_WEIGHT = {"Green": 0.5, "Orange": 3.0, "Red": 9.0}
HAZARD_TYPES = {"FL": "flood", "EQ": "earthquake", "WF": "wildfire", "DR": "drought", "TC": "tropical cyclone", "VO": "volcano"}


GDACS_NAMES = {"CZ": "Czech Republic"}


def _gdacs_pages(query: str, start: str, end: str) -> list[dict]:
    events, page = [], 1
    while True:
        params = {"eventlist": ";".join(HAZARD_TYPES), "fromdate": start, "todate": end, "country": query,
                  "alertlevel": "Green;Orange;Red", "pagesize": 100, "pagenumber": page}
        r = _get(SOURCES["hazards"], params=params, tries=4, timeout=120)
        feats = r.json().get("features", []) if r.status_code == 200 and r.text.strip() else []
        events += [f["properties"] for f in feats]
        if len(feats) < 100:
            return events
        page += 1


def _affected(e: dict) -> set[str]:
    codes = {c.get("iso3") for c in (e.get("affectedcountries") or []) if c.get("iso3")}
    return codes | ({e["iso3"]} if e.get("iso3") else set())


def load_hazard_events(start_year: int, end_year: int, refresh: bool = False, verbose: bool = False) -> pd.DataFrame:
    folder = RAW_DIR / "gdacs"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for iso2, (iso3, name) in EU27.items():
        path = folder / f"{iso3}.json"
        if path.exists() and not refresh:
            events = json.loads(path.read_text())
        else:
            found = {}
            for q in dict.fromkeys([GDACS_NAMES.get(iso2, name), iso3]):
                for e in _gdacs_pages(q, f"{start_year}-01-01", f"{end_year}-12-31"):
                    if iso3 in _affected(e):
                        found[(e.get("eventtype"), e.get("eventid"), e.get("episodeid"))] = e
            events = [{k: e.get(k) for k in ("eventid", "episodeid", "eventtype", "alertlevel", "fromdate", "todate", "name")}
                      for e in found.values()]
            path.write_text(json.dumps(events))
            if verbose:
                print(f"  {iso2}: {len(events)} events", flush=True)
            time.sleep(0.5)
        rows += [{"iso2": iso2, **e} for e in events]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["iso2", "eventid", "eventtype", "alertlevel", "start", "end", "name"])
    df["alertlevel"] = df["alertlevel"].astype(str).str.capitalize()
    df = df.sort_values("episodeid").drop_duplicates(["iso2", "eventtype", "eventid"], keep="last")
    df["start"] = pd.to_datetime(df["fromdate"], errors="coerce")
    df["end"] = pd.to_datetime(df["todate"], errors="coerce").fillna(df["start"])
    return df[["iso2", "eventid", "eventtype", "alertlevel", "start", "end", "name"]].reset_index(drop=True)


def hazard_load(events: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    rows = []
    for e in events.itertuples():
        if pd.isna(e.start):
            continue
        for y in range(max(e.start.year, years[0]), min(e.end.year, years[-1]) + 1):
            rows.append((e.iso2, y, e.eventtype, ALERT_WEIGHT.get(e.alertlevel, 1.0), e.alertlevel in ("Orange", "Red")))
    df = pd.DataFrame(rows, columns=["iso2", "year", "eventtype", "weight", "major"])
    df["major_weight"] = df["weight"].where(df["major"], 0.0)
    grid = pd.MultiIndex.from_product([list(EU27), years], names=["iso2", "year"]).to_frame(index=False)
    load = df.groupby(["iso2", "year"]).agg(hazard_load=("weight", "sum"), hazard_events=("weight", "size"),
                                            major_load=("major_weight", "sum"), major_events=("major", "sum")).reset_index()
    by_type = df.pivot_table(index=["iso2", "year"], columns="eventtype", values="weight", aggfunc="size", fill_value=0)
    by_type.columns = [f"events_{HAZARD_TYPES.get(c, c).replace(' ', '_')}" for c in by_type.columns]
    out = grid.merge(load, on=["iso2", "year"], how="left").merge(by_type.reset_index(), on=["iso2", "year"], how="left")
    return out.fillna(0)


# ---------------------------------------------------------------- manual + boundaries
def load_volunteers() -> pd.DataFrame:
    """Optional manual file data/manual/volunteers.csv (iso2, year, volunteers); created with headers if absent."""
    path = MANUAL_DIR / "volunteers.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("iso2,year,volunteers\n")
    return pd.read_csv(path, dtype={"iso2": str})


def load_boundaries(refresh: bool = False):
    import geopandas as gpd
    raw = _cached_bytes(RAW_DIR / "gisco_countries_20m.geojson", SOURCES["boundaries"], refresh)
    gdf = gpd.read_file(io.BytesIO(raw))
    gdf["iso2"] = gdf["CNTR_ID"].replace({"EL": "GR"})
    return gdf[gdf["iso2"].isin(EU27)][["iso2", "geometry"]]
