from __future__ import annotations

import io
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import EU27, NAMES
from .fires import AQUA_START, ROW_END, ROW_START, VIIRS_START, load_detections

ISO = list(EU27)
SENSOR_CODE = {"T": 0, "A": 1, "V": 2}
SENSOR_NAME = ["MODIS Terra", "MODIS Aqua", "VIIRS S-NPP"]
BASE_YEARS = (2001, 2020)
FIRST_YEAR = 2001
MAX_CSV_ROWS = 1_000_000


def _conf_code(conf: pd.Series, sensor: pd.Series) -> np.ndarray:
    num = pd.to_numeric(conf, errors="coerce")
    letter = conf.astype(str).str[0].str.lower().map({"n": -1, "h": -2, "l": -3})
    return np.where(sensor == "V", letter.fillna(-1), num.fillna(-1)).astype(np.int8)


def build_store(out: Path, verbose: bool = True) -> int:
    parts = []
    for k, iso2 in enumerate(ISO):
        d = load_detections(iso2, extra=True)
        parts.append(pd.DataFrame({
            "lat": d["lat"].astype(np.float32), "lon": d["lon"].astype(np.float32),
            "day": (d["date"] - ROW_START).dt.days.astype(np.int16), "time": d["time"].fillna(0).astype(np.int16),
            "sensor": d["sensor"].map(SENSOR_CODE).astype(np.uint8), "conf": _conf_code(d["confidence"], d["sensor"]),
            "cy": d["cy"].astype(np.int16), "cx": d["cx"].astype(np.int16),
            "frp": d["frp"].astype(np.float32), "night": (d["daynight"].astype(str) == "N").astype(np.uint8),
            "country": np.full(len(d), k, np.uint8)}))
        if verbose:
            print(f"{iso2}: {len(d):,}", flush=True)
    df = pd.concat(parts, ignore_index=True).sort_values(["day", "lat", "lon"], kind="stable")
    np.savez_compressed(out / "detections.npz", **{c: df[c].to_numpy() for c in df.columns})
    return len(df)


def _month_end(d: date) -> date:
    return (date(d.year + (d.month == 12), d.month % 12 + 1, 1) - timedelta(days=1))


def _same(d: date, year: int) -> date:
    try:
        return d.replace(year=year)
    except ValueError:
        return d.replace(year=year, day=28)


class AreaData:
    def __init__(self, folder: Path, events: pd.DataFrame, meta: dict):
        z = np.load(folder / "detections.npz")
        self.d = {k: z[k] for k in z.files}
        self.cy = self.d["cy"].astype(np.int32)
        self.cx = self.d["cx"].astype(np.int32)
        self.start = ROW_START.date()
        self.end = ROW_END.date()
        self.n = (self.end - self.start).days + 1
        days = pd.date_range(ROW_START, ROW_END, freq="D")
        self.month_of_day = days.month.to_numpy() - 1
        self.pre_aqua = days < AQUA_START
        self.viirs_from = (VIIRS_START - ROW_START).days
        self.ratio = np.array([meta["ratio"][iso2] for iso2 in ISO])
        self.terra = np.array(meta["terra_factor"])
        self.validation = meta.get("validation", {})
        self.events = events

    def idx(self, d: date) -> int:
        return (d - self.start).days

    def select(self, bbox: tuple[float, float, float, float] | None, country: str | None) -> np.ndarray:
        if country:
            return self.d["country"] == ISO.index(country)
        w, s, e, n = bbox
        lat, lon = self.d["lat"], self.d["lon"]
        return (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)

    def series(self, mask: np.ndarray) -> dict[str, np.ndarray]:
        day = self.d["day"][mask].astype(np.int64)
        sensor = self.d["sensor"][mask]
        country = self.d["country"][mask]
        viirs = (sensor == 2).astype(np.int64)
        key = (((viirs << 15) | (self.cy[mask].astype(np.int64) + 9000)) << 17 | (self.cx[mask].astype(np.int64) + 18000)) << 14 | day
        _, first = np.unique(key, return_index=True)
        cd_day, cd_viirs, cd_country = day[first], viirs[first].astype(bool), country[first]
        modis_cd = np.bincount(cd_day[~cd_viirs], minlength=self.n)[:self.n].astype(float)
        viirs_cd = np.bincount(cd_day[cd_viirs], minlength=self.n)[:self.n].astype(float)
        md = cd_day[~cd_viirs]
        mi = self.month_of_day[md]
        weight = self.ratio[cd_country[~cd_viirs], mi] * np.where(self.pre_aqua[md], self.terra[mi], 1.0)
        modis_h = np.bincount(md, weights=weight, minlength=self.n)[:self.n]
        harm = np.where(np.arange(self.n) < self.viirs_from, modis_h, viirs_cd)
        return {"harmonized": harm, "modis": modis_cd, "viirs": viirs_cd}

    def window(self, arr: np.ndarray, a: date, b: date) -> float:
        i, j = max(0, self.idx(a)), min(self.n - 1, self.idx(b))
        return float(arr[i:j + 1].sum()) if j >= i else 0.0

    def calendar(self, s: dict, d: date) -> list[dict]:
        out = []
        for y in range(FIRST_YEAR, d.year + 1):
            for m in range(1, 13):
                a = date(y, m, 1)
                if a > d:
                    out.append({"year": y, "month": m, "harmonized": None, "raw": None, "modelled": y < 2012, "partial": False})
                    continue
                b = min(_month_end(a), d)
                raw = s["modis"] if y < 2012 else s["viirs"]
                out.append({"year": y, "month": m, "harmonized": round(self.window(s["harmonized"], a, b), 1),
                            "raw": round(self.window(raw, a, b), 1), "modelled": y < 2012, "partial": b < _month_end(a)})
        return out

    def normal(self, arr: np.ndarray, d: date) -> dict:
        a = d.replace(day=1)
        now = self.window(arr, a, d)
        years = list(range(FIRST_YEAR, d.year + 1))
        whole = {y: self.window(arr, _same(a, y), _month_end(_same(a, y))) for y in years if y != d.year}
        full = [v for y, v in whole.items() if BASE_YEARS[0] <= y <= BASE_YEARS[1]]
        med = float(np.median(full)) if full else 0.0
        return {"year": d.year, "month": d.month, "value": round(now), "median": round(med), "ratio": round(now / max(med, 1.0), 1),
                "rank": 1 + sum(1 for v in whole.values() if v > now), "years": len(years), "since": years[0],
                "usual_low": round(float(np.percentile(full, 10))) if full else 0,
                "usual_high": round(float(np.percentile(full, 90))) if full else 0}

    def range_normal(self, arr: np.ndarray, a: date, b: date) -> dict | None:
        if (b - a).days > 365:
            return None
        now = self.window(arr, a, b)
        span = b - a
        past = {}
        for y in range(FIRST_YEAR, b.year + 1):
            shift = y - b.year
            pa = _same(a, a.year + shift)
            pb = pa + span
            if y == b.year or pa < self.start:
                continue
            past[y] = self.window(arr, pa, pb)
        base = [v for y, v in past.items() if BASE_YEARS[0] <= y <= BASE_YEARS[1]]
        med = float(np.median(base)) if base else 0.0
        mean = float(np.mean(base)) if base else 0.0
        sd = float(np.std(base)) if base else 0.0
        return {"value": round(now, 1), "median": round(med, 1), "mean": round(mean, 1),
                "ratio": round(now / med, 2) if med > 0 else None,
                "anomaly_pct": round((now - mean) / mean * 100) if mean > 0 else None,
                "z_score": round((now - mean) / sd, 2) if sd > 0 else None,
                "rank": 1 + sum(1 for v in past.values() if v > now), "years": len(past) + 1,
                "by_year": [{"year": y, "value": round(v, 1)} for y, v in sorted(past.items())] + [{"year": b.year, "value": round(now, 1)}]}

    def outlook(self, arr: np.ndarray, d: date, weeks: int = 6) -> dict:
        shares, weekly = [], []
        for y in range(FIRST_YEAR, d.year):
            a = _same(d, y)
            total = self.window(arr, date(y, 1, 1), date(y, 12, 31))
            if total <= 0:
                continue
            shares.append(self.window(arr, a, a + timedelta(days=7 * weeks - 1)) / total)
            weekly += [self.window(arr, a + timedelta(days=7 * k), a + timedelta(days=7 * k + 6)) for k in range(weeks)]
        return {"weeks": weeks, "until": (d + timedelta(days=7 * weeks)).isoformat(),
                "share": round(float(np.median(shares)), 3) if shares else 0.0,
                "typical_week": round(float(np.median(weekly))) if weekly else 0,
                "bad_week": round(float(np.percentile(weekly, 90))) if weekly else 0}

    def analog(self, arr: np.ndarray, d: date, ev: pd.DataFrame) -> dict | None:
        def vec(y):
            return np.log1p([self.window(arr, date(y, m, 1), min(_month_end(date(y, m, 1)), _same(d, y))) for m in range(1, d.month + 1)])
        cur = vec(d.year)
        best = min(((float(np.linalg.norm(vec(y) - cur)), y) for y in range(FIRST_YEAR, d.year)), default=None)
        if best is None:
            return None
        y = best[1]
        top = ev[(ev["start_d"] >= date(y, 1, 1)) & (ev["start_d"] <= date(y, 12, 31))].sort_values("area_ha", ascending=False).head(1)
        return {"year": y, "largest_fire": None if top.empty else top.iloc[0]["name"]}

    def area_events(self, bbox, country) -> pd.DataFrame:
        ev = self.events
        if country:
            return ev[ev["iso2"] == country]
        w, s, e, n = bbox
        return ev[(ev["lat"] >= s) & (ev["lat"] <= n) & (ev["lon"] >= w) & (ev["lon"] <= e)]

    def query(self, bbox, country, a: date, b: date, asof: date | None = None) -> dict:
        asof = max(asof or b, b)
        mask = self.select(bbox, country)
        s = self.series(mask)
        ev = self.area_events(bbox, country)
        i, j = self.idx(a), self.idx(b)
        in_range = mask & (self.d["day"] >= i) & (self.d["day"] <= j)
        sensors = self.d["sensor"][in_range]
        frp = self.d["frp"][in_range]
        hit = ev[(ev["start_d"] <= b) & (ev["end_d"] >= a)].sort_values("detections", ascending=False)
        big = ev[(ev["start_d"] >= date(2012, 1, 1)) & (ev["start_d"] <= asof)].sort_values("area_ha", ascending=False).head(4)
        span = (b - a).days + 1
        if span <= 400:
            step = [{"date": (a + timedelta(days=k)).isoformat(), "harmonized": round(float(s["harmonized"][i + k]), 1),
                     "modis": int(s["modis"][i + k]), "viirs": int(s["viirs"][i + k])} for k in range(span)]
            unit = "day"
        else:
            step, cur = [], a.replace(day=1)
            while cur <= b:
                lo, hi = max(cur, a), min(_month_end(cur), b)
                step.append({"date": cur.isoformat(), "harmonized": round(self.window(s["harmonized"], lo, hi), 1),
                             "modis": int(self.window(s["modis"], lo, hi)), "viirs": int(self.window(s["viirs"], lo, hi))})
                cur = _month_end(cur) + timedelta(days=1)
            unit = "month"
        shares = np.bincount(self.d["country"][in_range], minlength=len(ISO))
        an = self.analog(s["harmonized"], asof, ev)
        return {
            "months": self.calendar(s, asof),
            "normal": self.normal(s["harmonized"], asof),
            "range_normal": self.range_normal(s["harmonized"], a, b),
            "outlook": self.outlook(s["harmonized"], asof),
            "analog": an,
            "largest_fires": [{"name": e.name, "start": e.start, "area_ha": int(e.area_ha), "detections": int(e.detections)} for e in big.itertuples()],
            "summary": {
                "days": span,
                "harmonized_cell_days": round(self.window(s["harmonized"], a, b), 1),
                "modis_cell_days": int(self.window(s["modis"], a, b)),
                "viirs_cell_days": int(self.window(s["viirs"], a, b)),
                "detections": int(in_range.sum()),
                "detections_by_sensor": {SENSOR_NAME[k]: int((sensors == k).sum()) for k in range(3)},
                "night_share": round(float(self.d["night"][in_range].mean()), 3) if in_range.any() else 0.0,
                "mean_frp": round(float(frp.mean()), 1) if len(frp) else 0.0,
                "max_frp": round(float(frp.max()), 1) if len(frp) else 0.0,
                "fires": int(len(hit)),
                "busiest_day": max(step, key=lambda r: r["harmonized"])["date"] if unit == "day" and step else None,
            },
            "series": {"unit": unit, "values": step},
            "fires": [{"name": e.name, "iso2": e.iso2, "start": e.start, "end": e.end, "lat": e.lat, "lon": e.lon,
                       "detections": int(e.detections), "area_ha": int(e.area_ha)} for e in hit.head(10).itertuples()],
            "countries": [{"iso2": ISO[k], "name": NAMES[ISO[k]], "detections": int(c)} for k, c in sorted(enumerate(shares), key=lambda t: -t[1]) if c > 0],
        }

    def hotspots_csv(self, bbox, country, a: date, b: date) -> tuple[str, int, bool]:
        mask = self.select(bbox, country) & (self.d["day"] >= self.idx(a)) & (self.d["day"] <= self.idx(b))
        rows = np.flatnonzero(mask)
        cut = len(rows) > MAX_CSV_ROWS
        rows = rows[:MAX_CSV_ROWS]
        day = pd.to_datetime(self.start) + pd.to_timedelta(self.d["day"][rows], unit="D")
        conf = self.d["conf"][rows]
        sensor = self.d["sensor"][rows]
        conf_text = np.where(sensor == 2, np.select([conf == -2, conf == -3], ["high", "low"], "nominal"), conf.astype(str))
        cy = self.cy[rows]
        cx = self.cx[rows]
        df = pd.DataFrame({
            "latitude": np.round(self.d["lat"][rows], 5), "longitude": np.round(self.d["lon"][rows], 5),
            "acq_date": day.strftime("%Y-%m-%d"), "acq_time": [f"{t:04d}" for t in self.d["time"][rows]],
            "sensor": np.array(SENSOR_NAME)[sensor], "confidence": conf_text, "frp_mw": np.round(self.d["frp"][rows], 1),
            "daynight": np.where(self.d["night"][rows] == 1, "N", "D"),
            "country": np.array(ISO)[self.d["country"][rows]],
            "cell_id": [f"{y}_{x}" for y, x in zip(cy, cx)],
            "harmonized_series": np.where(sensor == 2, "VIIRS", "MODIS"),
        })
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue(), len(rows), cut


def bbox_area_km2(bbox) -> float:
    w, s, e, n = bbox
    r = 6371.0
    return abs(math.radians(e - w) * (math.sin(math.radians(n)) - math.sin(math.radians(s)))) * r * r
