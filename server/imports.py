from __future__ import annotations

import hashlib
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from firms import config as C

log = logging.getLogger("server.imports")

REQUIRED = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "daynight"]
VIIRS_FAMILY = {"N": "VIIRS_SNPP", "SNPP": "VIIRS_SNPP", "1": "VIIRS_NOAA20", "N20": "VIIRS_NOAA20", "NOAA-20": "VIIRS_NOAA20",
                "N21": "VIIRS_NOAA21", "NOAA-21": "VIIRS_NOAA21"}
COUNTRY_URL = "https://firms.modaps.eosdis.nasa.gov/data/country/{folder}/{year}/{folder}_{year}_{name}.csv"
EU27 = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech_Republic", "Denmark", "Estonia", "Finland",
        "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
        "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"]
ARCHIVE_FOLDERS = {"modis": "modis", "viirs": "viirs-snpp"}


class ImportError_(ValueError):
    pass


def _sensor(cols: set[str]) -> str:
    if {"brightness", "bright_t31"} <= cols:
        return "MODIS"
    if {"bright_ti4", "bright_ti5"} <= cols:
        return "VIIRS"
    raise ImportError_("not a FIRMS MODIS or VIIRS file: expected brightness/bright_t31 or bright_ti4/bright_ti5 columns")


def split_firms_csv(text: str) -> dict[str, pd.DataFrame]:
    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str)
    except Exception as exc:
        raise ImportError_(f"could not read the CSV: {exc}") from exc
    raw.columns = [c.strip().lower() for c in raw.columns]
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ImportError_(f"missing FIRMS columns: {', '.join(missing)}")
    if raw.empty:
        return {}
    sensor = _sensor(set(raw.columns))
    for col in ("scan", "track"):
        if col not in raw.columns:
            raw[col] = ""
    if "version" not in raw.columns:
        raw["version"] = ""
    nrt = raw["version"].fillna("").str.upper().str.contains("NRT")
    sat = raw["satellite"].fillna("").str.strip()
    if sensor == "MODIS":
        family = pd.Series("MODIS", index=raw.index)
    else:
        family = sat.map(VIIRS_FAMILY)
        if family.isna().any():
            bad = sorted(sat[family.isna()].unique().tolist())
            raise ImportError_(f"unknown VIIRS satellite codes: {', '.join(bad[:5])}")
    out: dict[str, pd.DataFrame] = {}
    for (fam, is_nrt), rows in raw.groupby([family, nrt]):
        spec = C.FAMILIES[fam]
        source = spec["nrt"] if is_nrt or not spec["sp"] else spec["sp"]
        out[source] = pd.concat([out[source], rows]) if source in out else rows
    return out


def save_import(text: str, name: str = "upload", raw_dir: Path | None = None) -> list[dict]:
    raw_dir = raw_dir or C.RAW_DIR
    parts = split_firms_csv(text)
    if not parts:
        raise ImportError_("the file has no detections")
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(name).stem)[:40] or "upload"
    saved = []
    for source, rows in sorted(parts.items()):
        path = raw_dir / source / f"import_{stem}_{digest}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        rows.to_csv(path, index=False)
        dates = rows["acq_date"].dropna()
        saved.append({"source": source, "rows": int(len(rows)), "file": str(path.relative_to(raw_dir)),
                      "first": dates.min() if len(dates) else None, "last": dates.max() if len(dates) else None,
                      "replaced": existed})
    log.info("imported %s: %s", name, ", ".join(f"{s['source']} {s['rows']}" for s in saved))
    return saved


def _download(task: tuple[str, int, str]) -> tuple[str, int, str, str | None]:
    folder, year, country = task
    url = COUNTRY_URL.format(folder=folder, year=year, name=country)
    for attempt in range(4):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
            if r.status_code == 404:
                return folder, year, country, None
            r.raise_for_status()
            return folder, year, country, r.text
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return folder, year, country, None


def seed_country_archives(years: list[int], sensors: tuple[str, ...] = ("modis", "viirs"), countries: list[str] | None = None,
                          raw_dir: Path | None = None, workers: int = 6) -> dict:
    tasks = [(ARCHIVE_FOLDERS[s], y, c) for s in sensors for y in years for c in (countries or EU27)]
    rows = files = 0
    with ThreadPoolExecutor(workers) as pool:
        for folder, year, country, text in pool.map(_download, tasks):
            if not text or len(text) < 100:
                continue
            try:
                for item in save_import(text, f"{folder}_{year}_{country}", raw_dir):
                    rows += item["rows"]
                    files += 1
            except ImportError_ as exc:
                log.warning("skipped %s %s %s: %s", folder, year, country, exc)
    return {"files": files, "rows": rows, "requests": len(tasks)}


def seed_from_dir(folder: Path, raw_dir: Path | None = None, years: list[int] | None = None) -> dict:
    rows = files = 0
    for path in sorted(folder.rglob("*.csv")):
        if years and not any(str(y) in path.name for y in years):
            continue
        text = path.read_text()
        if len(text) < 100:
            continue
        try:
            for item in save_import(text, path.stem, raw_dir):
                rows += item["rows"]
                files += 1
        except ImportError_ as exc:
            log.warning("skipped %s: %s", path, exc)
    return {"files": files, "rows": rows}


SNAPSHOT_MARK = ".snapshot"


def on_snapshot(db_path: Path) -> bool:
    return (Path(db_path).parent / SNAPSHOT_MARK).exists()


def restore_snapshot(url: str, db_path: Path, force: bool = False) -> dict:
    import shutil
    import subprocess
    db_path = Path(db_path)
    mark = db_path.parent / SNAPSHOT_MARK
    if mark.exists() and mark.read_text().strip() == url and db_path.exists() and not force:
        return {"restored": False, "reason": "already restored"}
    if not shutil.which("zstd"):
        raise RuntimeError("zstd is not installed")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    packed = db_path.parent / "snapshot.zst.part"
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=600, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(packed, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                digest.update(chunk)
    expected = None
    try:
        expected = requests.get(url + ".sha256", timeout=60, headers={"User-Agent": "Mozilla/5.0"}).text.split()[0]
    except Exception:
        log.warning("no checksum next to the snapshot, skipping the check")
    if expected and expected != digest.hexdigest():
        packed.unlink(missing_ok=True)
        raise RuntimeError("snapshot checksum does not match")
    fresh = db_path.with_suffix(".db.restore")
    subprocess.run(["zstd", "-d", "-f", "-q", str(packed), "-o", str(fresh)], check=True)
    packed.unlink(missing_ok=True)
    for extra in (db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm")):
        extra.unlink(missing_ok=True)
    fresh.replace(db_path)
    mark.write_text(url)
    return {"restored": True, "bytes": db_path.stat().st_size, "sha256": digest.hexdigest()}
