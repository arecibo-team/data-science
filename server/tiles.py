"""Heat tiles: the overview of views that hold too many detections to draw one by one.

A tile is a 256 px Web Mercator PNG. Density is drawn on a log scale, hue follows the layers
(MODIS orange, VIIRS blue, confirmed detections pulled toward red), and the three layer toggles are a
bit mask (`MODIS=1, VIIRS=2, COMMON=4`) so switching a layer just requests other tiles.

Where the counts come from, by zoom level (LOD):
    z <= BIN_MAX_ZOOM   `bin_day`, detections already summed per day and 0.1 degree cell. Far fewer rows.
    above               `detections` inside the tile's box, using the (ts, lat, lon) covering index.

Rendering draws onto a canvas padded by PAD px and crops, so blurred spots do not break at tile edges.

Cache: PNGs under TILE_DIR/<data version>/<from>_<to>/<mask>/<z>/<x>/<y>.png. The data version is the id of
the last finished `stats` run, so any pipeline rerun starts a fresh cache; the `tiles` step deletes the old
versions and pre-renders the low zoom levels for the common ranges.
"""
from __future__ import annotations

import math
import shutil
import struct
import sys
import zlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from . import db, settings

TILE = 256
PAD = 24                     # extra pixels drawn around the tile, cropped afterwards
BIN_MAX_ZOOM = 7             # up to here tiles are drawn from bin_day
MAX_ZOOM = 18
PRERENDER_ZOOM = 6           # the tiles step renders z0..this for the common ranges
MASK_ALL = 7                 # MODIS | VIIRS | COMMON
M_BIT, V_BIT, C_BIT = 1, 2, 4

ORANGE = np.array([249, 115, 22], dtype=np.float32)   # MODIS, as web/src/components/PointLayer.ts
BLUE = np.array([37, 99, 235], dtype=np.float32)      # VIIRS
RED = np.array([220, 38, 38], dtype=np.float32)       # confirmed inside an incident

_EMPTY: bytes | None = None


# ---------------------------------------------------------------- PNG

def encode_png(rgba: np.ndarray) -> bytes:
    """RGBA uint8 array (h, w, 4) to a PNG, with no dependency beyond zlib."""
    h, w, _ = rgba.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgba.reshape(h, w * 4)], axis=1).tobytes()

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))


def empty_tile() -> bytes:
    global _EMPTY
    if _EMPTY is None:
        _EMPTY = encode_png(np.zeros((TILE, TILE, 4), np.uint8))
    return _EMPTY


# ---------------------------------------------------------------- geometry

def _world(z: int) -> float:
    return TILE * 2.0 ** z


def project(lat: np.ndarray, lon: np.ndarray, z: int) -> tuple[np.ndarray, np.ndarray]:
    """Web Mercator world pixel coordinates at zoom z."""
    w = _world(z)
    la = np.radians(np.clip(lat, -85.0511, 85.0511))
    px = (lon + 180.0) / 360.0 * w
    py = (0.5 - np.log(np.tan(math.pi / 4 + la / 2)) / (2 * math.pi)) * w
    return px, py


def unproject(px: float, py: float, z: int) -> tuple[float, float]:
    w = _world(z)
    lon = px / w * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * py / w))))
    return lat, lon


def tile_box(z: int, x: int, y: int, pad_px: int = PAD) -> tuple[float, float, float, float]:
    """(south, north, west, east) of a tile grown by pad_px on every side."""
    north, west = unproject(x * TILE - pad_px, y * TILE - pad_px, z)
    south, east = unproject((x + 1) * TILE + pad_px, (y + 1) * TILE + pad_px, z)
    return max(south, -90.0), min(north, 90.0), max(west, -180.0), min(east, 180.0)


# ---------------------------------------------------------------- rendering

def _sigma(z: int) -> float:
    """Blur radius in px: a bin cell's footprint at low zoom, a fixed small halo for single detections."""
    if z <= BIN_MAX_ZOOM:
        deg_per_px = 360.0 / _world(z)
        return float(np.clip(db.BIN_DEG / deg_per_px * 0.4, 1.0, 10.0))
    return 1.4


def render(z: int, x: int, y: int, mask: int, lat, lon, m, v, cm, cv) -> bytes | None:
    """Draw one tile from points or bin centres with per-point counts. None when nothing is visible."""
    if len(lat) == 0:
        return None
    S = TILE + 2 * PAD
    px, py = project(np.asarray(lat, float), np.asarray(lon, float), z)
    ix = np.floor(px - x * TILE + PAD).astype(np.int64)
    iy = np.floor(py - y * TILE + PAD).astype(np.int64)
    ok = (ix >= 0) & (ix < S) & (iy >= 0) & (iy < S)
    if not ok.any():
        return None
    flat = iy[ok] * S + ix[ok]

    M, V, C = bool(mask & M_BIT), bool(mask & V_BIT), bool(mask & C_BIT)
    m, v, cm, cv = (np.asarray(a, float)[ok] for a in (m, v, cm, cv))
    layers = {
        "m": m if M else 0 * m,
        "v": v if V else 0 * v,
        # Confirmed detections already counted in an enabled sensor layer are not counted twice.
        "c": (cm * (not M) + cv * (not V)) if C else 0 * cm,
        "conf": (cm + cv) if C else 0 * cm,
    }
    sigma = _sigma(z)
    gain = 2 * math.pi * sigma * sigma  # makes the blur of one unit have peak 1
    out = {}
    for k, w in layers.items():
        grid = np.bincount(flat, weights=w, minlength=S * S).reshape(S, S)
        out[k] = gaussian_filter(grid, sigma, mode="constant") * gain
    total = out["m"] + out["v"] + out["c"]
    if total.max() < 0.08:
        return None

    safe = np.maximum(total, 1e-9)[..., None]
    rgb = (out["m"][..., None] * ORANGE + out["v"][..., None] * BLUE + out["c"][..., None] * RED) / safe
    frac = np.clip(out["conf"] / np.maximum(total, 1e-9), 0, 1)[..., None]
    rgb = rgb * (1 - 0.6 * frac) + RED * (0.6 * frac)
    # Opacity follows log density against a reference count: bin cells hold days of detections, points one.
    ref = 1500.0 if z <= BIN_MAX_ZOOM else 25.0
    strength = np.clip(np.log1p(total) / math.log1p(ref), 0, 1)
    alpha = np.where(total >= 0.08, 0.12 + 0.83 * strength ** 0.9, 0.0)
    img = np.empty((S, S, 4), np.uint8)
    img[..., :3] = np.clip(rgb, 0, 255)
    img[..., 3] = (alpha * 235).astype(np.uint8)
    img = img[PAD:PAD + TILE, PAD:PAD + TILE]
    return encode_png(np.ascontiguousarray(img)) if img[..., 3].any() else None


# ---------------------------------------------------------------- data access

def _ts(day: str, plus: int = 0) -> int:
    d = date.fromisoformat(day) + timedelta(days=plus)
    return int((d - date(1970, 1, 1)).days * 86400)


def load_cells(con, d0: str, d1: str, box: tuple[float, float, float, float] | None = None):
    """Summed bin_day cells in a date range, optionally inside a (south, north, west, east) box."""
    where, args = "date >= ? AND date <= ?", [d0, d1]
    if box:
        s, n, w, e = box
        deg = db.BIN_DEG
        where += " AND r BETWEEN ? AND ? AND c BETWEEN ? AND ?"
        args += [int(math.floor((s + 90) / deg)), int(math.floor((n + 90) / deg)),
                 int(math.floor((w + 180) / deg)), int(math.floor((e + 180) / deg))]
    rows = con.execute(f"SELECT r, c, SUM(m), SUM(v), SUM(cm), SUM(cv) FROM bin_day WHERE {where} "
                       "GROUP BY r, c", args).fetchall()
    a = np.array(rows, dtype=float).reshape(-1, 6)
    return (-90 + (a[:, 0] + 0.5) * db.BIN_DEG, -180 + (a[:, 1] + 0.5) * db.BIN_DEG, *a[:, 2:].T)


def load_points(con, d0: str, d1: str, box):
    s, n, w, e = box
    rows = con.execute(
        "SELECT lat, lon, sensor = 1, sensor = 2, sensor = 1 AND incident_id IS NOT NULL, "
        "sensor = 2 AND incident_id IS NOT NULL FROM detections "
        "WHERE ts >= ? AND ts < ? AND lat >= ? AND lat <= ? AND lon >= ? AND lon <= ?",
        (_ts(d0), _ts(d1, 1), s, n, w, e)).fetchall()
    a = np.array(rows, dtype=float).reshape(-1, 6)
    return tuple(a.T)


def render_from_db(con, z: int, x: int, y: int, d0: str, d1: str, mask: int) -> bytes | None:
    box = tile_box(z, x, y)
    cells = load_cells(con, d0, d1, box) if z <= BIN_MAX_ZOOM else load_points(con, d0, d1, box)
    return render(z, x, y, mask, *cells)


# ---------------------------------------------------------------- cache

def data_version(con) -> int:
    r = con.execute("SELECT id FROM pipeline_steps WHERE step='stats' AND status='done' "
                    "ORDER BY id DESC LIMIT 1").fetchone()
    return r[0] if r else 0


def tile_path(version: int, d0: str, d1: str, mask: int, z: int, x: int, y: int) -> Path:
    return Path(settings.TILE_DIR) / str(version) / f"{d0}_{d1}" / str(mask) / str(z) / str(x) / f"{y}.png"


_writes = 0


def get_tile(con, z: int, x: int, y: int, d0: str, d1: str, mask: int) -> bytes:
    """Cached tile bytes, rendering and storing on a miss. Empty tiles are never written to disk."""
    global _writes
    path = tile_path(data_version(con), d0, d1, mask, z, x, y)
    try:
        return path.read_bytes()
    except FileNotFoundError:
        pass
    png = render_from_db(con, z, x, y, d0, d1, mask)
    if png is None:
        return empty_tile()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(png)
    tmp.replace(path)
    _writes += 1
    if _writes % 500 == 0:
        prune(settings.TILE_CACHE_MB * 1024 * 1024)
    return png


def clear_old_versions(keep: int) -> None:
    root = Path(settings.TILE_DIR)
    if root.is_dir():
        for d in root.iterdir():
            if d.is_dir() and d.name != str(keep):
                shutil.rmtree(d, ignore_errors=True)


def prune(max_bytes: int) -> int:
    """Delete the least recently written tiles until the cache fits. Returns bytes left."""
    files = [(p.stat().st_mtime, p.stat().st_size, p) for p in Path(settings.TILE_DIR).rglob("*.png")]
    total = sum(f[1] for f in files)
    for _, size, p in sorted(files, key=lambda f: f[0]):
        if total <= max_bytes:
            break
        p.unlink(missing_ok=True)
        total -= size
    return total


def common_ranges(first: str, last: str) -> list[tuple[str, str]]:
    """Ranges the web app offers as presets, pre-rendered by the tiles step."""
    end = date.fromisoformat(last)
    out = [(first, last)]
    for days in (365, 30):
        start = max(date.fromisoformat(first), end - timedelta(days=days - 1))
        if (start.isoformat(), last) not in out:
            out.append((start.isoformat(), last))
    return out


def prerender(con, first: str, last: str, max_zoom: int = PRERENDER_ZOOM) -> int:
    """Render and cache z0..max_zoom for the common ranges with all layers on. Returns tiles written."""
    version = data_version(con)
    written = 0
    for d0, d1 in common_ranges(first, last):
        cells = load_cells(con, d0, d1)
        lat, lon = cells[0], cells[1]
        for z in range(0, max_zoom + 1):
            px, py = project(lat, lon, z)
            tx, ty = np.floor(px / TILE).astype(int), np.floor(py / TILE).astype(int)
            for x, y in sorted({*zip(tx.tolist(), ty.tolist())} | _neighbours(tx, ty, z)):
                path = tile_path(version, d0, d1, MASK_ALL, z, x, y)
                if path.exists() or not (0 <= x < 2 ** z and 0 <= y < 2 ** z):
                    continue
                png = render(z, x, y, MASK_ALL, *_slice(cells, z, x, y))
                if png is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(png)
                    written += 1
    return written


def _neighbours(tx, ty, z):
    """Tiles next to occupied ones: blurred spots spill a few pixels over the border."""
    return {(x + dx, y + dy) for x, y in zip(tx.tolist(), ty.tolist()) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}


def _slice(cells, z: int, x: int, y: int):
    s, n, w, e = tile_box(z, x, y)
    lat, lon = cells[0], cells[1]
    keep = (lat >= s - db.BIN_DEG) & (lat <= n + db.BIN_DEG) & (lon >= w - db.BIN_DEG) & (lon <= e + db.BIN_DEG)
    return tuple(a[keep] for a in cells)


if __name__ == "__main__":
    sys.exit("tiles are served by python -m server")
