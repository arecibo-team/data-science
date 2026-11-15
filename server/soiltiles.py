"""Bare-ground (soil) overlay tiles drawn from the embedded `soil_squares` table."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from . import context, tiles

# One colour per class, in the order of context.SOIL_CLASSES.
COLORS = np.array([
    [250, 204, 21],    # sand, beaches, dunes
    [120, 113, 108],   # bare rock
    [132, 204, 22],    # sparse vegetation
    [28, 25, 23],      # burnt
    [186, 230, 253],   # glacier, snow
    [20, 184, 166],    # intertidal flats
    [168, 85, 247],    # quarries, gravel pits
    [236, 72, 153],    # dump sites
    [244, 63, 94],     # construction sites
], dtype=np.float32)


def render(con, z: int, x: int, y: int, cls: int | None = None) -> bytes | None:
    """PNG of the squares in a tile, or None when empty. cls limits it to one class index."""
    s, n, w, e = tiles.tile_box(z, x, y)
    q = "SELECT lat, lon, class, all_bare_pct FROM soil_squares WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?"
    args: list = [s, n, w, e]
    if cls is not None:
        q += " AND class = ?"
        args.append(cls)
    rows = np.array(con.execute(q, args).fetchall(), dtype=np.float64).reshape(-1, 4)
    if not len(rows):
        return None
    S = tiles.TILE + 2 * tiles.PAD
    px, py = tiles.project(rows[:, 0], rows[:, 1], z)
    ix = np.floor(px - x * tiles.TILE + tiles.PAD).astype(np.int64)
    iy = np.floor(py - y * tiles.TILE + tiles.PAD).astype(np.int64)
    ok = (ix >= 0) & (ix < S) & (iy >= 0) & (iy < S)
    if not ok.any():
        return None
    flat, k, wt = iy[ok] * S + ix[ok], rows[ok, 2].astype(int), rows[ok, 3] / 100.0
    sigma = float(np.clip(1.6 / (40075.0 / tiles._world(z)) * 0.6, 0.8, 8.0))  # a square's footprint in px
    gain = 2 * np.pi * sigma * sigma
    grids = np.stack([gaussian_filter(np.bincount(flat, weights=wt * (k == i), minlength=S * S).reshape(S, S),
                                      sigma, mode="constant") * gain for i in range(len(COLORS))])
    total = grids.sum(axis=0)
    rgb = (grids[..., None] * COLORS[:, None, None, :]).sum(axis=0) / np.maximum(total[..., None], 1e-6)
    alpha = np.clip(total, 0, 1) ** 0.7 * 0.85
    img = np.dstack([rgb, alpha * 255]).clip(0, 255).astype(np.uint8)
    img = img[tiles.PAD:tiles.PAD + tiles.TILE, tiles.PAD:tiles.PAD + tiles.TILE]
    return tiles.encode_png(img) if img[..., 3].any() else None
