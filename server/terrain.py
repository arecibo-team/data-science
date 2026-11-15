"""Terrain and vegetation per 0.1 degree cell from Copernicus Data Space (Sentinel Hub Process API).

Static fuel proxies for the forecast model, stored in `terrain_cells`:
  elev, relief, slope        Copernicus DEM (90 m), mean elevation, spread of elevation and mean slope inside the cell
  ndvi_s, ndmi_s             Sentinel-2 summer 2023 greenness and canopy moisture (cloud free pixels, least cloudy scene)
  ndvi_w                     Sentinel-2 winter 2023/24 greenness (low in snow or bare ground, high for evergreens)
Rasters are fetched at 0.01 degrees (about 1 km) in 25 degree tiles and averaged per cell.

Needs CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in .env (a Sentinel Hub OAuth client of the Copernicus Data Space
Ecosystem). Without them the step is skipped.
"""
from __future__ import annotations

import io
import logging
import time

import numpy as np
import requests
from dotenv import dotenv_values

from firms import config as C
from . import db

log = logging.getLogger("server.terrain")

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
RES = 0.01
TILE = 2500                      # pixels per side, the Process API limit
COLS = ["elev", "relief", "slope", "ndvi_s", "ndmi_s", "ndvi_w"]

DEM_SCRIPT = """//VERSION=3
function setup() { return { input: ["DEM"], output: { bands: 1, sampleType: "FLOAT32" } }; }
function evaluatePixel(s) { return [s.DEM]; }"""

S2_SCRIPT = """//VERSION=3
function setup() { return { input: ["B04", "B08", "B11", "SCL", "dataMask"], output: { bands: 2, sampleType: "FLOAT32" } }; }
function evaluatePixel(s) {
  var bad = s.dataMask == 0 || s.SCL == 3 || s.SCL == 8 || s.SCL == 9 || s.SCL == 10 || s.SCL == 1 || s.SCL == 6;
  if (bad) return [NaN, NaN];
  return [(s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6), (s.B08 - s.B11) / (s.B08 + s.B11 + 1e-6)];
}"""


def credentials() -> tuple[str, str] | None:
    env = dotenv_values(C.ROOT / ".env")
    cid, sec = env.get("CDSE_CLIENT_ID"), env.get("CDSE_CLIENT_SECRET")
    return (cid, sec) if cid and sec else None


class _Client:
    def __init__(self, cid: str, secret: str):
        self.cid, self.secret, self._tok, self._exp = cid, secret, None, 0.0

    def token(self) -> str:
        if not self._tok or time.time() > self._exp - 60:
            r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials", "client_id": self.cid,
                                               "client_secret": self.secret}, timeout=30)
            r.raise_for_status()
            j = r.json()
            self._tok, self._exp = j["access_token"], time.time() + j.get("expires_in", 600)
        return self._tok

    def raster(self, bbox, dataset: dict, script: str) -> np.ndarray:
        """(bands, h, w) float32 for bbox W,S,E,N at RES degrees per pixel (NaN where there is no data)."""
        w, s, e, n = bbox
        body = {"input": {"bounds": {"bbox": [w, s, e, n], "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                          "data": [dataset]},
                "output": {"width": round((e - w) / RES), "height": round((n - s) / RES),
                           "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
                "evalscript": script}
        for attempt in range(6):
            r = requests.post(PROCESS_URL, json=body, headers={"Authorization": f"Bearer {self.token()}"}, timeout=300)
            if r.status_code == 200:
                break
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(f"Process API {r.status_code}: {r.text[:300]}")
            time.sleep(15 * (attempt + 1))
        import rasterio
        with rasterio.MemoryFile(r.content) as mf, mf.open() as ds:
            return ds.read().astype(np.float32)


def _tiles(bbox):
    w, s, e, n = bbox
    step = TILE * RES
    out = []
    x = w
    while x < e - 1e-9:
        y = s
        while y < n - 1e-9:
            out.append((x, y, min(x + step, e), min(y + step, n)))
            y += step
        x += step
    return out


def _cell_means(arr: np.ndarray, tile, func=np.nanmean) -> dict[tuple[int, int], float]:
    """Mean per 0.1 degree cell of a (h, w) raster whose top-left is (W, N); cells are bin_day's r, c indices."""
    w, s, e, n = tile
    h, wd = arr.shape
    out = {}
    k = round(0.1 / RES)                                   # 10 pixels per cell
    r0, c0 = round((s + 90) / 0.1), round((w + 180) / 0.1)
    rows, cols = h // k, wd // k
    blocks = arr[: rows * k, : cols * k].reshape(rows, k, cols, k)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(rows, cols, k * k)
    with np.errstate(all="ignore"):
        vals = func(blocks, axis=2)
    for i in range(rows):                                  # raster row 0 is the northern edge
        for j in range(cols):
            if np.isfinite(vals[i, j]):
                out[(r0 + rows - 1 - i, c0 + j)] = float(vals[i, j])
    return out


def _slope_deg(dem: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(dem, RES * 111_000)               # metres per pixel, close enough for a feature
    return np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)


def run(should_stop=lambda: False) -> int:
    """Fetch and store terrain_cells for all of Europe. Returns cells written."""
    cred = credentials()
    if not cred:
        raise RuntimeError("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET missing in .env")
    cl = _Client(*cred)
    bbox = C.EUROPE_BBOX
    cells: dict[tuple[int, int], dict[str, float]] = {}
    layers = [
        ("dem", {"type": "dem", "dataFilter": {"demInstance": "COPERNICUS_90"}, "processing": {"downsampling": "BILINEAR"}}, DEM_SCRIPT),
        ("s2_summer", {"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": "2023-06-01T00:00:00Z", "to": "2023-08-31T23:59:59Z"},
                                                               "mosaickingOrder": "leastCC", "maxCloudCoverage": 70},
                       "processing": {"downsampling": "BILINEAR"}}, S2_SCRIPT),
        ("s2_winter", {"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": "2023-12-01T00:00:00Z", "to": "2024-02-29T23:59:59Z"},
                                                               "mosaickingOrder": "leastCC", "maxCloudCoverage": 70},
                       "processing": {"downsampling": "BILINEAR"}}, S2_SCRIPT),
    ]
    for tile in _tiles(bbox):
        for name, data, script in layers:
            if should_stop():
                return len(cells)
            ras = cl.raster(tile, data, script)
            log.info("terrain tile %s %s", tile, name)
            if name == "dem":
                dem = ras[0]
                dem = np.where(dem < -500, np.nan, dem)
                for key, f in (("elev", np.nanmean), ("relief", np.nanstd)):
                    for cell, v in _cell_means(dem, tile, f).items():
                        cells.setdefault(cell, {})[key] = v
                for cell, v in _cell_means(_slope_deg(np.nan_to_num(dem)), tile).items():
                    cells.setdefault(cell, {})["slope"] = v
            elif name == "s2_summer":
                for cell, v in _cell_means(ras[0], tile).items():
                    cells.setdefault(cell, {})["ndvi_s"] = v
                for cell, v in _cell_means(ras[1], tile).items():
                    cells.setdefault(cell, {})["ndmi_s"] = v
            else:
                for cell, v in _cell_means(ras[0], tile).items():
                    cells.setdefault(cell, {})["ndvi_w"] = v
    rows = [(r, c, *(vals.get(k) for k in COLS)) for (r, c), vals in cells.items()]
    with db.session() as con:
        con.execute("DELETE FROM terrain_cells")
        con.executemany(f"INSERT INTO terrain_cells VALUES (?,?,{','.join('?' * len(COLS))})", rows)
    return len(rows)
