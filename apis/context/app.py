from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

DATA = Path(os.environ.get("CONTEXT_DATA", Path(__file__).resolve().parent / "data"))
ISO3_TO_ISO2 = {
    "AUT": "AT", "BEL": "BE", "BGR": "BG", "HRV": "HR", "CYP": "CY", "CZE": "CZ", "DNK": "DK", "EST": "EE", "FIN": "FI",
    "FRA": "FR", "DEU": "DE", "GRC": "GR", "HUN": "HU", "IRL": "IE", "ITA": "IT", "LVA": "LV", "LTU": "LT", "LUX": "LU",
    "MLT": "MT", "NLD": "NL", "POL": "PL", "PRT": "PT", "ROU": "RO", "SVK": "SK", "SVN": "SI", "ESP": "ES", "SWE": "SE",
    "ALB": "AL", "ALD": "AX", "AND": "AD", "BIH": "BA", "BLR": "BY", "CHE": "CH", "FRO": "FO", "GBR": "GB", "GGY": "GG",
    "IMN": "IM", "ISL": "IS", "JEY": "JE", "KOS": "XK", "LIE": "LI", "MDA": "MD", "MKD": "MK", "MNE": "ME", "NOR": "NO",
    "RUS": "RU", "SMR": "SM", "SRB": "RS", "TUR": "TR", "UKR": "UA",
}
ISO2_TO_ISO3 = {v: k for k, v in ISO3_TO_ISO2.items()}
EU27 = {"AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT",
        "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"}


def _load(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


LAND, LAND_GRID = _load("landcover_countries"), _load("landcover_grid")
TERRAIN, TERRAIN_GRID = _load("terrain_countries"), _load("terrain_grid")
SOIL, SOIL_GRID = _load("soil_moisture_countries"), _load("soil_moisture_grid")
VEG, VEG_GRID = _load("vegetation_countries"), _load("vegetation_grid")
MONTHS: list[str] = SOIL["months"]


def _month_index(month: str | None) -> int:
    if month is None:
        return len(MONTHS) - 1
    if month in MONTHS:
        return MONTHS.index(month)
    if month < MONTHS[0]:
        raise HTTPException(404, f"Context data starts in {MONTHS[0]}")
    return len(MONTHS) - 1


def _num(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _at(series, i: int) -> float | None:
    return _num(series[i]) if series and i < len(series) else None


def dryness(soil_z1: float | None, soil_z2: float | None, lst_z: float | None) -> float | None:
    parts = [x for x in (-soil_z1 if soil_z1 is not None else None, -soil_z2 if soil_z2 is not None else None,
                         lst_z) if x is not None]
    return round(sum(parts) / len(parts), 2) if parts else None


def dryness_label(z: float | None) -> str:
    if z is None:
        return "No data"
    if z >= 1.5:
        return "Much drier than usual"
    if z >= 0.5:
        return "Drier than usual"
    if z > -0.5:
        return "Close to normal"
    if z > -1.5:
        return "Wetter than usual"
    return "Much wetter than usual"


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class Fuel(BaseModel):
    wildland_fuel_pct: float | None
    forest_pct: float | None
    shrub_pct: float | None
    grassland_pct: float | None
    cropland_pct: float | None
    urban_pct: float | None
    burnt_2018_km2: float | None


class Terrain(BaseModel):
    elevation_m: float | None
    elevation_p90_m: float | None
    slope_deg: float | None
    steep_pct: float | None
    north_facing_pct: float | None


class Moisture(BaseModel):
    soil_moisture: float | None
    soil_moisture_z: float | None
    deep_soil_moisture_z: float | None
    ndvi: float | None
    ndvi_z: float | None
    lst_c: float | None
    lst_z: float | None


class CountryContext(BaseModel):
    iso2: str
    iso3: str
    name: str
    eu: bool
    month: str
    fuel: Fuel
    terrain: Terrain
    moisture: Moisture
    dryness_z: float | None
    dryness: str
    fuel_risk: float | None
    rank: int | None = None


class SeriesPoint(BaseModel):
    month: str
    soil_moisture_z: float | None
    ndvi_z: float | None
    lst_z: float | None
    dryness_z: float | None


class CountryDetail(CountryContext):
    series: list[SeriesPoint]


def _country(iso3: str, i: int) -> CountryContext:
    land = LAND["countries"].get(iso3, {})
    ter = TERRAIN["countries"].get(iso3, {})
    soil = SOIL["countries"].get(iso3, {})
    veg = VEG["countries"].get(iso3, {})
    name = land.get("name") or ter.get("name") or soil.get("name") or veg.get("name") or iso3
    z1, z2, lz = _at(soil.get("swvl1_z"), i), _at(soil.get("swvl2_z"), i), _at(veg.get("lst_c_z"), i)
    dz = dryness(z1, z2, lz)
    fuel_pct = _num(land.get("wildland_fuel_pct"))
    risk = round(fuel_pct * _phi(dz), 1) if fuel_pct is not None and dz is not None else None
    elev = ter.get("elevation_m") or {}
    iso2 = ISO3_TO_ISO2.get(iso3, iso3[:2])
    return CountryContext(
        iso2=iso2, iso3=iso3, name=name, eu=iso2 in EU27, month=MONTHS[i],
        fuel=Fuel(wildland_fuel_pct=fuel_pct, forest_pct=_num(land.get("forest_pct")), shrub_pct=_num(land.get("shrub_pct")),
                  grassland_pct=_num(land.get("grassland_pct")), cropland_pct=_num(land.get("cropland_pct")),
                  urban_pct=_num(land.get("urban_pct")), burnt_2018_km2=_num(land.get("burnt_2018_km2"))),
        terrain=Terrain(elevation_m=_num(elev.get("p50")), elevation_p90_m=_num(elev.get("p90")),
                        slope_deg=_num(ter.get("slope_deg_mean")), steep_pct=_num(ter.get("steep_pct")),
                        north_facing_pct=_num(ter.get("north_facing_pct"))),
        moisture=Moisture(soil_moisture=_at(soil.get("swvl1"), i), soil_moisture_z=z1, deep_soil_moisture_z=z2,
                          ndvi=_at(veg.get("ndvi"), i), ndvi_z=_at(veg.get("ndvi_z"), i), lst_c=_at(veg.get("lst_c"), i), lst_z=lz),
        dryness_z=dz, dryness=dryness_label(dz), fuel_risk=risk)


def _resolve(code: str) -> str:
    code = code.upper()
    if code == "EL":
        code = "GR"
    iso3 = ISO2_TO_ISO3.get(code, code)
    if iso3 not in SOIL["countries"] and iso3 not in LAND["countries"]:
        raise HTTPException(404, f"No context data for '{code}'")
    return iso3


def _nearest(points: np.ndarray, lat: float, lon: float) -> tuple[int, float]:
    k = math.cos(math.radians(lat))
    d = (points[:, 0] - lat) ** 2 + ((points[:, 1] - lon) * k) ** 2
    i = int(np.argmin(d))
    return i, float(math.sqrt(d[i]))


@lru_cache(maxsize=None)
def _list_grid(name: str) -> tuple[np.ndarray, list]:
    grid = LAND_GRID if name == "land" else TERRAIN_GRID
    cells = grid["cells"]
    return np.array([[c[0], c[1]] for c in cells]), cells


@lru_cache(maxsize=None)
def _dict_grid(name: str) -> np.ndarray:
    grid = SOIL_GRID if name == "soil" else VEG_GRID
    return np.array([[c["lat"], c["lon"]] for c in grid["cells"]])


app = FastAPI(title="Arecibo context API",
              description="Fuel, terrain and dryness context for fire activity: CORINE land cover, Copernicus DEM, "
                          "ERA5-Land soil moisture and MODIS NDVI and land surface temperature anomalies.")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "months": [MONTHS[0], MONTHS[-1]], "countries": len(SOIL["countries"]),
            "grids": {"land": len(LAND_GRID["cells"]), "terrain": len(TERRAIN_GRID["cells"]),
                      "soil": len(SOIL_GRID["cells"]), "vegetation": len(VEG_GRID["cells"])}}


@app.get("/months")
def months():
    return {"months": MONTHS, "baseline": SOIL["baseline"],
            "sources": {"land": LAND["source"], "terrain": TERRAIN["source"], "soil": SOIL["source"], "vegetation": VEG["source"]},
            "method": {"dryness_z": "mean of the topsoil and root-zone soil moisture deficits and the land surface "
                                    "temperature anomaly, all as z-scores against 2001-2020 for the same month",
                       "fuel_risk": "wildland fuel share of the land times the probability that a normal month is "
                                    "wetter than this one (0 to 100)"}}


@app.get("/countries", response_model=list[CountryContext])
def countries(month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"), eu: bool = True):
    i = _month_index(month)
    codes = sorted(set(SOIL["countries"]) | set(LAND["countries"]))
    rows = [_country(c, i) for c in codes]
    rows = [r for r in rows if r.eu] if eu else rows
    ranked = sorted([r for r in rows if r.dryness_z is not None], key=lambda r: -r.dryness_z)
    for n, r in enumerate(ranked, 1):
        r.rank = n
    return sorted(rows, key=lambda r: (r.rank is None, r.rank or 0, r.name))


@app.get("/countries/{code}", response_model=CountryDetail)
def country(code: str, month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$")):
    iso3 = _resolve(code)
    base = _country(iso3, _month_index(month))
    soil = SOIL["countries"].get(iso3, {})
    veg = VEG["countries"].get(iso3, {})
    series = []
    for k, m in enumerate(MONTHS):
        z1, z2, lz = _at(soil.get("swvl1_z"), k), _at(soil.get("swvl2_z"), k), _at(veg.get("lst_c_z"), k)
        series.append(SeriesPoint(month=m, soil_moisture_z=z1, ndvi_z=_at(veg.get("ndvi_z"), k), lst_z=lz, dryness_z=dryness(z1, z2, lz)))
    return CountryDetail(**base.model_dump(), series=series)


@app.get("/point")
def point(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180),
          month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$")):
    i = None if month is not None and month < MONTHS[0] else _month_index(month)
    cal = int(MONTHS[i][5:7]) - 1 if i is not None else 0
    out = {"lat": lat, "lon": lon, "month": MONTHS[i] if i is not None else None}
    pts, cells = _list_grid("land")
    j, dist = _nearest(pts, lat, lon)
    if dist <= LAND_GRID["res_deg"]:
        c = dict(zip(LAND_GRID["fields"], cells[j]))
        fuel = sum(c.get(k, 0) or 0 for k in ("forest_pct", "shrub_pct", "grassland_pct"))
        out["land"] = {**{k: c[k] for k in LAND_GRID["fields"] if k not in ("lat", "lon")}, "wildland_fuel_pct": round(fuel, 1)}
    pts, cells = _list_grid("terrain")
    j, dist = _nearest(pts, lat, lon)
    if dist <= TERRAIN_GRID["res_deg"]:
        out["terrain"] = {k: v for k, v in zip(TERRAIN_GRID["fields"], cells[j]) if k not in ("lat", "lon")}
    j, dist = _nearest(_dict_grid("soil"), lat, lon)
    if i is not None and dist <= SOIL_GRID["res_deg"]:
        c = SOIL_GRID["cells"][j]
        z = c["z10"][i] / 10 if c["z10"][i] is not None else None
        out["soil"] = {"normal": c["normal"][cal], "z": z}
    j, dist = _nearest(_dict_grid("veg"), lat, lon)
    if i is not None and dist <= VEG_GRID["res_deg"]:
        c = VEG_GRID["cells"][j]
        nz = c["ndvi_z10"][i] / 10 if c["ndvi_z10"][i] is not None else None
        lz = c["lst_z10"][i] / 10 if c["lst_z10"][i] is not None else None
        out["vegetation"] = {"ndvi_normal": c["ndvi_normal"][cal], "ndvi_z": nz, "lst_normal": c["lst_normal"][cal], "lst_z": lz}
    soil_z = out.get("soil", {}).get("z")
    lst_z = out.get("vegetation", {}).get("lst_z")
    out["dryness_z"] = dryness(soil_z, None, lst_z)
    out["dryness"] = dryness_label(out["dryness_z"])
    return out


@app.get("/grid/{layer}")
def grid(layer: Literal["dryness", "soil", "ndvi", "lst", "fuel", "steep"], month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"),
         west: float = -32, south: float = 27, east: float = 45, north: float = 72):
    i = _month_index(month)
    inside = lambda lat, lon: south <= lat <= north and west <= lon <= east
    if layer in ("fuel", "steep"):
        src = LAND_GRID if layer == "fuel" else TERRAIN_GRID
        f = src["fields"]
        if layer == "fuel":
            idx = [f.index(k) for k in ("forest_pct", "shrub_pct", "grassland_pct")]
            cells = [[c[0], c[1], round(sum(c[k] or 0 for k in idx), 1)] for c in src["cells"] if inside(c[0], c[1])]
        else:
            k = f.index("steep_pct")
            cells = [[c[0], c[1], c[k]] for c in src["cells"] if inside(c[0], c[1]) and c[k] is not None]
        return {"layer": layer, "month": None, "res_deg": src["res_deg"], "cells": cells}
    if layer == "soil":
        cells = [[c["lat"], c["lon"], c["z10"][i] / 10] for c in SOIL_GRID["cells"]
                 if inside(c["lat"], c["lon"]) and c["z10"][i] is not None]
        return {"layer": layer, "month": MONTHS[i], "res_deg": SOIL_GRID["res_deg"], "cells": cells}
    if layer in ("ndvi", "lst"):
        key = f"{layer}_z10"
        cells = [[c["lat"], c["lon"], c[key][i] / 10] for c in VEG_GRID["cells"]
                 if inside(c["lat"], c["lon"]) and c[key][i] is not None]
        return {"layer": layer, "month": MONTHS[i], "res_deg": VEG_GRID["res_deg"], "cells": cells}
    veg_pts = _dict_grid("veg")
    cells = []
    for c in SOIL_GRID["cells"]:
        if not inside(c["lat"], c["lon"]) or c["z10"][i] is None:
            continue
        j, dist = _nearest(veg_pts, c["lat"], c["lon"])
        lz = VEG_GRID["cells"][j]["lst_z10"][i] if dist <= 1.0 else None
        z = dryness(c["z10"][i] / 10, None, lz / 10 if lz is not None else None)
        if z is not None:
            cells.append([c["lat"], c["lon"], z])
    return {"layer": layer, "month": MONTHS[i], "res_deg": SOIL_GRID["res_deg"], "cells": cells}
