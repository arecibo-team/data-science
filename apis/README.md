# APIs

Every HTTP service of the project, the models behind them, and where the web app uses each endpoint. Three services:

| Service | Code | Port | Image |
| --- | --- | --- | --- |
| Amber API | `amber-index/api/main.py` | 8001 | `amber-index/Dockerfile` |
| FIRMS server | `server/app.py` | 8002 | `Dockerfile.server` |
| Context API | `apis/context/app.py` | 8003 | `apis/context/Dockerfile` |

## Run them with Docker

From this folder:

```bash
docker compose up --build -d
docker compose ps                       # all three healthy after about a minute
curl -s localhost:8001/health | jq .status
curl -s localhost:8002/api/pipeline | jq .running
curl -s localhost:8003/health | jq .status
```

Open the interactive docs at http://localhost:8001/docs, http://localhost:8002/docs and http://localhost:8003/docs.

The FIRMS server seeds itself on first start from the yearly FIRMS country archives (no key needed, years in
`FIRMS_SEED_YEARS`, default `2024,2025`) and then runs its pipeline in the background. Heat tiles and counts appear
after the `stats` step, about five minutes later. `docker compose logs -f firms-server` shows progress.

To run the web app on top of them, use the `arecibo` repository, which adds the frontend behind nginx and routes
`/amber/*`, `/api/*` and `/context/*` to these three services.

### Run one service without Docker

```bash
# Amber API
cd amber-index && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api.main:app --port 8001

# FIRMS server
uv sync && uv run python -m server seed --years 2024,2025
PIPELINE_SKIP=fetch,sentinel2,terrain,weather,forecast uv run python -m server serve --port 8002

# Context API
cd apis/context && pip install -r requirements.txt && uvicorn app:app --port 8003
```

### Tests

```bash
cd amber-index && .venv/bin/pytest                         # Amber API
uv run pytest                                              # FIRMS server
cd apis/context && pytest test_app.py                      # Context API
```

## Amber API (8001)

Long harmonized fire record 2001–2025 from the FIRMS country archives, the Amber index and firefighting power, the
regional firefighter estimates, simulations and the assistance plan. Loads committed artifacts in `amber-index/artifacts`.

| Endpoint | Model or data | Used in the app |
| --- | --- | --- |
| `GET /health`, `GET /model` | LightGBM burned-area model, cross-validation | Method dialog |
| `GET /replay?date=` | Active fires, area ratios, power and Amber index for a day | Map, Active fires, area panel |
| `GET /regions?date=` | Regional firefighter estimates × this month's fire share → regional power | Map layer **Regions**, compass card in the area panel |
| `GET /area` and `/area/{hotspots,daily,calendar}.csv` | Harmonized calendar, range comparison, exports for any box or country | Area panel (Calendar, Range), timeline, downloads |
| `GET /history/{iso2}` and `.csv` | Country calendar (older, superseded by `/area`) | not used |
| `GET /scores`, `GET /scores/{iso2}` | Amber index per year, SHAP reasons, rank range, archetype | **Why this score** card |
| `GET /scores/{iso2}/history` | Index and power per year | **Over the years** card |
| `POST /score/custom-weights` | Re-rank with your own pillar weights | **Weigh the Amber index yourself** dialog |
| `POST /predict` | Score raw indicators for a hypothetical country-year | not used in the UI, API only |
| `GET /simulate/presets`, `POST /simulate` | Hazard shocks on top of a replay day | Simulate panel |
| `GET /live` | GDACS alerts of the last 14 days | **Hazards happening now** scenario |
| `POST /plan` | Fire mutual aid in three stages (local, national, European): each unit sustains one sixth of its firefighters, donors serve their own fires first, a MILP picks the fastest arrivals | Who can help panel, animated flows on the map |
| `POST /optimize/assistance`, `GET /live/assistance` | Older yearly optimizer | not used in the UI, API only |
| `POST /admin/reload` | Reload artifacts | operations |

## FIRMS server (8002)

Detection-level MODIS and VIIRS harmonization: every detection matched across sensors (1 km, 3 h), tagged with
independent incidents (Copernicus EMS, EFFIS, optional Sentinel-2 burn scars). SQLite in `/app/data`.

| Endpoint | Model or data | Used in the app |
| --- | --- | --- |
| `GET /api/meta`, `/api/pipeline`, `/api/sources` | Store and pipeline state | Data panel |
| `GET /api/tiles/{z}/{x}/{y}.png` | Heat tiles (MODIS, VIIRS, confirmed) | Map layer **Detections** |
| `GET /api/incidents` | EMS, EFFIS, Sentinel-2 polygons | Map layer **Detections** → Incidents |
| `GET /api/summary`, `/api/counts` | Counts by sensor, platform, match status, confidence | Data panel |
| `GET /api/export.csv` | Harmonized detections | Data panel → Export |
| `POST /api/import` | Any FIRMS CSV into the pipeline | Data panel → Import |
| `GET /api/weather?lat&lon&time` | Fire weather from NASA POWER or DWD ICON | Fire card in Active fires |
| `GET /api/points` | Binary detections for a view | not used (tiles are used instead) |
| `GET /api/forecast`, `GET /api/risk` | LightGBM cell forecast and country risk | Data panel shows whether a model is trained |
| `POST /api/model/train`, `/api/model/runs`, `/api/model/stream` | Training jobs for the forecast | Data panel lists runs |
| `GET /api/regions`, `/api/soil/*` | Country outlines, bare-ground layer | not used |

The forecast models need the `weather` step (NASA POWER on a 1° grid) before training, which takes hours on first
run, so the Docker setup skips it. Remove `weather` and `forecast` from `PIPELINE_SKIP`, wait for the steps, then
`curl -X POST localhost:8002/api/model/train`.

## Context API (8003)

Fuel, terrain and dryness context from the research repository's context data, served for the first time:

- CORINE Land Cover 2018 (country shares and a 0.25° grid)
- Copernicus DEM GLO-90 terrain (country statistics and a 0.25° grid)
- ERA5-Land soil moisture, monthly, Jan 2024 to Oct 2026, with anomalies against 2001–2020 (1° grid)
- MODIS Terra NDVI and land surface temperature, monthly, same period and baseline (1° grid)

Model: **dryness** = mean of the topsoil and root-zone soil moisture deficits and the land surface temperature anomaly,
as z-scores; **fuel risk** = wildland fuel share × the probability that a normal month is wetter.

| Endpoint | Returns | Used in the app |
| --- | --- | --- |
| `GET /health`, `GET /months` | Coverage, sources, method | Method dialog |
| `GET /countries?month=YYYY-MM` | Fuel, terrain, moisture anomalies, dryness and rank per EU country | API only |
| `GET /countries/{iso2 or iso3}?month=` | One country with its monthly series | **Fuel and dryness** card |
| `GET /point?lat&lon&month=` | Land cover, terrain and anomalies at a point; land and terrain for any date | Fire card, custom areas |
| `GET /grid/{dryness,soil,ndvi,lst,fuel,steep}?month=&west&south&east&north` | Grid cells for a map layer | Map layer **Dryness** |

Months outside Jan 2024 to Oct 2026 return 404 for the monthly endpoints; `/point` then returns land and terrain only.

## Regional firefighters

`amber-index/amber/firefighters_regions.csv` holds the regional estimates (north, east, south, west) of the national
firefighter totals for 26 EU countries, source period 2025 except Slovenia (2010–2024) and Ireland (2024). Austria's
regional values add up to 1,947 against a national base of 5,200; their proportions are kept and scaled to the base
(`rescaled: true` in `/regions`). Malta has no estimate. Region shapes and anchors are in
`amber-index/amber/regions_geo.json`, built from the frontend's borders by `frontend/scripts/build-regions.mjs`.
