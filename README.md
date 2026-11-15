# data-science

Data playground for risk scores. Current project: **NASA FIRMS active-fire hotspots**, harmonizing
MODIS (Terra/Aqua) and VIIRS (S-NPP, NOAA-20, NOAA-21) detections into one schema, with historical
analysis and a simple React map.

## Setup

1. Get a free FIRMS MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key
2. Put it in `.env` (comma-separate several keys to spread the rate limit):
   ```
   KEY=your_key_here
   ```
3. Python: [uv](https://docs.astral.sh/uv/) manages the environment from `pyproject.toml` / `uv.lock`:
   ```
   uv sync
   ```
4. Web app: [bun](https://bun.sh) manages dependencies from `web/bun.lock`:
   ```
   cd web && bun install
   ```

## Running

```
./start.sh                        # installs, builds the web app if needed, serves http://127.0.0.1:8000
```

Or by hand: `cd web && bun run build`, then `uv run python -m server`.

The Python server (FastAPI + SQLite, `data/firms.db`) runs every pipeline step on start in a background
thread and serves the map and its API meanwhile. For web development run `bun run dev` in `web/`;
it proxies `/api` to the server.

### Pipeline steps and the ledger

| Step | Does | Reruns when |
| --- | --- | --- |
| `fetch` | FIRMS CSVs into `data/raw` (needs `KEY` in `.env`) | the fetch window changes (end = today, so once a day) |
| `normalize` | raw CSVs into `data/normalized` parquet | the set of raw files changes |
| `harmonize` | MODIS/VIIRS clusters, writes the `detections` and `clusters` tables | `normalize` reran, or the match radius/window changed |
| `ems` | Copernicus EMS areas into `incidents` | once a day |
| `sentinel2` | Sentinel-2 burn scars into `incidents` (slow, resumable) | `harmonize` reran |
| `tag` | marks detections inside an incident | `harmonize` reran or the incidents changed |
| `stats` | per-day counts, per-cell counts (`bin_day`), per-incident counts | `tag` reran |

The `pipeline_steps` table records each run (inputs, status, duration, rows, error). A step is skipped when a
`done` row with the same inputs exists. Dependents include the upstream run id, so rerunning a step reruns
what follows it.

```
uv run python -m server status             # what ran, what is pending
uv run python -m server run [step ...]     # run pending steps without serving
uv run python -m server reset harmonize    # delete its rows and those after it; next run redoes them
```

Settings (environment): `FIRMS_START`, `FIRMS_BBOX` (default `-25,-20,45,72`), `FIRMS_FAMILIES`,
`FIRMS_RADIUS_KM`, `FIRMS_WINDOW_MIN`, `FIRMS_S2_MAX_EVENTS`, `PIPELINE_SKIP=fetch,sentinel2` to leave steps out
(e.g. offline), `PIPELINE_ON_START=0` to serve without running anything, `FIRMS_DB` for the database path.

### API

`/api/meta`, `/api/pipeline`, `/api/points`, `/api/counts`, `/api/incidents`, `/api/tiles/{z}/{x}/{y}.png`,
`/api/tiles/status`, `/api/weather?lon=&lat=&time=` (fire-weather features for a point and hour: NASA POWER for
times over five days old, DWD ICON via Open-Meteo for recent days and up to six days ahead; `source=nasa|dwd`
forces one; provider responses are cached in the `weather_cache` table, NASA 30 days, DWD 15 minutes). The map asks for the
current view (date range plus bounding box). `/api/points` returns binary columns; when a view holds more than
`FIRMS_MAX_POINTS` (60k) detections the map shows heat tiles instead (density on a log scale, orange MODIS,
blue VIIRS, red = confirmed inside an incident).

Tiles are cached on disk under `data/tiles/<data version>/<from>_<to>/<layer mask>/<z>/<x>/<y>.png`, one file
per zoom level (a blank tile is a zero-byte file), so a repeat view costs about a millisecond. Zoom 0-7 is drawn
from the per-day, per-0.1° `bin_day` table, higher zooms from the detections themselves. A new pipeline result
starts a new cache version; the old one is deleted. The cache is capped by `FIRMS_TILE_CACHE_MB` (default 2000).

**Tiles are generated in the background, and never block a request.** A thread in the server
(`server/tilegen.py`) renders every zoom level from 0 up to `FIRMS_TILE_BG_MAX_ZOOM` (default 9; about 16k tiles,
roughly an hour at low priority) for the three presets in the sidebar (all data, last year, last 30 days), lowest
zoom first so each level is usable early. It stands by while the pipeline is rewriting detections and starts over
when new data arrives. A tile that has not been generated yet is rendered on request, in the request itself, and
cached, so the map never waits for the background work; the generator skips tiles that already exist. Deeper
zooms, other date ranges and other layer combinations are rendered on first request only. Progress is at
`/api/tiles/status` and shown in the sidebar. `FIRMS_TILE_BG=0` turns the generator off (tiles still render on
request).

### Independent European incidents

NASA detections are only checked against sources that do not use MODIS or VIIRS data: Copernicus EMS and
Sentinel-2 burn scars. EFFIS burnt areas (`firms/effis.py`, 2016 onwards) are loaded as a third incident source, but
they are mapped from MODIS and Sentinel-2 imagery, so they only say which detections belong to a mapped wildfire (the
forecast label) and must not be used to judge how accurate the detections are.

- **Copernicus EMS** rapid-mapping wildfire activations (`category=fire`), using the AOI extents from the
  dashboard API. These are event-level mapped regions made from separate imagery, not burn perimeters.
  Activations outside Europe are dropped. Some non-rapid `EMSN` activations return 403 and are skipped.
- **Sentinel-2 burn scars** (`firms/sentinel2.py`), computed here. Detections are linked into fire events
  (within 2 km and 2 days). For each event, a clear Sentinel-2 L2A scene before the first detection and one
  after the last are taken from Microsoft Planetary Computer. Burn severity is dNBR = NBR(pre) - NBR(post),
  NBR = (B8A - B12) / (B8A + B12), with a 0.27 threshold on clear land. A scar counts when it is at least
  5 ha and within 1 km of a detection. Each event gets a verdict (`confirmed`, `not_confirmed`,
  `unverifiable`, `too_large`) in `data/incidents/s2_verdicts.csv`. Only confirmed scars become incidents.
  Scars confirm burning days to weeks after the fact, so `not_confirmed` does not prove a false alarm.
- **Checked and rejected**: EGIF (Spain), the JRC EFFIS fire database and BDIFF (France) either stop before
  2024 or lag too far behind to cover 2024-2026. Portugal's ANEPC/fogos.pt feed had no stable documented
  API.

The `tag` step tags each NASA detection with the incident it falls inside. The matching buffer and date
slack depend on the source: EMS uses 1 km and 2 days, Sentinel-2 uses 0.3 km and an exact window from the
pre to post scene. Buffers are in degrees, so longitude is slightly under-counted at high latitude.
Unmatched detections are kept but marked unconfirmed.

`--bbox` takes `W,S,E,N` or `world`. Standard-processing (`_SP`) archives are used where available,
and NRT for the most recent days. The FIRMS API accepts at most 5 days per request, so the fetcher
splits ranges into 5-day windows.

### Imports, exports and seeding without a key

- `POST /api/import?name=<file>` with a FIRMS CSV as the body (MODIS or VIIRS, S-NPP, NOAA-20 or NOAA-21, SP or NRT,
  from the area API, the archive download tool or the yearly country files). Rows are split by source into
  `data/raw/<SOURCE>/import_<name>_<hash>.csv` and the pipeline starts. Importing the same file again replaces it.
- `GET /api/export.csv?from=&to=&west=&south=&east=&north=[&sensor=MODIS|VIIRS]` streams the harmonized detections:
  position, time, sensor, platform, product, day/night, confidence, FRP, brightness, cross-sensor status, cluster and
  incident.
- `GET /api/summary?from=&to=&...` counts by platform, match status and confidence, per-day MODIS/VIIRS counts, and
  the date span of the whole store. `GET /api/sources` lists the raw files.
- `python -m server seed --years 2024,2025` fills `data/raw` from the yearly FIRMS country archives of the EU-27
  (no MAP_KEY needed); `--from-dir <folder>` does the same from CSVs on disk, `--if-empty` only when `data/raw` is empty.

### Docker

`Dockerfile.server` builds the FIRMS server (seeds on first start, then serves on port 8002, data in `/app/data`).
`amber-index/Dockerfile` builds the Amber API (port 8001). Both are wired together with the frontend in the
`arecibo` repository's `docker-compose.yml`.

## Outputs

| Path | Contents |
| --- | --- |
| `data/firms.db` | SQLite: `detections`, `incidents`, `clusters`, `day_stats`, `bin_day`, `pipeline_steps`, `weather_grid`, `weather_issued`, `country_outlook` |
| `data/risk/` | country risk models (one per lead and feature set), `outlook.npz` and `meta.json` for the report |
| `data/tiles/` | cached heat tiles (regenerable) |
| `data/raw/<SOURCE>/<YYYYMMDD>_<days>.csv` | raw FIRMS responses (gitignored) |
| `data/normalized/<SOURCE>/*.parquet` | one common schema per detection (`firms/config.py: SCHEMA`) |
| `data/incidents/raw/` | cached EMS and per-event Sentinel-2 results, so runs resume |
| `data/incidents/s2_verdicts.csv` | one row per checked fire event with its Sentinel-2 verdict |

## Harmonization notes

- **Confidence:** MODIS 0–100 maps to low (<30), nominal (30–79), high (≥80). VIIRS l/n/h maps directly.
  The raw value is kept.
- **Brightness:** `bt_mid_k` is MODIS `brightness` (band 21) or VIIRS `bright_ti4`. `bt_secondary_k` is
  MODIS `bright_t31` or VIIRS `bright_ti5`. These are close, but not identical, channels.
- **Matching:** a MODIS and a VIIRS detection are linked when they are within `--radius-km` and
  `--window-min` of each other. Clusters are connected components of those links, so chains of
  nearby fires can merge into one large cluster. Check `clusters.csv` if that matters for your analysis.
- **Dedupe:** NRT and SP copies of the same detection are collapsed, keeping SP.

## Tests

```
uv run pytest
```

## Fire forecast

The **Forecast** tab has a timeline over all days (plus six future days), the detections of the selected day,
and the cells where a LightGBM model expects a COMMON fire (a detection inside an EU incident) in the next
`FIRMS_HORIZON` (3) days. For past days it marks hits, misses and false alarms against what was detected.

- Train from the UI (Train / Stop / Resume). Progress and loss stream from `GET /api/model/stream` (SSE).
  Weights go to `model_checkpoints` every `FIRMS_CHECKPOINT_EVERY` (25) rounds and on stop; after a crash the run is
  `interrupted` and Resume continues from its checkpoint. Predictions use the newest `done` run.
- Train on days up to `FIRMS_TRAIN_END` (2025-12-31), with a random 20% of those cell-days for validation;
  every later day is the test set (PR-AUC, precision and recall are shown per run).
- `FIRMS_LABEL=both` (default: a fire seen by MODIS and VIIRS, consistent for every year since 2012) or `confirmed`
  (the UI's COMMON layer, a detection inside an EU incident; incident data only reaches back to 2024).
- **Every cell is scored on every day.** Candidate cells are all 0.1 degree cells with a detection before the train
  end. Training keeps every positive, 10% of the negatives in cells with a fire in the last 90 days and 2% of the
  rest, each weighted by 1/keep, so losses, thresholds and probabilities stand for all cell-days.
- Features: fire history per cell (`bin_day`), neighbours, season, CORINE bare ground, daily NASA POWER weather on a
  1 degree grid (every cell touching Europe, from 2012), and the weather of the next 3 days (`fc_*`: peak wind,
  peak temperature anomaly, rain, dry air). The model is trained with the reanalysis of those days (what a perfect
  forecast would say); in use, days beyond the reanalysis come from the `forecast` step (DWD ICON via Open-Meteo,
  `weather_forecast`, daily). Past days are therefore scored with perfect weather forecasts, which flatters them.
  The `context` step embeds `context-data/` into `soil_cells` and `soil_squares`. The `weather` step is slow on
  first run and resumable (`FIRMS_WEATHER=0` or `PIPELINE_SKIP=weather,forecast` skips both).
- Older history: `FIRMS_START=2012-01-01` fetches MODIS and VIIRS from 2012. Skip `ems,sentinel2` when you only
  want the NASA data (`PIPELINE_SKIP=ems,sentinel2`).
- The map colours each country from green to red by its risk: the expected number of fire cells in the next days
  (sum of the cell probabilities), ranked against that country's own year (`country_risk`, `/api/risk?date=`).
- API: `/api/model/{train,stop,resume/{id},runs,stream}`, `/api/forecast?date=`, `/api/risk?date=`, `/api/regions`,
  `/api/soil/tiles/{z}/{x}/{y}.png?cls=`.

## Country fire risk, 1 to 7 days ahead (`server/risk`)

The question this model answers: **on day D, how many 0.1 degree cells of a country will burn on each of the next
seven days**, and is that unusual for the country. It was built to test whether the Greek fires of 2025 could have
been seen coming, so it is strict about time:

- Trained on days up to `2024-12-31` only. Every "normal" (usual fires for the day of the year, usual temperature,
  the cell's usual fire weather) is computed from those days too.
- The number of trees is picked on 2023 and 2024, held out; those held-out predictions also set each country's alert
  threshold (its 90th percentile over May to October). Nothing from the tested season tunes anything.
- Training uses the weather that came. A replay is fed the **forecasts issued at the time**: ECMWF IFS runs made 1 to
  7 days earlier (Open-Meteo Previous Runs API, from March 2024, `weather_issued`), mapped onto the NASA POWER
  reanalysis with a per-cell line fitted on 2024.

Unit: a patch, the part of a 1 degree weather cell inside one country. One LightGBM (Poisson) model per lead predicts
the burning cells of a patch on D+k; a country is the sum of its patches. Features: fire history of the patch, its
neighbours and its country up to D; dryness on D (Canadian FWI moisture codes, rain of 30 days, days since rain, soil
wetness, heat of the last week); the forecast for D+k (fire weather index carried forward through the forecast days,
wind, heat against normal, humidity, rain on the way); season and place. A second model per lead (`conditions`) gets
no fire history at all, to show what weather alone can foresee.

```
uv run python -m server.risk.forecasts --country Greece --from 2025-04-01 --to 2025-10-31   # issued forecasts (resumable)
uv run python -m server.risk.forecasts --country Greece --from 2024-04-01 --to 2024-10-31   # the year the correction is fitted on
uv run python -m server.risk.model train        # about 30 minutes; writes data/risk/ and the country_outlook table
uv run python -m server.risk.model predict      # reuse the saved models (after fetching more forecasts or new days)
uv run python -m server.risk.report             # Greece 2025 against persistence, the usual, and the fire weather index
uv run python -m server.risk.report --country Spain --year 2026 --target mv
```

Once `country_outlook` is filled, `/api/risk?date=` (the country colours of the Forecast tab) answers from this model:
expected burning cell-days over the next 7 days, ranked against the country's own stored days. Each country says
whether its numbers came from issued forecasts (`weather: "issued"`) or from the weather that came (`"perfect"`,
which flatters past days). Without it the endpoint falls back to the cell model's sums.

Limits: issued forecasts are fetched per country (the free service is rate limited), so other countries replay with
perfect weather. The state on day D uses the reanalysis of D, which in live use arrives days late. The label is a
fire seen by both MODIS and VIIRS, which includes some crop burning.
