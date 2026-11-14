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
cd web && bun run build && cd ..   # once, and after web changes
uv run python -m server            # http://127.0.0.1:8000
```

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
| `stats` | per-day and per-incident counts | `tag` reran |

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

`/api/meta`, `/api/pipeline`, `/api/points`, `/api/grid`, `/api/incidents`. The map asks for the current view
(date range plus bounding box). `/api/points` returns binary columns; when a view holds more than
`FIRMS_MAX_POINTS` (250k) detections the map gets per-cell counts from `/api/grid` instead.

### Independent European incidents

NASA detections are only checked against sources that do not use MODIS or VIIRS data. EFFIS
burnt areas were dropped because they are derived from those same sensors, so matching against
them would be circular.

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

## Outputs

| Path | Contents |
| --- | --- |
| `data/firms.db` | SQLite: `detections`, `incidents`, `clusters`, `day_stats`, `pipeline_steps` |
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
