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

## Pipeline

```
# 1. download raw CSVs (resumable; existing files are skipped)
uv run python -m firms.fetch --start 2025-08-01 --end 2025-08-31 --bbox 15,-20,35,-5

# 2. independent European incidents
uv run python -m firms.incidents                   # Copernicus EMS areas + confirmed Sentinel-2 scars
uv run python -m firms.sentinel2 --max-events 300  # optional, slow: Sentinel-2 burn-scar check (resumable)

# 3. normalize -> harmonize -> incident match -> analysis tables -> web export
uv run python -m firms.build                       # 1 km, 180 min match window
uv run python -m firms.build --radius-km 0.75 --window-min 120

# 4. run the map
cd web && bun run dev         # or: bun run build && bun run preview
```

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

`firms.build` tags each NASA detection with the incident it falls inside. The matching buffer and date
slack depend on the source: EMS uses 1 km and 2 days, Sentinel-2 uses 0.3 km and an exact window from the
pre to post scene. Buffers are in degrees, so longitude is slightly under-counted at high latitude.
Unmatched detections are kept but marked unconfirmed.

`--bbox` takes `W,S,E,N` or `world`. Standard-processing (`_SP`) archives are used where available,
and NRT for the most recent days. The FIRMS API accepts at most 5 days per request, so the fetcher
splits ranges into 5-day windows.

## Outputs

| Path | Contents |
| --- | --- |
| `data/raw/<SOURCE>/<YYYYMMDD>_<days>.csv` | raw FIRMS responses (gitignored) |
| `data/normalized/<SOURCE>/*.parquet` | one common schema per detection (`firms/config.py: SCHEMA`) |
| `data/incidents/incidents.geojson` | independent incident polygons: EMS areas and confirmed Sentinel-2 scars (gitignored) |
| `data/incidents/s2_verdicts.csv` | one row per checked fire event with its Sentinel-2 verdict |
| `data/incidents/raw/s2/` | per-event Sentinel-2 results, cached so runs resume (gitignored) |
| `data/analysis/harmonized.parquet` | all detections with `cluster_id`, `harmonized_status` and `incident_id` |
| `data/analysis/clusters.csv` | one row per MODIS+VIIRS cluster |
| `data/analysis/daily_summary.csv` | per UTC day: counts, FRP and matched share per sensor |
| `web/public/data/` | `index.json`, `days/YYYY-MM-DD.json` (detections) and `incident-days/YYYY-MM-DD.json` (incidents) for the map (gitignored) |

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
