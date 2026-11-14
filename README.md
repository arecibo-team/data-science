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

# 2. European incidents (EFFIS perimeters + Copernicus EMS wildfire areas)
uv run python -m firms.incidents                   # resumable; refreshes the current season

# 3. normalize -> harmonize -> incident match -> analysis tables -> web export
uv run python -m firms.build                       # 1 km, 180 min match window
uv run python -m firms.build --radius-km 0.75 --window-min 120

# 4. run the map
cd web && bun run dev         # or: bun run build && bun run preview
```

### European incidents

`firms.incidents` harvests real fire incidents in Europe:

- **EFFIS** burnt-area perimeters (JRC/Copernicus WFS, layers `ms:modis.ba.poly.<year>` and `ms:modis.ba.poly.season`).
  Only fires of about 30 ha or more are mapped. Countries are restricted to Europe (`firms/config.py: EUROPE_COUNTRIES`);
  Turkey and Russia are excluded.
- **Copernicus EMS** rapid-mapping wildfire activations (`category=fire`), using the AOI extents from the
  dashboard API. Activations with a non-European continent are dropped. Some non-rapid `EMSN` activations
  return 403 and are skipped.
- **National reports**: not included. No stable machine-readable download was available.

`firms.build` tags each NASA detection with the incident it falls inside: within 1 km of a perimeter (the
buffer is in degrees, so longitude is slightly under-counted at high latitude) and dated within two days of
the incident's start and end. Unmatched detections are kept but marked unconfirmed. The map's
"Only confirmed European incidents" filter hides them.

`--bbox` takes `W,S,E,N` or `world`. Standard-processing (`_SP`) archives are used where available,
and NRT for the most recent days. The FIRMS API accepts at most 5 days per request, so the fetcher
splits ranges into 5-day windows.

## Outputs

| Path | Contents |
| --- | --- |
| `data/raw/<SOURCE>/<YYYYMMDD>_<days>.csv` | raw FIRMS responses (gitignored) |
| `data/normalized/<SOURCE>/*.parquet` | one common schema per detection (`firms/config.py: SCHEMA`) |
| `data/incidents/incidents.geojson` | European incident polygons from EFFIS and EMS (gitignored) |
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
