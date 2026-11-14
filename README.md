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

# 2. normalize -> harmonize -> analysis tables -> web export
uv run python -m firms.build                       # 1 km, 180 min match window
uv run python -m firms.build --radius-km 0.75 --window-min 120

# 3. run the map
cd web && bun run dev         # or: bun run build && bun run preview
```

`--bbox` takes `W,S,E,N` or `world`. Standard-processing (`_SP`) archives are used where available,
and NRT for the most recent days. The FIRMS API accepts at most 5 days per request, so the fetcher
splits ranges into 5-day windows.

## Outputs

| Path | Contents |
| --- | --- |
| `data/raw/<SOURCE>/<YYYYMMDD>_<days>.csv` | raw FIRMS responses (gitignored) |
| `data/normalized/<SOURCE>/*.parquet` | one common schema per detection (`firms/config.py: SCHEMA`) |
| `data/analysis/harmonized.parquet` | all detections with `cluster_id` and `harmonized_status` |
| `data/analysis/clusters.csv` | one row per MODIS+VIIRS cluster |
| `data/analysis/daily_summary.csv` | per UTC day: counts, FRP and matched share per sensor |
| `web/public/data/` | `index.json` and `days/YYYY-MM-DD.json` for the map (gitignored) |

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
