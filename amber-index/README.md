# Amber Index

**Where is Europe's wildfire danger outgrowing its firefighting readiness?** The Amber Index scores every EU-27 country
and year from 0 to 100: higher means more fire danger relative to the people, money and stations available to fight it.
It combines burned-area records (GWIS), Eurostat statistics on firefighters, fire-protection spending, population and
area, and OpenStreetMap fire stations. A LightGBM model learns which of these predict next year's burned area, and its
SHAP values inform (half of) the weights. Bands: **Green** < 40 ≤ **Amber** < 60 ≤ **Red**. The scores, explanations and a
weight-slider endpoint are served by a FastAPI service for a front-end.

## How the score works

| Pillar | Direction | Indicators (all scaled 0–1) |
|---|---|---|
| Danger | raises | burned area as % of land |
| Hazards | raises | GDACS alerts in the country (floods, earthquakes, droughts, storms, volcanoes, wildfires): all alerts per 100,000 km² weighted Green 0.5, Orange 3, Red 9, plus major (Orange/Red) alerts |
| Workforce | lowers | professional firefighters per 100k inhabitants (+ volunteers per 100k if `data/manual/volunteers.csv` is filled) |
| Capability | lowers | fire-protection spending per inhabitant; fire stations per 1,000 km² and per 100k inhabitants |

`Amber = 100 · sigmoid(k · (Danger − Readiness − c))` where Danger − Readiness = w_D·D + w_H·H − w_W·W − w_C·C with pillar
weights summing to 1, `c` is the median of the training years and `k` maps their 5th–95th percentile to about 5–95.
Preprocessing (log1p, 5th/95th percentile winsorising, min-max) and the `IterativeImputer` are fitted on 2014–2020 only.

**Power** (0–100) estimates the firefighting capacity a country can actually use: readiness reduced by the hazard
pressure it is under at the same time, `power = 100 · readiness · (1 − 0.5 · pressure)`, with pressure = ½ danger + ½ hazards.
A country with strong services but many simultaneous hazards has less power to spare.

**Simulations** force hazard events on any country. The annual hazard indicators rise, and the emergency ties up crews:
each event demands firefighters by severity (Green 50, Orange 500, Red 2,000, an assumption), and power falls by the share
of the national workforce that demand consumes (capped at 95%; the rest is reported as unmet demand). In the latest year a
multi-hazard emergency (Red flood, Red wildfire, Orange earthquake) drops the three strongest countries by 4 to 11 places;
Cyprus, the smallest service, is left with about 3,200 firefighters of unmet demand.

**Assistance optimizer.** When a country needs firefighters it cannot cover, an integer program splits the request
across the other 26 countries. Each donor can send at most 5% of its workforce, reduced by its own hazard pressure and
by crews tied up in its own emergencies; donors with power below 20 send none; nobody covers more than 40% of the
request; a donor sends at least a team of 20 or nothing. Each firefighter costs
`w · distance / max distance + (1 − w) · (1 − power / 100)`, so `w` trades **location** (distance from the donor's
centroid to the incident) against **power** (prefer donors with capacity to spare). With `w = 1` help for Portugal
comes from France, Italy and Ireland; with `w = 0` from Poland, Germany and Czechia.

## Run the notebook

```sh
python3.12 -m venv .venv          # any Python 3.11+; LightGBM on macOS also needs: brew install libomp
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/amber_index.ipynb --output amber_index.ipynb
```

All downloads are cached in `data/raw/`; set `REFRESH = True` in the first cell to fetch again. The OpenStreetMap
station count (Overpass API) is slow and the public server is often busy: it retries with backoff, tries mirrors, saves
progress per country and only runs once. Countries it could not reach stay missing and are imputed; to fill them in
later, run the line below, then rerun the notebook:

```sh
.venv/bin/python -c "from amber.data import load_osm_stations; load_osm_stations(retry_missing=True, budget_s=3600)"
```

## Start the API

```sh
.venv/bin/uvicorn api.main:app --reload
```

The trained model ships in `artifacts/` (written by the last section of the notebook), so the API starts without
retraining. Rerun the notebook to refresh it; the API refuses to start if any artifact is missing.

```sh
curl localhost:8000/health
curl localhost:8000/countries
curl "localhost:8000/scores?year=2024"
curl localhost:8000/scores/GR
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"burned_ha": 25000, "area_km2": 238398, "population": 19000000, "firefighters": null, "spending_meur": 900, "stations": null}'
curl -X POST localhost:8000/score/custom-weights -H 'Content-Type: application/json' \
  -d '{"year": 2024, "danger": 2, "workforce": 1, "capability": 1}'
curl -X POST localhost:8000/simulate -H 'Content-Type: application/json' \
  -d '{"shocks": [{"iso2": "CZ", "preset": "multi_hazard"}, {"iso2": "CY", "events": [{"type": "WF", "alert": "Red", "count": 1}]}]}'
curl -X POST localhost:8000/optimize/assistance -H 'Content-Type: application/json' \
  -d '{"recipient": "CY", "shocks": [{"iso2": "CY", "preset": "multi_hazard"}], "weight_location": 0.5}'
curl -X POST localhost:8000/optimize/assistance -H 'Content-Type: application/json' \
  -d '{"recipient": "PT", "need": 3000, "incident_lat": 40.1, "incident_lon": -8.0, "weight_location": 1}'
```

| Endpoint | Returns |
|---|---|
| `GET /health` | status, training years, CV metrics |
| `GET /model` | the exported model: features, pillars, weights (ML, equal, final), score scale, CV metrics |
| `GET /countries` | the 27 ISO2 codes and names |
| `GET /scores?year=` | every country for a year (default latest), sorted by score |
| `GET /scores/{iso2}?year=` | one country with its top 3 reasons, e.g. "High burned area pushes the score up 12 pts" |
| `POST /predict` | raw indicators (any may be null) → predicted next-year burned share, score, band, SHAP contributions |
| `POST /score/custom-weights` | pillar weights → recomputed scores and bands (for weight sliders) |
| `GET /simulate/presets` | named hazard shocks (major flood, earthquake, wildfire season, drought, multi-hazard) |
| `POST /simulate` | force hazards on countries → power, ranks, crew demand and unmet demand before and after |
| `POST /optimize/assistance` | recipient + need (or shocks that create one) → which countries send how many firefighters |

Tests: `.venv/bin/python -m pytest`.

## Validation

See section 7 of the notebook and `artifacts/metadata.json`. Models predict next year's burned area (% of land) and
are compared by leave-one-country-out and a time split (train years ≤ 2020, test ≥ 2021), with a naive persistence
forecast (next year = this year) as a reference. Rows are never shuffled.

## Limitations

- **Firefighter data is thin.** Eurostat's ad hoc LFS extractions cover the EU-27 only and exist for 2022, 2023 and 2024.
  Other years copy the nearest year (flagged `imputed`); Austria, Ireland, Luxembourg and Malta have no publishable value
  and are imputed. Several values carry Eurostat's low-reliability flag `u`.
- **Volunteers are not counted by Eurostat.** Countries that rely on volunteer brigades (e.g. Austria, Germany) look
  under-staffed unless `data/manual/volunteers.csv` is filled in.
- **OpenStreetMap station counts vary in completeness** between countries and are a single current snapshot applied to
  every year. Countries whose Overpass query failed are imputed.
- **GDACS coverage is uneven.** It logs every M4.5+ earthquake (so seismic countries like Greece collect many Green
  alerts) and its flood and wildfire alerting expanded over the years. Green alerts are down-weighted for that reason.
- **Burned area covers wildfires, not building fires**, and only fires large enough for satellites (GWIS).
- Fire-protection spending (COFOG GF0302) covers all fire services, not only wildfire.
- Crew demand per event (Green 50, Orange 500, Red 2,000), the 5% deployable share and the 40% cap are planning
  assumptions, not measured values; the API lets you change all of them except the demand table.
- Distances are straight lines between country centroids (or to a given incident point), not travel times.
- With 27 countries and about ten years, the model has a few hundred rows; its weights are a guide, which is why the
  final weights blend 50% model and 50% equal.
