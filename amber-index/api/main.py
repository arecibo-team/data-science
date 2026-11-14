"""Amber Index API. Serves the artifacts written by notebooks/amber_index.ipynb.

Run:  uvicorn api.main:app --reload
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from amber import ARTIFACTS_DIR, NAMES
from amber.features import FEATURES, PILLARS, compute_rates
from amber.model import load_artifacts
from amber.power import power_index
from amber.optimize import AssistanceParams, donor_table, optimize_assistance
from amber.simulate import PRESETS, run_scenario
from amber.scoring import amber_score, band, contributions, gap, pillar_scores, reason_text

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("AMBER_ARTIFACTS", ARTIFACTS_DIR))
    try:
        art = load_artifacts(path)
    except FileNotFoundError as e:
        raise RuntimeError(f"{e} The API needs these files: run the notebook first.") from e
    import shap
    w = art["weights"]
    feats = art["metadata"]["feature_order"]
    STATE.update(art)
    STATE["features"] = feats
    STATE["final_weights"] = {"within": {f: w[f]["w_final"] for f in feats},
                              "pillar": {p: w["_pillars"][p]["w_final"] for p in PILLARS}}
    STATE["scale"] = art["metadata"]["scale"]
    STATE["reference"] = pd.Series(art["metadata"]["reference"])[feats]
    STATE["df"] = pd.DataFrame(art["scores"])
    STATE["explainer"] = shap.TreeExplainer(art["model"])
    yield
    STATE.clear()


app = FastAPI(title="Amber Index API", version="0.1.0", lifespan=lifespan,
              description="Wildfire protection-gap score (0-100) for EU-27 countries. Higher = more danger relative to readiness.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- schemas
class CVMetric(BaseModel):
    model: str
    validation: str
    MAE: float
    Spearman: float


class Health(BaseModel):
    status: str
    created: str
    training_years: list[int]
    model_years: list[int]
    latest_year: int
    features: list[str]
    cv_metrics: list[CVMetric]


class Country(BaseModel):
    iso2: str
    name: str


class Score(BaseModel):
    iso2: str
    name: str
    year: int
    amber: float
    band: str
    rank: int
    rank_lo: int
    rank_hi: int
    danger: float
    hazards: float
    workforce: float
    capability: float
    readiness: float
    hazard_pressure: float
    power: float
    power_rank: int
    hazard_events: dict[str, int]
    archetype: str
    imputed: dict[str, bool]
    firefighters_low_reliability: bool


class Reason(BaseModel):
    feature: str
    points: float = Field(description="Points this feature adds to the score vs an average country-year")
    text: str


class ScoreDetail(Score):
    reasons: list[Reason]


class PredictRequest(BaseModel):
    """Raw indicators for one country-year. Any value may be null; nulls are imputed."""
    iso2: str | None = None
    burned_ha: float | None = Field(None, ge=0)
    area_km2: float | None = Field(None, gt=0)
    population: float | None = Field(None, gt=0)
    firefighters: float | None = Field(None, ge=0)
    volunteers: float | None = Field(None, ge=0)
    spending_meur: float | None = Field(None, ge=0)
    stations: float | None = Field(None, ge=0)
    hazard_load: float | None = Field(None, ge=0, description="Alert-weighted GDACS events this year: Green 0.5, Orange 3, Red 9")
    major_load: float | None = Field(None, ge=0, description="Alert-weighted Orange and Red GDACS events this year")


class ShapItem(BaseModel):
    feature: str
    value: float = Field(description="Contribution to the model output (log1p of next-year burned share)")


class PredictResponse(BaseModel):
    predicted_burned_share_pct_next_year: float
    amber: float
    band: str
    power: float
    pillars: dict[str, float]
    imputed_features: list[str]
    shap_base_value: float
    shap: list[ShapItem]


class CustomWeights(BaseModel):
    year: int | None = None
    danger: float = Field(ge=0)
    workforce: float = Field(ge=0)
    capability: float = Field(ge=0)
    hazards: float | None = Field(None, ge=0)

    @model_validator(mode="after")
    def not_all_zero(self):
        if self.danger + self.workforce + self.capability + (self.hazards or 0) <= 0:
            raise ValueError("at least one pillar weight must be positive")
        return self


class CustomScore(BaseModel):
    iso2: str
    name: str
    amber: float
    band: str
    rank: int


class CustomWeightsResponse(BaseModel):
    year: int
    weights: dict[str, float]
    scores: list[CustomScore]


# ---------------------------------------------------------------- helpers
def _year(year: int | None) -> int:
    df = STATE["df"]
    y = year or int(df["year"].max())
    if y not in set(df["year"]):
        raise HTTPException(404, f"No scores for {y}. Available: {int(df.year.min())}-{int(df.year.max())}")
    return y


def _score(row: pd.Series) -> dict:
    return {k: row[k] for k in Score.model_fields}


# ---------------------------------------------------------------- endpoints
@app.get("/health", response_model=Health)
def health():
    m = STATE["metadata"]
    return Health(status="ok", created=m["created"], training_years=m["training_years"], model_years=m["model_years"],
                  latest_year=m["latest_year"], features=m["feature_order"], cv_metrics=m["cv_metrics"])


class PillarWeight(BaseModel):
    w_ml: float
    w_equal: float
    w_final: float


class FeatureInfo(BaseModel):
    feature: str
    label: str
    pillar: str
    direction: int
    w_ml: float
    w_equal: float
    w_final: float


class ModelInfo(BaseModel):
    model: str
    target: str
    features: list[FeatureInfo]
    pillars: dict[str, PillarWeight]
    scale: dict[str, float]
    alpha: float
    bands: dict[str, float]
    training_years: list[int]
    model_years: list[int]
    cv_metrics: list[CVMetric]
    created: str


@app.get("/model", response_model=ModelInfo)
def model_info():
    m, w = STATE["metadata"], STATE["weights"]
    feats = [FeatureInfo(feature=f, label=FEATURES[f][3], pillar=FEATURES[f][0], direction=FEATURES[f][1],
                         w_ml=w[f]["w_ml"], w_equal=w[f]["w_equal"], w_final=w[f]["w_final"]) for f in STATE["features"]]
    return ModelInfo(model=type(STATE["model"]).__name__, target=m["target"], features=feats,
                     pillars={p: PillarWeight(**v) for p, v in w["_pillars"].items()}, scale=m["scale"],
                     alpha=m.get("alpha", 0.5), bands=m["bands"], training_years=m["training_years"],
                     model_years=m["model_years"], cv_metrics=m["cv_metrics"], created=m["created"])


@app.get("/countries", response_model=list[Country])
def countries():
    return [Country(iso2=k, name=v) for k, v in sorted(NAMES.items(), key=lambda kv: kv[1])]


@app.get("/scores", response_model=list[Score])
def scores(year: int | None = Query(None, description="Defaults to the latest year")):
    y = _year(year)
    df = STATE["df"]
    return [_score(r) for _, r in df[df.year == y].sort_values("amber", ascending=False).iterrows()]


@app.get("/scores/{iso2}", response_model=ScoreDetail)
def score_one(iso2: str, year: int | None = None):
    iso2 = iso2.upper().replace("EL", "GR")
    if iso2 not in NAMES:
        raise HTTPException(404, f"Unknown country '{iso2}'. Use an EU-27 ISO2 code, see /countries")
    y = _year(year)
    df = STATE["df"]
    row = df[(df.iso2 == iso2) & (df.year == y)].iloc[0]
    x = pd.Series(row["features"])[STATE["features"]]
    pts = contributions(x, STATE["reference"], STATE["final_weights"], STATE["scale"])
    top = pts.reindex(pts.abs().sort_values(ascending=False).index)[:3]
    reasons = [Reason(feature=f, points=round(float(p), 1), text=reason_text(f, float(p), float(x[f]))) for f, p in top.items()]
    return ScoreDetail(**_score(row), reasons=reasons)


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    feats = STATE["features"]
    raw = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in body.model_dump(exclude={"iso2"}).items()}])
    rates = compute_rates(raw)[feats]
    X = STATE["preprocessor"].transform(rates)
    pred_log = float(STATE["model"].predict(X)[0])
    sv = STATE["explainer"].shap_values(X)[0]
    w, scale = STATE["final_weights"], STATE["scale"]
    P = pillar_scores(X, w)
    score = float(amber_score(gap(P, w), scale)[0])
    return PredictResponse(
        predicted_burned_share_pct_next_year=round(max(0.0, float(np.expm1(pred_log))), 5),
        amber=round(score, 2), band=band(score), power=round(float(power_index(P, w, STATE["metadata"].get("alpha", 0.5)).iloc[0]), 2), pillars={p: round(float(P[p].iloc[0]), 4) for p in PILLARS},
        imputed_features=[f for f in feats if pd.isna(rates[f].iloc[0])],
        shap_base_value=float(np.ravel(STATE["explainer"].expected_value)[0]),
        shap=[ShapItem(feature=f, value=round(float(v), 5)) for f, v in sorted(zip(feats, sv), key=lambda t: -abs(t[1]))])


@app.post("/score/custom-weights", response_model=CustomWeightsResponse)
def custom_weights(body: CustomWeights):
    y = _year(body.year)
    df = STATE["df"][STATE["df"].year == y]
    raw = {"danger": body.danger, "workforce": body.workforce, "capability": body.capability,
           "hazards": STATE["final_weights"]["pillar"].get("hazards", 0.0) if body.hazards is None else body.hazards}
    tot = sum(raw.values())
    pw = {k: v / tot for k, v in raw.items()}
    w = {"within": STATE["final_weights"]["within"], "pillar": pw}
    X = pd.DataFrame(list(df["features"]), index=df.index)[STATE["features"]]
    s = pd.Series(amber_score(gap(pillar_scores(X, w), w), STATE["scale"]), index=df.index)
    order = s.sort_values(ascending=False)
    out = [CustomScore(iso2=df.loc[i, "iso2"], name=df.loc[i, "name"], amber=round(float(v), 2), band=band(float(v)), rank=r + 1)
           for r, (i, v) in enumerate(order.items())]
    return CustomWeightsResponse(year=y, weights={k: round(v, 4) for k, v in pw.items()}, scores=out)


class HazardEvent(BaseModel):
    type: Literal["FL", "EQ", "WF", "DR", "TC", "VO"]
    alert: Literal["Green", "Orange", "Red"]
    count: int = Field(1, ge=1, le=50)


class Shock(BaseModel):
    iso2: str
    preset: str | None = None
    events: list[HazardEvent] = []

    @model_validator(mode="after")
    def has_events(self):
        if not self.preset and not self.events:
            raise ValueError("give a preset or a list of events")
        if self.preset and self.preset not in PRESETS:
            raise ValueError(f"unknown preset, use one of {sorted(PRESETS)}")
        return self


class SimulateRequest(BaseModel):
    year: int | None = None
    alpha: float = Field(0.5, ge=0, le=1)
    shocks: list[Shock] = Field(min_length=1)


class SimRow(BaseModel):
    iso2: str
    name: str
    shocked: bool
    power_before: float
    power_after: float
    power_change: float
    power_rank_before: int
    power_rank_after: int
    rank_change: int
    amber_before: float
    amber_after: float
    band_before: str
    band_after: str
    firefighters: float
    crew_demand: float
    committed_share: float
    unmet_demand: float


class SimulateResponse(BaseModel):
    year: int
    rows: list[SimRow]


def _raw_table(year: int) -> pd.DataFrame:
    df = STATE["df"][STATE["df"].year == year]
    raw = pd.DataFrame([{**{f: (np.nan if r["raw"].get(f) is None else r["raw"][f]) for f in STATE["features"]},
                         "area_km2": r["area_km2"], "hazard_load": r["hazard_load"], "major_load": r["major_load"],
                         "firefighters_est": r["firefighters_est"]}
                        for r in df.to_dict("records")],
                       index=df["iso2"].to_list())
    return raw


@app.get("/simulate/presets")
def simulate_presets() -> dict[str, list[HazardEvent]]:
    return {k: [HazardEvent(type=t, alert=a, count=c) for t, a, c in v] for k, v in PRESETS.items()}


@app.post("/simulate", response_model=SimulateResponse)
def simulate(body: SimulateRequest):
    y = _year(body.year)
    shocks = {}
    for sh in body.shocks:
        iso2 = sh.iso2.upper().replace("EL", "GR")
        if iso2 not in NAMES:
            raise HTTPException(404, f"Unknown country '{iso2}'")
        events = list(PRESETS.get(sh.preset, [])) + [(e.type, e.alert, e.count) for e in sh.events]
        shocks[iso2] = shocks.get(iso2, []) + events
    res = run_scenario(_raw_table(y), shocks, STATE["preprocessor"], STATE["features"], STATE["final_weights"],
                       STATE["scale"], body.alpha, y)
    rows = [SimRow(iso2=i, name=NAMES[i], shocked=bool(r.shocked), power_before=round(r.power_before, 2),
                   power_after=round(r.power_after, 2), power_change=round(r.power_change, 2),
                   power_rank_before=int(r.power_rank_before), power_rank_after=int(r.power_rank_after),
                   rank_change=int(r.rank_change), amber_before=round(r.amber_before, 2), amber_after=round(r.amber_after, 2),
                   band_before=r.band_before, band_after=r.band_after, firefighters=round(r.firefighters),
                   crew_demand=r.crew_demand, committed_share=round(r.committed_share, 4),
                   unmet_demand=round(r.unmet_demand)) for i, r in res.iterrows()]
    return SimulateResponse(year=y, rows=rows)


class AssistanceRequest(BaseModel):
    recipient: str
    year: int | None = None
    need: float | None = Field(None, gt=0, description="Firefighters requested; defaults to the recipient's unmet demand after the shocks")
    shocks: list[Shock] = []
    incident_lat: float | None = Field(None, ge=-90, le=90)
    incident_lon: float | None = Field(None, ge=-180, le=180)
    weight_location: float = Field(0.5, ge=0, le=1, description="1 = only distance matters, 0 = only donor power matters")
    deploy_share: float = Field(0.05, gt=0, le=0.5)
    max_share_per_donor: float = Field(0.4, gt=0, le=1)
    min_power: float = Field(20.0, ge=0, le=100)
    max_distance_km: float | None = Field(None, gt=0)
    min_team: int = Field(20, ge=1, le=500, description="A donor sends at least this many firefighters, or none")

    @model_validator(mode="after")
    def need_or_shocks(self):
        if self.need is None and not self.shocks:
            raise ValueError("give a need or shocks that create one")
        if (self.incident_lat is None) != (self.incident_lon is None):
            raise ValueError("give both incident_lat and incident_lon")
        return self


class Donation(BaseModel):
    iso2: str
    name: str
    sent: int
    share_of_need: float
    share_of_own_crews: float
    distance_km: float
    power_before: float
    power_after: float


class AssistanceResponse(BaseModel):
    year: int
    recipient: str
    need: float
    covered: int
    shortfall: float
    mean_distance_km: float | None
    available_capacity: float
    incident: list[float]
    donors: list[Donation]


def _shock_dict(shocks: list[Shock]) -> dict:
    out = {}
    for sh in shocks:
        iso2 = sh.iso2.upper().replace("EL", "GR")
        if iso2 not in NAMES:
            raise HTTPException(404, f"Unknown country '{iso2}'")
        out[iso2] = out.get(iso2, []) + list(PRESETS.get(sh.preset, [])) + [(e.type, e.alert, e.count) for e in sh.events]
    return out


@app.post("/optimize/assistance", response_model=AssistanceResponse)
def optimize(body: AssistanceRequest):
    y = _year(body.year)
    recipient = body.recipient.upper().replace("EL", "GR")
    if recipient not in NAMES:
        raise HTTPException(404, f"Unknown country '{recipient}'")
    scen = run_scenario(_raw_table(y), _shock_dict(body.shocks), STATE["preprocessor"], STATE["features"],
                        STATE["final_weights"], STATE["scale"], STATE["metadata"].get("alpha", 0.5), y)
    need = body.need if body.need is not None else float(scen.loc[recipient, "unmet_demand"])
    if need <= 0:
        raise HTTPException(422, f"{NAMES[recipient]} can cover this demand itself; pass an explicit need to plan anyway")
    cent = pd.DataFrame(STATE["metadata"]["centroids"], index=["lat", "lon"]).T
    params = AssistanceParams(weight_location=body.weight_location, deploy_share=body.deploy_share,
                              max_share_per_donor=body.max_share_per_donor, min_power=body.min_power,
                              max_distance_km=body.max_distance_km, min_team=body.min_team)
    incident = (body.incident_lat, body.incident_lon) if body.incident_lat is not None else None
    plan, info = optimize_assistance(donor_table(scen, cent), recipient, need, incident, params)
    donors = [Donation(iso2=i, name=NAMES[i], sent=int(r.sent), share_of_need=round(r.share_of_need, 4),
                       share_of_own_crews=round(r.share_of_own_crews, 4), distance_km=round(r.distance_km, 1),
                       power_before=round(r.power, 2), power_after=round(r.power_after, 2)) for i, r in plan.iterrows()]
    return AssistanceResponse(year=y, recipient=recipient, need=round(need, 1), covered=info["covered"],
                              shortfall=round(info["shortfall"], 1), mean_distance_km=info["mean_distance_km"],
                              available_capacity=round(info["available_capacity"], 1), incident=info["incident"], donors=donors)
