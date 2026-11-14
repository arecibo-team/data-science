"""Feature engineering: the same pure functions are used for training (notebook) and serving (API)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from . import EU27, SEED

# feature -> (pillar, sign inside the index, log1p?, human label)
FEATURES = {
    "burned_share_pct": ("danger", +1, True, "Burned area"),
    "hazard_alerts": ("hazards", +1, False, "Concurrent hazards, percentile among EU countries that year"),
    "major_hazards": ("hazards", +1, False, "Major hazards (Orange/Red), percentile among EU countries that year"),
    "ff_per_100k": ("workforce", -1, True, "Professional firefighters"),
    "volunteers_per_100k": ("workforce", -1, True, "Volunteer firefighters"),
    "spend_per_capita": ("capability", -1, True, "Fire-protection spending"),
    "stations_per_1000km2": ("capability", -1, True, "Fire-station density by area"),
    "stations_per_100k": ("capability", -1, True, "Fire stations per population"),
}
PILLARS = ("danger", "hazards", "workforce", "capability")
PILLAR_SIGN = {"danger": 1, "hazards": 1, "workforce": -1, "capability": -1}
RAW_COLUMNS = ["burned_ha", "area_km2", "population", "firefighters", "volunteers", "spending_meur", "stations", "hazard_load"]


def active_features(panel: pd.DataFrame) -> list[str]:
    """All features, minus volunteers when no volunteer data was provided."""
    feats = list(FEATURES)
    if "volunteers" not in panel or panel["volunteers"].notna().sum() == 0:
        feats.remove("volunteers_per_100k")
    if "hazard_load" not in panel or panel["hazard_load"].notna().sum() == 0:
        feats.remove("hazard_alerts")
        feats.remove("major_hazards")
    return feats


def nearest_year_fill(df: pd.DataFrame, years: list[int], col: str) -> pd.DataFrame:
    """Expand an observed [iso2, year, col] table to every year by copying the nearest observed year.
    Adds `<col>_imputed` (True where the value was copied)."""
    out = []
    for iso2 in EU27:
        obs = df[df["iso2"] == iso2].dropna(subset=[col]).set_index("year")[col]
        for y in years:
            if y in obs.index:
                out.append((iso2, y, obs[y], False))
            elif len(obs):
                near = min(obs.index, key=lambda t: (abs(t - y), -t))
                out.append((iso2, y, obs[near], True))
            else:
                out.append((iso2, y, np.nan, False))
    return pd.DataFrame(out, columns=["iso2", "year", col, f"{col}_imputed"])


def compute_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """Raw counts -> per-capita / per-area rates. Input has RAW_COLUMNS, output has FEATURES columns."""
    p = panel
    out = pd.DataFrame(index=p.index)
    out["burned_share_pct"] = p["burned_ha"] / p["area_km2"]          # ha / km2 = % of land (1 km2 = 100 ha)
    out["hazard_alerts"] = p.get("hazard_pct", np.nan)
    out["major_hazards"] = p.get("major_pct", np.nan)
    out["ff_per_100k"] = p["firefighters"] / p["population"] * 1e5
    out["volunteers_per_100k"] = p.get("volunteers", np.nan) / p["population"] * 1e5
    out["spend_per_capita"] = p["spending_meur"] * 1e6 / p["population"]
    out["stations_per_1000km2"] = p["stations"] / p["area_km2"] * 1000
    out["stations_per_100k"] = p["stations"] / p["population"] * 1e5
    return out


@dataclass
class Preprocessor:
    """log1p -> winsorise (5th/95th pct) -> min-max to 0-1 -> IterativeImputer. Fitted on training rows only."""
    features: list[str]
    lo: dict = field(default_factory=dict)
    hi: dict = field(default_factory=dict)
    imputer: IterativeImputer | None = None

    def _scale(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for f in self.features:
            v = X[f].astype(float)
            if FEATURES[f][2]:
                v = np.log1p(v.clip(lower=0))
            span = (self.hi[f] - self.lo[f]) or 1.0
            out[f] = (v.clip(self.lo[f], self.hi[f]) - self.lo[f]) / span
        return out

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        for f in self.features:
            v = X[f].astype(float)
            if FEATURES[f][2]:
                v = np.log1p(v.clip(lower=0))
            self.lo[f], self.hi[f] = float(np.nanpercentile(v, 5)), float(np.nanpercentile(v, 95))
        self.imputer = IterativeImputer(random_state=SEED, max_iter=25, min_value=0.0, max_value=1.0, keep_empty_features=True)
        self.imputer.fit(self._scale(X).to_numpy())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns scaled, imputed features in [0, 1] (same index as X)."""
        scaled = self._scale(X)
        filled = self.imputer.transform(scaled.to_numpy())
        return pd.DataFrame(np.clip(filled, 0, 1), index=X.index, columns=self.features)

    def inverse(self, feature: str, scaled) -> np.ndarray:
        v = self.lo[feature] + np.asarray(scaled, dtype=float) * (self.hi[feature] - self.lo[feature])
        return np.expm1(v) if FEATURES[feature][2] else v

    def missing_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features].isna()


def build_panel(years: list[int], spending: pd.DataFrame, population: pd.DataFrame, area: pd.DataFrame,
                burned: pd.DataFrame, firefighters: pd.DataFrame, stations: pd.DataFrame,
                volunteers: pd.DataFrame | None = None, hazards: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (iso2, year) with raw indicator columns and `<indicator>_imputed` flags.

    - firefighters: nearest-year copy for years without an LFS file (flagged imputed), 'u' flag kept
    - stations: one OpenStreetMap snapshot applied to every year
    - area: latest available value applied to earlier years if needed
    Remaining gaps stay NaN and are filled later by the IterativeImputer (also flagged)."""
    grid = pd.MultiIndex.from_product([list(EU27), years], names=["iso2", "year"]).to_frame(index=False)
    ren = lambda d: d.rename(columns={"geo": "iso2"})
    panel = (grid.merge(ren(spending), on=["iso2", "year"], how="left")
                 .merge(ren(population), on=["iso2", "year"], how="left")
                 .merge(burned, on=["iso2", "year"], how="left"))
    a = ren(area).sort_values("year").groupby("iso2")["area_km2"].last()
    panel["area_km2"] = panel["iso2"].map(a)
    sp = nearest_year_fill(ren(spending)[["iso2", "year", "spending_meur"]], years, "spending_meur")
    panel = panel.drop(columns="spending_meur").merge(sp, on=["iso2", "year"], how="left")
    ff = nearest_year_fill(firefighters[["iso2", "year", "firefighters"]], years, "firefighters")
    panel = panel.merge(ff, on=["iso2", "year"], how="left")
    lowrel = firefighters.set_index(["iso2", "year"])["low_reliability"]
    panel["firefighters_low_reliability"] = [bool(lowrel.get((i, y), False)) for i, y in zip(panel.iso2, panel.year)]
    panel["stations"] = panel["iso2"].map(stations.set_index("iso2")["stations"])
    vol = volunteers if volunteers is not None and len(volunteers) else pd.DataFrame(columns=["iso2", "year", "volunteers"])
    panel = panel.merge(vol.astype({"year": int}) if len(vol) else vol, on=["iso2", "year"], how="left") if len(vol) \
        else panel.assign(volunteers=np.nan)
    if hazards is not None and len(hazards):
        panel = panel.merge(hazards, on=["iso2", "year"], how="left")
        panel[[c for c in hazards.columns if c not in ("iso2", "year")]] = panel[[c for c in hazards.columns if c not in ("iso2", "year")]].fillna(0)
    else:
        panel["hazard_load"] = np.nan
    for col in ["population", "burned_ha", "area_km2", "stations", "volunteers", "hazard_load"]:
        panel[f"{col}_imputed"] = panel[col].isna()
    panel["spending_meur_imputed"] = panel["spending_meur_imputed"].fillna(False) | panel["spending_meur"].isna()
    panel["firefighters_imputed"] = panel["firefighters_imputed"].fillna(False) | panel["firefighters"].isna()
    return panel


def tidy_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """Long table: indicators(iso2, year, indicator, value, source, imputed)."""
    src = {"spending_meur": "Eurostat gov_10a_exp (GF0302)", "population": "Eurostat demo_pjan",
           "area_km2": "Eurostat reg_area3", "burned_ha": "OWID / GWIS", "firefighters": "Eurostat LFS extraction",
           "stations": "OpenStreetMap (Overpass)", "volunteers": "manual", "hazard_load": "GDACS alerts"}
    rows = []
    for col, s in src.items():
        part = panel[["iso2", "year", col, f"{col}_imputed"]].rename(columns={col: "value", f"{col}_imputed": "imputed"})
        rows.append(part.assign(indicator=col, source=s))
    return pd.concat(rows, ignore_index=True)[["iso2", "year", "indicator", "value", "source", "imputed"]]
