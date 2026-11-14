"""Model training, validation, SHAP weights and artifact save/load."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

from . import ARTIFACTS_DIR, SEED
from .features import FEATURES, PILLARS, Preprocessor


def ml_table(panel_rates: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Rows (iso2, year t) with features at t and the target: burned share (% of land) at t+1."""
    df = pd.concat([meta, panel_rates], axis=1)
    nxt = df[["iso2", "year", "burned_share_pct"]].copy()
    nxt["year"] -= 1
    df = df.merge(nxt.rename(columns={"burned_share_pct": "target_next"}), on=["iso2", "year"], how="left")
    return df.dropna(subset=["target_next"]).reset_index(drop=True)


def make_lgbm(features: list[str]) -> LGBMRegressor:
    mono = [{"danger": 1, "hazards": 0}.get(FEATURES[f][0], -1) for f in features]
    return LGBMRegressor(max_depth=3, num_leaves=7, n_estimators=300, learning_rate=0.03, min_child_samples=10,
                         subsample=0.9, subsample_freq=1, colsample_bytree=0.9, monotone_constraints=mono,
                         random_state=SEED, verbose=-1)


def make_enet() -> ElasticNet:
    return ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)


def _fit_predict(model_name: str, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Fit the preprocessor and model on train only; predict test. Target is log1p(burned share)."""
    pre = Preprocessor(features).fit(train[features])
    Xtr, Xte = pre.transform(train[features]), pre.transform(test[features])
    y = np.log1p(train["target_next"])
    model = make_lgbm(features) if model_name == "LightGBM" else make_enet()
    model.fit(Xtr, y)
    return np.expm1(model.predict(Xte)).clip(min=0)


def _metrics(y_true, y_pred) -> dict:
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "Spearman": float(spearmanr(y_true, y_pred).statistic)}


def evaluate(table: pd.DataFrame, features: list[str], split_year: int = 2020) -> pd.DataFrame:
    """Leave-one-country-out and time split (train t <= split_year, test t > split_year). Never shuffles rows."""
    rows = []
    for name in ("ElasticNet", "LightGBM", "Persistence"):
        # leave one country out
        preds = pd.Series(index=table.index, dtype=float)
        for iso2 in table["iso2"].unique():
            te = table["iso2"] == iso2
            preds[te] = (table.loc[te, "burned_share_pct"].to_numpy() if name == "Persistence"
                         else _fit_predict(name, table[~te], table[te], features))
        rows.append({"model": name, "validation": "Leave one country out", **_metrics(table["target_next"], preds)})
        # time split
        tr, te = table["year"] <= split_year, table["year"] > split_year
        p = (table.loc[te, "burned_share_pct"].to_numpy() if name == "Persistence"
             else _fit_predict(name, table[tr], table[te], features))
        rows.append({"model": name, "validation": f"Time split (train t<={split_year}, test t>={split_year + 1})",
                     **_metrics(table.loc[te, "target_next"], p)})
    return pd.DataFrame(rows)


def shap_weights(model: LGBMRegressor, X: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Mean |SHAP| per feature -> weights within each pillar, and pillar weights = share of total |SHAP|."""
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    within = {}
    for p in PILLARS:
        fs = [f for f in X.columns if FEATURES[f][0] == p]
        tot = imp[fs].sum()
        within.update({f: float(imp[f] / tot) if tot > 0 else 1 / len(fs) for f in fs})
    pillar = {p: float(imp[[f for f in X.columns if FEATURES[f][0] == p]].sum() / imp.sum()) if imp.sum() else 0.0 for p in PILLARS}
    return {"within": within, "pillar": pillar}, pd.DataFrame(sv, columns=X.columns, index=X.index)


def blend(a: dict, b: dict, alpha: float = 0.5) -> dict:
    return {"within": {f: alpha * a["within"][f] + (1 - alpha) * b["within"][f] for f in a["within"]},
            "pillar": {p: alpha * a["pillar"][p] + (1 - alpha) * b["pillar"][p] for p in PILLARS}}


# ---------------------------------------------------------------- artifacts
FILES = ("model.joblib", "preprocessor.joblib", "weights.json", "scores.json", "metadata.json")


def save_artifacts(model, preprocessor, weights_table: dict, scores: list[dict], metadata: dict,
                   out: Path = ARTIFACTS_DIR) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    joblib.dump(preprocessor, out / "preprocessor.joblib")
    (out / "weights.json").write_text(json.dumps(weights_table, indent=2))
    (out / "scores.json").write_text(json.dumps(scores, indent=1))
    import lightgbm, shap, sklearn  # versions for the record
    metadata = {**metadata, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                             "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "shap": shap.__version__}}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return [out / f for f in FILES]


def load_artifacts(path: Path = ARTIFACTS_DIR) -> dict:
    missing = [f for f in FILES if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts {missing} in {path}. Run notebooks/amber_index.ipynb first.")
    return {"model": joblib.load(path / "model.joblib"), "preprocessor": joblib.load(path / "preprocessor.joblib"),
            "weights": json.loads((path / "weights.json").read_text()), "scores": json.loads((path / "scores.json").read_text()),
            "metadata": json.loads((path / "metadata.json").read_text())}
