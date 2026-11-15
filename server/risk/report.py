"""How the country risk model did on a season it never saw.

    uv run python -m server.risk.report                         # Greece, fire season 2025
    uv run python -m server.risk.report --country Spain --year 2026

For every lead it compares the model with three simple forecasts of the same day: persistence (today's number of
burning cells), the country's usual number for the day of the year, and the classic fire weather index. Scores:
  rank     Spearman correlation between the forecast and the cells that burned, over the days of the season
  auc      how well the forecast separates surge days from the others (0.5 = no better than chance)
  hit      share of the surge days that were under alert
  alarms   alert days in the season, and the share of them followed by fire activity above the usual
A surge day is a day at or above the country's 95th percentile of fire-season days in the training years. A country
is under alert when the forecast is above its own 90th percentile over the two held-out validation seasons, so both
thresholds were fixed before the tested season began.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import model

SEASON = (5, 10)          # May to October
LEADS = (1, 3, 5, 7)


def _season(start: dt.date, n: int, years) -> np.ndarray:
    d = np.array([start + dt.timedelta(days=i) for i in range(n)])
    return np.array([x.year in years and SEASON[0] <= x.month <= SEASON[1] for x in d])


def evaluate(country: str = "Greece", year: int = 2025, target: str = "b") -> dict:
    meta = json.loads((model.DIR / "meta.json").read_text())
    z = np.load(model.DIR / "outlook.npz")
    start, T = dt.date.fromisoformat(meta["start"]), meta["n_days"]
    ci = meta["countries"].index(country)
    y = z[f"actual_{target}"][ci].astype(float)
    te_year = int(meta["train_end"][:4])
    train = _season(start, T, set(range(2012, te_year + 1)))
    val = _season(start, T, {te_year - 1, te_year})
    test = _season(start, T, {year})
    surge = float(max(np.percentile(y[train], 95), 1))
    usual = float(np.percentile(y[train], 75))
    clim = z["clim"][ci][z["doy"][:T]]
    issued_ok = country == meta["issued_for"]
    out = {"country": country, "year": year, "target": target, "surge_at": surge, "train_end": meta["train_end"],
           "surge_days": int((y[test] >= surge).sum()), "season_days": int(test.sum()), "leads": {}}
    days = np.flatnonzero(test)
    series = {"date": [(start + dt.timedelta(days=int(t))).isoformat() for t in days], "actual": y[days].tolist()}
    for k in LEADS:
        t = days[days - k >= 0]
        cands = {"model (perfect weather)": z["perfect_full"][k - 1, ci], "weather only (perfect weather)": z["perfect_conditions"][k - 1, ci],
                 "fire weather index (perfect)": z["perfect_fwi"][k - 1, ci], "persistence": y, "usual for the day": None}
        if issued_ok:
            cov = z["cover"][k - 1][t - k]
            cands = {"model (issued forecasts)": z["issued_full"][k - 1], "weather only (issued forecasts)": z["issued_conditions"][k - 1],
                     "fire weather index (issued)": z["issued_fwi"][k - 1], **cands}
        rows = {}
        for name, f in cands.items():
            if f is None:
                s, sv = clim[t], clim[np.flatnonzero(val)]
            else:
                s = f[t - k]                                   # issued on D = t - k, about day t
                tv = np.flatnonzero(val)
                ref = z["perfect_" + {"model": "full", "weather only": "conditions", "fire weather index": "fwi"}[name.split(" (")[0]]][k - 1, ci] \
                    if "(" in name else f
                sv = ref[tv[tv - k >= 0] - k]
            m = cov if "issued" in name else np.ones(len(t), bool)
            yy, ss = y[t][m], s[m]
            thr = float(np.percentile(sv, 90))
            alert = ss >= thr
            sg = yy >= surge
            rows[name] = {"days": int(m.sum()), "rank": float(spearmanr(ss, yy)[0]),
                          "auc": float(roc_auc_score(sg, ss)) if 0 < sg.sum() < len(sg) else None,
                          "hit": float((alert & sg).sum() / max(sg.sum(), 1)), "surge_days": int(sg.sum()),
                          "alarms": int(alert.sum()), "alarms_useful": float((alert & (yy > usual)).sum() / max(alert.sum(), 1)),
                          "threshold": thr}
            if name.startswith("model"):
                full = np.full(len(days), np.nan)
                full[np.isin(days, t[m])] = ss
                series[f"{name} +{k}"] = [None if np.isnan(v) else round(float(v), 2) for v in full]
                series[f"alert {name} +{k}"] = [bool(v >= thr) if not np.isnan(v) else None for v in full]
        out["leads"][k] = rows
    out["series"] = series
    return out


def show(r: dict) -> None:
    print(f"\n{r['country']} {r['year']}, May to October: {r['surge_days']} surge days of {r['season_days']} "
          f"(a surge day has {r['surge_at']:.0f} or more burning cells; model trained up to {r['train_end']})")
    for k, rows in r["leads"].items():
        print(f"\n  {k} day{'s' if k > 1 else ''} ahead" + " " * 22 + "rank    auc    hit   alarms (useful)")
        for name, v in rows.items():
            auc = f"{v['auc']:.2f}" if v["auc"] is not None else "  - "
            print(f"    {name:<34} {v['rank']:5.2f}  {auc}  {v['hit']:5.0%}   {v['alarms']:3d} ({v['alarms_useful']:.0%})")
    s = r["series"]
    key = next((k for k in ("model (issued forecasts)", "model (perfect weather)") if f"{k} +1" in s), None)
    big = [i for i, v in enumerate(s["actual"]) if v >= r["surge_at"]]
    print(f"\n  Surge days, and what the {key} said 1, 3, 5 and 7 days before (* = alert)")
    for i in big:
        cells = []
        for k in LEADS:
            v, a = s[f"{key} +{k}"][i], s[f"alert {key} +{k}"][i]
            cells.append("   n/a " if v is None else f"{v:5.1f}{'*' if a else ' '} ")
        print(f"    {s['date'][i]}  burned {s['actual'][i]:3.0f}   " + " ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="Greece")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--target", default="b", choices=["b", "mv", "cc", "new"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = evaluate(a.country, a.year, a.target)
    if a.json:
        print(json.dumps(r))
    else:
        show(r)


if __name__ == "__main__":
    main()
