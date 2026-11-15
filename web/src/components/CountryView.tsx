import { useEffect, useMemo, useState, type ReactNode } from "react";
import { MapView } from "./MapView";
import { CountryLayer, MapSync, riskColor, riskValue, riskWord, WAR_COUNTRIES, type RiskMode } from "./countries";
import { loadRegions, loadRisk, type RiskDay } from "../data";

const FUTURE_DAYS = 6;

function addDays(d: string, n: number): string {
  const t = new Date(`${d}T00:00:00Z`);
  t.setUTCDate(t.getUTCDate() + n);
  return t.toISOString().slice(0, 10);
}

const noop = () => undefined;

interface Props {
  /** Days with data, ascending. */
  dates: string[];
  /** The Map / Forecast / Countries switch, rendered at the top of the sidebar. */
  header: ReactNode;
}

/** Per-country page: the model's country risk and the real (confirmed) country risk, side by side, on one timeline. */
export function CountryView({ dates, header }: Props) {
  const last = dates[dates.length - 1];
  const days = useMemo(() => [...dates, ...Array.from({ length: FUTURE_DAYS }, (_, i) => addDays(last, i + 1))], [dates, last]);
  const [idx, setIdx] = useState(dates.length - 1);
  const [mode, setMode] = useState<RiskMode>("own");
  const [skipWar, setSkipWar] = useState(false);
  const [regions, setRegions] = useState<GeoJSON.FeatureCollection | null>(null);
  const [risk, setRisk] = useState<RiskDay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const day = days[idx];
  // The real side is what really happened in the days after the forecast day, so it only exists while those days are in the data.
  const realOn = day <= last;

  useEffect(() => { loadRegions().then(setRegions).catch(() => undefined); }, []);
  useEffect(() => {
    const ctl = new AbortController();
    const t = setTimeout(() => {
      loadRisk(day, ctl.signal)
        .then((r) => { setRisk(r); setError(null); })
        .catch((e: Error) => { if (!ctl.signal.aborted) setError(e.message); });
    }, 120);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [day]);

  const rows = useMemo(() => {
    if (!risk) return [];
    return risk.countries
      .filter((c) => !(skipWar && WAR_COUNTRIES.has(c.name)))
      .map((c) => ({ c, model: riskValue(c, mode, "model") ?? 0, real: realOn ? riskValue(c, mode, "real") : null }))
      .filter((x) => x.model >= 0.15 || (x.real ?? 0) >= 0.15)
      .sort((a, b) => Math.max(b.model, b.real ?? 0) - Math.max(a.model, a.real ?? 0))
      .slice(0, 15);
  }, [risk, mode, skipWar, realOn]);

  const horizon = risk?.horizon ?? 3;
  const realWindow = `${addDays(day, 1)} to ${addDays(day, horizon)}`;

  return (
    <div className="app">
      <aside className="sidebar">
        {header}

        <fieldset>
          <legend>Timeline (UTC)</legend>
          <div className="timeline-day"><strong>{day}</strong></div>
          <input
            type="range" min={0} max={days.length - 1} value={idx} className="slider"
            onChange={(e) => setIdx(Number(e.target.value))} aria-label="Day"
          />
          <div className="presets">
            <button type="button" onClick={() => setIdx(Math.max(0, idx - 1))}>◀ Day</button>
            <button type="button" onClick={() => setIdx(Math.min(days.length - 1, idx + 1))}>Day ▶</button>
            <button type="button" onClick={() => setIdx(dates.length - 1)}>Latest</button>
          </div>
          {risk && (
            <p className="muted small">
              Model made on {risk.feature_date}{risk.feature_date !== day ? " (the last day with data)" : ""}, for the next {horizon} days.
              {realOn ? ` Real: confirmed fires ${realWindow}${risk.complete ? "" : " (partly in the data)"}.` : " Real: those days are not in the data yet."}
            </p>
          )}
        </fieldset>

        <fieldset>
          <legend>Country risk</legend>
          <label className="check">
            <input type="checkbox" checked={skipWar} onChange={(e) => setSkipWar(e.target.checked)} /> Leave out Ukraine and Russia (war fires)
          </label>
          <label>
            Colour countries
            <select value={mode} onChange={(e) => setMode(e.target.value as RiskMode)}>
              <option value="own">Against their own season</option>
              <option value="compare">Compared across countries</option>
            </select>
          </label>
          <div className="ramp-bar" />
          <div className="ramp-labels small muted">
            {mode === "own" ? <><span>OK</span><span>Really bad</span></> : <><span>0.01 per 1,000 km²</span><span>1 per 1,000 km²</span></>}
          </div>
          <p className="muted small">
            Left: the model's expected fire cells in each country. Right: the confirmed fires that really happened in the same
            days, measured the same way. Both use the same colours, so the two maps can be compared country by country.
          </p>
        </fieldset>

        <fieldset>
          <legend>Countries, side by side</legend>
          <table className="country-table">
            <thead>
              <tr><th>Country</th><th>Model</th><th>Real</th></tr>
            </thead>
            <tbody>
              {rows.map(({ c, model, real }) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  <td>
                    <span className="dot" style={{ background: riskColor(model) }} /> {riskWord(model)}
                    <span className="muted"> {c.expected.toFixed(1)} cells</span>
                  </td>
                  <td>
                    {real === null ? <span className="muted">not in yet</span> : (
                      <>
                        <span className="dot" style={{ background: riskColor(real) }} /> {riskWord(real)}
                        <span className="muted"> {c.real ? `${c.real.expected} cells${c.real.complete ? "" : " so far"}` : "no fire"}</span>
                      </>
                    )}
                  </td>
                </tr>
              ))}
              {rows.length === 0 && <tr><td colSpan={3} className="muted">Every country is at normal or low risk.</td></tr>}
            </tbody>
          </table>
        </fieldset>

        {error && <p className="error small">{error}</p>}
      </aside>

      <main className="map-wrap split">
        <div className="half">
          <div className="map-title">Model: expected fire cells</div>
          <MapView key="model" visible={null} tileUrl={null} incidents={[]} muted soilUrl={null} onPick={noop} onView={noop}>
            <MapSync group="countries" />
            <CountryLayer regions={regions} risk={risk} mode={mode} skipWar={skipWar} />
          </MapView>
        </div>
        <div className="half">
          <div className="map-title">Real: confirmed fires</div>
          <MapView key="real" visible={null} tileUrl={null} incidents={[]} muted soilUrl={null} onPick={noop} onView={noop}>
            <MapSync group="countries" />
            <CountryLayer regions={regions} risk={realOn ? risk : null} mode={mode} skipWar={skipWar} source="real" />
          </MapView>
        </div>
      </main>
    </div>
  );
}
