import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import { MapView } from "./MapView";
import { SoilControl, type SoilState } from "./SoilControl";
import { TrainPanel } from "./TrainPanel";
import { EventCheck } from "./EventCheck";
import { loadForecast, loadRegions, loadRisk, loadRuns, soilUrl, tileUrl, type CountryRisk, type Forecast, type ModelRun, type RiskDay } from "../data";

const FUTURE_DAYS = 6;
const CELL_KM = 11; // a 0.1 degree cell

function addDays(d: string, n: number): string {
  const t = new Date(`${d}T00:00:00Z`);
  t.setUTCDate(t.getUTCDate() + n);
  return t.toISOString().slice(0, 10);
}

// ---- what the model saw on each day ---------------------------------------

type DayState = "train" | "event" | "test" | "future";

const STATE_INFO: Record<DayState, { color: string; label: string; note: string }> = {
  train: { color: "#2563eb", label: "Training data", note: "The model trained on this day, so a good match here is not a fair test." },
  event: { color: "#dc2626", label: "Held out event", note: "Removed from training on purpose: the model has never seen this period." },
  test: { color: "#d97706", label: "Never trained on", note: "After the training period: a fair test of the model." },
  future: { color: "#9ca3af", label: "Future", note: "No data yet: a real forecast." },
};

function dayState(day: string, last: string, run: ModelRun | null): DayState | null {
  if (day > last) return "future";
  if (!run?.params.train_end) return null;
  const ev = run.params.holdout_event;
  if (ev && day >= addDays(ev.first, -run.params.horizon) && day <= ev.last) return "event";
  return day <= run.params.train_end ? "train" : "test";
}

/** A coloured strip under the timeline: which days the active model trained on, held out, or never saw. */
function TrainStrip({ days, last, run }: { days: string[]; last: string; run: ModelRun | null }) {
  const segments = useMemo(() => {
    const out: { state: DayState; from: number; to: number }[] = [];
    days.forEach((d, i) => {
      const s = dayState(d, last, run);
      if (!s) return;
      const prev = out[out.length - 1];
      if (prev && prev.state === s && prev.to === i - 1) prev.to = i;
      else out.push({ state: s, from: i, to: i });
    });
    return out;
  }, [days, last, run]);
  if (!segments.length) return null;
  const n = days.length;
  return (
    <div className="strip" aria-hidden="true">
      {segments.map((s) => (
        <span key={`${s.state}${s.from}`} title={`${STATE_INFO[s.state].label}: ${days[s.from]} to ${days[s.to]}`}
          style={{ left: `${(s.from / n) * 100}%`, width: `${((s.to - s.from + 1) / n) * 100}%`, background: STATE_INFO[s.state].color }} />
      ))}
    </div>
  );
}

// ---- colours ---------------------------------------------------------------

const P_LOW = 0.001;
/** Probability to colour on a log scale from 0.1% (pale) to 100% (deep red). */
function probT(p: number): number {
  return Math.min(1, Math.max(0, Math.log(Math.max(p, P_LOW) / P_LOW) / Math.log(1 / P_LOW)));
}
const probColor = (p: number) => `hsl(${55 - 55 * probT(p)}, 95%, ${62 - 20 * probT(p)}%)`;
const REAL = "#0369a1";

type Outcome = "hit" | "miss" | "false" | "alert" | "low";
function outcome(f: Forecast, i: number): Outcome {
  const alert = f.p[i] >= f.threshold;
  if (f.actual[i]) return alert ? "hit" : "miss";
  if (alert) return f.complete ? "false" : "alert";
  return "low";
}

function radiusFor(zoom: number, lat: number): number {
  const mPerPx = (156543.03 * Math.cos((lat * Math.PI) / 180)) / 2 ** zoom;
  return Math.min(40, Math.max(5, (CELL_KM * 1000) / 2 / mPerPx));
}

/** Markers that follow the zoom. Order puts the strongest on top. */
function useMarkers(forecast: Forecast | null, pick: (i: number) => boolean, make: (i: number, r: number) => L.CircleMarker, order: (a: number, b: number) => number) {
  const map = useMap();
  useEffect(() => {
    if (!forecast) return;
    const group = L.featureGroup();
    const radius = (i: number) => radiusFor(map.getZoom(), forecast.lat[i]);
    const markers: [number, L.CircleMarker][] = [];
    Array.from({ length: forecast.p.length }, (_, i) => i).filter(pick).sort(order).forEach((i) => {
      const m = make(i, radius(i));
      m.addTo(group);
      markers.push([i, m]);
    });
    const resize = () => markers.forEach(([i, m]) => m.setRadius(radius(i) + ((m.options as { extra?: number }).extra ?? 0)));
    map.on("zoomend", resize);
    group.addTo(map);
    return () => {
      map.off("zoomend", resize);
      group.remove();
    };
  }, [map, forecast, pick, make, order]);
}

/** Layer 1, the model: one dot per cell, coloured by predicted probability. Cells over the alert threshold are solid. */
function ModelLayer({ forecast, onlyAlerts }: { forecast: Forecast | null; onlyAlerts: boolean }) {
  const pick = useCallback((i: number) => !!forecast && (!onlyAlerts || forecast.p[i] >= forecast.threshold), [forecast, onlyAlerts]);
  const order = useCallback((a: number, b: number) => (forecast ? forecast.p[a] - forecast.p[b] : 0), [forecast]);
  const make = useCallback((i: number, r: number) => {
    const f = forecast!;
    const p = f.p[i], alert = p >= f.threshold;
    const m = L.circleMarker([f.lat[i], f.lon[i]], {
      radius: r, stroke: alert, color: "#fff", weight: 1.5, fillColor: probColor(p),
      fillOpacity: alert ? 0.95 : 0.25 + 0.5 * probT(p),
    });
    const happened = f.actual[i] ? "a confirmed fire followed" : f.complete ? "no confirmed fire followed" : "outcome not known yet";
    m.bindTooltip(`Model: ${(p * 100).toFixed(p < 0.01 ? 2 : 0)}%${alert ? " (alert)" : ""}<br>Real data: ${happened}`, { sticky: true });
    return m;
  }, [forecast]);
  useMarkers(forecast, pick, make, order);
  return null;
}

/** Layer 2, real data: a blue ring around every cell where a confirmed (COMMON) fire really happened in the window. */
function RealLayer({ forecast }: { forecast: Forecast | null }) {
  const pick = useCallback((i: number) => !!forecast && forecast.actual[i] === 1, [forecast]);
  const order = useCallback(() => 0, []);
  const make = useCallback((i: number, r: number) => {
    const f = forecast!;
    const m = L.circleMarker([f.lat[i], f.lon[i]], {
      radius: r + 3, color: REAL, weight: 3, fill: false, extra: 3,
    } as L.CircleMarkerOptions);
    m.bindTooltip(`Real data: a confirmed fire happened here<br>Model had ${(f.p[i] * 100).toFixed(f.p[i] < 0.01 ? 2 : 0)}%${f.p[i] >= f.threshold ? " (alert: hit)" : " (below the alert threshold: miss)"}`, { sticky: true });
    return m;
  }, [forecast]);
  useMarkers(forecast, pick, make, order);
  return null;
}

// ---- country risk -----------------------------------------------------------

/** Green (0) through yellow to red (1). */
const riskColor = (s: number) => `hsl(${120 - 120 * Math.min(1, Math.max(0, s))}, 70%, 45%)`;
const riskWord = (s: number) => (s >= 0.9 ? "Very high" : s >= 0.7 ? "High" : s >= 0.4 ? "Elevated" : s >= 0.15 ? "Normal" : "Low");

/** own: each country against its own season. compare: fire per surface, one scale for every country. */
export type RiskMode = "own" | "compare";
const DENSITY_LO = 0.01;   // expected cells per 1,000 km2 at the bottom of the compare scale
const DENSITY_HI = 1;      // ... and at the top; the scale is logarithmic
const riskValue = (c: CountryRisk, mode: RiskMode): number => {
  if (mode === "own") return c.score;
  if (!c.density || c.density <= 0) return 0;
  const t = (Math.log10(c.density) - Math.log10(DENSITY_LO)) / (Math.log10(DENSITY_HI) - Math.log10(DENSITY_LO));
  return Math.min(1, Math.max(0, t));
};

/** Countries at war: their fires are often artificial (shelling, burning fields near the front), so they can be left out. */
const WAR_COUNTRIES = new Set(["Ukraine", "Russia"]);

/** Layer 0: every country coloured by its risk, under the cell layers. */
function CountryLayer({ regions, risk, mode, skipWar }: { regions: GeoJSON.FeatureCollection | null; risk: RiskDay | null; mode: RiskMode; skipWar: boolean }) {
  const map = useMap();
  useEffect(() => {
    if (!regions || !risk) return;
    const by = new Map<string, CountryRisk>(risk.countries.map((c) => [c.name, c]));
    if (!map.getPane("risk")) {
      const pane = map.createPane("risk");
      pane.style.zIndex = "350";
    }
    const layer = L.geoJSON(regions, {
      pane: "risk",
      filter: (f) => !(skipWar && WAR_COUNTRIES.has(f?.properties?.name)),
      style: (f) => {
        const c = by.get(f?.properties?.name);
        return c ? { color: "#fff", weight: 1, fillColor: riskColor(riskValue(c, mode)), fillOpacity: 0.55 }
                 : { color: "#fff", weight: 0.5, fillColor: "#9ca3af", fillOpacity: 0.12 };
      },
      onEachFeature: (f, l) => {
        const c = by.get(f.properties?.name);
        const real = c && c.burned !== null ? `<br>Real data: ${c.burned} cells had a confirmed fire` : "";
        const v = c ? riskValue(c, mode) : 0;
        const head = `<strong>${f.properties?.name}</strong>: ${riskWord(v)} (${Math.round(v * 100)}/100)<br>`;
        const body = mode === "compare"
          ? `Expected fire cells per 1,000 km²: ${c?.density != null ? c.density.toFixed(3) : "n/a"} `
            + `(${c?.expected.toFixed(1)} cells on ${c?.area_km2 ? Math.round(c.area_km2).toLocaleString() : "?"} km²)${real}`
          : `Expected fire cells: ${c?.expected.toFixed(1)}${c?.pct != null ? `, more than ${Math.round(c.pct * 100)}% of its usual days` : ""}${real}`;
        l.bindTooltip(c ? head + body : `<strong>${f.properties?.name}</strong>: no recent fire activity`, { sticky: true });
      },
    }).addTo(map);
    return () => { layer.remove(); };
  }, [map, regions, risk, mode, skipWar]);
  return null;
}

// ---- two maps that move together ------------------------------------------

const groups = new Map<string, Set<L.Map>>();
let syncing = false;
function MapSync({ group }: { group: string }) {
  const map = useMap();
  useEffect(() => {
    const set = groups.get(group) ?? new Set<L.Map>();
    groups.set(group, set);
    set.add(map);
    const onMove = () => {
      if (syncing) return;
      syncing = true;
      set.forEach((m) => m !== map && m.setView(map.getCenter(), map.getZoom(), { animate: false }));
      syncing = false;
    };
    map.on("move", onMove);
    return () => {
      map.off("move", onMove);
      set.delete(map);
    };
  }, [map, group]);
  return null;
}

function MapTitle({ children }: { children: ReactNode }) {
  return <div className="map-title">{children}</div>;
}

interface Props {
  /** Days with data, ascending. */
  dates: string[];
  generatedUtc: string;
  /** The Map / Forecast switch, rendered at the top of the sidebar. */
  header: ReactNode;
}

/** Second map view: a timeline, the model's prediction and what really happened, shown over each other or side by side. */
export function ForecastView({ dates, generatedUtc, header }: Props) {
  const last = dates[dates.length - 1];
  const days = useMemo(() => [...dates, ...Array.from({ length: FUTURE_DAYS }, (_, i) => addDays(last, i + 1))], [dates, last]);
  const [idx, setIdx] = useState(dates.length - 1);
  const [playing, setPlaying] = useState(false);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [missing, setMissing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [soil, setSoil] = useState<SoilState>({ on: false, cls: null });
  const [modelVersion, setModelVersion] = useState(0);
  const [run, setRun] = useState<ModelRun | null>(null);
  const [showModel, setShowModel] = useState(true);
  const [showReal, setShowReal] = useState(true);
  const [showCountries, setShowCountries] = useState(true);
  const [riskMode, setRiskMode] = useState<RiskMode>("own");
  const [skipWar, setSkipWar] = useState(false);
  const [regions, setRegions] = useState<GeoJSON.FeatureCollection | null>(null);
  const [risk, setRisk] = useState<RiskDay | null>(null);
  const [onlyAlerts, setOnlyAlerts] = useState(false);
  const [split, setSplit] = useState(false);
  const [realWindow, setRealWindow] = useState<"next" | "day">("next");
  const [allDetections, setAllDetections] = useState(false);
  const day = days[idx];

  useEffect(() => {
    loadRuns().then((r) => setRun(r.runs.find((x) => x.id === r.active) ?? null)).catch(() => undefined);
  }, [modelVersion]);

  useEffect(() => { loadRegions().then(setRegions).catch(() => undefined); }, []);
  useEffect(() => {
    const ctl = new AbortController();
    const t = setTimeout(() => {
      loadRisk(day, ctl.signal).then(setRisk).catch(() => undefined);
    }, 120);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [day, modelVersion]);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setIdx((i) => (i + 1 >= days.length ? (setPlaying(false), i) : i + 1)), 900);
    return () => clearInterval(t);
  }, [playing, days.length]);

  const seq = useRef(0);
  useEffect(() => {
    const ctl = new AbortController();
    const mine = ++seq.current;
    const timer = setTimeout(() => {
      loadForecast(day, ctl.signal)
        .then((f) => {
          if (mine !== seq.current) return;
          setForecast(f);
          setMissing(f === null);
          setError(null);
        })
        .catch((e: Error) => {
          if (!ctl.signal.aborted) setError(e.message);
        });
    }, 120);
    return () => {
      clearTimeout(timer);
      ctl.abort();
    };
  }, [day, modelVersion]);

  const onModelChanged = useCallback(() => setModelVersion((v) => v + 1), []);
  const noop = useCallback(() => undefined, []);
  const state = dayState(day, last, run);

  const tally = useMemo(() => {
    const t = { hit: 0, miss: 0, falseAlarm: 0, alerts: 0 };
    if (!forecast) return t;
    for (let i = 0; i < forecast.p.length; i++) {
      const o = outcome(forecast, i);
      if (o === "hit") t.hit++;
      else if (o === "miss") t.miss++;
      else if (o === "false") t.falseAlarm++;
      if (forecast.p[i] >= forecast.threshold) t.alerts++;
    }
    return t;
  }, [forecast]);

  // The real-data heat tiles cover the days the prediction is about, or just the selected day.
  const horizon = forecast?.horizon ?? run?.params.horizon ?? 3;
  const realFrom = realWindow === "next" ? addDays(day, 1) : day;
  const realTo = realWindow === "next" ? addDays(day, horizon) : day;
  const realKnown = realFrom <= last;
  const realTiles = realKnown ? tileUrl(realFrom, realTo > last ? last : realTo, allDetections ? 7 : 4, generatedUtc) : null;
  const soilLayer = soil.on ? soilUrl(soil.cls) : null;

  const modelMap = (
    <MapView key="model" visible={null} tileUrl={null} incidents={[]} muted soilUrl={soilLayer} onPick={noop} onView={noop}>
      {split && <MapSync group="forecast" />}
      {showCountries && <CountryLayer regions={regions} risk={risk} mode={riskMode} skipWar={skipWar} />}
      <ModelLayer forecast={forecast} onlyAlerts={onlyAlerts} />
    </MapView>
  );
  const realMap = (
    <MapView key="real" visible={null} tileUrl={realTiles} incidents={[]} muted soilUrl={soilLayer} onPick={noop} onView={noop}>
      {split && <MapSync group="forecast" />}
      <RealLayer forecast={forecast} />
    </MapView>
  );
  const overlayMap = (
    <MapView key="overlay" visible={null} tileUrl={showReal ? realTiles : null} incidents={[]} muted soilUrl={soilLayer} onPick={noop} onView={noop}>
      {showCountries && <CountryLayer regions={regions} risk={risk} mode={riskMode} skipWar={skipWar} />}
      {showModel && <ModelLayer forecast={forecast} onlyAlerts={onlyAlerts} />}
      {showReal && <RealLayer forecast={forecast} />}
    </MapView>
  );

  return (
    <div className="app">
      <aside className="sidebar">
        {header}

        <fieldset>
          <legend>Timeline (UTC)</legend>
          <div className="timeline-day">
            <strong>{day}</strong>
            {state && <span className="state-badge" style={{ background: STATE_INFO[state].color }}>{STATE_INFO[state].label}</span>}
          </div>
          <input
            type="range" min={0} max={days.length - 1} value={idx} className="slider"
            onChange={(e) => { setPlaying(false); setIdx(Number(e.target.value)); }}
            aria-label="Day"
          />
          <TrainStrip days={days} last={last} run={run} />
          {run?.params.train_end && (
            <div className="strip-legend small">
              {(Object.keys(STATE_INFO) as DayState[]).filter((s) => s !== "event" || run.params.holdout_event).map((s) => (
                <span key={s}><span className="dot" style={{ background: STATE_INFO[s].color }} /> {STATE_INFO[s].label}</span>
              ))}
            </div>
          )}
          {state && <p className="muted small">{STATE_INFO[state].note}</p>}
          <div className="presets">
            <button type="button" onClick={() => { setPlaying(false); setIdx(Math.max(0, idx - 1)); }}>◀ Day</button>
            <button type="button" onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>
            <button type="button" onClick={() => { setPlaying(false); setIdx(Math.min(days.length - 1, idx + 1)); }}>Day ▶</button>
            <button type="button" onClick={() => { setPlaying(false); setIdx(dates.length - 1); }}>Latest</button>
          </div>
          {forecast && (
            <p className="muted small">
              Prediction made on {forecast.feature_date}{forecast.feature_date !== day ? " (the last day with data)" : ""}, for
              the next {forecast.horizon} days
              {forecast.complete ? "" : "; not all of those days are in the data yet, so the comparison is partial"}.
            </p>
          )}
        </fieldset>

        <fieldset>
          <legend>Layers</legend>
          <label className="check">
            <input type="checkbox" checked={split || showModel} disabled={split} onChange={(e) => setShowModel(e.target.checked)} />
            <span className="dot" style={{ background: probColor(0.2) }} /> Model prediction
          </label>
          <label className="check">
            <input type="checkbox" checked={split || showReal} disabled={split} onChange={(e) => setShowReal(e.target.checked)} />
            <span className="ring" /> Real data
          </label>
          <label className="check">
            <input type="checkbox" checked={showCountries} onChange={(e) => setShowCountries(e.target.checked)} />
            <span className="ramp" /> Country risk
          </label>
          <label className="check">
            <input type="checkbox" checked={split} onChange={(e) => setSplit(e.target.checked)} /> Side by side
          </label>
          <label className="check">
            <input type="checkbox" checked={onlyAlerts} onChange={(e) => setOnlyAlerts(e.target.checked)} /> Only cells over the alert threshold
          </label>
          <label>
            Real data covers
            <select value={realWindow} onChange={(e) => setRealWindow(e.target.value as "next" | "day")}>
              <option value="next">The days the prediction is about</option>
              <option value="day">The selected day only</option>
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={allDetections} onChange={(e) => setAllDetections(e.target.checked)} /> Include unconfirmed MODIS and VIIRS detections
          </label>
          <p className="muted small">
            Model: dot colour is the predicted chance of a confirmed fire, pale at 0.1% to deep red at 100%; solid
            dots are over the alert threshold{forecast ? ` (${(forecast.threshold * 100).toFixed(0)}%)` : ""}.
            Real data: a blue ring where a confirmed fire (inside an EU incident) really happened{realKnown ? "" : "; those days are not in the data yet"}, over the heat map of those detections.
            A ring around a solid dot is a hit; a ring alone is a miss; a solid dot alone is a false alarm.
          </p>
        </fieldset>

        {risk && showCountries && (
          <fieldset>
            <legend>Country risk, next {risk.horizon} days</legend>
            <label className="check">
              <input type="checkbox" checked={skipWar} onChange={(e) => setSkipWar(e.target.checked)} /> Leave out Ukraine and Russia (war fires)
            </label>
            <label>
              Colour countries
              <select value={riskMode} onChange={(e) => setRiskMode(e.target.value as RiskMode)}>
                <option value="own">Against their own season</option>
                <option value="compare">Compared across countries</option>
              </select>
            </label>
            <div className="ramp-bar" />
            <div className="ramp-labels small muted">
              {riskMode === "own" ? <><span>OK</span><span>Really bad</span></> : <><span>0.01 per 1,000 km²</span><span>1 per 1,000 km²</span></>}
            </div>
            <ul className="risk-list">
              {risk.countries.filter((c) => !(skipWar && WAR_COUNTRIES.has(c.name)))
                .map((c) => ({ c, v: riskValue(c, riskMode) })).filter((x) => x.v >= 0.15)
                .sort((a, b) => b.v - a.v).slice(0, 8).map(({ c, v }) => (
                <li key={c.name}>
                  <span className="dot" style={{ background: riskColor(v) }} /> {c.name}
                  <span className="muted"> {riskWord(v)} · {c.expected.toFixed(1)} cells
                    {riskMode === "compare" && c.density != null ? ` · ${c.density.toFixed(3)} per 1,000 km²` : ""}
                    {c.burned !== null ? ` · ${c.burned} real` : ""}</span>
                </li>
              ))}
              {!risk.countries.some((c) => !(skipWar && WAR_COUNTRIES.has(c.name)) && riskValue(c, riskMode) >= 0.15) && <li className="muted">Every country is at normal or low risk.</li>}
            </ul>
            <p className="muted small">
              {riskMode === "own"
                ? "Against their own season: the model's expected fire cells in the country, ranked against the same time of year in that country's history."
                : "Compared across countries: the expected fire cells divided by the country's surface, on one scale for every country."}
              {" "}Real: cells that really had a confirmed fire in the window.
            </p>
          </fieldset>
        )}

        {missing && <p className="small">No completed model yet. Train one below.</p>}
        {forecast && (
          <fieldset>
            <legend>Model against real data, this day</legend>
            <div className="legend">
              <span><span className="dot" style={{ background: "#16a34a" }} /> Hits (alert and a fire followed) <strong>{tally.hit}</strong></span>
              <span><span className="dot" style={{ background: "#dc2626" }} /> Misses (fire, no alert) <strong>{tally.miss}</strong></span>
              <span><span className="dot" style={{ background: "#9ca3af" }} /> False alarms (alert, no fire) <strong>{tally.falseAlarm}</strong></span>
            </div>
            <p className="muted small">{tally.alerts.toLocaleString()} cells (0.1°) were over the alert threshold.</p>
          </fieldset>
        )}
        {error && <p className="error small">{error}</p>}

        <EventCheck version={modelVersion} onJump={(date) => { setPlaying(false); const i = days.indexOf(date); if (i >= 0) setIdx(i); }} />
        <TrainPanel horizon={horizon} onModelChanged={onModelChanged} />
        <SoilControl soil={soil} onChange={setSoil} />
      </aside>

      <main className={`map-wrap${split ? " split" : ""}`}>
        {split ? (
          <>
            <div className="half"><MapTitle>Model prediction</MapTitle>{modelMap}</div>
            <div className="half"><MapTitle>Real data</MapTitle>{realMap}</div>
          </>
        ) : overlayMap}
      </main>
    </div>
  );
}
