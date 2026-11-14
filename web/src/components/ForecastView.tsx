import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import { MapView } from "./MapView";
import { SoilControl, type SoilState } from "./SoilControl";
import { TrainPanel } from "./TrainPanel";
import { loadForecast, soilUrl, tileUrl, type Forecast } from "../data";

const FUTURE_DAYS = 6;
const CELL = 0.1; // degrees, the bin_day cell

const HIT = "#16a34a";
const MISS = "#dc2626";
const FALSE_ALARM = "#9ca3af";

function addDays(d: string, n: number): string {
  const t = new Date(`${d}T00:00:00Z`);
  t.setUTCDate(t.getUTCDate() + n);
  return t.toISOString().slice(0, 10);
}

/** Heat colour for a probability: yellow to red. */
function probColor(p: number): string {
  const t = Math.min(1, Math.max(0, p / 0.5));
  return `hsl(${55 - 55 * t}, 95%, ${55 - 10 * t}%)`;
}

interface Tally {
  hit: number;
  miss: number;
  falseAlarm: number;
  predicted: number;
}

function classify(f: Forecast, i: number): "hit" | "miss" | "false" | "predicted" | "low" {
  const hot = f.p[i] >= f.threshold;
  if (f.actual[i]) return hot ? "hit" : "miss";
  if (hot) return f.complete ? "false" : "predicted";
  return "low";
}

function tally(f: Forecast | null): Tally {
  const t: Tally = { hit: 0, miss: 0, falseAlarm: 0, predicted: 0 };
  if (!f) return t;
  for (let i = 0; i < f.p.length; i++) {
    const c = classify(f, i);
    if (c === "hit") t.hit++;
    else if (c === "miss") t.miss++;
    else if (c === "false") t.falseAlarm++;
    if (f.p[i] >= f.threshold) t.predicted++;
  }
  return t;
}

/** One rectangle per predicted cell, on a canvas renderer so thousands of cells stay fast. */
function ForecastCells({ forecast }: { forecast: Forecast | null }) {
  const map = useMap();
  useEffect(() => {
    if (!forecast) return;
    const renderer = L.canvas({ padding: 0.3 });
    const group = L.featureGroup();
    for (let i = 0; i < forecast.p.length; i++) {
      const kind = classify(forecast, i);
      const color = kind === "hit" ? HIT : kind === "miss" ? MISS : kind === "false" ? FALSE_ALARM : probColor(forecast.p[i]);
      const op = kind === "low" ? 0.15 + forecast.p[i] : kind === "predicted" ? 0.6 : 0.75;
      const la = forecast.lat[i], lo = forecast.lon[i];
      L.rectangle([[la - CELL / 2, lo - CELL / 2], [la + CELL / 2, lo + CELL / 2]], {
        renderer, stroke: false, fillColor: color, fillOpacity: Math.min(op, 0.85), interactive: false,
      }).addTo(group);
    }
    group.addTo(map);
    return () => {
      group.remove();
    };
  }, [map, forecast]);
  return null;
}

interface Props {
  /** Days with data, ascending. */
  dates: string[];
  generatedUtc: string;
  /** The Map / Forecast switch, rendered at the top of the sidebar. */
  header: ReactNode;
}

/** Second map view: scroll a timeline, see what was detected on a day and what the model predicts next. */
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
  const day = days[idx];
  const future = day > last;

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setIdx((i) => (i + 1 >= days.length ? (setPlaying(false), i) : i + 1)), 900);
    return () => clearInterval(t);
  }, [playing, days.length]);

  // Fetch the forecast for the selected day; a quick scrub cancels the requests it skipped.
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
  const t = useMemo(() => tally(forecast), [forecast]);
  const noop = useCallback(() => undefined, []);
  const heat = future ? null : tileUrl(day, day, 7, generatedUtc);

  return (
    <div className="app">
      <aside className="sidebar">
        {header}
        <p className="muted small">
          Orange, blue and red: detections on the selected day. Squares: where the model expects a COMMON fire
          (a detection inside an EU incident) in the days after it.
        </p>

        <fieldset>
          <legend>Timeline (UTC)</legend>
          <div className="timeline-day">
            <strong>{day}</strong>
            {future && <span className="muted"> · forecast</span>}
          </div>
          <input
            type="range" min={0} max={days.length - 1} value={idx} className="slider"
            onChange={(e) => { setPlaying(false); setIdx(Number(e.target.value)); }}
            aria-label="Day"
          />
          <div className="presets">
            <button type="button" onClick={() => { setPlaying(false); setIdx(Math.max(0, idx - 1)); }}>◀ Day</button>
            <button type="button" onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>
            <button type="button" onClick={() => { setPlaying(false); setIdx(Math.min(days.length - 1, idx + 1)); }}>Day ▶</button>
            <button type="button" onClick={() => { setPlaying(false); setIdx(dates.length - 1); }}>Latest</button>
          </div>
          {forecast && (
            <p className="muted small">
              Prediction made on {forecast.feature_date}{forecast.feature_date !== day ? " (the last day with data)" : ""} for the
              next {forecast.horizon} days
              {forecast.complete ? "" : "; those days are not all in the data yet, so hits and misses are partial"}.
            </p>
          )}
        </fieldset>

        {missing && <p className="small">No completed model yet. Train one below.</p>}
        {forecast && (
          <fieldset>
            <legend>Forecast vs what happened</legend>
            <div className="legend">
              <span><span className="dot" style={{ background: HIT }} /> Hit: predicted and a COMMON fire occurred <strong>{t.hit}</strong></span>
              <span><span className="dot" style={{ background: MISS }} /> Miss: fire occurred, not predicted <strong>{t.miss}</strong></span>
              <span><span className="dot" style={{ background: FALSE_ALARM }} /> No fire followed <strong>{t.falseAlarm}</strong></span>
              <span><span className="dot" style={{ background: probColor(0.3) }} /> Predicted, outcome not known yet</span>
            </div>
            <p className="muted small">
              {t.predicted.toLocaleString()} cells (0.1°) above the model's alert threshold {forecast.threshold.toFixed(4)} on this day.
              Faint squares are lower probabilities.
            </p>
          </fieldset>
        )}
        {error && <p className="error small">{error}</p>}

        <TrainPanel horizon={forecast?.horizon ?? 3} onModelChanged={onModelChanged} />
        <SoilControl soil={soil} onChange={setSoil} />
      </aside>

      <main className="map-wrap">
        <MapView
          visible={null}
          tileUrl={heat}
          incidents={[]}
          soilUrl={soil.on ? soilUrl(soil.cls) : null}
          onPick={noop}
          onView={noop}
        >
          <ForecastCells forecast={forecast} />
        </MapView>
      </main>
    </div>
  );
}
