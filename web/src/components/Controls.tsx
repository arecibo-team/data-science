import type { PipelineStep } from "../types";
import { type ConfFilter, type Filters, type LayerCounts, type LayerKey, type StatusFilter } from "../model";

interface Props {
  dates: string[];
  filters: Filters;
  onChange: (f: Filters) => void;
  /** Counts in the loaded map view and date range, shown next to each layer. Null in the heat overview. */
  counts: LayerCounts | null;
  /** Visible detections and how many are matched by both sensors. Null in grid view. */
  stats: { total: number; matched: number } | null;
  /** True while the server is being asked for the current view. */
  loading: boolean;
  /** Set when the view holds too many detections to draw one by one; heat tiles are shown instead. */
  overview: { total: number } | null;
  incidentCount: number;
  /** Pipeline steps while the server is still building data; null when idle. */
  pipeline: PipelineStep[] | null;
}

const PRESETS: [string, number][] = [["3 days", 3], ["30 days", 30], ["1 year", 365], ["All", 0]];

/** Range ending at the latest day. 0 days means everything. The server pre-renders 30 days, 1 year and all. */
function preset(dates: string[], days: number): Pick<Filters, "from" | "to"> {
  const to = dates[dates.length - 1];
  if (!days) return { from: dates[0], to };
  const start = new Date(`${to}T00:00:00Z`);
  start.setUTCDate(start.getUTCDate() - (days - 1));
  const want = start.toISOString().slice(0, 10);
  return { from: dates.find((d) => d >= want) ?? to, to };
}

export function Controls({ dates, filters, onChange, counts, stats, loading, overview, incidentCount, pipeline }: Props) {
  const set = (patch: Partial<Filters>) => onChange({ ...filters, ...patch });
  const setLayer = (key: LayerKey, on: boolean) => set({ layers: { ...filters.layers, [key]: on } });
  const grid = overview !== null;
  const n = (v: number | undefined) => (v === undefined ? "–" : v.toLocaleString());

  // Any range is allowed. Keep it ordered by pulling the other end along if needed.
  const setFrom = (from: string) => set({ from, to: filters.to < from ? from : filters.to });
  const setTo = (to: string) => set({ to, from: filters.from > to ? to : filters.from });

  return (
    <div className="controls">
      {pipeline && (
        <fieldset>
          <legend>Server is updating</legend>
          <ul className="steps small">
            {pipeline.map((s) => (
              <li key={s.step} className={`step-${s.state}`}>{s.title}: {s.state.replace("_", " ")}</li>
            ))}
          </ul>
        </fieldset>
      )}

      <fieldset>
        <legend>Date range (UTC)</legend>
        <div className="presets">
          {PRESETS.map(([label, days]) => (
            <button key={label} type="button" onClick={() => set(preset(dates, days))}>{label}</button>
          ))}
        </div>
        <label>
          From
          <select value={filters.from} onChange={(e) => setFrom(e.target.value)}>
            {dates.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </label>
        <label>
          To
          <select value={filters.to} onChange={(e) => setTo(e.target.value)}>
            {dates.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </label>
        <p className="muted small" role="status">
          {loading
            ? "Loading from the server…"
            : grid
              ? `${overview.total.toLocaleString()} detections in view: shown as a heat map (red = confirmed inside an incident). Zoom in or shorten the range for single points.`
              : "Showing every detection in view."}
        </p>
      </fieldset>

      <fieldset>
        <legend>Layers</legend>
        <label className="check">
          <input type="checkbox" checked={filters.layers.MODIS} onChange={(e) => setLayer("MODIS", e.target.checked)} />
          <span className="dot dot-modis" /> MODIS
          <span className="muted"> {n(counts?.MODIS)}</span>
        </label>
        <label className="check">
          <input type="checkbox" checked={filters.layers.VIIRS} onChange={(e) => setLayer("VIIRS", e.target.checked)} />
          <span className="dot dot-viirs" /> VIIRS
          <span className="muted"> {n(counts?.VIIRS)}</span>
        </label>
        <label className="check">
          <input type="checkbox" checked={filters.layers.COMMON} onChange={(e) => setLayer("COMMON", e.target.checked)} />
          <span className="dot dot-common" /> COMMON
          <span className="muted"> {n(counts?.COMMON)}</span>
        </label>
        <p className="muted small">
          COMMON is NASA detections confirmed inside a European incident, from either sensor.
        </p>
        <label className="check">
          <input
            type="checkbox"
            checked={filters.layers.INCIDENTS}
            onChange={(e) => setLayer("INCIDENTS", e.target.checked)}
          />
          <span className="swatch" /> EU Incidents
          <span className="muted"> {n(incidentCount)}</span>
        </label>
        <p className="muted small">
          Independent checks only, none derived from NASA data: Copernicus EMS wildfire mapping areas
          and Sentinel-2 burn scars (dNBR) from Planetary Computer.
        </p>
      </fieldset>

      <fieldset>
        <legend>Confidence</legend>
        <select
          value={filters.confidence}
          disabled={grid}
          onChange={(e) => set({ confidence: e.target.value as ConfFilter })}
        >
          <option value="all">All</option>
          <option value="high">High</option>
          <option value="nominal">Nominal and high</option>
          <option value="low">Low only</option>
        </select>
        <p className="muted small">
          {grid
            ? "Needs individual detections: zoom in or shorten the range."
            : "Nominal and high include detections at nominal or high confidence. Low only selects low."}
        </p>
      </fieldset>

      <fieldset>
        <legend>Cross-sensor match</legend>
        <select
          value={filters.status}
          disabled={grid}
          onChange={(e) => set({ status: e.target.value as StatusFilter })}
        >
          <option value="all">All detections</option>
          <option value="both">Matched by both sensors</option>
          <option value="single">Single-sensor only</option>
        </select>
        <p className="muted small">
          {stats
            ? `${stats.matched.toLocaleString()} of ${stats.total.toLocaleString()} shown are matched by both sensors.`
            : "Needs individual detections: zoom in or shorten the range."}
        </p>
      </fieldset>
    </div>
  );
}
