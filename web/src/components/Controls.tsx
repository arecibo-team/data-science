import type { Confidence } from "../types";

export type StatusFilter = "all" | "both" | "single";
export type ConfFilter = "all" | Confidence;
export type LayerKey = "MODIS" | "VIIRS" | "COMMON" | "INCIDENTS";

export interface Filters {
  from: string;
  to: string;
  /** Each layer toggles independently. A detection is drawn if any enabled layer includes it. */
  layers: Record<LayerKey, boolean>;
  confidence: ConfFilter;
  status: StatusFilter;
}

interface Props {
  dates: string[];
  filters: Filters;
  onChange: (f: Filters) => void;
  /** Counts within the selected date range, shown next to each layer. */
  counts: Record<LayerKey, number>;
  stats: { total: number; matched: number };
  loading: boolean;
}

export function Controls({ dates, filters, onChange, counts, stats, loading }: Props) {
  const set = (patch: Partial<Filters>) => onChange({ ...filters, ...patch });
  const setLayer = (key: LayerKey, on: boolean) => set({ layers: { ...filters.layers, [key]: on } });

  // Any range is allowed. Keep it ordered by pulling the other end along if needed.
  const setFrom = (from: string) => set({ from, to: filters.to < from ? from : filters.to });
  const setTo = (to: string) => set({ to, from: filters.from > to ? to : filters.from });

  return (
    <div className="controls">
      <fieldset>
        <legend>Date range (UTC)</legend>
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
        {loading && <p className="muted">Loading…</p>}
      </fieldset>

      <fieldset>
        <legend>Layers</legend>
        <label className="check">
          <input type="checkbox" checked={filters.layers.MODIS} onChange={(e) => setLayer("MODIS", e.target.checked)} />
          <span className="dot dot-modis" /> MODIS
          <span className="muted"> {counts.MODIS.toLocaleString()}</span>
        </label>
        <label className="check">
          <input type="checkbox" checked={filters.layers.VIIRS} onChange={(e) => setLayer("VIIRS", e.target.checked)} />
          <span className="dot dot-viirs" /> VIIRS
          <span className="muted"> {counts.VIIRS.toLocaleString()}</span>
        </label>
        <label className="check">
          <input type="checkbox" checked={filters.layers.COMMON} onChange={(e) => setLayer("COMMON", e.target.checked)} />
          <span className="dot dot-common" /> COMMON
          <span className="muted"> {counts.COMMON.toLocaleString()}</span>
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
          <span className="muted"> {counts.INCIDENTS.toLocaleString()}</span>
        </label>
        <p className="muted small">
          EFFIS burnt-area perimeters (30 ha or larger) and Copernicus EMS wildfire mapping areas.
        </p>
      </fieldset>

      <fieldset>
        <legend>Confidence</legend>
        <select value={filters.confidence} onChange={(e) => set({ confidence: e.target.value as ConfFilter })}>
          <option value="all">All</option>
          <option value="high">High</option>
          <option value="nominal">Nominal and high</option>
          <option value="low">Low only</option>
        </select>
        <p className="muted small">
          Nominal and high include detections at nominal or high confidence. Low only selects low.
        </p>
      </fieldset>

      <fieldset>
        <legend>Cross-sensor match</legend>
        <select value={filters.status} onChange={(e) => set({ status: e.target.value as StatusFilter })}>
          <option value="all">All detections</option>
          <option value="both">Matched by both sensors</option>
          <option value="single">Single-sensor only</option>
        </select>
        <p className="muted small">
          {stats.matched.toLocaleString()} of {stats.total.toLocaleString()} shown are matched by both sensors.
        </p>
      </fieldset>
    </div>
  );
}
