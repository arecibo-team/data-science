import type { Confidence, Sensor } from "../types";

export type StatusFilter = "all" | "both" | "single";
export type ConfFilter = "all" | Confidence;

export interface Filters {
  from: string;
  to: string;
  sensors: Record<Sensor, boolean>;
  confidence: ConfFilter;
  status: StatusFilter;
  /** Hide NASA detections not inside a known European incident. */
  confirmedOnly: boolean;
  /** Draw incident perimeters for the selected date range. */
  showIncidents: boolean;
}

interface Props {
  dates: string[];
  filters: Filters;
  onChange: (f: Filters) => void;
  stats: { total: number; MODIS: number; VIIRS: number; matched: number; confirmed: number };
  incidentCount: number;
  loading: boolean;
  maxDays: number;
}

export function Controls({ dates, filters, onChange, stats, incidentCount, loading, maxDays }: Props) {
  const set = (patch: Partial<Filters>) => onChange({ ...filters, ...patch });

  const setFrom = (from: string) => {
    // Keep the range within maxDays by moving "to" if needed.
    let to = filters.to;
    if (to < from) to = from;
    const limit = dates[Math.min(dates.indexOf(from) + maxDays - 1, dates.length - 1)];
    if (to > limit) to = limit;
    set({ from, to });
  };
  const setTo = (to: string) => {
    let from = filters.from;
    if (from > to) from = to;
    const floor = dates[Math.max(dates.indexOf(to) - maxDays + 1, 0)];
    if (from < floor) from = floor;
    set({ from, to });
  };

  return (
    <div className="controls">
      <fieldset>
        <legend>Date range (UTC, max {maxDays} days)</legend>
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
        <legend>Sensor</legend>
        {(["MODIS", "VIIRS"] as Sensor[]).map((s) => (
          <label key={s} className="check">
            <input
              type="checkbox"
              checked={filters.sensors[s]}
              onChange={(e) => set({ sensors: { ...filters.sensors, [s]: e.target.checked } })}
            />
            <span className={`dot dot-${s.toLowerCase()}`} /> {s}
            <span className="muted"> {stats[s].toLocaleString()}</span>
          </label>
        ))}
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
        <legend>European incidents</legend>
        <label className="check">
          <input
            type="checkbox"
            checked={filters.confirmedOnly}
            onChange={(e) => set({ confirmedOnly: e.target.checked })}
          />
          Only confirmed European incidents
          <span className="muted"> {stats.confirmed.toLocaleString()}</span>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={filters.showIncidents}
            onChange={(e) => set({ showIncidents: e.target.checked })}
          />
          Show incident perimeters
          <span className="muted"> {incidentCount.toLocaleString()}</span>
        </label>
        <p className="muted small">
          Confirmed means the detection lies within an EFFIS burnt-area perimeter (30 ha or larger) or a
          Copernicus EMS wildfire mapping area, dated within two days of the incident.
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
