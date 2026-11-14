import { type ConfFilter, type Filters, type LayerCounts, type LayerKey, type Mode, type StatusFilter } from "../model";

interface Props {
  dates: string[];
  filters: Filters;
  onChange: (f: Filters) => void;
  /** Counts within the selected date range, shown next to each layer. */
  counts: LayerCounts;
  /** Visible detections and how many are matched by both sensors. Null in grid view. */
  stats: { total: number; matched: number } | null;
  /** Files finished so far while a new range loads; null when idle. */
  progress: { done: number; total: number } | null;
}

export function Controls({ dates, filters, onChange, counts, stats, progress }: Props) {
  const set = (patch: Partial<Filters>) => onChange({ ...filters, ...patch });
  const setLayer = (key: LayerKey, on: boolean) => set({ layers: { ...filters.layers, [key]: on } });
  const grid = filters.view === "grid";

  // Any range is allowed. Keep it ordered by pulling the other end along if needed.
  const setFrom = (from: string) => set({ from, to: filters.to < from ? from : filters.to });
  const setTo = (to: string) => set({ to, from: filters.from > to ? to : filters.from });
  const setView = (view: Mode) => set({ view });

  return (
    <div className="controls">
      <fieldset>
        <legend>View</legend>
        <select value={filters.view} onChange={(e) => setView(e.target.value as Mode)}>
          <option value="points">Points (single detections)</option>
          <option value="grid">Grid (1° cell counts)</option>
        </select>
        <p className="muted small">
          {grid
            ? "Grid counts per cell. Fast for long ranges. Confidence and cross-sensor filters need the points view."
            : "Every detection as a point. Long ranges take longer to draw."}
        </p>
      </fieldset>

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
        {progress && (
          <div className="loading" role="status">
            <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={progress.total} aria-valuenow={progress.done}>
              <div className="progress-bar" style={{ width: `${(progress.done / progress.total) * 100}%` }} />
            </div>
            <p className="muted small">
              Loading {progress.done.toLocaleString()} / {progress.total.toLocaleString()} files…
            </p>
          </div>
        )}
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
            ? "Available in the points view."
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
            : "Available in the points view."}
        </p>
      </fieldset>
    </div>
  );
}
