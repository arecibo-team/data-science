import { useCallback, useEffect, useMemo, useState } from "react";
import { loadGrid, loadIncidents, loadIndex, loadMonth, mapLimit, pointAt, pruneCaches } from "./data";
import { MapView } from "./components/MapView";
import { Controls } from "./components/Controls";
import { SENSOR_COLOR } from "./components/PointLayer";
import { aggregateGrid, scanMonths, type Filters, type LayerCounts, type Mode } from "./model";
import type { GridMonth, Index, IncidentFeature, IncidentProperties, MonthData, Point } from "./types";

const NO_COUNTS: LayerCounts = { MODIS: 0, VIIRS: 0, COMMON: 0, INCIDENTS: 0 };

export default function App() {
  const [index, setIndex] = useState<Index | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters | null>(null);
  const [months, setMonths] = useState<MonthData[]>([]);
  const [grids, setGrids] = useState<GridMonth[]>([]);
  const [incidents, setIncidents] = useState<IncidentFeature[]>([]);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [dataVersion, setDataVersion] = useState(0);
  const [selected, setSelected] = useState<Point | null>(null);

  const dates = useMemo(() => (index ? Object.keys(index.days).sort() : []), [index]);
  const from = filters?.from ?? "";
  const to = filters?.to ?? "";
  const mode: Mode = filters?.view ?? "points";

  // Startup: index, then the European incidents (loaded once for the whole archive).
  useEffect(() => {
    loadIndex()
      .then((idx) => {
        setIndex(idx);
        const all = Object.keys(idx.days).sort();
        if (all.length === 0) return;
        setFilters({
          from: all[0],
          to: all[Math.min(2, all.length - 1)],
          view: "points",
          layers: { MODIS: true, VIIRS: true, COMMON: true, INCIDENTS: true },
          confidence: "all",
          status: "all",
        });
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!index) return;
    loadIncidents(index.incidents.file)
      .then(setIncidents)
      .catch((e: Error) => setError(`Incidents: ${e.message}`));
  }, [index]);

  // Load the months (or per-month grids) that overlap the range. Layer toggles do not reload.
  useEffect(() => {
    if (!index || !from) return;
    const infos = Object.entries(index.months)
      .filter(([, info]) => info.days.some((d) => d >= from && d <= to))
      .sort(([a], [b]) => a.localeCompare(b));
    let cancelled = false;
    let done = 0;
    const total = infos.length;
    setError(null);
    setProgress(total ? { done: 0, total } : null);
    const tick = () => {
      done++;
      if (!cancelled) setProgress({ done, total });
    };

    const load = async () => {
      if (mode === "grid") {
        const gs = await mapLimit(infos, 4, async ([m, info]) => {
          const g = await loadGrid(m, info.grid);
          tick();
          return g;
        });
        if (cancelled) return;
        setGrids(gs);
        setMonths([]);
      } else {
        const ms = await mapLimit(infos, 4, async ([m, info]) => {
          const d = await loadMonth(m, info.file);
          tick();
          return d;
        });
        if (cancelled) return;
        setMonths(ms);
        setGrids([]);
      }
      pruneCaches(new Set(infos.map(([m]) => m)));
      setDataVersion((v) => v + 1);
    };

    load()
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setProgress(null));
    return () => {
      cancelled = true;
    };
  }, [index, from, to, mode]);

  const enums = index?.enums;

  // Points mode: one pass over the loaded months with all filters applied.
  const scan = useMemo(
    () => (filters && enums && mode === "points" ? scanMonths(months, enums, filters) : null),
    [filters, enums, months, mode],
  );

  // Grid mode: per-cell counts over the range, with the layer toggles applied.
  const agg = useMemo(
    () => (filters && index && mode === "grid" ? aggregateGrid(grids, index.grid_cell_deg, filters) : null),
    [filters, index, grids, mode],
  );

  // Incidents overlapping the selected range.
  const rangeIncidents = useMemo(
    () => incidents.filter((f) => f.properties.start_date <= to && f.properties.end_date >= from),
    [incidents, from, to],
  );

  const counts: LayerCounts = useMemo(
    () => ({ ...(scan?.counts ?? agg?.counts ?? NO_COUNTS), INCIDENTS: rangeIncidents.length }),
    [scan, agg, rangeIncidents],
  );

  const stats = scan ? { total: scan.total, matched: scan.matched } : null;
  const showIncidents = filters?.layers.INCIDENTS ? rangeIncidents : [];

  const incidentById = useMemo(
    () => new Map(incidents.map((f) => [f.properties.incident_id, f.properties])),
    [incidents],
  );

  const onPick = useCallback(
    (i: number | null) => {
      if (i === null || !scan || !enums) {
        setSelected(null);
        return;
      }
      setSelected(pointAt(months[scan.visible.chunk[i]], scan.visible.row[i], enums));
    },
    [scan, months, enums],
  );

  if (error && !filters) {
    return <main className="page"><h1>FIRMS hotspots</h1><p className="error">{error}</p></main>;
  }
  if (!filters || !index) {
    return <main className="page"><h1>FIRMS hotspots</h1><p>Loading index…</p></main>;
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>FIRMS hotspots</h1>
        <p className="muted small">
          MODIS (Terra/Aqua) and VIIRS (SNPP/NOAA-20/NOAA-21) active fires. Generated {index.generated_utc}.
        </p>
        <Controls
          dates={dates}
          filters={filters}
          onChange={setFilters}
          counts={counts}
          stats={stats}
          progress={progress}
        />
        {error && <p className="error">{error}</p>}

        <div className="legend">
          <span><span className="dot dot-modis" /> MODIS</span>
          <span><span className="dot dot-viirs" /> VIIRS</span>
          <span><span className="dot dot-common" /> COMMON: confirmed inside an EU incident</span>
          <span><span className="swatch" /> EU Incidents (EMS / Sentinel-2)</span>
          <span className="muted small">Marker size grows with fire radiative power (FRP).</span>
        </div>

        <section className="detail">
          <h2>Selected detection</h2>
          {selected ? (
            <dl>
              <dt>Sensor</dt><dd style={{ color: SENSOR_COLOR[selected.sensor] }}>{selected.sensor}</dd>
              <dt>Platform</dt><dd>{selected.platform} ({selected.product})</dd>
              <dt>Date / time (UTC)</dt><dd>{selected.date} {selected.time}</dd>
              <dt>Day / night</dt><dd>{selected.daynight ?? "–"}</dd>
              <dt>Confidence</dt><dd>{selected.confidence ?? "–"}</dd>
              <dt>FRP (MW)</dt><dd>{selected.frp ?? "–"}</dd>
              <dt>Brightness (K)</dt><dd>{selected.bt ?? "–"}</dd>
              <dt>Cross-sensor</dt>
              <dd>{selected.status === "both" ? `matched, cluster ${selected.cluster}` : selected.status.replace("_", " ")}</dd>
              <dt>Location</dt><dd>{selected.lat.toFixed(4)}, {selected.lon.toFixed(4)}</dd>
              <dt>European incident</dt>
              <dd>{selected.incident ? <IncidentInfo p={incidentById.get(selected.incident)} id={selected.incident} /> : "none (unconfirmed)"}</dd>
            </dl>
          ) : (
            <p className="muted small">Click a point on the map.</p>
          )}
        </section>
      </aside>

      <main className="map-wrap">
        <MapView
          visible={scan?.visible ?? null}
          cells={agg?.cells ?? null}
          incidents={showIncidents}
          onPick={onPick}
          fitKey={`${from}|${to}|${dataVersion}`}
        />
      </main>
    </div>
  );
}

function IncidentInfo({ p, id }: { p: IncidentProperties | undefined; id: string }) {
  if (!p) return <>{id}</>;
  return (
    <>
      {p.name}
      <br />
      <span className="muted small">
        {p.source} · {p.country} · {p.start_date}
        {p.end_date !== p.start_date ? ` to ${p.end_date}` : ""}
        {p.area_ha !== null ? ` · ${Math.round(p.area_ha).toLocaleString()} ha` : ""}
      </span>
    </>
  );
}
