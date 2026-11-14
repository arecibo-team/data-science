import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { loadCounts, loadIncidents, loadMeta, loadPipeline, loadPoints, pointAt, tileUrl, type Counts, type Query } from "./data";
import { MapView } from "./components/MapView";
import { Controls } from "./components/Controls";
import { SENSOR_COLOR } from "./components/PointLayer";
import { scanPoints, type Filters, type LayerCounts } from "./model";
import type { Bounds, IncidentFeature, IncidentProperties, Meta, PipelineStep, Point, PointsData } from "./types";

const PAD = 0.25; // fetch this much beyond the view on each side, so small pans need no request
const DEBOUNCE_MS = 250;

/** What the loaded data covers. Pans inside `bounds` do not trigger a request. */
interface Loaded {
  from: string;
  to: string;
  bounds: Bounds;
  zoom: number;
  version: number;
}

const inside = (v: Bounds, b: Bounds) => v.west >= b.west && v.east <= b.east && v.south >= b.south && v.north <= b.north;
const area = (b: Bounds) => (b.east - b.west) * (b.north - b.south);

function pad(b: Bounds): Bounds {
  const dx = (b.east - b.west) * PAD;
  const dy = (b.north - b.south) * PAD;
  return {
    west: Math.max(-180, b.west - dx),
    east: Math.min(180, b.east + dx),
    south: Math.max(-90, b.south - dy),
    north: Math.min(90, b.north + dy),
  };
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters | null>(null);
  const [points, setPoints] = useState<PointsData | null>(null);
  // Set when the view holds more detections than the server sends as points: heat tiles are shown instead.
  const [overview, setOverview] = useState<{ total: number; counts: Counts } | null>(null);
  const [incidents, setIncidents] = useState<IncidentFeature[]>([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Point | null>(null);
  const [view, setView] = useState<{ bounds: Bounds; zoom: number } | null>(null);
  const [steps, setSteps] = useState<PipelineStep[]>([]);
  // Bumped when the pipeline finishes a pass, so the map asks the server again.
  const [dataVersion, setDataVersion] = useState(0);
  const loaded = useRef<Loaded | null>(null);

  const dates = useMemo(() => (meta ? Object.keys(meta.days).sort() : []), [meta]);
  const from = filters?.from ?? "";
  const to = filters?.to ?? "";

  const refreshMeta = useCallback(
    () =>
      loadMeta()
        .then((m) => {
          setMeta(m);
          const all = Object.keys(m.days).sort();
          // Start on the latest three days, which is the part people open the map for.
          if (all.length) {
            setFilters((f) => f ?? {
              from: all[Math.max(0, all.length - 3)],
              to: all[all.length - 1],
              layers: { MODIS: true, VIIRS: true, COMMON: true, INCIDENTS: true },
              confidence: "all",
              status: "all",
            });
          }
          return m;
        })
        .catch((e: Error) => {
          setError(e.message);
          return null;
        }),
    [],
  );

  useEffect(() => {
    refreshMeta();
  }, [refreshMeta]);

  // While the server runs its startup pipeline, poll it. When it finishes, reload everything.
  const running = meta?.pipeline_running ?? false;
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => {
      loadPipeline()
        .then((p) => {
          setSteps(p.steps);
          if (!p.running) {
            refreshMeta().then(() => setDataVersion((v) => v + 1));
          }
        })
        .catch(() => undefined);
    }, 3000);
    return () => clearInterval(timer);
  }, [running, refreshMeta]);

  // Ask the server for what is in view. Debounced; a pan inside the loaded box asks for nothing.
  useEffect(() => {
    if (!meta || !filters || !view) return;
    const prev = loaded.current;
    const sameRange = prev && prev.from === from && prev.to === to && prev.version === dataVersion;
    // In the overview, zooming in a lot may bring the count under the point limit: ask again.
    const stillGood =
      sameRange && inside(view.bounds, prev.bounds) && (overview ? area(view.bounds) > area(prev.bounds) / 3 : true);
    if (stillGood) {
      setLoading(false); // an earlier in-flight request was cancelled by this effect's cleanup
      return;
    }

    const ctl = new AbortController();
    const timer = setTimeout(async () => {
      const bounds = pad(view.bounds);
      const q: Query = { from, to, ...bounds };
      setLoading(true);
      setError(null);
      try {
        const [pts, inc] = await Promise.all([loadPoints(q, ctl.signal), loadIncidents(q, ctl.signal)]);
        // Too many to send as points: the sidebar numbers come from the server instead.
        const counts = pts.overflow ? await loadCounts({ from, to, ...view.bounds }, ctl.signal) : null;
        if (ctl.signal.aborted) return;
        setPoints(pts.overflow ? null : pts);
        setOverview(counts ? { total: pts.total, counts } : null);
        setIncidents(inc);
        setSelected(null);
        loaded.current = { from, to, bounds, zoom: view.zoom, version: dataVersion };
      } catch (e) {
        if (!ctl.signal.aborted) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!ctl.signal.aborted) setLoading(false);
      }
    }, DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      ctl.abort();
    };
    // `overview` only decides whether a refetch is needed; it must not trigger one.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta, from, to, view, dataVersion]);

  const enums = meta?.enums;

  // Points: one pass over the response with all filters applied.
  const scan = useMemo(
    () => (filters && enums && points ? scanPoints(points, enums, filters) : null),
    [filters, enums, points],
  );

  // Points: counted from the loaded detections. Heat overview: the server's totals for the visible box.
  const counts: LayerCounts = useMemo(() => {
    const c = overview?.counts ?? scan?.counts ?? { MODIS: 0, VIIRS: 0, COMMON: 0 };
    return { MODIS: c.MODIS, VIIRS: c.VIIRS, COMMON: c.COMMON, INCIDENTS: incidents.length };
  }, [overview, scan, incidents]);

  const layerMask = filters ? (filters.layers.MODIS ? 1 : 0) | (filters.layers.VIIRS ? 2 : 0) | (filters.layers.COMMON ? 4 : 0) : 7;
  const stats = scan
    ? { total: scan.total, matched: scan.matched }
    : overview
      ? { total: overview.counts.total, matched: overview.counts.matched }
      : null;
  const showIncidents = filters?.layers.INCIDENTS ? incidents : [];

  const incidentById = useMemo(
    () => new Map(incidents.map((f) => [f.properties.incident_id, f.properties])),
    [incidents],
  );

  const onView = useCallback((bounds: Bounds, zoom: number) => setView({ bounds, zoom }), []);

  const onPick = useCallback(
    (i: number | null) => {
      if (i === null || !scan || !enums || !points) {
        setSelected(null);
        return;
      }
      setSelected(pointAt(points, scan.visible.row[i], enums));
    },
    [scan, points, enums],
  );

  if (error && !meta) {
    return <main className="page"><h1>FIRMS hotspots</h1><p className="error">{error}</p></main>;
  }
  if (!meta) {
    return <main className="page"><h1>FIRMS hotspots</h1><p>Connecting to the server…</p></main>;
  }
  if (!filters) {
    return (
      <main className="page">
        <h1>FIRMS hotspots</h1>
        <p>{meta.pipeline_running ? "The server is building the database. This page updates when it is ready." : "No detections yet. Run the pipeline: python -m server run"}</p>
        <PipelineList steps={steps} />
      </main>
    );
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>FIRMS hotspots</h1>
        <p className="muted small">
          MODIS (Terra/Aqua) and VIIRS (SNPP/NOAA-20/NOAA-21) active fires. Data updated {meta.generated_utc ?? "never"}.
        </p>
        <Controls
          dates={dates}
          filters={filters}
          onChange={setFilters}
          counts={counts}
          stats={stats}
          loading={loading}
          overview={overview}
          pipeline={running ? steps : null}
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
          tileUrl={overview ? tileUrl(from, to, layerMask, meta.generated_utc ?? "0") : null}
          incidents={showIncidents}
          onPick={onPick}
          onView={onView}
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

function PipelineList({ steps }: { steps: PipelineStep[] }) {
  if (!steps.length) return null;
  return (
    <ul className="muted small">
      {steps.map((s) => (
        <li key={s.step}>{s.title}: {s.state.replace("_", " ")}</li>
      ))}
    </ul>
  );
}
