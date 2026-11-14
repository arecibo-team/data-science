import { useCallback, useEffect, useMemo, useState } from "react";
import { loadDay, loadIncidentDay, loadIndex } from "./data";
import { MapView } from "./components/MapView";
import { Controls, type Filters } from "./components/Controls";
import { SENSOR_COLOR } from "./components/PointLayer";
import type { Index, IncidentFeature, IncidentProperties, Point } from "./types";

const MAX_DAYS = 7;

export default function App() {
  const [index, setIndex] = useState<Index | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Point | null>(null);
  const [incidents, setIncidents] = useState<IncidentFeature[]>([]);

  const dates = useMemo(() => (index ? Object.keys(index.days).sort() : []), [index]);

  useEffect(() => {
    loadIndex()
      .then((idx) => {
        setIndex(idx);
        const all = Object.keys(idx.days).sort();
        if (all.length === 0) return;
        setFilters({
          from: all[0],
          to: all[Math.min(2, all.length - 1)],
          sensors: { MODIS: true, VIIRS: true },
          confidence: "all",
          status: "all",
          confirmedOnly: false,
          showIncidents: true,
        });
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  // Load the days in the selected range.
  const rangeKey = filters ? `${filters.from}|${filters.to}` : "";
  useEffect(() => {
    if (!filters) return;
    const days = dates.filter((d) => d >= filters.from && d <= filters.to);
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all(days.map(loadDay))
      .then((parts) => {
        if (!cancelled) setPoints(parts.flat());
      })
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // filters.from/to are captured via rangeKey; dates is stable after index loads.
  }, [rangeKey, dates]);

  // Load the European incidents active on the days in the selected range.
  useEffect(() => {
    if (!filters || !index) return;
    const days = dates.filter((d) => d >= filters.from && d <= filters.to);
    let cancelled = false;
    Promise.all(days.map((d) => loadIncidentDay(d, index.incidents.dir)))
      .then((parts) => {
        if (cancelled) return;
        // A fire spanning several days appears in each day file; keep one copy.
        const byId = new Map<string, IncidentFeature>();
        for (const part of parts) for (const f of part.features) byId.set(f.properties.incident_id, f);
        setIncidents([...byId.values()]);
      })
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
    // Same trigger as the points effect above.
  }, [rangeKey, dates, index]);

  const incidentById = useMemo(
    () => new Map(incidents.map((f) => [f.properties.incident_id, f.properties])),
    [incidents],
  );

  const visible = useMemo(() => {
    if (!filters) return [];
    return points.filter((p) => {
      if (filters.confirmedOnly && !p.incident) return false;
      if (!filters.sensors[p.sensor]) return false;
      if (filters.confidence === "high" && p.confidence !== "high") return false;
      if (filters.confidence === "nominal" && p.confidence === "low") return false;
      if (filters.confidence === "low" && p.confidence !== "low") return false;
      if (filters.status === "both" && p.status !== "both") return false;
      if (filters.status === "single" && p.status === "both") return false;
      return true;
    });
  }, [points, filters]);

  const stats = useMemo(() => {
    const s = { total: visible.length, MODIS: 0, VIIRS: 0, matched: 0, confirmed: 0 };
    for (const p of visible) {
      s[p.sensor]++;
      if (p.status === "both") s.matched++;
      if (p.incident) s.confirmed++;
    }
    return s;
  }, [visible]);

  const onPick = useCallback((p: Point | null) => setSelected(p), []);

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
          stats={stats}
          incidentCount={incidents.length}
          loading={loading}
          maxDays={MAX_DAYS}
        />
        {error && <p className="error">{error}</p>}

        <div className="legend">
          <span><span className="dot dot-modis" /> MODIS</span>
          <span><span className="dot dot-viirs" /> VIIRS</span>
          <span><span className="swatch" /> EFFIS / EMS incident</span>
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
          points={visible}
          incidents={filters.showIncidents ? incidents : []}
          onPick={onPick}
          fitKey={`${rangeKey}:${points.length}`}
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
