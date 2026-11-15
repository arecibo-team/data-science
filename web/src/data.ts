import type { Bounds, Enums, IncidentCollection, Meta, PipelineStep, Point, PointsData, Status } from "./types";

const base = import.meta.env.BASE_URL;
// In dev the Vite proxy forwards /api to the Python server; in production it is served from the same origin.
const api = (path: string, q?: Record<string, string | number>) => {
  const url = new URL(`${base}api/${path}`, window.location.href);
  for (const [k, v] of Object.entries(q ?? {})) url.searchParams.set(k, String(v));
  return url.href;
};

export interface Query extends Bounds {
  from: string;
  to: string;
}

const params = (q: Query) => ({ from: q.from, to: q.to, west: q.west, south: q.south, east: q.east, north: q.north });

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`${new URL(url).pathname}: HTTP ${r.status}`);
  return r.json();
}

/** GET /api/tiles/status: progress of the server's background tile generation. */
export interface TileStatus {
  enabled: boolean;
  state: "starting" | "waiting" | "paused" | "building" | "idle" | "error" | "off";
  done?: number;
  total?: number;
  zoom?: number | null;
  max_zoom?: number;
}

export const loadTileStatus = (): Promise<TileStatus> => getJson(api("tiles/status"));

export interface Counts {
  MODIS: number;
  VIIRS: number;
  COMMON: number;
  total: number;
  matched: number;
  confidence: { low: number; nominal: number; high: number };
}

/** Totals for the view, used in the heat overview where single detections are not loaded. */
export const loadCounts = (q: Query, signal: AbortSignal): Promise<Counts> => getJson(api("counts", params(q)), signal);

export const loadMeta = (): Promise<Meta> => getJson(api("meta"));
export const loadPipeline = (): Promise<{ running: boolean; steps: PipelineStep[] }> => getJson(api("pipeline"));

/**
 * Heat tile URL template for Leaflet. `layers` is a bit mask (MODIS 1, VIIRS 2, COMMON 4) and `version`
 * changes when the server has new data, so cached tiles are not reused across updates.
 */
export function tileUrl(from: string, to: string, layers: number, version: string): string {
  const u = api("tiles/0/0/0.png", { from, to, layers, v: version });
  return u.replace("tiles/0/0/0.png", "tiles/{z}/{x}/{y}.png").replace(/%7B/g, "{").replace(/%7D/g, "}");
}
export const loadIncidents = (q: Query, signal: AbortSignal): Promise<IncidentCollection["features"]> =>
  getJson<IncidentCollection>(api("incidents", params(q)), signal).then((c) => c.features);

// ---- worker client -------------------------------------------------------
// Point responses are decoded into typed arrays off the main thread and transferred back.

type Pending = { resolve: (v: PointsData) => void; reject: (e: Error) => void };
let worker: Worker | null = null;
const pending = new Map<number, Pending>();
let nextId = 0;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./loader.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<{ id: number; data?: PointsData; error?: string }>) => {
      const p = pending.get(e.data.id);
      if (!p) return;
      pending.delete(e.data.id);
      if (e.data.error !== undefined) p.reject(new Error(e.data.error));
      else p.resolve(e.data.data as PointsData);
    };
  }
  return worker;
}

/** Detections in the range and box. Aborting drops the result; the worker finishes its fetch in the background. */
export function loadPoints(q: Query, signal: AbortSignal): Promise<PointsData> {
  const id = nextId++;
  return new Promise<PointsData>((resolve, reject) => {
    const abort = () => {
      pending.delete(id);
      reject(new DOMException("aborted", "AbortError"));
    };
    if (signal.aborted) return abort();
    signal.addEventListener("abort", abort, { once: true });
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, url: api("points", params(q)) });
  });
}

// ---- row access ----------------------------------------------------------

/** Build the detail record for one detection. Only called for the selected point. */
export function pointAt(d: PointsData, i: number, enums: Enums): Point {
  const c = d.cols;
  const frp = c.frp[i];
  const bt = c.bt[i];
  const inc = c.incident[i];
  const iso = new Date(c.ts[i] * 1000).toISOString();
  return {
    lat: c.lat[i],
    lon: c.lon[i],
    sensor: enums.sensor[c.sensor[i]] ?? "MODIS",
    platform: enums.platform[c.platform[i]] ?? "",
    product: enums.product[c.product[i]] ?? "",
    date: iso.slice(0, 10),
    time: iso.slice(11, 16),
    daynight: enums.daynight[c.daynight[i]] ?? null,
    confidence: enums.confidence[c.confidence[i]] ?? null,
    frp: Number.isNaN(frp) ? null : frp,
    bt: Number.isNaN(bt) ? null : bt,
    status: (enums.status[c.status[i]] ?? "both") as Status,
    cluster: c.cluster[i] < 0 ? null : c.cluster[i],
    incident: inc ? (d.incidents[inc - 1] ?? null) : null,
  };
}

// ---- forecast model ------------------------------------------------------

/** Replay of the event held out of training, computed after a run with that run's own model. */
export interface EventReplay {
  event: { name: string; first: string; last: string; bbox: number[]; spike: string[] };
  horizon: number;
  threshold: number;
  europe: { cell_days: number; positives: number; pr_auc: number | null; baseline_pr_auc: number | null; roc_auc: number | null; recall: number; precision: number };
  /** Per day in the window, for the event's region: cells that burned in the next `horizon` days, alerts, real alerts, expected burning cells. */
  days: { date: string; burning: number; alerts: number; real: number; risk: number }[];
  spike_cells: { in_region: number; scoreable: number; active: number };
  /** Spike cells flagged k days before the spike, by group (new = quiet the 10 days before, burning = already hot). */
  lead: Record<"new" | "burning", { cells: number; flagged: Record<string, number> }>;
  lead_days: number[];
}

export interface RunMetrics {
  benchmark?: EventReplay;
  threshold: number;
  horizon: number;
  label?: string;
  train_end: string;
  test_rows: number;
  pr_auc?: number;
  base_rate?: number;
  precision?: number;
  recall?: number;
  test_positives?: number;
}

export interface ModelRun {
  id: number;
  status: "running" | "done" | "failed" | "stopped" | "interrupted";
  started_at: string;
  finished_at: string | null;
  rounds_done: number;
  rounds_total: number;
  metrics: RunMetrics | null;
  message: string | null;
  checkpoints: number;
  last_checkpoint_round: number | null;
  params: {
    horizon: number; lr: number; label?: string; train_end?: string;
    holdout_event?: { name: string; first: string; last: string; bbox: number[]; spike: string[] } | null;
  };
}

export interface Runs {
  running: boolean;
  current: number | null;
  /** The run used for predictions: the newest completed one. */
  active: number | null;
  runs: ModelRun[];
}

export type TrainEvent =
  | { type: "start"; run: number; total: number; resumed?: boolean }
  | { type: "progress"; round: number; total: number; train_loss: number | null; val_loss: number | null }
  | { type: "checkpoint"; round: number }
  | { type: "done"; run: number; metrics: RunMetrics }
  | { type: "stopped"; run: number }
  | { type: "error"; run: number; message: string };

export const loadRuns = (): Promise<Runs> => getJson<Runs>(api("model/runs"));
export const modelStreamUrl = () => api("model/stream");

async function post(path: string, body?: unknown): Promise<void> {
  const r = await fetch(api(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => null);
    throw new Error(detail?.detail ?? `HTTP ${r.status}`);
  }
}
export const startTraining = (rounds: number, horizon: number) => post("model/train", { rounds, horizon });
export const stopTraining = () => post("model/stop");
export const resumeTraining = (id: number) => post(`model/resume/${id}`);

export interface Forecast {
  run: number;
  date: string;
  /** The day the features come from: the forecast date, or the last day with data. */
  feature_date: string;
  horizon: number;
  threshold: number;
  /** True when every day of the window is already in the data, so hits and misses are final. */
  complete: boolean;
  lat: Float32Array;
  lon: Float32Array;
  p: Float32Array;
  /** 1 where a COMMON fire really happened in the window. */
  actual: Uint8Array;
}

/** GET /api/forecast: uint32 header length, JSON header, then lat, lon, p (f4) and actual (u1). */
export async function loadForecast(date: string, signal?: AbortSignal): Promise<Forecast | null> {
  const r = await fetch(api("forecast", { date }), { signal });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`forecast: HTTP ${r.status}`);
  const buf = await r.arrayBuffer();
  const hl = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(buf.slice(4, 4 + hl)));
  const n: number = head.n;
  let off = 4 + hl;
  const f32 = () => {
    const a = new Float32Array(buf.slice(off, off + 4 * n)); // copy: the offset need not be 4-aligned
    off += 4 * n;
    return a;
  };
  const lat = f32(), lon = f32(), p = f32();
  const actual = new Uint8Array(buf.slice(off, off + n));
  return { ...head, lat, lon, p, actual };
}

// ---- soil ----------------------------------------------------------------

export interface SoilClasses {
  classes: string[];
  colors: number[][];
}
export const loadSoilClasses = (): Promise<SoilClasses> => getJson<SoilClasses>(api("soil/classes"));
/** Leaflet URL template of the bare-ground overlay. cls limits it to one class index. */
export function soilUrl(cls: number | null): string {
  const u = api("soil/tiles/0/0/0.png", cls === null ? {} : { cls });
  return u.replace("soil/tiles/0/0/0.png", "soil/tiles/{z}/{x}/{y}.png").replace(/%7B/g, "{").replace(/%7D/g, "}");
}

// ---- country risk -----------------------------------------------------------

export interface CountryRisk {
  name: string;
  /** Expected number of 0.1 degree cells with a confirmed fire in the window. */
  expected: number;
  /** Rank against the country's own history, 0..1; null for countries that never see fire. */
  pct: number | null;
  /** 0 (green) to 1 (red). */
  score: number;
  active: number;
  /** Cells that really burned in the window, null while the window is not complete. */
  burned: number | null;
  /** Expected cells per 1,000 km2 of the country's surface; absent from the older fallback. */
  density?: number | null;
  area_km2?: number | null;
}
export interface RiskDay {
  feature_date: string;
  horizon: number;
  complete: boolean;
  countries: CountryRisk[];
}
export async function loadRisk(date: string, signal?: AbortSignal): Promise<RiskDay | null> {
  const r = await fetch(api("risk", { date }), { signal });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`risk: HTTP ${r.status}`);
  return r.json();
}
export const loadRegions = (): Promise<GeoJSON.FeatureCollection> => getJson<GeoJSON.FeatureCollection>(api("regions"));
