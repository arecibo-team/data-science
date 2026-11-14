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
