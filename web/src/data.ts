import type { Enums, GridMonth, Index, IncidentCollection, MonthData, Point, Status } from "./types";

const base = import.meta.env.BASE_URL;
const absolute = (path: string) => new URL(`${base}${path}`, window.location.href).href;

export async function loadIndex(): Promise<Index> {
  const r = await fetch(absolute("data/index.json"));
  if (!r.ok) throw new Error(`index.json: HTTP ${r.status}. Run "python -m firms.build" first.`);
  return r.json();
}

/**
 * Run `fn` over `items` with at most `limit` calls in flight at once. Results keep input order.
 * Loading a long range this way avoids opening hundreds of connections at the same time.
 */
export async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i]);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

// ---- worker client -------------------------------------------------------

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void };
let worker: Worker | null = null;
const pending = new Map<number, Pending>();
let nextId = 0;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./loader.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<{ id: number; data?: unknown; error?: string }>) => {
      const p = pending.get(e.data.id);
      if (!p) return;
      pending.delete(e.data.id);
      if (e.data.error !== undefined) p.reject(new Error(e.data.error));
      else p.resolve(e.data.data);
    };
  }
  return worker;
}

function request<T>(kind: "month" | "json", path: string): Promise<T> {
  const id = nextId++;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    getWorker().postMessage({ id, kind, url: absolute(path) });
  });
}

// ---- caches --------------------------------------------------------------
// A failed load is dropped from the cache so the next attempt retries it.

const monthCache = new Map<string, Promise<MonthData>>();
const gridCache = new Map<string, Promise<GridMonth>>();
let incidentCache: Promise<IncidentCollection> | null = null;

/** Detections of one UTC month. */
export function loadMonth(month: string, file: string): Promise<MonthData> {
  let p = monthCache.get(month);
  if (!p) {
    p = request<Omit<MonthData, "month">>("month", `data/${file}`).then((d) => ({ month, ...d }));
    monthCache.set(month, p);
    p.catch(() => monthCache.delete(month));
  }
  return p;
}

/** Per-day grid counts of one UTC month, used for long ranges. */
export function loadGrid(month: string, file: string): Promise<GridMonth> {
  let p = gridCache.get(month);
  if (!p) {
    p = request<GridMonth>("json", `data/${file}`);
    gridCache.set(month, p);
    p.catch(() => gridCache.delete(month));
  }
  return p;
}

/** All European incidents. Loaded once. */
export function loadIncidents(file: string): Promise<IncidentCollection["features"]> {
  if (!incidentCache) {
    incidentCache = request<IncidentCollection>("json", `data/${file}`);
    incidentCache.catch(() => (incidentCache = null));
  }
  return incidentCache.then((c) => c.features);
}

/** Drop cached months that are no longer in use, so memory tracks the selected range. */
export function pruneCaches(keep: Set<string>): void {
  for (const m of monthCache.keys()) if (!keep.has(m)) monthCache.delete(m);
  for (const m of gridCache.keys()) if (!keep.has(m)) gridCache.delete(m);
}

// ---- row access ----------------------------------------------------------

/** Build the detail record for one detection. Only called for the selected point. */
export function pointAt(m: MonthData, i: number, enums: Enums): Point {
  const c = m.cols;
  const frp = c.frp[i];
  const bt = c.bt[i];
  const minute = c.minute[i];
  const inc = c.incident[i];
  const hh = String(Math.floor(minute / 60)).padStart(2, "0");
  const mm = String(minute % 60).padStart(2, "0");
  return {
    lat: c.lat[i],
    lon: c.lon[i],
    sensor: enums.sensor[c.sensor[i]] ?? "MODIS",
    platform: enums.platform[c.platform[i]] ?? "",
    product: enums.product[c.product[i]] ?? "",
    date: `${m.month}-${String(c.day[i]).padStart(2, "0")}`,
    time: `${hh}:${mm}`,
    daynight: enums.daynight[c.daynight[i]] ?? null,
    confidence: enums.confidence[c.confidence[i]] ?? null,
    frp: Number.isNaN(frp) ? null : frp,
    bt: Number.isNaN(bt) ? null : bt,
    status: (enums.status[c.status[i]] ?? "both") as Status,
    cluster: c.cluster[i] < 0 ? null : c.cluster[i],
    incident: inc ? (m.incidents[inc - 1] ?? null) : null,
  };
}
