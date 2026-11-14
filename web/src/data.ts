import type { Cell, DayFile, Index, IncidentCollection, Point } from "./types";

const base = import.meta.env.BASE_URL;

export async function loadIndex(): Promise<Index> {
  const r = await fetch(`${base}data/index.json`);
  if (!r.ok) throw new Error(`index.json: HTTP ${r.status}. Run "python -m firms.build" first.`);
  return r.json();
}

/**
 * Fetch and parse JSON. Network-level failures (the browser's NetworkError, typically from
 * too many requests in flight) are retried with backoff; HTTP errors are not.
 */
async function fetchJson<T>(url: string, attempts = 4): Promise<T> {
  for (let i = 0; ; i++) {
    let r: Response;
    try {
      r = await fetch(url);
    } catch (e) {
      if (i + 1 >= attempts) throw e;
      await new Promise((res) => setTimeout(res, 250 * 2 ** i));
      continue;
    }
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return (await r.json()) as T;
  }
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

const dayCache = new Map<string, Promise<Point[]>>();

/** Load one day's points. Results are cached; failed loads are retried next time. */
export function loadDay(date: string): Promise<Point[]> {
  let p = dayCache.get(date);
  if (!p) {
    p = fetchJson<DayFile>(`${base}data/days/${date}.json`).then((file) => toPoints(date, file));
    dayCache.set(date, p);
    p.catch(() => dayCache.delete(date));
  }
  return p;
}

function toPoints(date: string, file: DayFile): Point[] {
  const at = Object.fromEntries(file.fields.map((name, i) => [name, i])) as Record<string, number>;
  const get = (row: Cell[], name: string): Cell => row[at[name]] ?? null;
  return file.rows.map((row) => ({
    lat: get(row, "lat") as number,
    lon: get(row, "lon") as number,
    sensor: get(row, "sensor") as Point["sensor"],
    platform: String(get(row, "platform") ?? ""),
    product: String(get(row, "product") ?? ""),
    date,
    time: String(get(row, "time_utc") ?? ""),
    daynight: get(row, "daynight") as string | null,
    confidence: get(row, "confidence") as Point["confidence"],
    frp: get(row, "frp_mw") as number | null,
    bt: get(row, "bt_mid_k") as number | null,
    status: get(row, "status") as Point["status"],
    cluster: get(row, "cluster_id") as number | null,
    incident: (get(row, "incident") as string | null) || null,
  }));
}

const incidentCache = new Map<string, Promise<IncidentCollection>>();

/** Incidents active on one UTC day. Cached; failed loads retry next time. */
export function loadIncidentDay(date: string, dir: string): Promise<IncidentCollection> {
  const key = `${dir}/${date}`;
  let p = incidentCache.get(key);
  if (!p) {
    p = fetchJson<IncidentCollection>(`${base}data/${dir}/${date}.json`);
    incidentCache.set(key, p);
    p.catch(() => incidentCache.delete(key));
  }
  return p;
}
