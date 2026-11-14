import type { Cell, DayFile, Index, IncidentCollection, Point } from "./types";

const base = import.meta.env.BASE_URL;

export async function loadIndex(): Promise<Index> {
  const r = await fetch(`${base}data/index.json`);
  if (!r.ok) throw new Error(`index.json: HTTP ${r.status}. Run "python -m firms.build" first.`);
  return r.json();
}

const dayCache = new Map<string, Promise<Point[]>>();

/** Load one day's points. Results are cached; failed loads are retried next time. */
export function loadDay(date: string): Promise<Point[]> {
  let p = dayCache.get(date);
  if (!p) {
    p = fetch(`${base}data/days/${date}.json`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`${date}: HTTP ${r.status}`);
        return toPoints(date, (await r.json()) as DayFile);
      });
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
    p = fetch(`${base}data/${dir}/${date}.json`).then(async (r) => {
      if (!r.ok) throw new Error(`incidents ${date}: HTTP ${r.status}`);
      return (await r.json()) as IncidentCollection;
    });
    incidentCache.set(key, p);
    p.catch(() => incidentCache.delete(key));
  }
  return p;
}
