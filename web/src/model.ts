import type { Enums, GridMonth, MonthData } from "./types";

export type LayerKey = "MODIS" | "VIIRS" | "COMMON" | "INCIDENTS";
export type ConfFilter = "all" | "high" | "nominal" | "low";
export type StatusFilter = "all" | "both" | "single";
export type Mode = "points" | "grid";

export interface Filters {
  from: string;
  to: string;
  /** "points": single detections (any range). "grid": per-cell counts, faster for long ranges. */
  view: Mode;
  /** Each layer toggles independently. A detection is drawn if any enabled layer includes it. */
  layers: Record<LayerKey, boolean>;
  confidence: ConfFilter;
  status: StatusFilter;
}

/** Detections per layer within the range, ignoring the layer toggles. */
export interface LayerCounts {
  MODIS: number;
  VIIRS: number;
  COMMON: number;
  INCIDENTS: number;
}

/** Detections that pass every filter, as typed arrays for the canvas layer. */
export interface VisibleSet {
  count: number;
  lat: Float32Array;
  lon: Float32Array;
  frp: Float32Array; // NaN when missing
  viirs: Uint8Array; // 1 for VIIRS, 0 for MODIS
  confirmed: Uint8Array; // 1 when inside a European incident
  /** Where each detection lives: index into the scanned months, and row within that month. */
  chunk: Uint16Array;
  row: Uint32Array;
}

export interface ScanResult {
  visible: VisibleSet;
  counts: LayerCounts;
  /** Visible detections and how many of them are matched by both sensors. */
  total: number;
  matched: number;
}

/** Filter every detection in the given months. One pass, no per-row allocation. */
export function scanMonths(months: MonthData[], enums: Enums, f: Filters): ScanResult {
  const cap = months.reduce((s, m) => s + m.n, 0);
  const lat = new Float32Array(cap);
  const lon = new Float32Array(cap);
  const frp = new Float32Array(cap);
  const viirs = new Uint8Array(cap);
  const confirmed = new Uint8Array(cap);
  const chunk = new Uint16Array(cap);
  const row = new Uint32Array(cap);

  const sensorOn = enums.sensor.map((s) => (s === null ? false : f.layers[s]));
  const sensorIsViirs = enums.sensor.map((s) => (s === "VIIRS" ? 1 : 0));
  const LOW = enums.confidence.indexOf("low");
  const HIGH = enums.confidence.indexOf("high");
  const BOTH = enums.status.indexOf("both");

  const counts: LayerCounts = { MODIS: 0, VIIRS: 0, COMMON: 0, INCIDENTS: 0 };
  let count = 0, total = 0, matched = 0;

  months.forEach((m, mi) => {
    const c = m.cols;
    // Which days of this month fall in the range.
    const dayOn = new Uint8Array(32);
    for (let d = 1; d <= 31; d++) {
      const date = `${m.month}-${String(d).padStart(2, "0")}`;
      dayOn[d] = date >= f.from && date <= f.to ? 1 : 0;
    }
    for (let i = 0; i < m.n; i++) {
      if (!dayOn[c.day[i]]) continue;
      const s = c.sensor[i];
      const conf = c.incident[i] !== 0;
      const name = enums.sensor[s];
      if (name === "MODIS") counts.MODIS++;
      else if (name === "VIIRS") counts.VIIRS++;
      if (conf) counts.COMMON++;

      if (!(sensorOn[s] || (f.layers.COMMON && conf))) continue;
      const confidence = c.confidence[i];
      if (f.confidence === "high" && confidence !== HIGH) continue;
      if (f.confidence === "nominal" && confidence === LOW) continue;
      if (f.confidence === "low" && confidence !== LOW) continue;
      const isBoth = c.status[i] === BOTH;
      if (f.status === "both" && !isBoth) continue;
      if (f.status === "single" && isBoth) continue;

      total++;
      if (isBoth) matched++;
      lat[count] = c.lat[i];
      lon[count] = c.lon[i];
      frp[count] = c.frp[i];
      viirs[count] = sensorIsViirs[s];
      confirmed[count] = conf ? 1 : 0;
      chunk[count] = mi;
      row[count] = i;
      count++;
    }
  });

  return {
    visible: {
      count,
      lat: lat.subarray(0, count),
      lon: lon.subarray(0, count),
      frp: frp.subarray(0, count),
      viirs: viirs.subarray(0, count),
      confirmed: confirmed.subarray(0, count),
      chunk: chunk.subarray(0, count),
      row: row.subarray(0, count),
    },
    counts,
    total,
    matched,
  };
}

/** One grid cell of the long-range overview, counting only the enabled layers. */
export interface GridCell {
  lat: number;
  lon: number;
  count: number;
  modis: number;
  viirs: number;
  confirmed: number;
}

/**
 * Sum the per-day grid counts over the range and apply the layer toggles.
 *
 * A detection can belong to a sensor layer and to COMMON. Counting each one once:
 * a confirmed MODIS detection is already in the MODIS count when that layer is on,
 * so COMMON only adds the confirmed detections whose sensor layer is off.
 */
export function aggregateGrid(
  grids: GridMonth[],
  cellDeg: number,
  f: Filters,
): { cells: GridCell[]; counts: LayerCounts } {
  const sums = new Map<number, [number, number, number, number]>(); // key = row * 1e6 + col
  const counts: LayerCounts = { MODIS: 0, VIIRS: 0, COMMON: 0, INCIDENTS: 0 };

  for (const g of grids) {
    for (const [date, rows] of Object.entries(g.days)) {
      if (date < f.from || date > f.to) continue;
      for (const [r, c, m, v, cm, cv] of rows) {
        counts.MODIS += m;
        counts.VIIRS += v;
        counts.COMMON += cm + cv;
        const key = r * 1e6 + c;
        const s = sums.get(key) ?? [0, 0, 0, 0];
        s[0] += m; s[1] += v; s[2] += cm; s[3] += cv;
        sums.set(key, s);
      }
    }
  }

  const M = f.layers.MODIS ? 1 : 0;
  const V = f.layers.VIIRS ? 1 : 0;
  const C = f.layers.COMMON ? 1 : 0;
  const cells: GridCell[] = [];
  for (const [key, [m, v, cm, cv]] of sums) {
    const count = M * m + V * v + C * (cm * (1 - M) + cv * (1 - V));
    if (count === 0) continue;
    const r = Math.floor(key / 1e6);
    const c = key - r * 1e6;
    cells.push({
      lat: -90 + (r + 0.5) * cellDeg,
      lon: -180 + (c + 0.5) * cellDeg,
      count,
      modis: M * m,
      viirs: V * v,
      confirmed: C * (cm + cv),
    });
  }
  return { cells, counts };
}
