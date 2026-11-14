import type { Enums, GridData, PointsData } from "./types";

export type LayerKey = "MODIS" | "VIIRS" | "COMMON" | "INCIDENTS";
export type ConfFilter = "all" | "high" | "nominal" | "low";
export type StatusFilter = "all" | "both" | "single";

export interface Filters {
  from: string;
  to: string;
  /** Each layer toggles independently. A detection is drawn if any enabled layer includes it. */
  layers: Record<LayerKey, boolean>;
  confidence: ConfFilter;
  status: StatusFilter;
}

/** Detections per layer in the loaded view, ignoring the layer toggles. */
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
  /** Row of each visible detection within the points response. */
  row: Uint32Array;
}

export interface ScanResult {
  visible: VisibleSet;
  counts: LayerCounts;
  /** Visible detections and how many of them are matched by both sensors. */
  total: number;
  matched: number;
}

/** Filter every detection of one /api/points response. One pass, no per-row allocation. */
export function scanPoints(data: PointsData, enums: Enums, f: Filters): ScanResult {
  const c = data.cols;
  const cap = data.n;
  const lat = new Float32Array(cap);
  const lon = new Float32Array(cap);
  const frp = new Float32Array(cap);
  const viirs = new Uint8Array(cap);
  const confirmed = new Uint8Array(cap);
  const row = new Uint32Array(cap);

  const sensorOn = enums.sensor.map((s) => (s === null ? false : f.layers[s]));
  const sensorIsViirs = enums.sensor.map((s) => (s === "VIIRS" ? 1 : 0));
  const LOW = enums.confidence.indexOf("low");
  const HIGH = enums.confidence.indexOf("high");
  const BOTH = enums.status.indexOf("both");

  const counts: LayerCounts = { MODIS: 0, VIIRS: 0, COMMON: 0, INCIDENTS: 0 };
  let count = 0, total = 0, matched = 0;

  for (let i = 0; i < data.n; i++) {
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
    row[count] = i;
    count++;
  }

  return {
    visible: {
      count,
      lat: lat.subarray(0, count),
      lon: lon.subarray(0, count),
      frp: frp.subarray(0, count),
      viirs: viirs.subarray(0, count),
      confirmed: confirmed.subarray(0, count),
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
 * Apply the layer toggles to the server's per-cell counts.
 *
 * A detection can belong to a sensor layer and to COMMON. Counting each one once:
 * a confirmed MODIS detection is already in the MODIS count when that layer is on,
 * so COMMON only adds the confirmed detections whose sensor layer is off.
 */
export function aggregateGrid(grid: GridData, f: Filters): { cells: GridCell[]; counts: LayerCounts } {
  const counts: LayerCounts = { MODIS: 0, VIIRS: 0, COMMON: 0, INCIDENTS: 0 };
  const M = f.layers.MODIS ? 1 : 0;
  const V = f.layers.VIIRS ? 1 : 0;
  const C = f.layers.COMMON ? 1 : 0;
  const cellDeg = grid.cell_deg;
  const cells: GridCell[] = [];
  for (const [r, c, m, v, cm, cv] of grid.cells) {
    counts.MODIS += m;
    counts.VIIRS += v;
    counts.COMMON += cm + cv;
    const count = M * m + V * v + C * (cm * (1 - M) + cv * (1 - V));
    if (count === 0) continue;
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
