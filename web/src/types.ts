export type Sensor = "MODIS" | "VIIRS";
export type Status = "both" | "modis_only" | "viirs_only";
export type Confidence = "low" | "nominal" | "high";

/** One detection, as shown in the detail panel. */
export interface Point {
  lat: number;
  lon: number;
  sensor: Sensor;
  platform: string;
  product: string;
  date: string;
  time: string; // HH:MM UTC
  daynight: string | null;
  confidence: Confidence | null;
  frp: number | null;
  bt: number | null;
  status: Status;
  cluster: number | null;
  /** incident_id of the European incident this detection falls inside, if any. */
  incident: string | null;
}

/** Properties of one European incident polygon (EFFIS perimeter or Copernicus EMS AOI). */
export interface IncidentProperties {
  incident_id: string;
  source: "EFFIS" | "Copernicus EMS";
  name: string;
  country: string;
  start_date: string;
  end_date: string;
  area_ha: number | null;
  /** Number of NASA detections matched to this incident across the whole archive. */
  detections: number;
}

export interface DayCounts {
  total: number;
  MODIS: number;
  VIIRS: number;
  matched: number;
  confirmed: number;
}

/** Value lists for categorical columns, from index.json. Code 0 is always null. */
export interface Enums {
  sensor: (Sensor | null)[];
  platform: (string | null)[];
  product: (string | null)[];
  daynight: (string | null)[];
  confidence: (Confidence | null)[];
  status: (Status | null)[];
}

export interface MonthInfo {
  file: string;
  grid: string;
  rows: number;
  days: string[];
}

export interface Index {
  version: number;
  generated_utc: string;
  grid_cell_deg: number;
  enums: Enums;
  incidents: { file: string; count: number; confirmed: number; sources: string[] };
  months: Record<string, MonthInfo>;
  days: Record<string, DayCounts>;
}

/** GeoJSON feature of one incident polygon, as written by firms/export.py. */
export interface IncidentFeature {
  type: "Feature";
  properties: IncidentProperties;
  geometry: { type: "Polygon" | "MultiPolygon"; coordinates: number[][][] | number[][][][] };
}

export interface IncidentCollection {
  type: "FeatureCollection";
  features: IncidentFeature[];
}

/** Typed columns of one month. Row i is one detection; see firms/export.py COLUMNS. */
export interface Columns {
  lat: Float32Array;
  lon: Float32Array;
  frp: Float32Array; // NaN when missing
  bt: Float32Array; // NaN when missing
  cluster: Int32Array; // -1 when unmatched
  minute: Uint16Array; // minutes since midnight UTC
  incident: Uint16Array; // 1-based code into `incidents`; 0 = not confirmed
  day: Uint8Array; // day of month
  sensor: Uint8Array;
  platform: Uint8Array;
  product: Uint8Array;
  daynight: Uint8Array;
  confidence: Uint8Array;
  status: Uint8Array;
}

export interface MonthData {
  /** "YYYY-MM" */
  month: string;
  n: number;
  /** incident_id by 1-based incident code, for this month only. */
  incidents: string[];
  cols: Columns;
}

/** Per-day grid counts of one month: day -> rows of [row, col, modis, viirs, confirmed modis, confirmed viirs]. */
export interface GridMonth {
  cell_deg: number;
  days: Record<string, [number, number, number, number, number, number][]>;
}
