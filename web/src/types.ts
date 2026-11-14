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

/** Properties of one independent incident polygon (Copernicus EMS AOI or Sentinel-2 burn scar). */
export interface IncidentProperties {
  incident_id: string;
  source: "Copernicus EMS" | "Sentinel-2";
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

export interface PipelineStep {
  step: string;
  title: string;
  state: "up_to_date" | "pending" | "running" | "failed" | "skipped";
  reason: string | null;
  last: { finished_at: string | null; message: string | null; row_count: number | null } | null;
}

/** GET /api/meta */
export interface Meta {
  version: number;
  generated_utc: string | null;
  /** Grid cell sizes (degrees) the server can aggregate to. */
  grid_cells: number[];
  enums: Enums;
  incidents: { count: number; confirmed: number; sources: string[] };
  days: Record<string, DayCounts>;
  pipeline_running: boolean;
}

/** Map view the data was requested for. */
export interface Bounds {
  west: number;
  south: number;
  east: number;
  north: number;
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

/** Typed columns of one /api/points response. Row i is one detection; see server/app.py COLUMNS. */
export interface Columns {
  lat: Float32Array;
  lon: Float32Array;
  frp: Float32Array; // NaN when missing
  bt: Float32Array; // NaN when missing
  cluster: Int32Array; // -1 when unmatched
  ts: Uint32Array; // unix seconds, UTC
  incident: Uint16Array; // 1-based code into `incidents`; 0 = not confirmed
  sensor: Uint8Array;
  platform: Uint8Array;
  product: Uint8Array;
  daynight: Uint8Array;
  confidence: Uint8Array;
  status: Uint8Array;
}

export interface PointsData {
  n: number;
  /** Detections in the requested box. When it is above the server limit, `overflow` is set and `n` is 0. */
  total: number;
  overflow: boolean;
  /** incident_id by 1-based incident code, for this response only. */
  incidents: string[];
  cols: Columns;
}

/** GET /api/grid: rows of [row, col, modis, viirs, confirmed modis, confirmed viirs]. */
export interface GridData {
  cell_deg: number;
  cells: [number, number, number, number, number, number][];
}
