export type Sensor = "MODIS" | "VIIRS";
export type Status = "both" | "modis_only" | "viirs_only";
export type Confidence = "low" | "nominal" | "high";

/** One normalized hotspot, as used by the map. */
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

export type Cell = string | number | null;

/** Per-day file written by firms/export.py. */
export interface DayFile {
  fields: string[];
  rows: Cell[][];
}

export interface DayCounts {
  total: number;
  MODIS: number;
  VIIRS: number;
  matched: number;
  confirmed: number;
}

export interface IncidentSummary {
  dir: string;
  count: number;
  confirmed: number;
  sources: string[];
}

export interface Index {
  generated_utc: string;
  sensors: Sensor[];
  fields: string[];
  incidents: IncidentSummary;
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
