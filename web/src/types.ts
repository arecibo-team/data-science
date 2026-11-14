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
}

export interface Index {
  generated_utc: string;
  sensors: Sensor[];
  fields: string[];
  days: Record<string, DayCounts>;
}
