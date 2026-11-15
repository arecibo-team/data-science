import { useEffect } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import type { CountryRisk, RiskDay } from "../data";

/** own: each country against its own season. compare: fire per surface, one scale for every country. */
export type RiskMode = "own" | "compare";
/** model: the model's expected cells. real: the confirmed fires that really happened in the same window. */
export type RiskSource = "model" | "real";

/** Green (0) through yellow to red (1). */
export const riskColor = (s: number) => `hsl(${120 - 120 * Math.min(1, Math.max(0, s))}, 70%, 45%)`;
export const riskWord = (s: number) => (s >= 0.9 ? "Very high" : s >= 0.7 ? "High" : s >= 0.4 ? "Elevated" : s >= 0.15 ? "Normal" : "Low");

const DENSITY_LO = 0.01;   // cells per 1,000 km2 at the bottom of the compare scale
const DENSITY_HI = 1;      // ... and at the top; the scale is logarithmic

interface Side {
  expected: number;
  pct: number | null;
  score: number;
  density: number | null | undefined;
  complete: boolean;
}

/** One side of a country: the model's numbers, or the real (confirmed) ones; null when there are none yet. */
export function sideOf(c: CountryRisk | undefined, source: RiskSource): Side | null {
  if (!c) return null;
  if (source === "model") return { expected: c.expected, pct: c.pct, score: c.score, density: c.density, complete: true };
  return c.real ?? null;
}

/** The 0..1 value that colours a country, or null when there is nothing to colour. */
export function riskValue(c: CountryRisk | undefined, mode: RiskMode, source: RiskSource): number | null {
  const s = sideOf(c, source);
  if (!s) return null;
  if (mode === "own") return s.score;
  if (!s.density || s.density <= 0) return 0;
  const t = (Math.log10(s.density) - Math.log10(DENSITY_LO)) / (Math.log10(DENSITY_HI) - Math.log10(DENSITY_LO));
  return Math.min(1, Math.max(0, t));
}

/** Countries at war: their fires are often artificial (shelling, burning fields near the front), so they can be left out. */
export const WAR_COUNTRIES = new Set(["Ukraine", "Russia"]);

/** Every country coloured by its risk, under the cell layers. source picks the model's numbers or the real ones. */
export function CountryLayer({ regions, risk, mode, skipWar, source = "model" }: {
  regions: GeoJSON.FeatureCollection | null; risk: RiskDay | null; mode: RiskMode; skipWar: boolean; source?: RiskSource;
}) {
  const map = useMap();
  useEffect(() => {
    if (!regions || !risk) return;
    const by = new Map<string, CountryRisk>(risk.countries.map((c) => [c.name, c]));
    if (!map.getPane("risk")) {
      const pane = map.createPane("risk");
      pane.style.zIndex = "350";
    }
    const layer = L.geoJSON(regions, {
      pane: "risk",
      filter: (f) => !(skipWar && WAR_COUNTRIES.has(f?.properties?.name)),
      style: (f) => {
        const v = riskValue(by.get(f?.properties?.name), mode, source);
        return v === null ? { color: "#fff", weight: 0.5, fillColor: "#9ca3af", fillOpacity: 0.12 }
          : { color: "#fff", weight: 1, fillColor: riskColor(v), fillOpacity: 0.55 };
      },
      onEachFeature: (f, l) => {
        const name = f.properties?.name;
        const c = by.get(name);
        const s = sideOf(c, source);
        if (!c || !s) {
          const word = source === "real" ? "no confirmed fire activity" : "no recent fire activity";
          l.bindTooltip(`<strong>${name}</strong>: ${c ? "real data not in yet" : word}`, { sticky: true });
          return;
        }
        const v = riskValue(c, mode, source) ?? 0;
        const area = c.area_km2 ? `${Math.round(c.area_km2).toLocaleString()} km²` : "?";
        const label = source === "model" ? "Expected fire cells" : "Confirmed fire cells";
        const body = mode === "compare"
          ? `${label} per 1,000 km²: ${s.density != null ? s.density.toFixed(3) : "n/a"} (${s.expected.toFixed(1)} cells on ${area})`
          : `${label}: ${s.expected.toFixed(1)}${s.pct != null ? `, more than ${Math.round(s.pct * 100)}% of its usual days` : ""}`;
        const partial = s.complete ? "" : "<br>Window not complete yet";
        const real = source === "model" && c.burned !== null ? `<br>Real data: ${c.burned} cells had a confirmed fire` : "";
        l.bindTooltip(`<strong>${name}</strong>: ${riskWord(v)} (${Math.round(v * 100)}/100)<br>${body}${partial}${real}`, { sticky: true });
      },
    }).addTo(map);
    return () => { layer.remove(); };
  }, [map, regions, risk, mode, skipWar, source]);
  return null;
}

// ---- two maps that move together ------------------------------------------

const groups = new Map<string, Set<L.Map>>();
let syncing = false;
export function MapSync({ group }: { group: string }) {
  const map = useMap();
  useEffect(() => {
    const set = groups.get(group) ?? new Set<L.Map>();
    groups.set(group, set);
    set.add(map);
    const onMove = () => {
      if (syncing) return;
      syncing = true;
      set.forEach((m) => m !== map && m.setView(map.getCenter(), map.getZoom(), { animate: false }));
      syncing = false;
    };
    map.on("move", onMove);
    return () => {
      map.off("move", onMove);
      set.delete(map);
    };
  }, [map, group]);
  return null;
}
