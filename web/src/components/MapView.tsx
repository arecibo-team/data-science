import { useEffect } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import { CanvasPointLayer } from "./PointLayer";
import type { IncidentFeature, Point } from "../types";

interface Props {
  points: Point[];
  incidents: IncidentFeature[];
  onPick: (p: Point | null) => void;
  fitKey: string;
}

const INCIDENT_STYLE = {
  EFFIS: { color: "#dc2626", fillColor: "#dc2626" }, // red: EFFIS burnt-area perimeters
  "Copernicus EMS": { color: "#7c3aed", fillColor: "#7c3aed" }, // violet: EMS mapping areas
} as const;

/** Outline + translucent fill for each incident, with a popup describing it. */
function IncidentsOverlay({ incidents }: Pick<Props, "incidents">) {
  const map = useMap();
  useEffect(() => {
    const layer = L.geoJSON(incidents as unknown as Parameters<typeof L.geoJSON>[0], {
      style: (f) => {
        const src = (f?.properties?.source ?? "EFFIS") as keyof typeof INCIDENT_STYLE;
        return { ...INCIDENT_STYLE[src], weight: 1.5, fillOpacity: 0.15 };
      },
      onEachFeature: (f, l) => {
        const p = f.properties;
        const box = document.createElement("div");
        const title = document.createElement("strong");
        title.textContent = p.name;
        const meta = document.createElement("div");
        meta.textContent = [
          p.source,
          p.country,
          p.start_date + (p.end_date !== p.start_date ? ` to ${p.end_date}` : ""),
          p.area_ha !== null ? `${Math.round(p.area_ha).toLocaleString()} ha` : null,
          `${p.detections.toLocaleString()} NASA detections`,
        ].filter(Boolean).join(" · ");
        box.append(title, meta);
        l.bindPopup(box);
      },
    });
    layer.addTo(map);
    return () => {
      layer.remove();
    };
  }, [map, incidents]);
  return null;
}

function PointsOverlay({ points, onPick }: Pick<Props, "points" | "onPick">) {
  const map = useMap();
  useEffect(() => {
    const layer = new CanvasPointLayer(points, onPick);
    layer.addTo(map);
    return () => {
      layer.remove();
    };
  }, [map, points, onPick]);
  return null;
}

/** Zoom to the data whenever the loaded dataset changes. */
function FitToPoints({ points, fitKey }: Props) {
  const map = useMap();
  useEffect(() => {
    if (points.length === 0) return;
    let s = 90, n = -90, w = 180, e = -180;
    for (const p of points) {
      if (p.lat < s) s = p.lat;
      if (p.lat > n) n = p.lat;
      if (p.lon < w) w = p.lon;
      if (p.lon > e) e = p.lon;
    }
    map.fitBounds(L.latLngBounds([s, w], [n, e]), { padding: [24, 24], maxZoom: 9 });
    // Re-run only when the selected dataset changes, not on every filter toggle.
  }, [map, fitKey]);
  return null;
}

export function MapView(props: Props) {
  return (
    <MapContainer center={[0, 20]} zoom={3} preferCanvas className="map" worldCopyJump>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      <IncidentsOverlay incidents={props.incidents} />
      <PointsOverlay points={props.points} onPick={props.onPick} />
      <FitToPoints {...props} />
    </MapContainer>
  );
}
