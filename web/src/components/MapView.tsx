import { useEffect, type ReactNode } from "react";
import { MapContainer, TileLayer, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import { CanvasPointLayer } from "./PointLayer";
import type { VisibleSet } from "../model";
import type { Bounds, IncidentFeature } from "../types";

interface Props {
  /** Individual detections (short ranges). Null when the grid is shown instead. */
  visible: VisibleSet | null;
  /** Heat tile URL template when the view is too dense for points; null otherwise. */
  tileUrl: string | null;
  incidents: IncidentFeature[];
  /** Bare-ground overlay URL template (server/soiltiles.py), or null/undefined when off. */
  soilUrl?: string | null;
  /** Grey, faded base map so overlays stand out (the forecast view). */
  muted?: boolean;
  /** Extra layers drawn inside the map, e.g. the forecast cells. */
  children?: ReactNode;
  onPick: (i: number | null) => void;
  /** Called with the visible bounds and zoom level on load and when the view settles. */
  onView: (bounds: Bounds, zoom: number) => void;
}

const INCIDENT_STYLE = {
  "Copernicus EMS": { color: "#7c3aed", fillColor: "#7c3aed" }, // violet: EMS mapping areas
  "Sentinel-2": { color: "#dc2626", fillColor: "#dc2626" }, // red: Sentinel-2 burn scars (dNBR)
} as const;

/** Outline + translucent fill for each incident, with a popup describing it. */
function IncidentsOverlay({ incidents }: Pick<Props, "incidents">) {
  const map = useMap();
  useEffect(() => {
    const layer = L.geoJSON(incidents as unknown as Parameters<typeof L.geoJSON>[0], {
      style: (f) => {
        const src = (f?.properties?.source ?? "Copernicus EMS") as keyof typeof INCIDENT_STYLE;
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

function PointsOverlay({ visible, onPick }: Pick<Props, "visible" | "onPick">) {
  const map = useMap();
  useEffect(() => {
    if (!visible) return;
    const layer = new CanvasPointLayer(visible, onPick);
    layer.addTo(map);
    return () => {
      layer.remove();
    };
  }, [map, visible, onPick]);
  return null;
}

/** Server-rendered heat tiles: the overview when a view holds too many detections to draw one by one. */
function HeatTiles({ url }: { url: string | null }) {
  if (!url) return null;
  return <TileLayer url={url} zIndex={420} maxZoom={18} tileSize={256} opacity={0.95} />;
}

/** Bare-ground tiles under the heat tiles. */
function SoilTiles({ url }: { url: string | null | undefined }) {
  if (!url) return null;
  return <TileLayer url={url} zIndex={410} maxZoom={18} tileSize={256} opacity={0.9} />;
}

/** Report the visible bounds and zoom on load and after every pan or zoom. */
function ViewReporter({ onView }: Pick<Props, "onView">) {
  const map = useMapEvents({
    moveend: () => report(),
  });
  const report = () => {
    const b = map.getBounds();
    onView({ west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() }, map.getZoom());
  };
  useEffect(report, [map]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

export function MapView(props: Props) {
  return (
    <MapContainer center={[46, 12]} zoom={5} preferCanvas className="map" worldCopyJump>
      <TileLayer
        className={props.muted ? "base-muted" : undefined}
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      <SoilTiles url={props.soilUrl} />
      <IncidentsOverlay incidents={props.incidents} />
      <PointsOverlay visible={props.visible} onPick={props.onPick} />
      <HeatTiles url={props.tileUrl} />
      <ViewReporter onView={props.onView} />
      {props.children}
    </MapContainer>
  );
}
