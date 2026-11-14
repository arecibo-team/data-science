import { useEffect } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import { CanvasPointLayer, SENSOR_COLOR } from "./PointLayer";
import type { GridCell, VisibleSet } from "../model";
import type { IncidentFeature } from "../types";

interface Props {
  /** Individual detections (short ranges). Null when the grid is shown instead. */
  visible: VisibleSet | null;
  /** Grid cells of counts (long ranges). Null when individual detections are shown. */
  cells: GridCell[] | null;
  incidents: IncidentFeature[];
  onPick: (i: number | null) => void;
  fitKey: string;
}

const INCIDENT_STYLE = {
  "Copernicus EMS": { color: "#7c3aed", fillColor: "#7c3aed" }, // violet: EMS mapping areas
  "Sentinel-2": { color: "#dc2626", fillColor: "#dc2626" }, // red: Sentinel-2 burn scars (dNBR)
} as const;

const CONFIRMED_RING = "#dc2626";

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

/** Grid cells, sized by count. Red ring: at least one confirmed detection in the cell. */
function GridOverlay({ cells }: Pick<Props, "cells">) {
  const map = useMap();
  useEffect(() => {
    if (!cells) return;
    const group = L.layerGroup();
    for (const c of cells) {
      const modisDominant = c.modis >= c.viirs;
      const fill = modisDominant ? SENSOR_COLOR.MODIS : SENSOR_COLOR.VIIRS;
      L.circleMarker([c.lat, c.lon], {
        radius: Math.min(18, 3 + 2.5 * Math.log10(c.count + 1)),
        color: c.confirmed > 0 ? CONFIRMED_RING : fill,
        weight: c.confirmed > 0 ? 1.5 : 1,
        fillColor: fill,
        fillOpacity: 0.7,
      })
        .bindPopup(() => {
          const box = document.createElement("div");
          const lines = [
            `${c.count.toLocaleString()} detections`,
            `MODIS ${c.modis.toLocaleString()} · VIIRS ${c.viirs.toLocaleString()}`,
            `Confirmed ${c.confirmed.toLocaleString()}`,
            `${c.lat.toFixed(1)}, ${c.lon.toFixed(1)} (1° cell)`,
          ];
          for (const line of lines) {
            const row = document.createElement("div");
            row.textContent = line;
            box.append(row);
          }
          return box;
        })
        .addTo(group);
    }
    group.addTo(map);
    return () => {
      group.remove();
    };
  }, [map, cells]);
  return null;
}

/** Zoom to the data whenever the loaded dataset changes. */
function FitToData({ visible, cells, fitKey }: Pick<Props, "visible" | "cells" | "fitKey">) {
  const map = useMap();
  useEffect(() => {
    let s = 90, n = -90, w = 180, e = -180;
    const add = (lat: number, lon: number) => {
      if (lat < s) s = lat;
      if (lat > n) n = lat;
      if (lon < w) w = lon;
      if (lon > e) e = lon;
    };
    if (visible && visible.count > 0) {
      for (let i = 0; i < visible.count; i++) add(visible.lat[i], visible.lon[i]);
    } else if (cells && cells.length > 0) {
      for (const c of cells) add(c.lat, c.lon);
    } else {
      return;
    }
    map.fitBounds(L.latLngBounds([s, w], [n, e]), { padding: [24, 24], maxZoom: 9 });
    // Re-run only when the loaded dataset changes, not on every filter toggle.
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
      <PointsOverlay visible={props.visible} onPick={props.onPick} />
      <GridOverlay cells={props.cells} />
      <FitToData visible={props.visible} cells={props.cells} fitKey={props.fitKey} />
    </MapContainer>
  );
}
