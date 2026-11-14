import { useEffect } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import { CanvasPointLayer } from "./PointLayer";
import type { Point } from "../types";

interface Props {
  points: Point[];
  onPick: (p: Point | null) => void;
  fitKey: string;
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
      <PointsOverlay points={props.points} onPick={props.onPick} />
      <FitToPoints {...props} />
    </MapContainer>
  );
}
