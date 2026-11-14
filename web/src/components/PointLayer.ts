import L from "leaflet";
import type { Point } from "../types";

export const SENSOR_COLOR: Record<Point["sensor"], string> = {
  MODIS: "#f97316", // orange
  VIIRS: "#2563eb", // blue
};

const CONFIRMED_RING = "#dc2626"; // red ring: detection confirmed inside an EU incident

/** Marker radius in px, growing with fire radiative power (MW). */
export function radiusFor(frp: number | null): number {
  if (frp === null || Number.isNaN(frp)) return 2;
  return Math.min(7, 2 + Math.sqrt(Math.max(0, frp)) * 0.5);
}

/**
 * Draws many points on a single canvas over the map. Far faster than one Leaflet marker
 * per detection. The canvas is redrawn on move/zoom/resize; clicks pick the nearest point.
 */
export class CanvasPointLayer extends L.Layer {
  private canvas = document.createElement("canvas");
  private mapRef: L.Map | null = null;
  private pending = 0;

  constructor(
    private points: Point[],
    private onPick: (p: Point | null) => void,
  ) {
    super();
    this.canvas.style.position = "absolute";
    this.canvas.style.left = "0";
    this.canvas.style.top = "0";
    this.canvas.style.pointerEvents = "none";
    this.canvas.style.zIndex = "450";
  }

  onAdd(map: L.Map): this {
    this.mapRef = map;
    map.getContainer().appendChild(this.canvas);
    map.on("move zoom resize viewreset", this.schedule, this);
    map.on("click", this.handleClick, this);
    this.draw();
    return this;
  }

  onRemove(map: L.Map): this {
    map.off("move zoom resize viewreset", this.schedule, this);
    map.off("click", this.handleClick, this);
    this.canvas.remove();
    this.mapRef = null;
    if (this.pending) cancelAnimationFrame(this.pending);
    return this;
  }

  private schedule = () => {
    if (this.pending) return;
    this.pending = requestAnimationFrame(() => {
      this.pending = 0;
      this.draw();
    });
  };

  private draw() {
    const map = this.mapRef;
    if (!map) return;
    const size = map.getSize();
    const ratio = window.devicePixelRatio || 1;
    if (this.canvas.width !== size.x * ratio || this.canvas.height !== size.y * ratio) {
      this.canvas.width = size.x * ratio;
      this.canvas.height = size.y * ratio;
      this.canvas.style.width = `${size.x}px`;
      this.canvas.style.height = `${size.y}px`;
    }
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size.x, size.y);

    // Only project points inside the (padded) view.
    const b = map.getBounds().pad(0.1);
    const south = b.getSouth(), north = b.getNorth(), west = b.getWest(), east = b.getEast();
    ctx.globalAlpha = 0.75;
    for (const p of this.points) {
      if (p.lat < south || p.lat > north || p.lon < west || p.lon > east) continue;
      const pt = map.latLngToContainerPoint([p.lat, p.lon]);
      ctx.fillStyle = SENSOR_COLOR[p.sensor];
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radiusFor(p.frp), 0, Math.PI * 2);
      ctx.fill();
      if (p.incident) {
        // COMMON: confirmed inside a European incident.
        ctx.strokeStyle = CONFIRMED_RING;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }

  private handleClick = (e: L.LeafletMouseEvent) => {
    const map = this.mapRef;
    if (!map) return;
    const click = e.containerPoint;
    let best: Point | null = null;
    let bestD2 = 8 * 8; // pick within 8 px
    for (const p of this.points) {
      const pt = map.latLngToContainerPoint([p.lat, p.lon]);
      const dx = pt.x - click.x, dy = pt.y - click.y;
      const d2 = dx * dx + dy * dy;
      if (d2 <= bestD2) {
        bestD2 = d2;
        best = p;
      }
    }
    this.onPick(best);
  };
}
