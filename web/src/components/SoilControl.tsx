import { useEffect, useState } from "react";
import { loadSoilClasses, type SoilClasses } from "../data";

export interface SoilState {
  on: boolean;
  /** Index into the soil classes, or null for all. */
  cls: number | null;
}

const LABELS: Record<string, string> = {
  sand: "Sand, beaches, dunes", rock: "Bare rock", sparse: "Sparse vegetation", burnt: "Burnt areas",
  glacier: "Glaciers, snow", intertidal: "Intertidal flats", quarry: "Quarries, gravel pits",
  dump: "Dump sites", construction: "Construction sites",
};

/** Toggle and class selector of the bare-ground overlay (CORINE, embedded from context-data). */
export function SoilControl({ soil, onChange }: { soil: SoilState; onChange: (s: SoilState) => void }) {
  const [info, setInfo] = useState<SoilClasses | null>(null);
  useEffect(() => {
    loadSoilClasses().then(setInfo).catch(() => undefined);
  }, []);
  return (
    <fieldset>
      <legend>Soil</legend>
      <label className="check">
        <input type="checkbox" checked={soil.on} onChange={(e) => onChange({ ...soil, on: e.target.checked })} />
        Bare ground (CORINE)
      </label>
      {soil.on && info && (
        <>
          <select
            value={soil.cls ?? ""}
            onChange={(e) => onChange({ ...soil, cls: e.target.value === "" ? null : Number(e.target.value) })}
          >
            <option value="">All classes</option>
            {info.classes.map((c, i) => <option key={c} value={i}>{LABELS[c] ?? c}</option>)}
          </select>
          <div className="legend soil-legend">
            {info.classes.map((c, i) => (
              <span key={c}>
                <span className="dot" style={{ background: `rgb(${info.colors[i].join(",")})` }} /> {LABELS[c] ?? c}
              </span>
            ))}
          </div>
        </>
      )}
      <p className="muted small">Europe only (EEA39 countries). Share of bare ground in 1.6 km squares with at least 25% bare ground.</p>
    </fieldset>
  );
}
