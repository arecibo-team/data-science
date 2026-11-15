import { useEffect, useState } from "react";
import { loadRuns, type EventReplay } from "../data";

const pct = (a: number, b: number) => (b ? `${Math.round((a / b) * 100)}%` : "n/a");

function addDays(d: string, n: number): string {
  const t = new Date(`${d}T00:00:00Z`);
  t.setUTCDate(t.getUTCDate() + n);
  return t.toISOString().slice(0, 10);
}

/** Bars: cells that really burned in the next N days. Line: the model's expected number (sum of probabilities). */
function Chart({ r }: { r: EventReplay }) {
  const W = 270, H = 110, ml = 22, mb = 16, mt = 6;
  const days = r.days;
  if (days.length < 2) return null;
  const max = Math.max(1, ...days.map((d) => Math.max(d.burning, d.risk)));
  const bw = (W - ml) / days.length;
  const x = (i: number) => ml + i * bw;
  const y = (v: number) => mt + (1 - v / max) * (H - mt - mb);
  const spike = r.event.spike;
  const line = days.map((d, i) => `${(x(i) + bw / 2).toFixed(1)},${y(d.risk).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="loss" role="img" aria-label="Cells that burned versus the model's expected number, per day of the held-out window">
      {days.map((d, i) => d.date >= spike[0] && d.date <= spike[1] && (
        <rect key={`s${d.date}`} x={x(i)} y={mt} width={bw} height={H - mt - mb} fill="var(--error)" opacity="0.1" />
      ))}
      {days.map((d, i) => (
        <rect key={d.date} x={x(i) + 1} y={y(d.burning)} width={Math.max(bw - 2, 1)} height={H - mb - y(d.burning)} fill="var(--muted)" opacity="0.55" />
      ))}
      <polyline points={line} fill="none" stroke="var(--modis)" strokeWidth="2" />
      <text x={2} y={mt + 8} fontSize="9" fill="var(--muted)">{Math.round(max)}</text>
      <text x={ml} y={H - 3} fontSize="9" fill="var(--muted)">{days[0].date.slice(5)}</text>
      <text x={W - 2} y={H - 3} fontSize="9" textAnchor="end" fill="var(--muted)">{days[days.length - 1].date.slice(5)}</text>
    </svg>
  );
}

interface Props {
  /** Bumped when a training run finishes, so the card reloads. */
  version: number;
  onJump: (date: string) => void;
}

/** Did the active model see the held-out event coming? Numbers come from the replay stored with the run. */
export function EventCheck({ version, onJump }: Props) {
  const [replay, setReplay] = useState<EventReplay | null | undefined>(undefined);
  const [runId, setRunId] = useState<number | null>(null);
  useEffect(() => {
    loadRuns()
      .then((r) => {
        const active = r.runs.find((x) => x.id === r.active);
        setRunId(active?.id ?? null);
        setReplay(active ? (active.metrics?.benchmark ?? null) : undefined);
      })
      .catch(() => setReplay(undefined));
  }, [version]);

  if (replay === undefined) return null; // no completed run yet
  if (replay === null) {
    return (
      <fieldset>
        <legend>Would it have seen the big fire?</legend>
        <p className="muted small">Run #{runId} was trained without a held-out event. Train a new run to get this check.</p>
      </fieldset>
    );
  }

  const { event, horizon } = replay;
  const dayBefore = addDays(event.spike[0], -1);
  const eve = replay.days.find((d) => d.date === dayBefore);
  const caught = eve ? eve.real : 0;
  const burned = eve ? eve.burning : 0;
  const share = burned ? caught / burned : 0;
  const verdict = share >= 0.7 ? "Yes" : share >= 0.4 ? "Partly" : "No";
  const color = share >= 0.7 ? "#16a34a" : share >= 0.4 ? "#d97706" : "#dc2626";
  const newG = replay.lead.new, hotG = replay.lead.burning;
  const k1 = replay.lead_days.includes(1) ? "1" : String(replay.lead_days[0]);
  const kMax = String(Math.max(...replay.lead_days));
  const sc = replay.spike_cells;

  return (
    <fieldset>
      <legend>Would it have seen the big fire?</legend>
      <p className="small">
        <strong>{event.name}</strong>: {event.first} to {event.last} was removed from training for all of Europe.
      </p>
      <p>
        <strong style={{ color, fontSize: 18 }}>{verdict}</strong>{" "}
        <span className="small">
          the day before the surge ({dayBefore}) it flagged {caught} of the {burned} cells that burned in the next {horizon} days ({pct(caught, burned)}).
        </span>
      </p>
      <Chart r={replay} />
      <p className="muted small">Grey bars: cells that really burned in the next {horizon} days. Orange line: the model's expected number of them. Red band: the surge ({event.spike[0]} to {event.spike[1]}).</p>

      <table className="small lead">
        <thead><tr><th>Surge cells flagged…</th>{replay.lead_days.map((k) => <th key={k}>{k}d before</th>)}</tr></thead>
        <tbody>
          <tr><td>New ({newG.cells}), quiet before</td>{replay.lead_days.map((k) => <td key={k}>{newG.flagged[String(k)]}/{newG.cells}</td>)}</tr>
          <tr><td>Already burning ({hotG.cells})</td>{replay.lead_days.map((k) => <td key={k}>{hotG.flagged[String(k)]}/{hotG.cells}</td>)}</tr>
        </tbody>
      </table>
      <p className="muted small">
        Newly ignited cells flagged {newG.flagged[k1]} of {newG.cells} one day ahead and {newG.flagged[kMax]} of {newG.cells} {kMax} days ahead.
        Only {sc.scoreable} of the {sc.in_region} cells that burned in the region could be scored at all, and {sc.active} had recent fire history.
      </p>
      <p className="muted small">
        Whole window, all of Europe: PR-AUC {replay.europe.pr_auc?.toFixed(2) ?? "n/a"} against {replay.europe.baseline_pr_auc?.toFixed(2) ?? "n/a"} for
        "a fire burned here last week"; recall {(replay.europe.recall * 100).toFixed(0)}%, precision {(replay.europe.precision * 100).toFixed(0)}%.
      </p>
      <div className="presets">
        <button type="button" onClick={() => onJump(dayBefore)}>Show {dayBefore} on the map</button>
      </div>
    </fieldset>
  );
}
