import { useCallback, useEffect, useRef, useState } from "react";
import { loadRuns, modelStreamUrl, resumeTraining, startTraining, stopTraining, type ModelRun, type Runs, type TrainEvent } from "../data";

interface Props {
  horizon: number;
  /** Called when the model used for predictions may have changed (a run finished). */
  onModelChanged: () => void;
}

interface Point {
  round: number;
  train: number | null;
  val: number | null;
}

/** Train / Stop / Resume, a live loss chart fed by server-sent events, and the list of runs. */
export function TrainPanel({ horizon, onModelChanged }: Props) {
  const [runs, setRuns] = useState<Runs | null>(null);
  const [rounds, setRounds] = useState(300);
  const [history, setHistory] = useState<Point[]>([]);
  const [total, setTotal] = useState(0);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const es = useRef<EventSource | null>(null);

  const refresh = useCallback(() => loadRuns().then(setRuns).catch((e: Error) => setError(e.message)), []);

  const connect = useCallback(() => {
    es.current?.close();
    const source = new EventSource(modelStreamUrl());
    es.current = source;
    source.onmessage = (m) => {
      const ev = JSON.parse(m.data) as TrainEvent & { replay?: boolean };
      if (ev.type === "start") {
        setTotal(ev.total);
        if (!ev.replay && !ev.resumed) setHistory([]);
        if (ev.replay) setHistory([]);
        setNote(null);
        refresh();
      } else if (ev.type === "progress") {
        setHistory((h) => [...h.filter((p) => p.round < ev.round), { round: ev.round, train: ev.train_loss, val: ev.val_loss }]);
        setTotal(ev.total);
      } else if (ev.type === "checkpoint") {
        refresh();
      } else {
        setNote(ev.type === "done" ? "Training finished." : ev.type === "stopped" ? "Stopped. A checkpoint is saved; Resume continues from it." : `Failed: ${ev.message}`);
        source.close();
        refresh();
        if (ev.type === "done") onModelChanged();
      }
    };
    source.onerror = () => {
      source.close();
      refresh();
    };
  }, [refresh, onModelChanged]);

  useEffect(() => {
    loadRuns()
      .then((r) => {
        setRuns(r);
        if (r.running) connect();
        else if (r.runs.length) {
          // Show the loss curve of the newest run.
          fetch(`api/model/runs/${r.runs[0].id}`).then((x) => x.json()).then((run) => {
            setHistory((run.loss_history as Point[]) ?? []);
            setTotal(run.rounds_total);
          }).catch(() => undefined);
        }
      })
      .catch((e: Error) => setError(e.message));
    return () => es.current?.close();
  }, [connect]);

  const act = (fn: () => Promise<void>) => {
    setError(null);
    fn().then(() => {
      connect();
      refresh();
    }).catch((e: Error) => setError(e.message));
  };

  const running = runs?.running ?? false;
  const resumable = runs?.runs.find((r) => ["stopped", "interrupted", "failed"].includes(r.status) && (r.last_checkpoint_round ?? 0) > 0 && r.id === runs.runs[0].id);
  const last = history[history.length - 1];

  return (
    <fieldset className="train">
      <legend>Model training</legend>
      <label>
        Boosting rounds
        <input type="number" min={1} max={5000} value={rounds} disabled={running} onChange={(e) => setRounds(Math.max(1, Number(e.target.value) || 1))} />
      </label>
      <div className="presets">
        <button type="button" disabled={running} onClick={() => act(() => startTraining(rounds, horizon))}>Train</button>
        <button type="button" disabled={!running} onClick={() => stopTraining().catch((e: Error) => setError(e.message))}>Stop</button>
        {resumable && <button type="button" disabled={running} onClick={() => act(() => resumeTraining(resumable.id))}>Resume #{resumable.id}</button>}
      </div>
      {(running || history.length > 0) && (
        <>
          <progress max={total || 1} value={last?.round ?? 0} />
          <p className="muted small">
            Round {last?.round ?? 0} of {total}
            {last?.train != null ? ` · train loss ${last.train.toFixed(4)}` : ""}
            {last?.val != null ? ` · validation loss ${last.val.toFixed(4)}` : ""}
          </p>
          <LossChart points={history} />
        </>
      )}
      {note && <p className="small">{note}</p>}
      {error && <p className="error small">{error}</p>}
      <RunList runs={runs} />
    </fieldset>
  );
}

function LossChart({ points }: { points: Point[] }) {
  if (points.length < 2) return null;
  const W = 260, H = 90, m = 4;
  const vals = points.flatMap((p) => [p.train, p.val]).filter((v): v is number => v != null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x0 = points[0].round, x1 = points[points.length - 1].round;
  const sx = (r: number) => m + ((r - x0) / Math.max(x1 - x0, 1)) * (W - 2 * m);
  const sy = (v: number) => H - m - ((v - lo) / Math.max(hi - lo, 1e-9)) * (H - 2 * m);
  const line = (key: "train" | "val") =>
    points.filter((p) => p[key] != null).map((p) => `${sx(p.round).toFixed(1)},${sy(p[key] as number).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="loss" role="img" aria-label="Training and validation loss per round">
      <polyline points={line("train")} fill="none" stroke="var(--viirs)" strokeWidth="1.5" />
      <polyline points={line("val")} fill="none" stroke="var(--modis)" strokeWidth="1.5" />
      <text x={m} y={10} fontSize="9" fill="var(--muted)">{hi.toFixed(3)}</text>
      <text x={m} y={H - 6} fontSize="9" fill="var(--muted)">{lo.toFixed(3)}</text>
      <text x={W - 4} y={10} fontSize="9" textAnchor="end" fill="var(--viirs)">train</text>
      <text x={W - 4} y={21} fontSize="9" textAnchor="end" fill="var(--modis)">validation</text>
    </svg>
  );
}

function RunList({ runs }: { runs: Runs | null }) {
  if (!runs?.runs.length) return <p className="muted small">No runs yet. Train a model to see forecasts.</p>;
  return (
    <ul className="runs small">
      {runs.runs.slice(0, 6).map((r: ModelRun) => (
        <li key={r.id} className={r.id === runs.active ? "active-run" : ""}>
          <strong>#{r.id}</strong> {r.status} · {r.rounds_done}/{r.rounds_total} rounds
          {r.id === runs.active ? " · in use" : ""}
          {r.last_checkpoint_round ? ` · checkpoint at ${r.last_checkpoint_round}` : ""}
          {r.metrics?.pr_auc != null && (
            <div className="muted">
              PR-AUC {r.metrics.pr_auc.toFixed(3)} (base rate {(r.metrics.base_rate ?? 0).toFixed(4)}) · precision {(r.metrics.precision ?? 0).toFixed(3)} · recall {(r.metrics.recall ?? 0).toFixed(3)} on days after {r.metrics.train_end}
            </div>
          )}
          {r.message && <div className="muted">{r.message}</div>}
        </li>
      ))}
    </ul>
  );
}
