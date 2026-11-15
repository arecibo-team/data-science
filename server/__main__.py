"""Command line.

    python -m server                      serve the API and the built web app (runs the pipeline on start)
    python -m server status               show which pipeline steps ran
    python -m server run [step ...]       run pending steps now, without starting the server
    python -m server reset STEP [--only]  forget a step and everything after it, so the next run redoes it
"""
from __future__ import annotations

import argparse
import logging


def _print_status() -> None:
    from . import db, pipeline
    db.init_db()
    for row in pipeline.status():
        last = row["last"]
        when = f"{last['finished_at'] or last['started_at']}  {last['row_count'] if last['row_count'] is not None else '-':>9}" if last else "never run"
        note = f"  ({row['reason']})" if row["reason"] else (f"  [{last['message']}]" if last and last["status"] == "failed" else "")
        print(f"{row['step']:<10} {row['state']:<11} {when}{note}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m server", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="start the server (default)")
    sub.add_parser("status", help="show the step ledger")
    run = sub.add_parser("run", help="run pending steps")
    run.add_argument("steps", nargs="*")
    reset = sub.add_parser("reset", help="delete a step's ledger rows so it runs again")
    reset.add_argument("step")
    reset.add_argument("--only", action="store_true", help="do not also reset the steps after it")
    for sp in (serve, p):
        sp.add_argument("--host", default="127.0.0.1")
        sp.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.cmd == "status":
        _print_status()
    elif args.cmd == "run":
        from . import db, pipeline
        db.init_db()
        print(pipeline.run_pending(args.steps or None))
    elif args.cmd == "reset":
        from . import db, pipeline
        db.init_db()
        print("reset:", ", ".join(pipeline.reset_step(args.step, cascade=not args.only)))
    else:
        import uvicorn
        uvicorn.run("server.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
