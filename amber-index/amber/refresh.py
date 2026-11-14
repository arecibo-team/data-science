from __future__ import annotations

import argparse
import subprocess
import sys

from . import ROOT
from .pipeline import load_sources, rescore


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m amber.refresh",
                                 description="Bring the Amber Index up to date: new data, new years, new events.")
    ap.add_argument("--offline", action="store_true", help="use cached downloads only")
    ap.add_argument("--stations", action="store_true", help="also re-count OpenStreetMap fire stations (slow)")
    ap.add_argument("--retrain", action="store_true", help="re-run the notebook: refit scaling, model, weights and archetypes")
    a = ap.parse_args(argv)
    if a.retrain:
        if not a.offline:
            load_sources(refresh=True, stations_refresh=a.stations)
        nb = ROOT / "notebooks" / "amber_index.ipynb"
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(nb),
                        "--output", nb.name], check=True, cwd=ROOT)
    else:
        rescore(refresh=not a.offline, stations_refresh=a.stations)


if __name__ == "__main__":
    main()
