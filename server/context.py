"""Embed context-data (CORINE bare ground) into the database, so the DB is self-contained."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import db

CONTEXT_DIR = Path(__file__).resolve().parent.parent / "context-data"
FILES = ("bareground_grid.json", "bareground_squares.json")
SOIL_CLASSES = ["sand", "rock", "sparse", "burnt", "glacier", "intertidal", "quarry", "dump", "construction"]
RES = 0.25


def digest() -> str | None:
    """Content hash of the context files, or None when they are missing."""
    if not all((CONTEXT_DIR / f).is_file() for f in FILES):
        return None
    h = hashlib.sha256()
    for f in FILES:
        h.update((CONTEXT_DIR / f).read_bytes())
    return h.hexdigest()[:16]


def cell_index(lat: float, lon: float) -> tuple[int, int]:
    return int((lat + 90) // RES), int((lon + 180) // RES)


def load(directory: Path | None = None) -> int:
    """Replace soil_cells and soil_squares from the JSON files. Returns the number of rows written."""
    d = directory or CONTEXT_DIR
    grid = json.loads((d / "bareground_grid.json").read_text())
    sq = json.loads((d / "bareground_squares.json").read_text())
    cells = []
    for row in grid["cells"]:
        r4, c4 = cell_index(row[0], row[1])
        cells.append((r4, c4, row[0], row[1], *row[2:11]))
    squares = [tuple(r) for r in sq["squares"]]
    with db.session() as con:
        con.execute("DELETE FROM soil_cells")
        con.executemany("INSERT OR REPLACE INTO soil_cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", cells)
        con.execute("DELETE FROM soil_squares")
        con.executemany("INSERT INTO soil_squares VALUES (?,?,?,?,?)", squares)
    return len(cells) + len(squares)
