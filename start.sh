#!/usr/bin/env bash
# Install dependencies, build the web app if needed, and start the server.
#   ./start.sh [--port 8000] [--host 127.0.0.1]
# Environment such as PIPELINE_SKIP=fetch,ems,sentinel2 is passed through.
set -euo pipefail
cd "$(dirname "$0")"

uv sync --quiet

if [ ! -d web/node_modules ]; then
  (cd web && bun install)
fi

# Rebuild when the built app is missing or any source file is newer than it.
if [ ! -f web/dist/index.html ] || [ -n "$(find web/src web/index.html web/package.json web/vite.config.ts -newer web/dist/index.html -print -quit)" ]; then
  (cd web && bun run build)
fi

exec uv run python -m server "$@"
