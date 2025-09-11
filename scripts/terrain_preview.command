#!/bin/bash
set -uo pipefail

# Echo commands for debugging when run via Finder
exec 1> >(sed 's/^/[preview.command] /') 2>&1
set -x

# Resolve project root relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Ensure Homebrew paths are available when launched from Finder
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Choose Python interpreter
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  if command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  elif [ -x "/opt/homebrew/bin/python3" ]; then
    PY="/opt/homebrew/bin/python3"
  elif [ -x "/usr/local/bin/python3" ]; then
    PY="/usr/local/bin/python3"
  else
    echo "Python 3 not found. Install Python 3 and retry."
    echo "Tip: brew install python@3.11"
    read -r _
    exit 1
  fi
fi

# Run preview
set +e
"$PY" scripts/terrain_preview.py "$@"
status=$?
set -e

# Open the generated image if it exists
if [ -f "assets/terrain_preview.png" ]; then
  open "assets/terrain_preview.png" || true
fi

# Report status and wait for user input when launched from Finder
if [ $status -ne 0 ]; then
  echo "Preview failed with exit code $status"
fi
set +x
echo "Done. Close this window or press Enter to exit."
read -r _