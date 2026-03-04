#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$SCRIPT_DIR/codeBase:$SCRIPT_DIR/src:$SCRIPT_DIR:$PYTHONPATH"
else
  export PYTHONPATH="$SCRIPT_DIR/codeBase:$SCRIPT_DIR/src:$SCRIPT_DIR"
fi

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --discovery-only)
      ARGS+=("--skip-analysis" "--skip-stats")
      shift
      ;;
    --analysis-only)
      ARGS+=("--skip-discovery" "--skip-stats")
      shift
      ;;
    --stats-only)
      ARGS+=("--skip-discovery" "--skip-analysis")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

exec "$PYTHON_BIN" "$SCRIPT_DIR/codeBase/run.py" "${ARGS[@]}"
