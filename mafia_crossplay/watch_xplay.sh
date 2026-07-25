#!/usr/bin/env bash
# Regenerate the xplay AgentViz manifest as eval games finish.
# Rescans all result dirs every $INTERVAL seconds; the converter is idempotent,
# so newly written game_*.json files just get added to the manifest/sessions.
#
#   ./watch_xplay.sh            # poll every 20s
#   INTERVAL=10 ./watch_xplay.sh
set -euo pipefail
cd "$(dirname "$0")"

INPUTS="results_coup,results_nr,results_sg2p,results_sg6p,results"
OUT="/workspace/allie/agentviz/dist/sessions/xplay"
TITLE="TextArena frontier cross-play (live)"
INTERVAL="${INTERVAL:-20}"
# TextArena venv has the env deps + matplotlib needed by analyze_xplay.py.
PY="${PY:-/workspace/allie/TextArena/.venv/bin/python}"

echo "[watch_xplay] regenerating $OUT every ${INTERVAL}s (Ctrl+C to stop)"
last_sig=""
while true; do
  # signature = count + latest mtime across all game files, to skip no-op rebuilds
  sig=$(find $(echo "$INPUTS" | tr ',' ' ') -name 'game_*.json' -printf '%T@\n' 2>/dev/null \
        | sort -n | tail -1)$(find $(echo "$INPUTS" | tr ',' ' ') -name 'game_*.json' 2>/dev/null | wc -l)
  if [ "$sig" != "$last_sig" ]; then
    ok=1
    python3 to_agentviz_general.py --inputs "$INPUTS" --out "$OUT" --title "$TITLE" 2>&1 || ok=0
    # Quantitative reports (needs the TextArena venv for env scoring tables).
    "$PY" analyze_xplay.py --inputs "$INPUTS" --out . \
        > /tmp/xplay-analyze.log 2>&1 || ok=0
    if [ "$ok" = 1 ]; then
      last_sig="$sig"
      echo "[watch_xplay] $(date +%H:%M:%S) manifest + reports refreshed"
    else
      echo "[watch_xplay] $(date +%H:%M:%S) refresh had errors (see /tmp/xplay-analyze.log), will retry"
    fi
  fi
  sleep "$INTERVAL"
done
