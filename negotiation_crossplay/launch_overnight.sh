#!/usr/bin/env bash
set -u
cd /workspace/allie/TextArena/negotiation_crossplay
PY=/workspace/allie/TextArena/.venv/bin/python
M5="qwen/qwen3.6-27b,qwen/qwen3.5-27b,google/gemma-4-31b-it,openai/gpt-5.5,anthropic/claude-sonnet-5"

echo "=== BATCH A: 3-player integrative (40 games) $(date) ==="
$PY run_crossplay.py --models "$M5" --players 3 --games 40 --turn-multiple 6 \
    --regime integrative --concurrency 10 --seed 100 --out results/xp_3p_integrative

echo "=== BATCH B: 4-player integrative (40 games) $(date) ==="
$PY run_crossplay.py --models "$M5" --players 4 --games 40 --turn-multiple 6 \
    --regime integrative --concurrency 10 --seed 200 --out results/xp_4p_integrative

echo "=== BATCH C: 3-player STOCK regime contrast (24 games) $(date) ==="
$PY run_crossplay.py --models "$M5" --players 3 --games 24 --turn-multiple 6 \
    --regime stock --concurrency 10 --seed 300 --out results/xp_3p_stock

echo "=== ALL DONE $(date) ==="
