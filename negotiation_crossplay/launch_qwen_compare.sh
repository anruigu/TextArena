#!/usr/bin/env bash
set -u
cd /workspace/allie/TextArena/negotiation_crossplay
PY=/workspace/allie/TextArena/.venv/bin/python
FRONT="anthropic/claude-sonnet-5,openai/gpt-5.5"

echo "=== qwen3.6-27b vs frontier (3p integrative, 24g) $(date) ==="
$PY run_crossplay.py --models "qwen/qwen3.6-27b,$FRONT" --players 3 --games 24 \
    --turn-multiple 6 --regime integrative --concurrency 10 --seed 500 \
    --out results/cmp_qwen36_vs_frontier

echo "=== qwen3.5-27b vs frontier (3p integrative, 24g, SAME seeds) $(date) ==="
$PY run_crossplay.py --models "qwen/qwen3.5-27b,$FRONT" --players 3 --games 24 \
    --turn-multiple 6 --regime integrative --concurrency 10 --seed 500 \
    --out results/cmp_qwen35_vs_frontier

echo "=== judging both $(date) ==="
$PY judge_xp.py --run-dir results/cmp_qwen36_vs_frontier --concurrency 10 >/dev/null 2>&1
$PY judge_xp.py --run-dir results/cmp_qwen35_vs_frontier --concurrency 10 >/dev/null 2>&1
echo "=== DONE $(date) ==="
