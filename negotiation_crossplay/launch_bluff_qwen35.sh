#!/usr/bin/env bash
# Bluff-prompt experiment for qwen/qwen3.5-9b on the TextArena Negotiation
# value-probe crossplay. 3 arms x {2p,3p}, SAME seeds + SAME opponent pool across
# arms, so any gain/leakage delta is attributable to the bluff prompting only.
#
#   off            -> native SYSTEM (baseline)
#   system         -> BLUFF_SYSTEM only
#   system_fewshot -> BLUFF_SYSTEM + few-shot bluff exemplars
#
# qwen3.5-9b is pinned into seat 0 every game (--pin-bluff) so we get a data point
# on it in each game. Opponents keep the native prompt.
set -u
cd /workspace/allie/TextArena/negotiation_crossplay
PY=/workspace/allie/TextArena/.venv/bin/python

MODELS="qwen/qwen3.5-9b,openai/gpt-5.5,anthropic/claude-sonnet-5,google/gemma-4-31b-it"
BLUFF=qwen/qwen3.5-9b
GAMES=${GAMES:-12}
SEED=${SEED:-700}
TM=${TM:-6}
CONC=${CONC:-6}

run_arm () {
  local mode="$1" tag="$2" players="$3"
  echo "=== arm=$mode ${players}p games=$GAMES seed=$SEED $(date) ==="
  $PY run_value_probe_ta.py --models "$MODELS" --players "$players" --games "$GAMES" \
      --turn-multiple $TM --regime integrative --concurrency $CONC --seed $SEED \
      --bluff-model "$BLUFF" --bluff-mode "$mode" --pin-bluff \
      --out "results/bluff_${tag}_${players}p"
}

for P in 2 3; do
  run_arm off            off     $P
  run_arm system         system  $P
  run_arm system_fewshot fewshot $P
done

echo "=== compare (gain vs leakage vs lie_rate) $(date) ==="
$PY bluff_compare.py --model "$BLUFF"
echo "=== DONE $(date) ==="
