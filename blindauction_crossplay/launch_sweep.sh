#!/bin/bash
# Blind auction cross-play: 7 frontier models, sweep conversation_rounds {1,3,6}.
# Same base seed across configs -> paired item/value draws per game index.
cd "$(dirname "$0")/.."
MODELS="openai/gpt-5.6-sol-pro,anthropic/claude-opus-4.8,deepseek/deepseek-v4-pro,meta-llama/llama-4-maverick,qwen/qwen3.7-max,google/gemini-3.6-flash,moonshotai/kimi-k3"
for R in 1 3 6; do
  echo "=== conversation_rounds=$R ==="
  .venv/bin/python blindauction_crossplay/run_blindauction.py \
    --models "$MODELS" \
    --games 14 --conversation-rounds "$R" --concurrency 4 --seed 0 \
    --env-file /workspace/allie/.env \
    --out "blindauction_crossplay/results/ba_7p_r$R"
done
echo "=== sweep complete ==="
