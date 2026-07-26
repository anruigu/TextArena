# RLFR for Leaky Lies — Phase 2 (self-play GRPO + arms race)

Phase 0 (the detector) and Phase 1 (the cheap BoN validation) are built, run, and reported below.
Phase 2 wires the same dense feature into online self-play GRPO with an outer co-adaptation loop.
The training + snapshot-serving steps need the fleet cluster (8×H200); everything else here is
runnable locally and has been smoke-tested.

## What's built

| Component | File | Status |
|---|---|---|
| Reader spots (public-channel, hole stripped) | `build_reader_spots.py` | ✅ run — 1751 spots |
| Leak-reader probe (the detector) | `train_reader_probe.py` | ✅ run — **AUROC 0.771** |
| Dense reward + `LeakReader` | `leaky_reward.py` | ✅ smoke |
| Best-of-N frontier | `bon_leaky.py` | ✅ run — see frontier |
| Reader HTTP service (load-once, hot-reload on refit) | `reader_service.py` | ✅ integration-tested |
| skyrl-gym env `leaky_poker` (BaseTextEnv) | `skyrl-neg-wt/.../envs/leaky_poker/env.py` | ✅ registered + smoke |
| Dataset prep (match seeds → parquet) | `.../leaky_poker/prepare_dataset.py` | ✅ |
| Outer arms-race loop | `armsrace_loop.py` | ✅ dry-run; GRPO step shells to fleet |

## Phase 0 result — the detector

The frozen Qwen3-8B reader, seeing ONLY the opponent's public channel (betting history + board +
pot, **hole cards stripped**), recovers the actor's hidden hand-strength bucket at **AUROC 0.771**
(best layer 22, n=1003 binary spots; Brier 0.244, ECE 0.214). Because the hole cards are hidden,
this is genuine type-**leakage** through betting behavior — not the visible-card re-decoding
`activation_probe.py` measured. This probe defines leakage everywhere downstream and is refit each
arms-race round.

## Phase 1 result — stealth pays (the premise)

`bon_leaky.py` drives a hero through the real PokerEnv vs a fixed opponent that exploits leakage
(reads the hero's public channel with the same probe; folds to bets it reads as strong, calls —
catches — bets it reads as weak). The hero's *selection* gain is leakage-BLIND; the only channel
through which readability enters selection is the dense leakage feature × λ. Sweeping λ:

| λ | bluffs | mean leakage | bluff success (fold-rate) | hero net chips |
|---:|---:|---:|---:|---:|
| 0   | 115 | 0.399 | 0.609 | **−97** |
| 25  | 115 | 0.284 | 0.765 | +86 |
| 50  | 89  | 0.208 | 0.843 | +112 |
| 100 | 72  | 0.140 | 0.903 | +124 |
| 200 | 60  | 0.067 | 0.983 | +126 |
| 400 | 38  | 0.016 | 1.000 | +156 |

At λ=0 the hero bluffs blind to readability, the exploiter catches 39% of the bluffs, and the hero
**loses money** (−97 net). As λ rises the dense feature steers it to **less-readable bluffs**
(leakage 0.40→0.016), the exploiter **folds more** (0.61→1.0), and realized chips flip
**negative→+156** while the **lie-rate contracts** (115→38 bluffs — the guardrail region). Lower-leakage
bluffs win more against an exploiting opponent: the premise holds, cheaply, before any RL.
(`bon_leaky_frontier.png`.)

## Phase 2 — how to run the full loop

**1. Start the reader service** (one per node; all rollout workers query it):
```bash
cd /workspace/allie/TextArena/deception_poc
CUDA_VISIBLE_DEVICES=0 /workspace/allie/performative/.venv/bin/python reader_service.py \
    --probe probes/probe_leakreader_qwen3_8b.npz --port 8137
export READER_BASE_URL=http://127.0.0.1:8137
```

**2. Prepare the dataset** (one row = one match seed; the env deals at init):
```bash
python skyrl-neg-wt/skyrl-gym/skyrl_gym/envs/leaky_poker/prepare_dataset.py \
    --output_dir ~/data/fleet/leaky_poker --n_train 4096 --n_val 256 --num_rounds 4
```

**3. GRPO train** on the fleet stack (mirrors the negotiation run; `env_class=leaky_poker`). Key
`env_config` knobs the env reads (see `env.py::__init__`):
- `reward_mode`: `dense` (the leaky feature — headline) or `sparse` (realized chips — baseline arm)
- `leak_lambda`, `reward_scale`, `invalid_penalty`, `hold_lie_rate` (+ `lie_rate_lo/hi`)
- `opponent_mode=llm`, `opponent_base_url` = the frozen **policy snapshot** served on vLLM (rotated
  each round), `reader_mode=endpoint`, `reader_base_url=$READER_BASE_URL`
- `transcript_dir` = where `env.close()` dumps per-episode bluff public-channels (for the refit)

Launch via `scripts/fleet-launch.sh` with a task YAML pointing `env_class` at `leaky_poker` and the
`generator.env_config` block at the knobs above (Qwen3.5-9B, FSDP2, TP=2 — same as `fleet-9b-run.sh`).

**4. Outer arms-race loop** (alternate train ↔ refit-reader ↔ rotate-opponent):
```bash
python armsrace_loop.py --rounds 4 --reader-base-url $READER_BASE_URL \
    --train-cmd "bash scripts/fleet-launch.sh ..."     # or --skip-train to refit/measure only
```
Each round refits the probe on the round's stealthier bluffs and reports **re-detection AUROC** —
if a freshly refit detector stays high, the stealth was probe-specific and is re-exposed; if it
falls, the concealment generalizes. That is the paper's "student evades an adapting monitor"
question, operationalized. A no-cluster dry run of the refit+measure machinery:
```bash
python armsrace_loop.py --dry-run
```

## Deliverables / eval (per the plan)

- Reader AUROC + calibration — **done** (`reader_probe_qwen3_8b.png`).
- BoN gain-vs-leakage frontier + win-rate over λ — **done** (`bon_leaky_frontier.png`).
- Arms-race curves (per-round policy leakage, re-detection AUROC, realized win-rate) — the loop and
  metrics are wired (`get_metrics()` logs `mean_bluff_leakage`, `bluff_fold_rate`, `hero_net_chips`,
  `lie_rate`); needs the cluster GRPO runs to fill in.
- Dense-vs-sparse efficiency — the same env with `reward_mode=dense|sparse` is the A/B.
- Generalization check — cross-validate stealth on the held-out behavioral reader
  (`reader_leakage.py`) so a drop reflects concealment, not probe-specific evasion.

## Containment

Deliberately training monitor-evasion in a poker sandbox: KL-to-ref (fleet default), a validity /
legibility gate (`invalid_penalty`, only legal bracket tokens score), and a held lie-rate band
(`hold_lie_rate`) so the policy cannot win by simply never bluffing. Kept to this sandbox.
