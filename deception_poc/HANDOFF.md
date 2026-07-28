# Deception-Efficacy Study — Handoff & Result Index

Handoff of the poker deception-metric design work (and the surrounding study). This is the
map of **what was run, where the results live, and which metric to trust**. Everything below is in
`TextArena/deception_poc/` on branch **`deception-poc`** (remote `fork` = `anruigu/TextArena`)
unless noted. Raw crossplay rollouts are mirrored to S3 (see [Raw data](#raw-data)).

## The thesis in one line
A *good* lie is one that gains value **without leaking your hidden type**. We measure this as two
terms per act of deception: **Term-1 = instrumental gain** (did the lie pay) and **Term-2 = leakage**
(how recoverable your true type is). Poker is the anchor game: it has **LLM-free exact ground truth**
(eval7 hand equity), so both terms are cleanest here.

## TL;DR — which metric to trust (the design conclusion)
- **Trust the causal counterfactual (Phase-2 Term-2b) as ground truth**, and the **behavioral reader
  probe** as the measurable, optimize-able leakage signal. See [Phase-2](#workstream-2--true-counterfactuals-phase-2)
  and [Reader](#workstream-3--behavioral-reader-probe--rlfr-leaky-lies).
- **Term-1 gain axis for poker is now Kelly log-growth**, not raw EV — variance-penalized,
  bankroll-normalized. See [Poker gain metric](#poker-gain-metric-design-ev--kelly).
- **Do NOT anchor on the cross-model correlation plots** (`deception_efficiency`, `best_conditioned`,
  `best_per_game`): n=5 models, sign-unstable (poker flips +0.64 vs −0.81 depending on the gain axis).
  They answer "who is a good deceiver," not "does reducing leakage cause better outcomes."
- **Term-2 = behavioral leakage** (readability to opponents), *not* the activation/introspective probe
  (which is self-representation — a complementary axis, see [Introspective probe](#workstream-4--introspective--activation-probe)).

## Environment
Run analysis in the performative venv (has `eval7`, `torch`, `sklearn`):
```bash
/workspace/allie/performative/.venv/bin/python <script>.py
```
`ci.py` provides the clustered (per-game) bootstrap CIs every gain number carries.

---

## Workstream 1 — Phase-1 analytic frontier (gain × leakage)
Analysis-only (no rollouts): read the crossplay transcripts, score each deception act.

| game | analyzer | results JSON | notes |
|---|---|---|---|
| Poker | `poker_bluff.py` | `poker_bluff_results.json` | bluff = Bet/Raise on equity<0.40; Term-1 `bluff_ev`/`bluff_kelly`, Term-2 `tell` AUROC |
| Kuhn | `kuhn_bluff.py` | `kuhn_bluff_results.json` | |
| Leduc | `leduc_vod.py` | `leduc_vod_results.json` | |
| LiarsDice | `liarsdice_vod.py` | `liarsdice_vod_results.json` | |
| Coup | `coup_leakage.py` | `coup_leakage_results.json` (+`coup_leakage.png`, `coup_leakage_grid.png`) | explicit-challenge game |
| Mafia | `mafia_leakage.py` | `mafia_leakage_results.json` (+`mafia_leakage.png`) | |
| BlindAuction | `blindauction_leakage.py`, `blindauction_vod.py` | `blindauction_leakage_results.json`, `blindauction_vod_results.json` (+`blindauction_leakage.png`) | |
| NewRecruit | `newrecruit_vod.py` | `newrecruit_vod_results.json` | integrative negotiation |
| ScorableGames | `scorablegames_vod.py` | `scorablegames_vod_results.json` | |
| Negotiation | `negotiation_vod.py` | `negotiation_vod_results.json` | different model pool |

**Frontier plots** (good deceiver = upper-left, gain while unreadable):
- `frontier.py` → **`deception_frontier.png`** — the headline gain-vs-leakage frontier. Now has **two
  poker panels: `Poker (EV)` and `Poker (Kelly)`** side by side.
- `frontier_scatter.py` → `deception_frontier_scatter.png` — per-instance (poker y is raw chips).
- Findings: **`POC_FINDINGS.md`**.

### Poker gain metric design (EV → Kelly)
The Term-1 poker gain axis was upgraded from risk-neutral EV to **Kelly / log-utility**:
- `action_ev(r)` — card-variance-free expected chips (fold ⇒ +pot; showdown ⇒ equity·pot). `bluff_ev`.
- `action_kelly(r)` — **`E[log(W'/W)]` per bluff**, W = chips behind at the decision (parsed via
  `parse_stack`). Same de-noising, but concave ⇒ penalizes variance + normalizes by bankroll. `bluff_kelly`.

Both are written per-model and per-game to `poker_bluff_results.json` with clustered-bootstrap CIs.
**Finding:** at these stack-to-bet ratios Kelly ≈ EV/W, so the model ranking is preserved (glm ≫
deepseek > gpt ≈ kimi > claude) — *except* claude, whose positive EV (+4.7 chips) collapses to ~0
under Kelly (CI straddles 0): its bluff "edge" is variance, not compounding growth. gpt is closest to
the efficient corner (lowest `tell` ≈ 0.57).
> Open: apply Kelly to the **Phase-2 counterfactual** gain (`phase2_aggregate.py`) — that's the causal
> axis where variance-reranking would matter most; not yet done.

---

## Workstream 2 — true counterfactuals (Phase-2)
Checkpoint-and-replay: restore the exact state at each deception fork, force only the fork action,
drive live to the horizon with opponents re-deciding. Turns Phase-1's analytic baselines into causal
numbers.

- Pipeline: `phase2_generate.py` (per-turn pickled checkpoints) → `counterfactual_replay.py`
  (restore + live rollouts, per-env adapters) → `phase2_aggregate.py`.
- Outputs: **`phase2_summary.json`**, per-fork rows in **`results/<env>_forks.jsonl`**,
  `deception_frontier_phase2.png` (Term-1 VoD vs leakage), `deception_term2b_phase2.png` (Term-2b bars).
- Findings: **`PHASE2_FINDINGS.md`**.

**Headlines:** poker Term-1 VoD **+73.9 chips/bluff** (67% positive); Term-2b leaked-type cost
**−218 chips** (being read hurts) and is **coupled to gain** (kimi gains most +162 / bleeds most −448).
The **sign of Term-2b flips with game type**: negative (adversarial poker) vs **positive** in
NewRecruit (+2562 — revealing your value table helps the integrative logroll).

---

## Workstream 3 — behavioral reader probe + RLFR leaky-lies
The **detector** and the **intervention** stack (the natural next phase for "fixing bad lying").

- Spots: `build_reader_spots.py` → `reader_spots.jsonl` (public channel, hole cards stripped).
- Detector: `train_reader_probe.py` → `probes/probe_leakreader_qwen3_8b.npz`,
  `reader_probe_qwen3_8b.json`, **`reader_probe_qwen3_8b.png`**. Frozen Qwen3-8B recovers hand-strength
  bucket from betting behavior alone at **AUROC 0.771** — genuine behavioral leakage.
- Behavioral reader (also feeds NL lie-counts): `reader_leakage.py` → `reader_leakage_results.json`.
- Reward + BoN validation: `leaky_reward.py`, `reader_service.py`, `bon_leaky.py` →
  `bon_leaky_results.json`, **`bon_leaky_frontier.png`**. Sweeping the leakage penalty λ: leakage
  0.40→0.016, bluff success 0.61→1.0, hero chips **−97 → +156** (stealth pays, causally, pre-RL).
- Online loop: `armsrace_loop.py` → `armsrace_results.json`; GRPO trainer `train_leaky_poker_tinker.py`;
  skyrl-gym env `leaky_poker` (under `skyrl-gym/skyrl_gym/envs/leaky_poker/`).
- Findings / how-to-run: **`PHASE2_LEAKY_LIES.md`**.

---

## Workstream 4 — introspective / activation probe
White-box self-representation meter (open models only, 8×H200). **Complementary** to Term-2, not a
replacement — measures whether the model *represents* its type internally, not whether opponents can
read it.

- `build_probe_spots.py` → `spots.jsonl` (1095 poker decisions + exact equity).
- `activation_probe.py` → `activation_probe_results_<tag>.json`, `activation_probe_<tag>.png`,
  **`activation_probe_summary.{png,json}`**, `probes/probe_pokerstrength_<tag>_last.npz`.
- Findings: **`ACTIVATION_PROBE_FINDINGS.md`**. Every model represents hand strength near-ceiling
  (activation AUROC 0.94–0.99) while behavioral leakage varies (0.51–0.88); the gap is a per-model
  concealment signature (steerable via the saved direction).

---

## Cross-game proxy analyses (`proxy_plots/`, `lie_frequency_plots/`)
- `proxy_plots/proxy_grid.py` → per-game `*_grid.png`, `proxy_combinations.md` (the A/B/C proxy audit).
- `proxy_plots/deception_efficiency.py` → `deception_efficiency.{png,md}` (per-lie MEAN-z, one number/model).
- `proxy_plots/best_conditioned.py` → `best_conditioned.{png,md}` (lie-conditioned class-A proxies).
- `proxy_plots/best_per_game.py` → `best_per_game.{png,md}` (best leakage×gain pairing per game).
- `proxy_plots/conditioned_leakage_heatmap.py` → `conditioned_leakage_heatmap.{png,md}`.
- `lie_frequency_plots/lie_frequency.py` → `lie_frequency.md`, `*_gain_vs_lie.png`, `lie_frequency_overview.png`.
> Caveat repeated: these are **cross-model (n=5) correlations** — directional only, sign-unstable. Use
> the counterfactual + reader-probe evidence for any causal claim.

---

## Raw data
Crossplay rollouts (per-game transcript JSON) live locally in `TextArena/<game>_crossplay/` and are
mirrored to S3:

**`s3://fleet-research/negotiation/`** — one prefix per game
(`poker_crossplay/`, `kuhn_crossplay/`, `leduc_crossplay/`, `liarsdice_crossplay/`,
`blindauction_crossplay/`, `mafia_crossplay/`, `negotiation_crossplay/`).

Poker specifically: `poker_crossplay/results/` + `poker_crossplay/results_more/` (30 games) — the input
`poker_bluff.py` reads.

## Suggested next steps for the collaborator
1. Apply Kelly to the Phase-2 counterfactual gain axis (`phase2_aggregate.py`) and re-plot the frontier.
2. Run the `leaky_poker` GRPO loop on the fleet (dense vs sparse reward A/B) — the intervention proper.
3. Extend the A-vs-B distinction: poker (leakage flows only through the behavioral channel) vs coup
   (explicit challenge) — test whether a poker-trained concealment skill transfers.
