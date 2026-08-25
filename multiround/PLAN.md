# Multi-Round Reputation Study — Negotiation / BlindAuction / NewRecruit

**Hypothesis (H-ratchet):** leaking your private values — or lying and getting caught — in
early rounds weakens you in later rounds of a repeated interaction: opponents carry your
revealed type forward and exploit it (squeeze the surplus split, snipe your targets), and
caught lies destroy the credibility of your later talk. This is the repeated-game /
ratchet-effect complement to the one-shot VoD work in `deception_poc/`. Sharpest existing
tension to resolve: Phase-2 Term-2b found leaking **helps** in one-shot NewRecruit
(+2562, logrolling) but hurts in poker (−218). Prediction: with repetition, the
distributive share of the leaker degrades even where integrative logrolling persists.

These three envs are chosen deliberately: small, fast, cheap (vs Coup/Diplomacy), already
instrumented (claim judge, value-inference probes, misrep metrics), and each isolates a
different mechanism.

## Common design (all three envs)

A multi-round **match** = T consecutive plays of the same env between the SAME seats.

What carries across rounds (the treatment channel):
- **Private values: FIXED across rounds within a match.** Without this, round-1 leakage
  is worthless in round 2 and the hypothesis is untestable. (Redrawn-values is a control
  arm, see below.)
- **Transcripts + outcomes**: each seat's context for round k includes rounds 1..k−1
  (its own view: prompts, public talk, whispers it saw, outcome announcements).
- **Cumulative score**, announced between rounds ("standings after round k"), so
  protecting future rounds is incentivized. Objective = sum of per-round payoffs.

What resets each round: inventories / capital / proposal state. (Carrying capital in the
auction creates a rich-get-richer confound — keep it out of v1.)

**Outcome-revelation knob** (`reveal` level) — controls how verifiable early lies are:
- `min`: winners/deal terms + prices only
- `bids`: + all submitted bids (auction) / both realized scores (NewRecruit) — makes
  "I don't care about item 3" objectively falsifiable next round
- `full`: + true values (BlindAuction's CURRENT announcement leaks the winner's true item
  values publicly — this must be patched to be configurable; `full` hands opponents ground
  truth for free and should only be a max-exploitation reference arm)

Default for the main runs: `bids`.

**Experimental arms** (causal manipulation of round-1 behavior, à la `bluff_prompts.py`
and the June prompt-arm runs — arm prompt applies in round 1 ONLY, all later rounds get
the neutral maximize prompt):
1. `honest` — truthfully state your values/priorities in round-1 talk
2. `conceal` — say nothing informative about values; talk process, not preferences
3. `leaky-liar` — actively misrepresent values (disclaim your targets, hype your duds)

Manipulation check: realized per-round misrep rate + leakage must actually separate by
arm (measure, don't assume — compliance varied in the bluff-arm runs).

**Controls:**
- **Memory ablation**: same arms, but opponents' round-1 context is wiped before round 2
  (hero keeps its own memory). Any late-round deficit that survives ablation is a skill
  confound of the arm prompt, not exploitation via carried knowledge. This is the key
  identification control.
- **Values-redrawn**: values re-randomized each round (memory kept). Kills the
  type-carryover channel, keeps the reputation/credibility channel — separates "they know
  my values" from "they know I lie".

Full grid is 3 arms × {carry, mem-ablate, val-redraw} — run `carry` for all arms first;
add the two controls for whichever contrast is significant.

## Per-env spec

### 1. Negotiation (resource trading, 2p first, 3p later)
- Round = one full env episode (fresh inventory, same `player_values` — pin by re-seeding
  the value draw; subclass like `IntegrativeNegotiationEnv` already shows the pattern).
  Keep the integrative regime (wide independent values) so there's surplus to steal.
- Between rounds inject a GAME observation: round-k trades executed + per-seat value
  gain (`min`: own gain only; `bids`: both gains).
- Exploitation signature: opponent's offer terms against the hero worsen (hero's
  surplus share of executed trades falls) after high round-1 leakage.
- Leakage metric per round: reuse the value-inference reader machinery
  (`negotiation_crossplay/run_value_probe_ta.py` / `reelicit_ta_posteriors.py`) —
  reader→target posterior Spearman computed on round-k talk only, and cumulatively.

### 2. BlindAuction (2–4p)
- Round = conversation phase + blind bids + results, same `player_item_values`
  (pin at match start; patch `reset` to accept injected values), capital reset.
- The objective lie-catch: round-k stated interest (existing `claim_judge.extract_stated`)
  vs round-k **revealed bids** — opponents can verify lies without any judge. This is the
  cleanest reputation env of the three.
- Exploitation signatures: (a) opponents' bids on the hero's true top-2 items rise in
  round k+1 as a function of hero's round-k leakage/misrep; (b) hero's profit share falls;
  (c) opponents' talk references the hero's past lies (qualitative, taxonomy pass).
- Patch `_announce_auction_results` to respect the `reveal` knob (strip "Value to
  Player X" lines below `full`).

### 3. NewRecruit (2p, fixed payoff table)
- Round = one full negotiation, same roles both seats (recruiter/candidate renegotiating
  an annual contract — natural cover story), payoffs summed across rounds. No-deal in a
  round = BATNA (0 above baseline) for both — no-deal-as-punishment stays available.
- The table is fixed, so priorities leak permanently — the purest type-carryover env, and
  the direct test of the Term-2b sign flip: does one-shot "+2562 leaking helps" become
  negative with repetition, or does logrolling keep dominating? Secondary prediction:
  integrative efficiency (joint points) stays high for honest arm across rounds while the
  leaker's SHARE degrades.
- `reveal=bids` here = announce both realized point totals after each round (makes "that
  deal was terrible for me" falsifiable).
- Leakage metric: stated issue importance (claim judge) vs true column spreads — already
  built in `deception_poc/newrecruit_vod.py`.

## Harness implementation

Do NOT rewrite the envs' step logic. New runner `multiround/run_multiround.py`:

1. Lift the seat-driver from `negotiation_crossplay/run_crossplay.py` (`chat()`,
   `render_obs`, OpenRouter client, seat rotation) into a shared module.
2. `MultiRoundMatch` loop: for k in 1..T → build env with pinned values (per-env
   `make_env(match_seed, round_k, values)` factory) → play episode collecting per-seat
   message histories → append inter-round bridge observation (results per `reveal` level +
   cumulative standings + "your valuations are unchanged; same opponents") → seed round
   k+1's per-seat context with the accumulated history.
3. Env-specific adapters: value pinning (Negotiation subclass; BlindAuction injected
   values; NewRecruit none needed), outcome extraction, reveal-level rendering.
4. Persist one ATIF-style JSON per match with `rounds: [...]` so the existing viewer and
   `claim_judge`/probe tooling work per-round (they key off steps; add `round` field).
5. Context budget: T=4 rounds × ~2–4k tokens/round transcript fits 64K comfortably for
   2p; for 3p auction add an optional "summarize rounds older than 2" fallback (off by
   default — raw transcripts preserve verifiability).

Est. new code: ~400–600 lines runner + ~100 lines env patches (reveal knob, value
injection) + analysis notebook. Reuse >70% of existing metric code.

## Metrics & analysis

Per seat × round: leakage (probe ρ / judge ρ), misrep rate, lie-caught events (stated vs
revealed-bid contradiction), gain (env-native: value_gain / profit / points−baseline),
surplus share, opponent-targeting stats (auction), and opponents' belief accuracy about
hero values (optional re-elicitation after each round, reusing `reelicit_ta_posteriors.py`
— directly measures the knowledge-carryover channel).

Primary tests (bootstrap CIs clustered by match, reuse `deception_poc/ci.py`):
- **H1 (main):** late-round gain (rounds 3..T) `leaky-liar` vs `conceal`, carry condition.
  H-ratchet ⇒ negative, and the deficit should VANISH under memory ablation.
- **H2 (dose-response):** within-arm regression of round-k gain on cumulative leakage
  through k−1.
- **H3 (mechanism split):** deficit present in val-redraw ⇒ reputation channel; present
  only in carry ⇒ type-exploitation channel.
- **H4 (NewRecruit sign flip):** sign of leakage→gain by round index; one-shot says +,
  H-ratchet says it decays/flips by round 3–4.

## Scale & cost (pilot)

Pilot: 2p only, T=4, arms×carry-only, per env: 24 matches × 2 seats. Models: one
mid-tier pair (e.g. qwen3.6-27b vs gpt-5.5-mini class) with seat/arm rotation; hero arm
assigned to one seat per match, opponent always neutral prompt.
≈ 24 matches × 3 arms × 3 envs = 216 matches = 864 episodes ≈ 2–3× one of the earlier
xplay batches — order $50–150 OpenRouter at mid-tier pricing, a weekend of wall-clock at
the usual concurrency. Controls (+mem-ablate, +val-redraw on the significant contrasts)
roughly double it.

## Phasing

- **P0 — harness (1–2 days):** shared driver extraction, MultiRoundMatch, Negotiation-2p
  adapter, reveal knob + value pinning patches, smoke with a cheap model, ATIF output
  renders in the viewer.
- **P1 — Negotiation pilot:** 3 arms × 24 matches, per-round probe leakage, H1/H2 read.
- **P2 — BlindAuction + NewRecruit adapters + pilots** (auction gives the objective
  lie-catch; NewRecruit gives the sign-flip test).
- **P3 — controls + analysis:** mem-ablation and val-redraw on significant contrasts,
  belief re-elicitation, plots (per-round gain trajectories by arm; leakage→next-round
  exploitation scatter), findings doc `MULTIROUND_FINDINGS.md`.

## Pitfalls / known confounds

- **Action-channel leakage is irreducible**: with fixed values, round-1 TRADES/BIDS reveal
  type even if talk is silent — `conceal` arm bounds talk-leakage only. Report
  talk-leakage and action-leakage separately (behavior_probe.py already does the action
  channel for Negotiation).
- **Arm prompts may alter skill, not just honesty** — that's exactly what the memory
  ablation control catches. Don't skip it for the headline contrast.
- **BlindAuction value reveal** (current env behavior) must be patched BEFORE any run, or
  round-2 exploitation is trivially confounded.
- **Judge noise**: claim-judge stated-interest extraction was the noisiest part of the
  one-shot PoC — keep the v4_checklist-style calibration habit; the revealed-bid
  contradiction metric needs no judge and should be the headline lie-catch stat.
- **Opponent compliance**: opponents get NO arm prompt and the neutral prompt must not
  hint that reputation matters, or we've assumed the conclusion. Use the same SYSTEM as
  one-shot xplay plus only the factual "this is a T-round match, same opponents, values
  fixed, cumulative score" statement — that statement is part of the game, not coaching.
