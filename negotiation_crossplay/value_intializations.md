# Leverage sweep for multiparty negotiation — plan

Goal: extend `superhuman_negotiator/skyrl_gym/envs/negotiation/multiparty/values.py`
into a **broad leverage sweep** so that *diverse strategies emerge from the
value/outside-option initialization alone* (no prompt or reward changes), then
run that sweep **cross-play across a pool of frontier models via OpenRouter**.

Two leverage families (confirmed scope):
1. **Value-structure** (keep the sum-100 per-party normalization): contestedness,
   per-party concentration/peakiness, cross-party asymmetry, bloc/alliance
   correlation, swan/spike strengths.
2. **Outside options / BATNA** (relax the `d = 0` disagreement point): each party
   gets a private walk-away value; no-deal pays that value, not 0. This is the
   classic hold-out lever and touches `game.py` + `prompts.py` end-to-end.

We are NOT relaxing sum-100 into heterogeneous budgets (deferred; it would break
cross-party score comparability and every normalized benchmark).

---

## Leverage axes (what we can dial at init)

- **Contestedness `alpha`** (exists): `raw = alpha*c + (1-alpha)*eps`. alpha->1
  everyone wants the same (distributive/fight); alpha->0 gains-from-trade
  (integrative). The discriminative axis in `FINDINGS_0720.md`.
- **Concentration / peakiness `gamma`** (new): exponentiate each party's raw
  vector (`raw**gamma`) before normalize. gamma>1 => peaked (strong preferences,
  logrolling leverage); gamma<1 => flat (indifferent, weak leverage).
- **Cross-party asymmetry `gamma_spread`** (new): draw each party's gamma from a
  spread so parties differ in how peaked they are (some strong-preference, some
  near-indifferent) — a structural power gap even under sum-100.
- **Bloc / alliance `n_blocs`, `bloc_strength`** (new `regime="bloc"`): a subset
  of parties shares an extra common sub-component (aligned subgroup); creates
  natural coalitions and exclusion leverage.
- **Black swan `swan_strength`** (exists): one item boosted in the *common*
  component — everyone wants it (the "one item everyone wants" case in the notes).
- **Spike `spike_strength`** (exists): one party wants one item a lot (the "one
  player uniquely wants it" case in the notes).
- **Outside options `outside_*`** (new): per-party reservation value on the 0..100
  scale. Modes: `none | uniform(low,high) | asymmetric | one_strong(party,value)`.
  A high-BATNA party can credibly refuse — the single strongest strategy inducer.

All axes are exposed as explicit kwargs plus a `LeverageSpec` dataclass, and a
`leverage_grid()` generator yields the sweep cells.

---

## Files to change

### 1. `.../multiparty/values.py` (structure axes + outside options)
- Add `gamma` (per-party or scalar) + `gamma_spread`, `regime="bloc"`
  (`n_blocs`, `bloc_strength`).
- Add outside-option sampling; extend `ValueDraw` with
  `outside_options: List[int]`, the `LeverageSpec`, and precomputed a-priori
  leverage metrics (see Phase 3). Keep `normalize_100` for the value rows.
- Add `LeverageSpec` dataclass, `sample_values(..., spec=...)`, and
  `leverage_grid(axes) -> Iterator[LeverageSpec]`.

### 2. `.../multiparty/game.py` (BATNA into scoring/benchmarks)
- `evaluate(..., outside_options=None)`:
  - No-deal: `scores[i] = outside_options[i]` (was 0).
  - Nash: disagreement point `d_i = outside_options[i]`; maximize
    `prod(max(0, u_i - d_i))`; report distance/ratio against that `d`.
  - New metrics: `ir_satisfied` (all realized >= outside), `ir_violations`,
    `surplus_over_outside` per party, `no_deal_value`.
- `efficiency_ratio` / `logrolling_*` stay as-is (items still fully allocated),
  so cross-cell comparability is preserved.

### 3. `.../multiparty/prompts.py` (surface the walk-away)
- Add a "Your walk-away value: if no agreement, you receive **W** points (not 0)"
  line to `SYSTEM_TEMPLATE` and adjust `FINAL_ROUND_WARNINGS` wording when
  outside options are active. `build_system_prompt(..., outside_option=None)`.

### 4. `.../multiparty/engine.py` (thread it through)
- Pass `draw.outside_options` into `build_system_prompt` and `game.evaluate`;
  add outside options to the trace `scenario`/`config`. Belief probes unchanged.

### 5. `.../multiparty/run_selfplay.py` (expose args + report)
- New args: `--gamma`, `--gamma-spread`, `--n-blocs`, `--bloc-strength`,
  `--outside-mode/--outside-low/--outside-high`, plus regime `bloc`.
- Add outside/IR metrics + score-inequality (Gini) to `summarize()`.

### 6. NEW `.../multiparty/run_leverage_sweep.py` (cross-play sweep)
- Replaces the rounds-only `run_sweep.py` pattern with a **leverage grid** driver.
- Args: `--models` (comma pool of frontier ids), `--parties`,
  `--episodes-per-cell`, grid lists (`--alpha-list`, `--gamma-list`,
  `--outside-list`, `--regime-list`), `--rounds`, `--agreement-mode`, `--out-root`.
- For each cell: call `run_selfplay.run` with the model **pool** (cross-play,
  seats rotate so each model plays each seat), matched seeds across cells.
- Emit `leverage_summary.json` + a per-cell table / heatmap of strategy-diversity
  metrics (below). Default pool (configurable) draws several OpenRouter frontier
  models.

### 7. `.../tests/test_multiparty.py`
- Unit tests: sum-100 invariant holds with gamma/bloc; outside-option no-deal pays
  `d_i`; Nash uses `d`; `leverage_grid` cardinality; IR metric correctness.

---

## Strategy-diversity readout (how we know init induced different play)

Per leverage cell, aggregated over cross-play episodes:
- `agreement_rate`, `no_deal_rate`, `mean_agreed_round`
- `efficiency_ratio`, `nash_distance_norm`, `nash_product_ratio`
- **score inequality (Gini)** and `ir_violation_rate` (BATNA hold-out signal)
- `pair_deal_rate` / `excluded_counts` (coalition/exclusion leverage; needs
  `--agreement-mode pair*`)
- `belief_mae_mean` / `belief_spearman` = value elicitation *quality* (does
  probing succeed / are others' values recoverable; already in-engine, free)
- per-model `offers/accepts/game` fingerprint (participation/initiative)
- `judge_traces.py` per cell — three judge passes over outward messages only
  (no CoT), each per party:
  - **value elicitation (the ACT)**: flags every move where a party asks/probes
    *another* party for their private values, typed `direct_question |
    indirect_probe | reciprocal_offer | test_probe` with a target. Aggregates:
    `elicitation_moves_per_party`, `elicitation_party_frac`,
    `elicitation_targets_per_party`, `elicitation_by_type`. This is the
    *initiative* signal, orthogonal to belief-MAE (which is the *success*
    signal) — high elicitation + low MAE = probing that paid off.
  - **value misrepresentation/deception**: extracts each claim a party makes
    about its OWN values, labels `honest | overstate | understate | fabrication`
    and whether it `benefited_speaker`. Aggregates:
    `deception_rate_micro/macro`, `deceptive_claims_benefited_frac`.
  - combined: `asymmetric_extractor_frac` = of parties that elicited, the
    fraction that also misrepresented themselves (one-sided info extraction:
    pump others while hiding own).

---

## Status (2026-08-12)

All five phases are **done**. Phases 1–3 are library work (90 tests, no API);
Phase 4 ran as a 7-cell proof-of-concept cross-play sweep and Phase 5 is written
up in `superhuman_negotiator/.../multiparty/REPORT_LEVERAGE_0812.md`.

**Answer to the headline question: yes** — initialization alone moved every
measurable behaviour with no prompt or reward change. Judged deception 7.6% →
14.9% pooled (7.5% → 29.2% for `claude-sonnet-5`), logrolling .385 → .703, pair
deals 1.00 → .64, inequality .268 → .377, belief MAE 6.4 → 8.6. The strongest
axis is `alpha`; the cleanest mechanism is the BATNA, where a credible walk-away
*substituted for* lying (lowest deception, lowest inequality, fastest agreement)
rather than becoming a bluffing instrument. Caveat: 12 episodes/cell, so only the
extremes separate statistically.

- Phase 1 ✅ `values.py`: `gamma`, `gamma_spread` (stratified log2 draws),
  `regime="bloc"` (`n_blocs`, `bloc_strength`), `LeverageSpec`, `leverage_grid`
  (`cross`/`oat`, dedupes by `cell_id`, supports compound axes).
- Phase 2 ✅ outside options end-to-end: `values.py` (4 modes) →
  `game.evaluate(outside_options=...)` (no-deal and non-signatories pay their
  BATNA; NBS over `prod(u_i - d_i)`; `ir_satisfied`/`ir_violations`/
  `surplus_over_outside`/`no_deal_value`/`ir_feasible`) → `prompts.py`
  (per-seat walk-away line, rule override, final-round warnings) →
  `engine.py` → `run_selfplay.py` (all axes as CLI args, IR + Gini metrics).
- Phase 3 ✅ NEW `leverage_metrics.py` (offline, free) + the gate result below.
- Phase 4 ✅ NEW `run_leverage_sweep.py` (with `--dry-run`: cell table + call
  estimate, spends nothing). Ran 7 cells × 12 episodes × 3 parties,
  `pair_or_grand` + 0.45 dividend, pool = claude-sonnet-5 / gpt-5.5 /
  gemini-3.1-pro-preview → `results/multiparty/leverage_poc`. 0 failures, 59 min.
  Deception measured by the optional `judge_traces.py` pass (judge
  `claude-sonnet-4.6`, outside the player pool).
- Phase 5 ✅ `REPORT_LEVERAGE_0812.md`.

**Phase 3 gate result** (300 draws/cell, 3 parties, one axis at a time): every
axis moves its own structure metric, and they are mutually orthogonal.

| axis | metric | range observed |
|---|---|---|
| `alpha` 0→1 | contestedness (mean pairwise Spearman) | −0.01 → 1.00 |
| `gamma` 0.5→4 | top-item share / concentration Gini | 0.25→0.55 / 0.11→0.51 |
| `gamma_spread` 0→2 | cross-party concentration spread | 0.12 → 0.31 |
| `bloc_strength` 0→1 | bloc cohesion (within − across rank corr) | 0.00 → 0.85 |
| `outside_*` | BATNA spread / egalitarian floor | 0→50 / 33.6→18.4 |

Orthogonality: `gamma` moves concentration without moving contestedness
(|Δ| < 0.02), and the BATNA cells leave every value-structure column
bit-identical (matched seeds — outside options are drawn last and consume no
randomness in mode `none`). `ir_feasible` stayed 1.00 everywhere up to
`one_strong(50)`, so no cell's no-deal rate will be arithmetically forced.

Two findings worth carrying into the sweep design:
- Cosine similarity saturates on non-negative value rows (unrelated draws still
  score ~0.9); rank correlation is the discriminating read, and is what
  `contest_rank` / `bloc_cohesion` use.
- `black_swan` is the most extreme cell offline (top-share 0.68, NBS Gini 0.31,
  floor 18.3) — worth including in the grid even though it was not in the
  suggested first grid.

## Phased execution

- **Phase 1 — values.py structure axes** (gamma, gamma_spread, bloc, LeverageSpec,
  leverage_grid) + tests. Pure library, no API.
- **Phase 2 — outside options end-to-end** (values -> game -> prompts -> engine ->
  run_selfplay) + tests. Pure library/logic, no API.
- **Phase 3 — offline characterization** (no API, cheap): compute a-priori
  leverage metrics per draw (contestedness index, concentration entropy/Gini +
  spread, gains-from-trade, pair-vs-grand coalition gain, outside-option spread,
  max attainable outcome inequality). Dump the grid and confirm the axes actually
  move structure BEFORE spending API budget.
- **Phase 4 — cross-play leverage sweep** (`run_leverage_sweep.py`) over a small
  validated grid with a frontier OpenRouter pool; per-cell summaries + aggregate.
- **Phase 5 — report**: map init -> induced strategy; identify which axes most
  reliably separate strategies (BATNA hold-out, coalition/exclusion, logrolling,
  elicitation, deception).

Suggested first grid (keep cell count modest for API cost): parties in {2,3};
alpha in {0.1, 0.4, 0.8}; gamma in {0.5, 1.0, 2.0}; outside in {none,
uniform(0,20), one_strong(30)}; regime in {cpi, spike, bloc}. Sweep one axis at a
time off a baseline rather than the full cross-product first.

---

## Original notes (preserved)

Power-asymmetry intent: contrast "an item everyone wants" (black_swan) vs "an item
one player uniquely values" (spike); brainstorm further asymmetries (concentration
spread, blocs, BATNA) — all covered above.

Behaviors to detect downstream from traces:
- **value elicitation**: how much players ask for others' values — the ACT of
  probing (`judge_traces.py` elicitation pass: typed moves + targets) alongside
  the belief-MAE success proxy.
- **value misrepresentation**: how much players misrepresent their own values
  (`judge_traces.py` deception labels).

Also relevant knobs already present: `max_rounds`, parties 2 vs 3,
`--agreement-mode` (unanimity/pair/pair_or_grand), `--grand-dividend`.
