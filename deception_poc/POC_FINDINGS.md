# Deception-efficacy PoC — early signal (analysis-only)

Operationalizes the VoD frame (`../deception_efficacy_plan.md`): **Term-1 instrumental
gain** vs **Term-2 type-leakage** (behavioral reader), plotted as an efficient-deception
frontier. No rollouts, no probes. Artifacts in `deception_poc/`:
`deception_frontier.png`, `{poker_bluff,blindauction_vod,newrecruit_vod}_results.json`.

![Efficient-deception frontier: gain vs type-leakage across Poker, BlindAuction, and New Recruit. Good deceiver = upper-left (high instrumental gain, low type-leakage).](deception_frontier.png)

*Efficient-deception frontier. x = type-leakage (how readable the hidden type is from the
public channel; higher = more transparent), y = instrumental gain. Upper-left = efficient
deceiver (gains while staying unreadable); lower-right = transparent loser.*

![Per-instance efficient-deception frontier: every scored game (faint dots) with model means ringed on top, for Poker, BlindAuction, and New Recruit.](deception_frontier_scatter.png)

*Per-instance view (`frontier_scatter.py`): faint = each scored game/player-game, ringed =
model mean (n shown). Makes the spread explicit — BlindAuction per-game profit is high-variance
around the means, and New Recruit has very few scored deals, so lean on the means with that in
mind.*

## What each game measures
| Game | Hidden type | Leakage meter (Term 2) | Gain (Term 1) | Ground truth |
|---|---|---|---|---|
| Poker | hand strength (equity) | AUROC: does "was aggressive" predict a strong hand? (0.5 unreadable→1 transparent) | mean end chips / bluff fold-rate | hole+board (eval7 equity), chips — **exact, no LLM** |
| BlindAuction | which items you value | ρ(judge-stated interest, true item values) | mean profit; "concealed wins" (disclaimed a top item then won it) | `meta` values/bids/winners/profits |
| New Recruit | private issue values | ρ(judge-stated importance, true column spread) | realized score − middle-deal baseline | fixed payoff table; scores from `reason` |

## Early signal
**Consistent direction: concealing your type (low leakage) tracks higher gain; the most
transparent model loses.** Seat-level Pearson(leakage, profit) in BlindAuction = **−0.20** (n=184).

- **Poker** — gpt-5.6-sol-pro sits upper-left: lowest tell (AUROC **0.59**, least readable) and
  most chips (**1158**); it bluffs 20% and its aggression doesn't reveal strength. deepseek/kimi/glm
  are transparent (tell 0.74–0.76). (matches the earlier gpt-5.6 sweep.)
- **BlindAuction** (richest, n=196 player-games) — **kimi-k3** upper-left: leakage −0.28, most
  concealed-wins (14), top profit (**+93**). **llama-4-maverick** far lower-right: leakage **+0.53**
  (states its true interests), misrep 18%, the **only negative profit (−10)**. qwen3.7 most
  anti-correlated (−0.44 → actively misrepresents).
- **New Recruit** — same direction but **noisy** (2 players, ≈6/10 games are invalid-move forfeits
  → few scored deals). qwen/kimi negative leakage; gemini transparent (0.91). Treat as directional.

## Model takeaways (cross-game)
- **gpt-5.6-sol-pro / kimi-k3** = efficient deceivers (gain at low leakage).
- **llama-4-maverick** = transparent loser everywhere (leaks its type; negative auction profit) —
  the same CoT/value-leakage behavior seen qualitatively.
- **deepseek / glm / gemini** = readable, middling.

## Caveats / what would harden it
- Small n where it bites: poker aggressive decisions 10–44/model; New Recruit deals 0–2/model.
  Poker + BlindAuction carry the signal; New Recruit needs more games (or ScorableGames).
- Leakage for the two NL games depends on the **claim judge** (gpt-5.4-mini, temp 0) — validate a
  sample by hand; swap in a stronger judge if noisy.
- Term-1 uses **analytic baselines** (middle-deal / raw profit), not the honest-replay
  counterfactual — that (true VoD) + the activation-probe leakage meter + repeated-game leakage
  cost are Phase 2.
- New Recruit Term-1 excludes forfeits (they conflate format compliance with skill).
