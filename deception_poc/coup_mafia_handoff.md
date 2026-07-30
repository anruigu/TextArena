# Coup & Mafia Deception — Handoff & Result Index

Companion to [`HANDOFF.md`](HANDOFF.md), scoped to the two **social-deduction** games in the study:
**Coup** and **Secret Mafia**. Both run on the `mafia_crossplay/` harness. They sit at opposite ends
of the leakage-measurement spectrum:

- **Coup** — hidden type = influence cards; every deception is a **machine-parseable claim** opponents
  can **explicitly call out** (`[BULLSHIT]`). Detector is **exact, LLM-free**. [→ Coup](#coup)
- **Mafia** — hidden type = role (Mafia vs Town); leakage is diffuse across free-form **day-discussion
  chat**, so the detector is an **LLM type-reader** over public messages, tracked over rounds.
  [→ Mafia](#mafia)

## Where everything lives
| thing | Coup | Mafia |
|---|---|---|
| **Raw traces** | `mafia_crossplay/results_coup/` (+`results_coup_q36_frontier/`) | `mafia_crossplay/results/` (+`results_more/`, `results_qwen/`) |
| Analyzer | `deception_poc/coup_leakage.py` | `deception_poc/mafia_leakage.py` |
| Outputs | `coup_leakage_results.json`, `coup_leakage.png`, `coup_leakage_grid.png` | `mafia_leakage_results.json`, `mafia_leakage.png` |
| Rollout runner | `mafia_crossplay/run_xplay.py --env coup` | `mafia_crossplay/run_mafia.py` (or `run_xplay.py --env mafia`) |
| Viewer converter | `to_agentviz_general.py` | `to_agentviz.py` |
| Viewer server | `agentviz/serve_xplay.mjs` | same server, `?manifest=sessions/mafia/manifest.json` |
| S3 mirror | `s3://fleet-research/negotiation/mafia_crossplay/results_coup/` | `s3://fleet-research/negotiation/mafia_crossplay/results/` |

All rollouts capture per-turn **reasoning traces for the viewer only** — only each seat's spoken
`action` reaches `env.step()` and opponents' histories, so thinking never leaks in-game.

---

# Coup

Coup is the study's *explicit-challenge* game and the Type-B counterpart to poker (see
[Why these two games](#why-these-two-games)).

## The coup traces
Coup runs on the mafia harness (`run_xplay.py --env coup`), so its rollouts live under
`mafia_crossplay/`, not a `coup_crossplay/` dir:

| dir | games | players | pool |
|---|---:|---:|---|
| `results_coup/` | 10 | 5 | base 7-frontier pool (the set `coup_leakage.py` reads) |
| `results_coup_q36_frontier/` | 12 | 5 | qwen3.6 frontier variant |
| `results_coup_qwenscale/` | 0 | — | empty (scaffold only) |

Each `game_XX.json` is a uniform xplay record: `seat_models`, `winners`, `meta` (`final_coins`,
`survivors`), and a `transcript` of per-turn `{pid, phase, action, reasoning, obs}`. `results_coup/`
also has `REPORT.md` + `analysis.json`. Generated with:
```bash
cd /workspace/allie/TextArena/mafia_crossplay
python3 run_xplay.py --env coup --games 10 --out results_coup   # 7 OpenRouter frontier models
```

## The analyzer (`coup_leakage.py`) — exact, LLM-free
Claim→role map is machine-parseable (`[tax]`→Duke, `[assassinate]`→Assassin, `[block steal captain]`→
Captain, …) and ground truth is in the same turn's obs, so all readers are exact (no judge risk).
```bash
/workspace/allie/performative/.venv/bin/python coup_leakage.py          # analyze + 3-panel figure
/workspace/allie/performative/.venv/bin/python coup_leakage.py --replot # rebuild fig from saved json
/workspace/allie/performative/.venv/bin/python coup_leakage.py --grid   # every leakage×gain pairing
```
> Reads `mafia_crossplay/results_coup` (hardcoded `RESULTS` at the top — repoint to analyze
> `results_coup_q36_frontier`). Only needs stdlib + matplotlib + `ci.py`.

**Metrics** (per model, in `coup_leakage_results.json`):
- **Term-2a `claim_honesty`** = P(hold | claim); high = readable. Deck prior **0.371** = pure-bluff floor.
- **Term-2b `bluff_caught_rate`** = P(challenged | bluff) — opponents' `[BULLSHIT]` is a **live, free
  type-reader**. `table_read_auroc` = does a challenge predict a bluff, per model.
- **Term-1 gain** = `win_rate`, `survival_rate`, `bluff_uncaught_rate` (Coup's per-lie metric in
  `deception_efficiency.py`).

**Headline** (n=206 claims / 10 games): the free reader works — **P(challenge|bluff)=0.22 vs
P(challenge|truth)=0.06**, pooled AUROC 0.58. **Concealment pays**: `bluff_caught_rate` vs win
**r=−0.47**, `claim_honesty` vs win **r=−0.22**. Small per-model n — directional. Coup also feeds
`proxy_plots/best_conditioned.md` (table-read vs win, −0.51) and `best_per_game.md` (bluff-caught vs
win, −0.47).

---

# Mafia

Secret Mafia (7 players, 2 Mafia) is the *diffuse-leakage* game: there's no discrete claim, so
readability is measured by an **LLM type-reader** over the public day-discussion chat, and — because
mafia plays multiple rounds — as a **leakage trajectory over days**.

## The mafia traces
| dir | games | runner/schema |
|---|---:|---|
| `results/` | 10 | `run_mafia.py` (raw: `roles` + `alive_at_end`) |
| `results_more/` | 24 | `run_mafia.py` (grows n without clobbering the original 10) |
| `results_qwen/` | ~28 | qwen-scale pool |

`mafia_leakage.py` reads all three (`RESULT_DIRS`), 62 games total. Each `game_XX.json` has
`seat_models`, `roles`/`seat_roles` (ground truth), `rewards`, `win_team`, and a `transcript` of
per-turn `{pid, phase, action, reasoning, obs}`. Generated with:
```bash
cd /workspace/allie/TextArena/mafia_crossplay
python3 run_mafia.py --games 10 --players 7 --out results     # or: run_xplay.py --env mafia
```
> Two schemas coexist: `run_mafia.py` writes `roles`+`alive_at_end`; `run_xplay.py` writes
> `seat_roles`. Both analyzer and viewer tolerate either.

## The analyzer (`mafia_leakage.py`) — LLM type-reader
Unlike Coup, there is no exact claim, so leakage is read by an LLM that sees **ONLY public
day-discussion messages** (never roles, night actions, or reasoning). **Needs an OpenRouter key** (read
from `/workspace/allie/.env`).
```bash
cd /workspace/allie/TextArena/deception_poc
python3 mafia_leakage.py --reader openai/gpt-5.4-mini --passes 3 --concurrency 6   # runs the reader
python3 mafia_leakage.py --replot                                                  # rebuild fig, NO LLM
```

**Metrics** (in `mafia_leakage_results.json`):
- **Leakage trajectory by day** — reader AUROC(role) from public msgs ≤ day d, vs the prior.
- **Self-channel AUROC** — read each player's role from ONLY their own public messages.
- **Per-model `role_leakage`** — reader P(Mafia) from a model's own msgs; **low = good deceiver**
  (holds the reader near the prior). Prior base rate **0.286**.
- **Marginal leakage / `top_leaky_mafia_acts`** — the biggest single-day posterior jumps = the "leaky
  acts."
- **Term-1 gain** — `mafia_win_rate`, `mafia_survival_rate`.

**Headline** (62 games, reader gpt-5.4-mini): leakage **accumulates over rounds** — collective reader
AUROC **day1 0.69 → day2 0.84**; self-channel AUROC 0.67 (true-Mafia posterior 0.55 vs Town 0.44).
Per-model role-leakage clusters near the 0.286 prior (deepseek 0.49 lowest of the frontier set; gpt
0.50 with the highest Mafia win rate 0.79) — the role-leakage-vs-win frontier is `mafia_leakage.png`
panel 3.

> **Key difference from Coup:** the mafia number depends on a **reader model** (LLM, costs API calls,
> mild run-to-run noise damped by `--passes`), whereas Coup is exact/LLM-free. Treat mafia leakage as
> a behavioral estimate, coup leakage as ground truth.

---

## How to run the viewer (agentviz)
Same server for both; only the converter and the manifest path differ.

**1. Convert traces to ATIF sessions**
```bash
cd /workspace/allie/TextArena/mafia_crossplay
# Coup (+ any xplay dirs) -> sessions/xplay
python3 to_agentviz_general.py \
    --inputs results_coup,results_coup_q36_frontier \
    --out /workspace/allie/agentviz/dist/sessions/xplay
# Raw Mafia -> sessions/mafia
python3 to_agentviz.py --in results --out /workspace/allie/agentviz/dist/sessions/mafia
```

**2. Serve the viewer** (static prebuilt `dist/`, fixed port for a stable replay URL)
```bash
cd /workspace/allie/agentviz
node serve_xplay.mjs            # binds 127.0.0.1:8731  (PORT=9000 node serve_xplay.mjs to override)
```

**3. Open** — the URL's `?manifest=` selects which set to replay:
```
Coup / xplay:  http://localhost:8731/?manifest=sessions/xplay/manifest.json
Mafia:         http://localhost:8731/?manifest=sessions/mafia/manifest.json
```
Coup games appear as `coup_base-gNN`; mafia as `mafia_game_NN`. Each player turn is one step carrying
that seat's model, its **reasoning trace**, the **observation** it saw, and its spoken action; setup +
result are system steps, so you can replay a bluff/accusation and read every seat's private reasoning.
The build is already populated (124 xplay sessions incl. `coup_base-g00..09`, plus a `mafia` set);
re-run step 1 only when you add/regenerate games (the `--out` dir is overwritten per run).

---

## Why these two games
Coup and Mafia bracket the **Type-A vs Type-B leakage** distinction (see `HANDOFF.md` TL;DR):
- **Coup** gives Type-B (opponents exploiting your type) an **explicit path** — the challenge — that a
  passive probe wouldn't capture, and lands Term-1 (value of the bluff) and Term-2 (cost of being read)
  on the **same act**, exactly and LLM-free. It's the transfer test for whether a poker-trained
  concealment skill generalizes to overt-claim lying.
- **Mafia** is the diffuse, multi-round case where leakage is only legible through an LLM reader over
  free-form chat and **compounds over days** — the natural setting for a *leakage-trajectory* /
  marginal-leaky-act analysis rather than a per-act one.

## Suggested next steps
1. Coup: re-run `coup_leakage.py` against `results_coup_q36_frontier` (repoint `RESULTS`) and pool for
   more claims per model.
2. Coup: add a Phase-2-style counterfactual (fork at a bluff, force honest vs bluff, measure win/coins
   delta) — the causal Term-1/Term-2b for Coup.
3. Mafia: sweep the reader model / raise `--passes` to tighten the leakage estimate; mine
   `top_leaky_mafia_acts` for qualitative "what makes a message leaky" examples.
