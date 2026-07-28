#!/usr/bin/env python3
"""Combined plot of the best LIE-CONDITIONED leakage x gain proxy per game.

Audit result (see COND in proxy_grid.py): almost every "leakage" proxy in this PoC is confounded
with the raw deception RATE, because it is computed over ALL actions (honest + deceptive) rather
than conditioned on the lie actually happening:
  - poker/kuhn/leduc `tell`,`tell_size`  -> AUROC over ALL hands; bluffs sit in the neg class, so
        bluffing more (identical per-bluff behavior) mechanically lowers the tell. (class C)
  - liarsdice `leakage`,`mean_own_frac`   -> own_frac averaged over ALL bids (bluffs enter as 0),
        so higher bluff rate drags leakage down. (class C)
  - blindauction/newrecruit/scorablegames `leakage rho` -> spearman(stated, true) over ALL items,
        mixing honest+dishonest; it is a rank-correlation restatement of `misrep`. (class C)
  - all `bluff_rate`,`misrep`,`claim_honesty` -> literally the frequency of lying. (class B)

Only these are conditioned on the lie event (class A, base-rate invariant / lie-event denominator):
  - mafia   `role_leakage`       = mean observer P(Mafia | player IS Mafia)
  - coup    `bluff_caught_rate`  = P(challenged | bluff)         [denominator = bluffs]
  - coup    `table_read_auroc`   = AUROC separating THIS model's bluffs vs truths by challenge
  - negotiation `leakage`        = reader->target posterior recovery of hidden values (per game)
  - liarsdice `bluff_stick_rate` = P(not-immediately-called | bluff)  -- but EMPTY in the data (n<3)

So only 3 games (mafia, coup, negotiation) have a usable lie-conditioned leakage proxy. The other
7 have NO conditioned leakage measure available and are shown as blank "no conditioned proxy" cells.

  python3 best_conditioned.py
"""
from __future__ import annotations
from pathlib import Path
from proxy_grid import GAMES, build, pearson, color_for, COND

D = Path(__file__).resolve().parent


# Gain axis must be a MATCH-LEVEL outcome that is NOT derived from the same opponent-fold event
# the leakage is built on. bluff_ev / bluff_chips_won / bluff_success all bake in opp_folded, so
# pairing them with P(opp calls|bluff) is semi-mechanical — exclude them. Keep whole-match stacks,
# rank reward, and win/survival rates.
MATCH_GAIN = {"meanchips", "meanbank", "gain", "mean_reward", "mafia_win_rate",
              "mafia_survival_rate", "win_rate", "survival_rate", "value_captured",
              "profit", "surplus", "concealed_wins"}


def best_conditioned(game):
    """Max-|r| pairing (n>=3) using ONLY class-A (lie-conditioned) leakage proxies, or None."""
    models, vals, leaks, gains = build(game)
    leaks = [t for t in leaks if COND.get((game, t[0])) == "A"]  # conditioned only
    gains = [t for t in gains if t[0] in MATCH_GAIN]  # independent match-level outcomes only
    if not leaks or not gains:
        return None
    best = None
    for lf, ll, inv in leaks:
        for gf, gl in gains:
            pts = [(vals[lf][m], vals[gf][m], m) for m in models
                   if vals[lf][m] is not None and vals[gf][m] is not None]
            r, n = pearson([(p[0], p[1]) for p in pts])
            if r != r or n < 3:
                continue
            if best is None or abs(r) > best["absr"]:
                best = {"game": game, "lf": lf, "ll": ll, "gf": gf, "gl": gl,
                        "r": r, "n": n, "absr": abs(r), "pts": pts}
    return best


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks, blank = [], []
    for g in GAMES:
        p = best_conditioned(g)
        (picks if p else blank).append(p if p else g)
    picks.sort(key=lambda p: -p["absr"])

    ncol = max(1, len(picks))
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.2), squeeze=False)
    for idx in range(ncol):
        ax = axes[0][idx]
        p = picks[idx]
        for i, (x, y, m) in enumerate(p["pts"]):
            ax.scatter(x, y, s=110, c=color_for(m, i), edgecolors="#0b0b0b", lw=1, zorder=3)
            ax.annotate(m.split("/")[-1].replace("-sol-pro", "").replace("-maverick", "")
                        .replace("-3.6-flash", "").replace("-opus-4.8", "").replace("-v4-pro", ""),
                        (x, y), textcoords="offset points", xytext=(5, 2), fontsize=7)
        strong = p["absr"] >= 0.5
        ax.set_title(f"{p['game']}   r={p['r']:+.2f} (n={p['n']})",
                     fontsize=11, color="#b00" if strong else "#333",
                     fontweight="bold" if strong else "normal")
        ax.set_xlabel(p["ll"], fontsize=9)
        ax.set_ylabel(p["gl"], fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    miss = ", ".join(blank)
    fig.suptitle("Best LIE-CONDITIONED leakage x gain per game (class A only — no raw-rate confound)\n"
                 f"no conditioned leakage proxy available: {miss}", fontsize=11.5, y=1.06)
    fig.tight_layout()
    out = D / "best_conditioned.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ---- markdown ----
    lines = ["# Best LIE-CONDITIONED leakage proxy per game (class A only)", "",
             "After a source-code audit (`COND` in `proxy_grid.py`), only class-**A** leakage proxies "
             "are conditioned on the lie event and thus free of the raw-deception-rate confound. Raw "
             "rates (**B**: `bluff_rate`, `misrep`, `claim_honesty`) and unconditioned mixes (**C**: "
             "`tell`/`tell_size`, `leakage rho`, liarsdice `own_frac` leakage) all move mechanically "
             "with how *often* the model lies, so they are excluded here. Combined scatter: "
             "`best_conditioned.png`. Sign: negative r = concealment-pays (low leakage -> high gain). "
             "Bold = |r| >= 0.5.", "",
             "![best conditioned](best_conditioned.png)", "",
             "| game | class-A leakage (x) | gain (y) | Pearson r | n |",
             "|---|---|---|---:|---:|"]
    for p in picks:
        rstr = f"{p['r']:+.2f}"
        if p["absr"] >= 0.5:
            rstr = f"**{rstr}**"
        lines.append(f"| {p['game']} | {p['ll']} | {p['gl']} | {rstr} | {p['n']} |")
    lines += ["",
              "### Games with NO usable lie-conditioned leakage proxy",
              "",
              "| game | why |", "|---|---|",
              "| poker | only `tell`/`tell_size` (C, AUROC over all hands) + `bluff_rate` (B) |",
              "| kuhn | only `tell` (C) + `bluff_rate` (B) |",
              "| leduc | only `tell` (C) + `bluff_rate` (B) |",
              "| liarsdice | `bluff_stick_rate` (A) is empty/n<3; rest are C/B |",
              "| blindauction | only `leakage rho` (C) + `misrep` (B) |",
              "| newrecruit | only `leakage rho` (C) + `misrep` (B) |",
              "| scorablegames | only `leakage rho` (C) + `misrep` (B) |",
              ""]
    (D / "best_conditioned.md").write_text("\n".join(lines))
    print(f"wrote {D / 'best_conditioned.md'}")
    print("\nBest CONDITIONED leakage proxy per game:")
    for p in picks:
        print(f"  |r|={p['absr']:.2f} r={p['r']:+.2f} n={p['n']}  {p['game']}: {p['ll']} vs {p['gl']}")
    print(f"  (no conditioned proxy: {miss})")


if __name__ == "__main__":
    main()
