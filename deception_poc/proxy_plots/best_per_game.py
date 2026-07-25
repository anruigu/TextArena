#!/usr/bin/env python3
"""Pick the single most-discerning Term-2 (leakage) x Term-1 (gain) proxy pairing per game
(largest |model-mean Pearson r| among computable pairings, n>=3) and render all games together
in ONE combined figure: best_per_game.png. Also writes best_per_game.md.

Pure re-analysis of ../*_results.json — no rollouts, no LLM. Reuses proxy_grid.py.

  python3 best_per_game.py
"""
from __future__ import annotations
from pathlib import Path
from proxy_grid import GAMES, build, pearson, color_for

D = Path(__file__).resolve().parent

# Term-2 axis must be genuine *leakage* = readability of the deceiver's own private
# type/hand/role/valuation to an observer. These fields are NOT leakage and are excluded
# from the "best leakage proxy" selection:
#   bluff_rate, misrep, bluff_stick_rate  -> raw deception *rates* (how often you act, not
#                                            how readable you are)
#   claim_honesty                          -> raw honesty rate (frequency, not readability)
#   table_read_auroc                       -> the model reading OTHERS (inference skill),
#                                            not its own signal leaking out
NON_LEAKAGE = {"bluff_rate", "misrep", "bluff_stick_rate", "claim_honesty", "table_read_auroc"}


def best_pairing(game):
    """Return dict for the max-|r| computable pairing (n>=3) over genuine-leakage x gain, or None."""
    models, vals, leaks, gains = build(game)
    leaks = [t for t in leaks if t[0] not in NON_LEAKAGE]  # drop raw-rate / inference proxies
    if not leaks:
        return None
    best = None
    for lf, ll, inv in leaks:
        for gf, gl in gains:
            pts = [(vals[lf][m], vals[gf][m], m) for m in models
                   if vals[lf][m] is not None and vals[gf][m] is not None]
            r, n = pearson([(p[0], p[1]) for p in pts])
            if r != r or n < 3:  # NaN or too few points
                continue
            if best is None or abs(r) > best["absr"]:
                best = {"game": game, "lf": lf, "ll": ll, "inv": inv, "gf": gf, "gl": gl,
                        "r": r, "n": n, "absr": abs(r), "pts": pts}
    return best


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = [best_pairing(g) for g in GAMES]
    picks = [p for p in picks if p is not None]
    picks.sort(key=lambda p: -p["absr"])  # most discerning first

    ncol = 5
    nrow = -(-len(picks) // ncol)  # ceil
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.6 * nrow), squeeze=False)
    for idx in range(nrow * ncol):
        ax = axes[idx // ncol][idx % ncol]
        if idx >= len(picks):
            ax.axis("off")
            continue
        p = picks[idx]
        for i, (x, y, m) in enumerate(p["pts"]):
            ax.scatter(x, y, s=95, c=color_for(m, i), edgecolors="#0b0b0b", lw=1, zorder=3)
            ax.annotate(m.split("/")[-1].replace("-sol-pro", "").replace("-maverick", "")
                        .replace("-3.6-flash", "").replace("-opus-4.8", "").replace("-v4-pro", ""),
                        (x, y), textcoords="offset points", xytext=(5, 2), fontsize=6)
        strong = p["absr"] >= 0.5
        ax.set_title(f"{p['game']}   r={p['r']:+.2f} (n={p['n']})",
                     fontsize=10.5, color="#b00" if strong else "#333",
                     fontweight="bold" if strong else "normal")
        ax.set_xlabel(p["ll"] + (" [inv]" if p["inv"] else ""), fontsize=8)
        ax.set_ylabel(p["gl"], fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Most-discerning *leakage* x gain proxy per game (readability only; raw rates "
                 "excluded; largest |model-mean r|, n>=3)", fontsize=12.5, y=1.002)
    fig.tight_layout()
    out = D / "best_per_game.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ---- markdown ----
    lines = ["# Best (most-discerning) *leakage* proxy pairing per game", "",
             "One row per game: the **genuine-leakage** (Term-2, readability) x gain (Term-1) "
             "pairing with the largest |model-mean Pearson r| (computable, n>=3). Raw-rate / "
             "inference proxies (`bluff_rate`, `misrep`, `bluff_stick_rate`, `claim_honesty`, "
             "`table_read_auroc`) are **excluded** — they measure how *often* a model deceives, or "
             "how well it reads others, not how readable it is. Combined scatter: "
             "`best_per_game.png`. Sign: leakage axes are *higher = more readable*; gain *higher = "
             "better*, so **negative r = concealment-pays** (low leakage -> high gain). "
             "Bold = |r| >= 0.5.", "",
             "![best per game](best_per_game.png)", "",
             "| game | Term-2 leakage (x) | Term-1 gain (y) | Pearson r | n |",
             "|---|---|---|---:|---:|"]
    for p in picks:
        ll = p["ll"] + (" (inv)" if p["inv"] else "")
        rstr = f"{p['r']:+.2f}"
        if p["absr"] >= 0.5:
            rstr = f"**{rstr}**"
        lines.append(f"| {p['game']} | {ll} | {p['gl']} | {rstr} | {p['n']} |")
    lines.append("")
    (D / "best_per_game.md").write_text("\n".join(lines))
    print(f"wrote {D / 'best_per_game.md'}")
    print("\nBest per game (ranked):")
    for p in picks:
        print(f"  |r|={p['absr']:.2f} r={p['r']:+.2f} n={p['n']}  {p['game']}: "
              f"{p['ll']}{' (inv)' if p['inv'] else ''} vs {p['gl']}")


if __name__ == "__main__":
    main()
