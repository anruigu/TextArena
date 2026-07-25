#!/usr/bin/env python3
"""Gain (Term-1) vs LIE FREQUENCY per game.

Lie frequency = how OFTEN a model deceives (a raw-rate proxy), as opposed to how *readable* it is
when it does (leakage, Term-2) or how *much* a lie buys (gain, Term-1). This is the axis the main
frontier / `best_per_game.md` deliberately EXCLUDES — here we look at it on its own: does bluffing
MORE OFTEN correlate with winning more?

Per game: x = lie frequency (per model), y = the game's Term-1 gain (per model), one point/model,
model-mean Pearson r. Reads the per-model numbers already in `../*_results.json` (no rollouts/LLM).

Outputs (in this dir): <game>_gain_vs_lie.png per game, lie_frequency_overview.png (grid),
lie_frequency.md (measures + explanation + numbers).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
POC = D.parent
COLOR = {"gpt-5.6-sol-pro": "#2a78d6", "claude-opus-4.8": "#008300", "kimi-k3": "#eda100",
         "deepseek-v4-pro": "#e87ba4", "qwen3.7-max": "#eb6834", "gemini-3.6-flash": "#4a3aa7",
         "llama-4-maverick": "#e34948", "glm-5.2": "#1baf7a"}

# game -> (results file, container, freq_key, gain_key, freq_label, gain_label, freq_desc)
GAMES = [
    ("Poker", "poker_bluff_results.json", "models", "bluff_rate", "meanchips",
     "bluff rate", "mean end chips",
     "share of aggressive actions (Bet/Raise) taken on a weak hand (eval7 equity < 0.40)"),
    ("KuhnPoker", "kuhn_bluff_results.json", "models", "bluff_rate", "meanchips",
     "bluff rate", "mean end chips",
     "share of [bet] actions made holding a J (the weakest of J/Q/K)"),
    ("LeducHoldem", "leduc_vod_results.json", "models", "bluff_rate", "meanbank",
     "bluff rate", "mean end bank",
     "share of aggressive actions on a weak card (enumeration equity < 0.40)"),
    ("LiarsDice", "liarsdice_vod_results.json", "models", "bluff_rate", "gain",
     "bluff rate", "mean rank reward",
     "share of bids on a face the player holds ZERO of (pure-air bids)"),
    ("BlindAuction", "blindauction_vod_results.json", "by_model", "misrep", "profit",
     "misrep rate", "mean profit",
     "share of items whose publicly-stated interest contradicts the true value "
     "(disclaimed a top-2 item, or hyped a bottom-2 item)"),
    ("NewRecruit", "newrecruit_vod_results.json", "by_model", "misrep", "surplus",
     "misrep rate", "surplus vs baseline",
     "share of issues whose stated importance contradicts the private point table"),
    ("ScorableGames", "scorablegames_vod_results.json", "by_model", "misrep", "surplus",
     "misrep rate", "surplus vs baseline",
     "share of issues whose stated importance contradicts the private scoring table"),
    ("Coup", "coup_leakage_results.json", "per_model", "bluff_rate", "win_rate",
     "bluff rate", "win rate",
     "share of role claims that are bluffs (asserted a role not in the hidden hand)"),
]


def load(fn, cont):
    fp = POC / fn
    if not fp.exists():
        return {}
    return json.load(open(fp)).get(cont, {})


def pearson(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** .5
    return num / den if den else None


def collect(g):
    _, fn, cont, fk, gk, *_ = g
    data = load(fn, cont)
    rows = []
    for m, d in data.items():
        f, gain = d.get(fk), d.get(gk)
        if f is None or gain is None:
            continue
        rows.append((m, float(f), float(gain)))
    return rows


def scatter(ax, g, rows):
    title, *_rest = g
    fl, gl = g[5], g[6]
    r = pearson([(f, gain) for _, f, gain in rows])
    for m, f, gain in rows:
        ax.scatter(f, gain, s=130, c=COLOR.get(m, "#888"), edgecolors="#0b0b0b", lw=0.8, zorder=3)
        ax.annotate(m.replace("-sol-pro", "").replace("-maverick", ""), (f, gain),
                    textcoords="offset points", xytext=(6, 3), fontsize=7.5)
    rtxt = f"r={r:+.2f}" if r is not None else "r=n/a"
    ax.set_title(f"{title}  ({rtxt}, n={len(rows)})", fontsize=10.5)
    ax.set_xlabel(f"lie frequency: {fl}", fontsize=8.5, color="#52514e")
    ax.set_ylabel(f"gain: {gl}", fontsize=8.5, color="#52514e")
    ax.grid(color="#e8e8e4", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return r


def main():
    results = {}
    # per-game standalone figures
    for g in GAMES:
        rows = collect(g)
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        r = scatter(ax, g, rows)
        fig.tight_layout()
        fig.savefig(D / f"{g[0].lower()}_gain_vs_lie.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        results[g[0]] = (g, rows, r)

    # overview grid
    gs = [g for g in GAMES if g[0] in results]
    ncol = 4
    nrow = -(-len(gs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.6, nrow * 4.3))
    axes = axes.flatten()
    for ax, g in zip(axes, gs):
        scatter(ax, g, results[g[0]][1])
    for ax in axes[len(gs):]:
        ax.set_visible(False)
    fig.suptitle("Gain (Term-1) vs LIE FREQUENCY — does deceiving more often win more? "
                 "(r>0 = more bluffing → more gain; r<0 = over-bluffing hurts)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(D / "lie_frequency_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {len(gs)} per-game figs + lie_frequency_overview.png")

    write_md(results)


def write_md(results):
    lines = [
        "# Lie frequency — how OFTEN each model deceives (Term-1 gain vs lie frequency)",
        "",
        "**Lie frequency** is a *raw-rate* proxy: how often a model takes a deceptive action — a",
        "distinct axis from **leakage** (Term-2: how readable it is *when* it deceives) and from",
        "**gain** (Term-1: how much a deception buys). It is deliberately *excluded* from the",
        "leakage↔gain frontier in `../proxy_plots/best_per_game.md` because a high rate says nothing",
        "about whether the deception is good — a model can bluff constantly and lose. These plots ask",
        "the separate question: **does deceiving more often correlate with winning more?**",
        "",
        "Per game: x = lie frequency, y = the game's Term-1 gain, one point per model; `r` = model-mean",
        "Pearson. Overview: `lie_frequency_overview.png`; per game: `<game>_gain_vs_lie.png`.",
        "Source: per-model numbers in `../*_results.json` (no rollouts/LLM).",
        "",
        "## Lie-frequency measure per game",
        "| game | lie-frequency measure | gain (y) | Pearson r | n |",
        "|---|---|---|---:|---:|",
    ]
    for g in GAMES:
        if g[0] not in results:
            continue
        _, rows, r = results[g[0]]
        lines.append(f"| {g[0]} | {g[7]} | {g[6]} | {r:+.2f} | {len(rows)} |" if r is not None
                     else f"| {g[0]} | {g[7]} | {g[6]} | n/a | {len(rows)} |")
    lines += [
        "",
        "**Not shown:** *Mafia* — a Mafia player's every public claim of innocence is a role-lie, so a",
        "per-turn 'lie frequency' isn't well-defined (frequency ≈ 1 by construction); *Negotiation* —",
        "the value-inference reader has no discrete lie act to count.",
        "",
        "## Per-model lie frequency (sorted, per game)",
    ]
    for g in GAMES:
        if g[0] not in results:
            continue
        _, rows, r = results[g[0]]
        lines.append(f"\n**{g[0]}** — {g[7]}:")
        for m, f, gain in sorted(rows, key=lambda t: -t[1]):
            lines.append(f"- {m}: **{f*100:.0f}%** ({g[6]} = {gain:.2f})")
    lines += [
        "",
        "## Reading it",
        "- **Sign is not consistent across games**, and that is the point: lie *frequency* is not a",
        "  virtue. Where bluffing is +EV and under-punished, more bluffing can track more gain (r>0);",
        "  where it is exploitable or the model over-bluffs weak spots, more bluffing tracks *less*",
        "  gain (r<0). Contrast with the leakage↔gain frontier, which is consistently negative",
        "  (concealment pays) in distributive games.",
        "- **Frequency ≠ efficacy.** A model high on this axis is not a good deceiver — it is a",
        "  frequent one. The efficient-deception story lives in leakage×gain, not here. Coup makes",
        "  this vivid: gpt-5.6 bluffs the most yet does not win the most (over-bluffing without",
        "  converting), so the frequency↔win slope is weak/negative even though its *concealment* is best.",
        "- Use these plots to separate **'lies a lot'** from **'lies well'** — the two are different models.",
    ]
    (D / "lie_frequency.md").write_text("\n".join(lines) + "\n")
    print("wrote lie_frequency.md")


if __name__ == "__main__":
    main()
