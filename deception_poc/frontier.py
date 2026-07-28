#!/usr/bin/env python3
"""Assemble the efficient-deception frontier across the PoC games.
x = leakage (how readable the agent's hidden type is from its public channel; higher =
more transparent), y = instrumental gain. Good deceiver = upper-left (gain, low leakage).

Audit fixes baked in:
  #1 the poker-family y-axis is now a DE-NOISED gain (card-variance-free bluff-EV, or the
     fold-rate a bluff induces), NOT raw realized chips — raw chips are dominated by card variance
     at this n (see the wide meanchips CIs in poker_bluff.py).
  #2 every poker-family point carries a 95% clustered-bootstrap CI (vertical error bar); where a
     bar crosses 0 the gain does NOT separate from variance — read those as null, not signal.
The NL games (BlindAuction / New Recruit / Negotiation) keep realized score/profit (the advice
says that metric is defensible there); their limitation is sample size, not variance-vs-metric.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
COLOR = {"gpt-5.6-sol-pro": "#2a78d6", "claude-opus-4.8": "#008300", "kimi-k3": "#eda100",
         "deepseek-v4-pro": "#e87ba4", "qwen3.7-max": "#eb6834", "gemini-3.6-flash": "#4a3aa7",
         "llama-4-maverick": "#e34948", "glm-5.2": "#1baf7a",
         "gpt-5.5": "#6699cc", "claude-sonnet-5": "#55aa55", "gemma-4-31b-it": "#cc66aa",
         "qwen3.6-27b": "#d9a300", "qwen3.5-9b": "#cc6633"}

# (title, file, leakage-key, gain-key, x-label, y-label, container, ci-key or None)
PANELS = [
    ("Poker (EV)", "poker_bluff_results.json", "tell", "bluff_ev",
     "leakage: hand tell (AUROC)", "bluff EV, chips (de-noised) ±95%CI", "models", "bluff_ev_ci"),
    ("Poker (Kelly)", "poker_bluff_results.json", "tell", "bluff_kelly",
     "leakage: hand tell (AUROC)", "bluff log-growth E[log(W'/W)] ±95%CI", "models", "bluff_kelly_ci"),
    ("KuhnPoker", "kuhn_bluff_results.json", "tell", "bluff_success",
     "leakage: bet→K tell (AUROC)", "bluff fold-rate (de-noised) ±95%CI", "models", "bluff_success_ci"),
    ("LeducHoldem", "leduc_vod_results.json", "tell", "bluff_ev",
     "leakage: aggression tell (AUROC)", "bluff EV, chips (de-noised) ±95%CI", "models", "bluff_ev_ci"),
    ("LiarsDice", "liarsdice_vod_results.json", "leakage", "gain",
     "leakage: bids reveal dice (ownFrac−1/6)", "mean rank reward ±95%CI", "models", "gain_ci"),
    ("BlindAuction", "blindauction_vod_results.json", "leakage", "profit",
     "leakage: ρ(stated interest, true value)", "mean profit", "by_model", None),
    ("New Recruit", "newrecruit_vod_results.json", "leakage", "surplus",
     "leakage: ρ(stated, true importance)", "surplus vs baseline", "by_model", None),
    ("Negotiation / trade  (diff. pool)", "negotiation_vod_results.json", "leakage", "gain",
     "leakage: reader→you ρ", "value gain", "by_model", None),
]


def main():
    ncol = 4
    nrow = -(-len(PANELS) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5.0, nrow * 4.8))
    axes = axes.flatten()
    for ax, (title, fn, lk, gk, xl, yl, mk, cik) in zip(axes, PANELS):
        fp = D / fn
        if not fp.exists():
            ax.set_title(f"{title} (no data)", fontsize=10); continue
        data = json.load(open(fp))[mk]
        for model, d in data.items():
            x, y = d.get(lk), d.get(gk)
            if x is None or y is None:
                continue
            yerr = None
            ci = d.get(cik) if cik else None
            if ci:
                yerr = [[max(0, y - ci["lo"])], [max(0, ci["hi"] - y)]]
                ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#9a9a94", elinewidth=1.1,
                            capsize=3, zorder=2)
            ax.scatter(x, y, s=120, c=COLOR.get(model, "#888"), edgecolors="#0b0b0b", lw=0.8, zorder=3)
            ax.annotate(model.replace("-sol-pro", "").replace("-maverick", ""), (x, y),
                        textcoords="offset points", xytext=(7, 3), fontsize=7.5, color="#0b0b0b")
        if cik:
            ax.axhline(0, color="#bbb", lw=1, zorder=1)  # gains crossing 0 = not separable from noise
        ax.set_title(title, fontsize=11, color="#0b0b0b")
        ax.set_xlabel(xl, fontsize=8.5, color="#52514e")
        ax.set_ylabel(yl, fontsize=8, color="#52514e")
        ax.grid(color="#dcdcd7", lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)
    for ax in axes[len(PANELS):]:
        ax.set_visible(False)
    fig.suptitle("Efficient-deception frontier (PoC) — DE-NOISED gain (±95% CI) vs type-leakage · "
                 "good deceiver = upper-left · error bar crossing 0 = gain not separable from variance",
                 fontsize=12, color="#0b0b0b", y=1.02)
    fig.tight_layout()
    p = D / "deception_frontier.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
