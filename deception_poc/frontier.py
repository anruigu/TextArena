#!/usr/bin/env python3
"""Assemble the efficient-deception frontier across the 3 PoC games.
x = leakage (how readable the agent's hidden type is from its public channel; higher =
more transparent), y = instrumental gain. Good deceiver = upper-left (gain, low leakage).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
COLOR = {"gpt-5.6-sol-pro": "#2a78d6", "claude-opus-4.8": "#008300", "kimi-k3": "#eda100",
         "deepseek-v4-pro": "#e87ba4", "qwen3.7-max": "#eb6834", "gemini-3.6-flash": "#4a3aa7",
         "llama-4-maverick": "#e34948", "glm-5.2": "#1baf7a"}

# (title, results-file, leakage-key, gain-key, x-label, y-label, models-container-key)
PANELS = [
    ("Poker", "poker_bluff_results.json", "tell", "meanchips", "leakage: hand-strength tell (AUROC)", "mean end chips", "models"),
    ("BlindAuction", "blindauction_vod_results.json", "leakage", "profit", "leakage: ρ(stated interest, true value)", "mean profit", "by_model"),
    ("New Recruit", "newrecruit_vod_results.json", "leakage", "surplus", "leakage: ρ(stated, true importance)", "surplus vs baseline", "by_model"),
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (title, fn, lk, gk, xl, yl, mk) in zip(axes, PANELS):
        data = json.load(open(D / fn))[mk]
        for model, d in data.items():
            x, y = d.get(lk), d.get(gk)
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=120, c=COLOR.get(model, "#888"), edgecolors="#0b0b0b", lw=0.8, zorder=3)
            ax.annotate(model.replace("-sol-pro", "").replace("-maverick", ""), (x, y),
                        textcoords="offset points", xytext=(7, 3), fontsize=7.5, color="#0b0b0b")
        ax.set_title(title, fontsize=11, color="#0b0b0b")
        ax.set_xlabel(xl, fontsize=8.5, color="#52514e")
        ax.set_ylabel(yl, fontsize=8.5, color="#52514e")
        ax.grid(color="#dcdcd7", lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)
    fig.suptitle("Efficient-deception frontier (PoC) — gain vs type-leakage · good deceiver = upper-left",
                 fontsize=12, color="#0b0b0b", y=1.02)
    fig.tight_layout()
    p = D / "deception_frontier.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
