#!/usr/bin/env python3
"""Per-instance efficient-deception frontier: every scored game/player-game as a point
(colored by model), model mean ringed. Outliers trimmed (5th-95th pct of gain per panel)
to zoom into the mass; mean labels decluttered with a vertical leader-line stack."""
import json
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
COLOR = {"gpt-5.6-sol-pro": "#2a78d6", "claude-opus-4.8": "#008300", "kimi-k3": "#eda100",
         "deepseek-v4-pro": "#e87ba4", "qwen3.7-max": "#eb6834", "gemini-3.6-flash": "#4a3aa7",
         "llama-4-maverick": "#e34948", "glm-5.2": "#1baf7a"}
PANELS = [
    ("Poker  (pt = one game/model)", "poker_bluff_results.json", "tell", "final_chips",
     "type-leakage: hand-strength tell (AUROC)", "end chips"),
    ("BlindAuction  (pt = one player-game)", "blindauction_vod_results.json", "leakage", "profit",
     "type-leakage: ρ(stated interest, true value)", "profit"),
    ("New Recruit  (pt = one scored deal)", "newrecruit_vod_results.json", "leakage", "surplus",
     "type-leakage: ρ(stated, true importance)", "surplus vs baseline"),
]


def short(m):
    return m.replace("-sol-pro", "").replace("-maverick", "")


def trim(pts, lo=5, hi=95):
    """Drop points whose gain (y) is outside [lo,hi] percentile. Skip if too few."""
    if len(pts) < 12:
        return pts
    ys = [y for _, y, _ in pts]
    ylo, yhi = np.percentile(ys, [lo, hi])
    return [(x, y, m) for x, y, m in pts if ylo <= y <= yhi]


def declutter(ax, means, xr, yr):
    """means: list of (mx,my,model). Place labels in a non-overlapping vertical stack,
    connected by thin leader lines."""
    ylim = ax.get_ylim()
    gap = 0.075 * (ylim[1] - ylim[0])
    order = sorted(means, key=lambda t: t[1])
    ly = ylim[0] - 1e9
    placed = []
    for mx, my, m in order:
        ly = max(my, ly + gap)
        placed.append((mx, my, ly, m))
    # if the stack overflows the top, shift the whole stack down
    over = placed[-1][2] - (ylim[1] - 0.03 * (ylim[1] - ylim[0]))
    if over > 0:
        placed = [(mx, my, ly - over, m) for mx, my, ly, m in placed]
    lx = ax.get_xlim()[1] - 0.30 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    for mx, my, ly, m in placed:
        ax.annotate(short(m), xy=(mx, my), xytext=(lx, ly), fontsize=8, color="#0b0b0b",
                    ha="left", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#9a9a94", lw=0.7,
                                    connectionstyle="arc3,rad=0.0"))


def main():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    seen = set()
    for ax, (title, fn, lk, gk, xl, yl) in zip(axes, PANELS):
        rows = json.load(open(D / fn))["rows"]
        pts = [(r[lk], r[gk], r["model"]) for r in rows
               if r.get(lk) is not None and r.get(gk) is not None]
        pts = trim(pts)
        for x, y, m in pts:
            ax.scatter(x, y, s=40, c=COLOR.get(m, "#888"), alpha=0.40, edgecolors="none", zorder=2)
            seen.add(m)
        by = {}
        for x, y, m in pts:
            by.setdefault(m, []).append((x, y))
        means = []
        for m, xy in by.items():
            mx, my = mean(a for a, _ in xy), mean(b for _, b in xy)
            means.append((mx, my, m))
            ax.scatter(mx, my, s=150, c=COLOR.get(m, "#888"), edgecolors="#0b0b0b", lw=1.4, zorder=4)
        ax.axhline(0, color="#dcdcd7", lw=1)
        xr = [p[0] for p in pts]; yr = [p[1] for p in pts]
        if xr:
            padx = 0.08 * (max(xr) - min(xr) or 1); pady = 0.10 * (max(yr) - min(yr) or 1)
            ax.set_xlim(min(xr) - padx, max(xr) + padx)
            ax.set_ylim(min(yr) - pady, max(yr) + pady)
        declutter(ax, means, xr, yr)
        ax.set_title(title, fontsize=10.5, color="#0b0b0b")
        ax.set_xlabel(xl, fontsize=8.5, color="#52514e")
        ax.set_ylabel(yl, fontsize=8.5, color="#52514e")
        ax.grid(color="#dcdcd7", lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=COLOR[m], mec="#0b0b0b",
               ms=8, label=short(m)) for m in COLOR if m in seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Efficient-deception frontier — per-instance (faint = each scored game · ringed = model mean · "
                 "gain outliers trimmed 5–95%)\ngood deceiver = upper-left (gain while unreadable)",
                 fontsize=11.5, color="#0b0b0b", y=1.05)
    fig.tight_layout()
    p = D / "deception_frontier_scatter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
