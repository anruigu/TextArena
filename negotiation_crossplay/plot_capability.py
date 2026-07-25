#!/usr/bin/env python3
"""Capability view for TextArena cross-play: value captured (own-value gain) and
pie-expansion-when-present (integrative ratio) — endowment-free alternatives to
the myopic win-rate chart."""
import json, glob
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FRONT = {"anthropic/claude-sonnet-5", "openai/gpt-5.5"}
def load(b): return [json.load(open(f)) for f in glob.glob(f"{b}/game_*.json")]

def per_model(b):
    gain = defaultdict(list); ratio = defaultdict(list)
    for g in load(b):
        for p, m in enumerate(g["seat_models"]):
            gain[m].append(g["gains"][str(p)])
        for m in set(g["seat_models"]):
            if g["integrative_ratio"] is not None:
                ratio[m].append(g["integrative_ratio"])
    return ({m: sum(v)/len(v) for m, v in gain.items()},
            {m: sum(v)/len(v) for m, v in ratio.items()})

g3, r3 = per_model("results/xp_3p_integrative")
g4, r4 = per_model("results/xp_4p_integrative")

ORDER = ["google/gemma-4-31b-it", "openai/gpt-5.5", "anthropic/claude-sonnet-5",
         "qwen/qwen3.5-27b", "qwen/qwen3.6-27b"]
LAB = {"google/gemma-4-31b-it": "gemma-4-31b", "openai/gpt-5.5": "gpt-5.5",
       "anthropic/claude-sonnet-5": "claude-sonnet-5", "qwen/qwen3.5-27b": "qwen3.5-27b",
       "qwen/qwen3.6-27b": "qwen3.6-27b"}
BLUE, GREEN = "#2a78d6", "#008300"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.4, 6.6), dpi=110,
                               gridspec_kw={"width_ratios": [1.35, 1]})
fig.patch.set_facecolor("#f7f7f2")
x = np.arange(len(ORDER)); w = 0.38

# --- Left: value captured (own-value gain), 3p + 4p ---
axL.set_facecolor("#f7f7f2")
for i, (gd, color, lab) in enumerate([(g3, BLUE, "3-player"), (g4, GREEN, "4-player")]):
    vals = [gd.get(m, 0) for m in ORDER]
    off = (i - 0.5) * w
    axL.bar(x + off, vals, w, color=color, label=lab, zorder=3)
    for xi, v in enumerate(vals):
        axL.text(x[xi] + off, v + 4, f"+{round(v)}", ha="center", va="bottom",
                 fontsize=11.5, fontweight="bold", color="#0b0b0b")
axL.set_ylabel("mean own-value gain (points)", fontsize=12, color="#52514e")
axL.set_title("Value captured  (skill: final − initial under own values)",
              fontsize=13, loc="left", color="#0b0b0b", pad=10)
axL.set_ylim(0, 330); axL.legend(frameon=False, fontsize=11.5, loc="upper right")
axL.set_xticks(x); axL.set_xticklabels([LAB[m] for m in ORDER], rotation=18, ha="right")

# --- Right: pie expansion when present (integrative ratio) ---
axR.set_facecolor("#f7f7f2")
for i, (rd, color, lab) in enumerate([(r3, BLUE, "3-player"), (r4, GREEN, "4-player")]):
    vals = [100*rd.get(m, 0) for m in ORDER]
    off = (i - 0.5) * w
    axR.bar(x + off, vals, w, color=color, label=lab, zorder=3)
    for xi, v in enumerate(vals):
        axR.text(x[xi] + off, v + 0.4, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=10.5, fontweight="bold", color="#0b0b0b")
axR.set_ylabel("pie captured when present (%)", fontsize=12, color="#52514e")
axR.set_title("Pie expansion when at the table  (non-zero-sum)",
              fontsize=13, loc="left", color="#0b0b0b", pad=10)
axR.set_ylim(65, 82); axR.set_xticks(x)
axR.set_xticklabels([LAB[m] for m in ORDER], rotation=18, ha="right")

for ax in (axL, axR):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(length=0)
    ax.grid(axis="y", color="#e1e0d9", lw=1, zorder=0); ax.set_axisbelow(True)

fig.suptitle("Negotiation CAPABILITY on TextArena cross-play (not win rate) — 2026-07-20",
             fontsize=15.5, x=0.012, ha="left", y=0.975)
fig.text(0.012, 0.015,
         "Left: how much value each model captures for itself (endowment-free). Right: how much the JOINT pie "
         "expands in games containing that model (shared, non-zero-sum). qwen3.6-27b tops the win-rate chart yet "
         "is LAST in value captured and lowest in pie expansion — win rate rewards passivity.",
         fontsize=9.5, color="#52514e")
plt.subplots_adjust(left=0.065, right=0.99, top=0.88, bottom=0.16, wspace=0.22)
out = "results/capability_textarena_0720.png"
fig.savefig(out, facecolor=fig.get_facecolor()); print("wrote", out)
