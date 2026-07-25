#!/usr/bin/env python3
"""Table-win-rate bar chart for TextArena 3-player cross-play (integrative vs stock)."""
import json, glob
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def winrates(batch):
    ap = defaultdict(int); win = defaultdict(int)
    for f in glob.glob(f"{batch}/game_*.json"):
        g = json.load(open(f))
        fv = {p: g["final_value"][str(p)] for p in range(g["players"])}
        top = max(fv.values()); winners = [p for p, v in fv.items() if v == top]
        for p, m in enumerate(g["seat_models"]):
            ap[m] += 1
            if len(winners) == 1 and winners[0] == p:
                win[m] += 1
    return ap, win

REGIMES = [("integrative", "results/xp_3p_integrative", "#2a78d6"),
           ("stock", "results/xp_3p_stock", "#008300")]
ORDER = ["openai/gpt-5.5", "anthropic/claude-sonnet-5", "google/gemma-4-31b-it",
         "qwen/qwen3.5-27b", "qwen/qwen3.6-27b"]
LABELS = {"openai/gpt-5.5": "gpt-5.5", "anthropic/claude-sonnet-5": "claude-sonnet-5",
          "google/gemma-4-31b-it": "gemma-4-31b", "qwen/qwen3.5-27b": "qwen3.5-27b",
          "qwen/qwen3.6-27b": "qwen3.6-27b"}

data = {}
for name, b, _ in REGIMES:
    data[name] = winrates(b)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
fig, ax = plt.subplots(figsize=(12.6, 6.6), dpi=110)
fig.patch.set_facecolor("#f7f7f2"); ax.set_facecolor("#f7f7f2")

x = np.arange(len(ORDER)); w = 0.38
for i, (name, b, color) in enumerate(REGIMES):
    ap, win = data[name]
    rates, fracs = [], []
    for m in ORDER:
        a = ap.get(m, 0); wn = win.get(m, 0)
        rates.append(100 * wn / a if a else 0); fracs.append(f"{wn}/{a}" if a else "0/0")
    off = (i - 0.5) * w
    bars = ax.bar(x + off, rates, w, color=color, label=name, zorder=3)
    for xi, (r, fr) in enumerate(zip(rates, fracs)):
        ax.text(x[xi] + off, r + 1.4, f"{round(r)}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#0b0b0b")
        ax.text(x[xi] + off, r + 0.1, fr, ha="center", va="bottom",
                fontsize=9.5, color="#898781")

ax.axhline(100/3, ls="--", color="#52514e", lw=1.3, zorder=2)
ax.text(-0.45, 100/3 + 0.6, "chance (3 players) 33%",
        ha="left", va="bottom", fontsize=11, color="#52514e")

ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in ORDER], fontsize=12)
ax.set_ylabel("table-win rate (%)", fontsize=12, color="#52514e")
ax.set_ylim(0, 68); ax.set_yticks(range(0, 61, 10))
ax.set_title("Who wins the 3-party negotiation table? TextArena cross-play, 2026-07-20",
             fontsize=15, pad=16, loc="left", color="#0b0b0b")
ax.legend(frameon=False, fontsize=12, loc="upper left", ncol=2)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(length=0)
ax.grid(axis="y", color="#e1e0d9", lw=1, zorder=0)
ax.set_axisbelow(True)

fig.text(0.012, 0.02,
         "win = strictly highest final inventory value in a game (endowment-influenced — value-GAIN is the "
         "skill metric; see FINDINGS_0720). integrative 40 games / stock 24 games, 5-way seat rotation "
         "(~24 / ~14 appearances per model).",
         fontsize=9.5, color="#52514e")
plt.subplots_adjust(left=0.075, right=0.985, top=0.9, bottom=0.13)
out = "results/winrates_textarena_0720.png"
fig.savefig(out, facecolor=fig.get_facecolor())
print("wrote", out)
