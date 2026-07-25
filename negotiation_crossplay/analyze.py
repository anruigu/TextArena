#!/usr/bin/env python3
"""Analyze cross-play negotiation batches.

For each results/<batch> dir: per-model value-gain, win rate, invalid rate,
trade conversion (did the model's offers/accepts execute), integrative gain +
ratio, and a head-to-head gain matrix (avg own-value gain of model R in games
that also contained model C). Prints tables and writes analysis.json.

Usage: python3 analyze.py results/xp_3p_integrative [results/xp_4p_integrative ...]
"""
from __future__ import annotations
import json, sys, glob
from collections import defaultdict
from pathlib import Path


def load(batch):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{batch}/game_*.json"))]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def trade_stats(g):
    """Per-seat: offers made, accepts issued, and executed trades touching seat."""
    made = defaultdict(int); acc = defaultdict(int); exe = defaultdict(int)
    for t in g["transcript"]:
        a = t["action"] or ""; pid = t["pid"]
        made[pid] += a.count("[Offer")
        acc[pid] += a.count("[Accept")
        for line in t["obs"].splitlines():
            if "ACCEPTED" in line and "Offer" in line:
                # executed trade is visible to both parties; count once per seat seeing own involvement
                exe[pid] += 1
    return made, acc, exe


def analyze_batch(batch):
    games = load(batch)
    if not games:
        return None
    per = defaultdict(lambda: {"gain": [], "win": [], "invalid": [], "offers": [], "accepts": []})
    h2h_gain = defaultdict(list)   # (row_model, col_model) -> row gains in games containing col
    intg_gain, intg_ratio = [], []
    for g in games:
        intg_gain.append(g["integrative_gain"])
        if g["integrative_ratio"] is not None:
            intg_ratio.append(g["integrative_ratio"])
        made, acc, _ = trade_stats(g)
        fv=g["final_value"]
        seat_models = g["seat_models"]
        present = set(seat_models)
        for pid, m in enumerate(seat_models):
            per[m]["gain"].append(g["gains"][str(pid)])
            per[m]["win"].append(1 if fv[str(pid)] == max(fv.values()) else 0)
            per[m]["invalid"].append(1 if g["invalid_moves"].get(str(pid)) else 0)
            per[m]["offers"].append(made[pid])
            per[m]["accepts"].append(acc[pid])
            for other in present:
                if other != m:
                    h2h_gain[(m, other)].append(g["gains"][str(pid)])
    models = sorted(per)
    summary = {
        "batch": batch, "games": len(games),
        "integrative_gain_mean": mean(intg_gain),
        "integrative_ratio_mean": mean(intg_ratio),
        "per_model": {
            m: {"n_seats": len(per[m]["gain"]), "value_gain_mean": mean(per[m]["gain"]),
                "win_rate": mean(per[m]["win"]), "invalid_rate": mean(per[m]["invalid"]),
                "offers_per_game": mean(per[m]["offers"]), "accepts_per_game": mean(per[m]["accepts"])}
            for m in models
        },
        "h2h_gain": {f"{r} vs {c}": mean(v) for (r, c), v in h2h_gain.items()},
    }
    return summary, models, h2h_gain


def print_batch(res):
    summary, models, h2h = res
    b = summary["batch"]
    print(f"\n{'='*70}\n{b}  ({summary['games']} games)  "
          f"integrative_gain={summary['integrative_gain_mean']} ratio={summary['integrative_ratio_mean']}")
    print(f"{'model':28s} {'gain':>8s} {'win':>6s} {'inval':>6s} {'off/g':>6s} {'acc/g':>6s}")
    for m in sorted(models, key=lambda x: -(summary['per_model'][x]['value_gain_mean'] or -1e9)):
        d = summary["per_model"][m]
        print(f"{m:28s} {str(d['value_gain_mean']):>8s} {str(d['win_rate']):>6s} "
              f"{str(d['invalid_rate']):>6s} {str(d['offers_per_game']):>6s} {str(d['accepts_per_game']):>6s}")
    print("\nhead-to-head avg own-value gain (row's gain in games containing col):")
    short = [m.split('/')[-1][:12] for m in models]
    print(f"{'':13s}" + "".join(f"{s:>13s}" for s in short))
    for r in models:
        row = [mean(h2h.get((r, c))) if h2h.get((r,c)) else None for c in models]
        cells = "".join(f"{('—' if (r==c or v is None) else str(v)):>13s}" for c, v in zip(models, row))
        print(f"{r.split('/')[-1][:12]:13s}{cells}")


def main():
    batches = sys.argv[1:] or glob.glob("results/xp_*")
    allsum = []
    for b in batches:
        res = analyze_batch(b)
        if res:
            print_batch(res)
            allsum.append(res[0])
            Path(f"{b}/analysis.json").write_text(json.dumps(res[0], indent=2))
    Path("results/analysis_all.json").write_text(json.dumps(allsum, indent=2))
    print(f"\nwrote per-batch analysis.json + results/analysis_all.json")


if __name__ == "__main__":
    main()
