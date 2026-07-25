#!/usr/bin/env python3
"""Adapt blindauction_crossplay game JSONs to the uniform xplay record schema
(run_xplay.py-style) so mafia_crossplay/to_agentviz_general.py can convert them
into ATIF sessions for the agentviz viewer on :8731.

Usage: ba_to_xplay_records.py  (writes results/xplay_adapted/r{1,3,6}/game_NN.json)
"""
import json, glob
from pathlib import Path

HERE = Path(__file__).resolve().parent


def short(model):
    return (model or "").split("/")[-1]


def adapt(g):
    rounds = g["conversation_rounds"]
    per_seat = {int(p): d for p, d in g["per_seat"].items()}
    winners = [p for p, d in per_seat.items() if d["win"]]
    if winners:
        w = winners[0]
        win_label = f"P{w}({short(per_seat[w]['model'])}) profit +{per_seat[w]['profit']}"
    else:
        win_label = "draw (tied net worth)"
    reason = ("Net worth: " + ", ".join(
        f"P{p}={d['net_worth']}" for p, d in sorted(per_seat.items()))
        + f" | efficiency={g['allocative_efficiency']} revenue={g['revenue']}"
        + f" unsold={g['items_unsold']}")
    return {
        "game_id": g["game_id"],
        "env": "blindauction",
        "players": g["players"],
        "seed": g["seed"],
        "game_config": f"r{rounds}",
        "seat_models": g["seat_models"],
        "seat_roles": {},
        "rewards": g["builtin_rewards"],
        "winners": winners,
        "winning_models": sorted({short(per_seat[p]["model"]) for p in winners}),
        "win_label": win_label,
        "reason": reason,
        "meta": {
            "conversation_rounds": rounds,
            "item_names": g["item_names"],
            "player_item_values": g["player_item_values"],
            "player_bids": g["player_bids"],
            "item_winners": g["item_winners"],
            "winning_bids": g["winning_bids"],
            "profits": {p: d["profit"] for p, d in sorted(per_seat.items())},
            "allocative_efficiency": g["allocative_efficiency"],
            "revenue": g["revenue"],
        },
        "n_steps": len(g["transcript"]),
        "timed_out": g["truncated"],
        "wall_seconds": g["wall_seconds"],
        "transcript": g["transcript"],
    }


for r in [1, 3, 6]:
    outdir = HERE / "results" / "xplay_adapted" / f"r{r}"
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in sorted(glob.glob(str(HERE / "results" / f"ba_7p_r{r}" / "game_*.json"))):
        g = json.load(open(fp))
        rec = adapt(g)
        (outdir / f"game_{rec['game_id']:02d}.json").write_text(
            json.dumps(rec, ensure_ascii=False))
        n += 1
    print(f"r{r}: adapted {n} games -> {outdir}")
