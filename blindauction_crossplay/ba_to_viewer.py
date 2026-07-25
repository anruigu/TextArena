#!/usr/bin/env python3
"""Port blindauction_crossplay game JSONs into the SkyRL trace viewer.

One viewer row per SEAT per game: the row's text is that seat's full chat
stream (game observations as user turns, the model's actions as assistant
turns, with saved reasoning restored as a <think> block so the viewer
highlights it). reward = seat profit; stop_reason = win/tie/loss.

Runs are stored one alias per sweep config, pseudo-step = conversation_rounds
(so the viewer's step slider walks the rounds sweep).

Usage: ba_to_viewer.py <results_dir e.g. results/ba_7p_r3> <alias>
Then:  python SkyRL-Fleet/tools/trace-viewer/build_manifest.py
"""
import json, sys, glob
from pathlib import Path

VIEWER_DATA = Path("/workspace/allie/SkyRL-Fleet/tools/trace-viewer/public/data")

results_dir, alias = sys.argv[1], sys.argv[2]
games = sorted(glob.glob(str(Path(results_dir) / "game_*.json")))
if not games:
    print(f"NO game files in {results_dir!r}"); sys.exit(2)

rows = []
step = None
for fp in games:
    g = json.load(open(fp))
    step = g["conversation_rounds"]
    players = g["players"]
    net = {p: g["per_seat"][str(p)]["net_worth"] for p in range(players)}
    best = max(net.values())
    for pid in range(players):
        seat = g["per_seat"][str(pid)]
        parts = []
        vals = g["player_item_values"][str(pid)]
        header = (f"BlindAuction game {g['game_id']} seat {pid} — {seat['model']}\n"
                  f"items={g['item_names']}  your_values={vals}  capital=1000")
        parts.append(f"<|im_start|>user\n{header}<|im_end|>")
        for t in g["transcript"]:
            if t["pid"] != pid:
                continue
            parts.append(f"<|im_start|>user\n{t['obs']}<|im_end|>")
            tx = t["action"] or "(empty)"
            if t.get("reasoning"):
                tx = f"<think>\n{t['reasoning']}\n</think>\n{tx}"
            elif "<think>" not in tx:
                tx = "<think>\n</think>\n" + tx
            parts.append(f"<|im_start|>assistant\n{tx}<|im_end|>")
        outcome = "win" if seat["win"] else ("tie" if net[pid] == best else "loss")
        rows.append({
            "env_key": "blindauction", "data_source": "blindauction_xp",
            "stop_reason": outcome,
            "reward": seat["profit"],
            "turns": sum(1 for t in g["transcript"] if t["pid"] == pid),
            "prompt": (f"{seat['model']} | game {g['game_id']} seat {pid} | r={step} | "
                       f"profit={seat['profit']} net={seat['net_worth']} items={seat['items_won']} "
                       f"spent={seat['spent']} overbids={seat['overbids']}"),
            "text": "\n".join(parts),
            "outcome_info": {
                "model": seat["model"], "profit": seat["profit"],
                "net_worth": seat["net_worth"], "items_won": seat["items_won"],
                "item_winners": g["item_winners"], "winning_bids": g["winning_bids"],
                "allocative_efficiency": g["allocative_efficiency"],
                "conversation_rounds": step,
            },
        })

outdir = VIEWER_DATA / alias
outdir.mkdir(parents=True, exist_ok=True)
# Only replace this config's step file — the same alias accumulates one
# step per conversation_rounds config, so the viewer slider walks the sweep.
with open(outdir / f"global_step_{step}.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        r2 = dict(r); r2["step"] = step
        f.write(json.dumps(r2, ensure_ascii=False) + "\n")
print(f"{alias}: {len(rows)} seat-traces (step={step}) in {outdir}")
