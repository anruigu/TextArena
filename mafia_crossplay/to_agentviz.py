#!/usr/bin/env python3
"""Convert raw Secret Mafia game transcripts (run_mafia.py output) into ATIF-v1.6
session files that agentviz can replay, plus a manifest.json.

Each player turn becomes an ATIF "agent" step carrying that seat's model_name,
its reasoning_content (thinking trace), the observation it saw, and its spoken
action as the message. GAME/setup/outcome lines become system steps.

Usage:
  python3 to_agentviz.py --in results --out /workspace/allie/agentviz/dist/sessions/mafia
"""
from __future__ import annotations
import argparse, glob, json
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc)


def ts(i):
    return (BASE + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")


def short(model):
    return model.split("/")[-1]


def to_atif(game):
    gid = game["game_id"]
    roles = game["roles"]
    seat_models = game["seat_models"]
    steps = []
    sid = 0

    def add(source, message, **extra):
        nonlocal sid
        sid += 1
        steps.append({"step_id": sid, "timestamp": ts(sid), "source": source,
                      "message": message, **extra})

    roster = "\n".join(
        f"  Player {p}: {roles[str(p)]:9}  [{short(seat_models[p])}]"
        for p in range(game["players"]))
    add("system", f"─── Secret Mafia · game {gid} · {game['players']} players ───\n"
                  f"Seat assignment (role · model):\n{roster}")

    for t in game["transcript"]:
        pid = t["pid"]
        header = f"Player {pid} · {t['role']} · {t['phase']} (day {t.get('day')})"
        msg = t["action"] or "(no spoken action)"
        add("agent",
            f"{header}\n{msg}",
            model_name=t["model"],
            reasoning_content=t.get("reasoning") or "",
            observation=t.get("obs") or "")

    add("system", f"─── RESULT ───\nWinner: {game['win_team']}\n{game['reason']}\n"
                  f"Winning seats: {game['winners']}"
                  + ("\n(reached max steps / timeout)" if game.get("timed_out") else ""))

    win_models = sorted({short(seat_models[p]) for p in game["winners"]})
    return {
        "schema_version": "ATIF-v1.6",
        "session_id": f"mafia-game-{gid:02d}",
        "agent": {
            "name": f"Secret Mafia cross-play · game {gid}",
            "model_name": "multi (7 frontier models)",
            "version": "1.0",
            "tool_definitions": [],
            "extra": {
                "env_key": "secret_mafia",
                "win_team": game["win_team"],
                "reason": game["reason"],
                "roles": roles,
                "seat_models": {str(p): seat_models[p] for p in range(game["players"])},
                "winners": game["winners"],
                "winning_models": win_models,
                "timed_out": game.get("timed_out", False),
            },
        },
        "steps": steps,
        "final_metrics": {"total_steps": len(steps), "player_turns": len(game["transcript"])},
        "notes": f"Secret Mafia cross-play game {gid}. Winner: {game['win_team']}. "
                 f"{game['reason']}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="results")
    ap.add_argument("--out", dest="outdir",
                    default="/workspace/allie/agentviz/dist/sessions/mafia")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(Path(args.indir) / "game_*.json")))
    if not files:
        raise SystemExit(f"no game_*.json in {args.indir}")

    sessions = []
    for fp in files:
        game = json.loads(Path(fp).read_text())
        atif = to_atif(game)
        gid = game["game_id"]
        fn = f"mafia_game_{gid:02d}.atif.json"
        (outdir / fn).write_text(json.dumps(atif, indent=1, ensure_ascii=False))
        win = game["win_team"] or "?"
        win_models = atif["agent"]["extra"]["winning_models"]
        sessions.append({
            "id": f"mafia-game-{gid:02d}",
            "name": f"Secret Mafia · game {gid} · {win} win",
            "url": fn,
            "format": "atif",
            "mtime": int((BASE + timedelta(seconds=gid)).timestamp() * 1000),
            "tags": [f"game {gid}", f"{win} win", *win_models],
        })

    manifest = {
        "generated": ts(0),
        "title": "Secret Mafia cross-play — 7 frontier models",
        "n_sessions": len(sessions),
        "sessions": sessions,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {len(sessions)} ATIF sessions + manifest.json -> {outdir}")


if __name__ == "__main__":
    main()
