#!/usr/bin/env python3
"""Convert uniform cross-play game records (run_xplay.py / run_mafia.py output)
into ATIF-v1.6 sessions + one combined manifest.json for agentviz.

Each player turn -> one ATIF "agent" step carrying that seat's model_name,
reasoning_content (thinking trace), the observation it saw, and its spoken
action. Setup + result become system steps.

Usage:
  python3 to_agentviz_general.py --inputs results_coup,results_nr,results_sg2p,results_sg6p,results \
      --out /workspace/allie/agentviz/dist/sessions/xplay
"""
from __future__ import annotations
import argparse, glob, json
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc)


def ts(i):
    return (BASE + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")


def short(model):
    return (model or "").split("/")[-1]


def win_label_of(game):
    """Xplay records carry win_label; raw mafia records carry win_team."""
    return game.get("win_label") or game.get("win_team") or "?"


def winning_models_of(game):
    wm = game.get("winning_models")
    if wm:
        return wm
    seat_models = game.get("seat_models") or []
    winners = game.get("winners") or []
    return sorted({short(seat_models[p]) for p in winners
                   if isinstance(p, int) and 0 <= p < len(seat_models)})


def to_atif(game, uid):
    # Tolerate both run_xplay.py (env/win_label/seat_roles) and raw
    # run_mafia.py (no env, win_team, roles) record schemas.
    env = game.get("env") or "secret_mafia"
    cfg = game.get("game_config")
    players = game["players"]
    roles = game.get("seat_roles") or game.get("roles") or {}
    seat_models = game["seat_models"]
    win_label = win_label_of(game)
    winning_models = winning_models_of(game)
    steps = []
    sid = 0

    def add(source, message, **extra):
        nonlocal sid
        sid += 1
        steps.append({"step_id": sid, "timestamp": ts(sid), "source": source,
                      "message": message, **extra})

    def rlabel(p):
        r = roles.get(str(p))
        return f"{r} · " if r else ""

    roster = "\n".join(
        f"  Player {p}: {rlabel(p)}{short(seat_models[p])}" for p in range(players))
    title = f"{env}" + (f"/{cfg}" if cfg else "")
    add("system", f"─── {title} · game {game['game_id']} · {players} players ───\n"
                  f"Seat assignment:\n{roster}")

    for t in game["transcript"]:
        pid = t["pid"]
        role = t.get("role")
        phase = t.get("phase") or ""
        head = f"Player {pid}" + (f" · {role}" if role else "") + (f" · {phase}" if phase else "")
        msg = t["action"] or "(no spoken action)"
        add("agent", f"{head}\n{msg}",
            model_name=t["model"],
            reasoning_content=t.get("reasoning") or "",
            observation=t.get("obs") or "")

    add("system", f"─── RESULT ───\nWinner(s): {win_label}\n{game.get('reason','')}"
                  + ("\n(reached max steps / timeout)" if game.get("timed_out") else ""))

    return {
        "schema_version": "ATIF-v1.6",
        "session_id": uid,
        "agent": {
            "name": f"{title} cross-play · game {game['game_id']}",
            "model_name": "multi (7 frontier models)",
            "version": "1.0",
            "tool_definitions": [],
            "extra": {
                "env_key": env, "game_config": cfg,
                "win_label": win_label, "reason": game.get("reason", ""),
                "seat_roles": roles,
                "seat_models": {str(p): seat_models[p] for p in range(players)},
                "winners": game["winners"], "winning_models": winning_models,
                "timed_out": game.get("timed_out", False),
                "meta": game.get("meta", {}),
            },
        },
        "steps": steps,
        "final_metrics": {"total_steps": len(steps), "player_turns": len(game["transcript"])},
        "notes": f"{title} cross-play game {game['game_id']}. Winner: {win_label}.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="comma-separated result dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="TextArena frontier cross-play")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    sessions = []
    n = 0
    for d in [x.strip() for x in args.inputs.split(",") if x.strip()]:
        for fp in sorted(glob.glob(str(Path(d) / "game_*.json"))):
            game = json.loads(Path(fp).read_text())
            env = game.get("env") or "secret_mafia"
            cfg = game.get("game_config")
            tag = env + (f"_{cfg}" if cfg else "")
            uid = f"{tag}-g{game['game_id']:02d}"
            atif = to_atif(game, uid)
            fn = f"{uid}.atif.json"
            (outdir / fn).write_text(json.dumps(atif, indent=1, ensure_ascii=False))
            win_label = win_label_of(game)
            tags = [env]
            if cfg:
                tags.append(cfg)
            tags += winning_models_of(game) or ["draw"]
            sessions.append({
                "id": uid,
                "name": f"{env}" + (f"/{cfg}" if cfg else "") + f" · g{game['game_id']} · {win_label[:40]}",
                "url": fn, "format": "atif",
                "mtime": int((BASE + timedelta(seconds=n)).timestamp() * 1000),
                "tags": tags,
            })
            n += 1

    manifest = {"generated": ts(0), "title": args.title,
                "n_sessions": len(sessions), "sessions": sessions}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {len(sessions)} ATIF sessions + manifest.json -> {outdir}")


if __name__ == "__main__":
    main()
