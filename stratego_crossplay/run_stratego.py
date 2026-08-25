#!/usr/bin/env python3
"""Frontier cross-play for TextArena Stratego (2 players).

One OpenRouter model per seat. With 5 models and 10 games the schedule is a
full round-robin over all C(5,2)=10 unique pairs, so every model meets every
other exactly once; seat orientation flips on repeat passes.

Thinking traces are captured per turn FOR THE VIEWER ONLY and never fed back
into the game -- only each seat's spoken bracket move reaches env.step() and
its own history, so reasoning never leaks to opponents.

Stratego has no in-env turn cap, so the harness enforces --max-steps; a game
that hits the cap is recorded as a timeout with the movable-material leader
noted in meta (counted as a draw in the summary).

Emits a per-game record (game_XX.json) + summary.json under --out.

Example:
  python3 run_stratego.py --games 10 --out results
"""
from __future__ import annotations
import argparse, asyncio, itertools, json, os, random, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import textarena as ta  # noqa: E402

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: pip install openai", file=sys.stderr); raise


DEFAULT_MODELS = [
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.2",
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.6-sol-pro",
]

SYSTEM = (
    "You are a world-class Stratego player. Read the board carefully: your own "
    "pieces are shown by rank abbreviation, opponent pieces as '?', lakes as '~'. "
    "Track what you learn about opponent pieces from battles and movement (a "
    "piece that moves is not a Bomb or Flag; a multi-square mover is a Scout). "
    "Protect your Flag, probe with expendable pieces, and attack only when the "
    "odds favor you. CRITICAL: respond with EXACTLY ONE move in the format "
    "'[A0 B0]' (source square then destination square), and it MUST be one of "
    "the moves in the 'Available Moves' list the game shows you -- an invalid "
    "move can forfeit the game. Do your thinking in your reasoning; your spoken "
    "output must be just the single bracketed move."
)


def _stratego_env(cfg):
    from textarena.envs.Stratego.env import StrategoEnv
    return StrategoEnv()


def _stratego_phase(env):
    return f"turn {env.state.turn}"


def _piece_counts(env):
    counts = {0: {"pieces": 0, "movable": 0}, 1: {"pieces": 0, "movable": 0}}
    for row in env.board:
        for cell in row:
            if isinstance(cell, dict):
                p = cell["player"]
                counts[p]["pieces"] += 1
                if cell["rank"] not in ("Bomb", "Flag"):
                    counts[p]["movable"] += 1
    return counts


def _stratego_meta(env):
    counts = _piece_counts(env)
    m0, m1 = counts[0]["movable"], counts[1]["movable"]
    leader = None if m0 == m1 else (0 if m0 > m1 else 1)
    return {"piece_counts": {str(p): c for p, c in counts.items()},
            "movable_material_leader": leader,
            "turns": env.state.turn}


def load_env_file(path="~/.env"):
    p = Path(path).expanduser()
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = re.sub(r"^export\s+", "", line)
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("\"'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def render_obs(obs) -> str:
    if isinstance(obs, str):
        return obs
    lines = []
    for entry in obs:
        frm, msg = entry[0], entry[1]
        who = "GAME" if frm == ta.GAME_ID else f"Player {frm}"
        lines.append(f"[{who}] {msg}")
    return "\n".join(lines)


def _extract_reasoning(msg):
    r = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if not r:
        r = (getattr(msg, "model_extra", None) or {}).get("reasoning")
    return (r or "").strip()


async def chat(client, model, messages, temperature=0.7, max_tokens=8000, retries=4):
    """max_tokens defaults to 8000 with retry-doubling on empty content:
    reasoning models can burn a small budget entirely on CoT and return an
    empty action (lesson from the blindauction runner)."""
    kw = {"model": model, "messages": messages, "max_tokens": max_tokens,
          "extra_body": {"reasoning": {"enabled": True}}}
    if temperature is not None:
        kw["temperature"] = temperature
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(**kw)
            if not getattr(resp, "choices", None):
                return "", ""
            m = resp.choices[0].message
            content = (m.content or "").strip()
            if not content and attempt < retries - 1:
                if "max_tokens" in kw:
                    kw["max_tokens"] = min(kw["max_tokens"] * 2, 32000)
                elif "max_completion_tokens" in kw:
                    kw["max_completion_tokens"] = min(kw["max_completion_tokens"] * 2, 32000)
                continue
            return content, _extract_reasoning(m)
        except Exception as e:  # noqa: BLE001
            em = str(e).lower()
            if "reasoning" in em and "extra_body" in kw:
                kw.pop("extra_body")
            elif "temperature" in em and "temperature" in kw:
                kw.pop("temperature")
            elif "max_tokens" in em and "max_tokens" in kw:
                kw["max_completion_tokens"] = kw.pop("max_tokens")
            if attempt == retries - 1:
                print(f"  chat FAILED [{model}]: {e!r}", file=sys.stderr, flush=True)
                return "", ""
            await asyncio.sleep(2.0 * (attempt + 1))
    return "", ""


def _safe(obj):
    try:
        json.dumps(obj); return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        return str(obj)


async def play_game(client, cfg, seat_models, seed, gid, temperature, max_steps):
    random.seed(seed * 100003 + gid)
    env = _stratego_env(cfg)
    env.reset(num_players=2, seed=seed + gid)

    histories = {pid: [{"role": "system", "content": SYSTEM}] for pid in range(2)}
    transcript = []
    done = False
    t0 = time.time()
    steps = 0
    while not done and steps < max_steps:
        steps += 1
        pid, obs = env.get_observation()
        obs_text = render_obs(obs)
        phase = _stratego_phase(env)
        histories[pid].append({"role": "user", "content": obs_text})
        model = seat_models[pid]
        content, reasoning = await chat(client, model, histories[pid], temperature)
        action = content or "[No action]"
        histories[pid].append({"role": "assistant", "content": action})
        transcript.append({
            "step": steps, "pid": pid, "model": model, "phase": phase,
            "obs": obs_text, "action": content, "reasoning": reasoning or "",
        })
        done, _ = env.step(action)

    rewards, game_info = env.close()
    rewards = rewards or {}
    meta = _safe(_stratego_meta(env))
    winners = [p for p, r in rewards.items() if r == 1]
    reason = ""
    if isinstance(game_info, dict):
        for v in game_info.values():
            if isinstance(v, dict) and v.get("reason"):
                reason = str(v["reason"]); break
    win_label = (", ".join(f"P{p}({seat_models[p].split('/')[-1]})" for p in winners)
                 if winners else "Draw / no winner")
    return {
        "game_id": gid, "env": "stratego", "players": 2, "seed": seed + gid,
        "seat_models": seat_models,
        "rewards": {str(p): rewards.get(p) for p in range(2)},
        "winners": winners,
        "winning_models": sorted({seat_models[p].split("/")[-1] for p in winners}),
        "win_label": win_label, "reason": reason,
        "meta": meta,
        "n_steps": steps, "timed_out": not done,
        "wall_seconds": round(time.time() - t0, 1),
        "transcript": transcript,
    }


def build_schedule(models, games):
    """Round-robin over unique pairs; seat orientation flips on repeats."""
    pairs = list(itertools.combinations(range(len(models)), 2))
    sched = []
    for g in range(games):
        i, j = pairs[g % len(pairs)]
        if (g // len(pairs)) % 2 == 1:
            i, j = j, i  # flip seats on the second pass through the pairs
        sched.append([models[i], models[j]])
    return sched


async def main_async(args):
    load_env_file(args.env_file)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (see --env-file)")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                         timeout=240.0, max_retries=2)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cfg = {}
    schedule = build_schedule(models, args.games)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(gid):
        async with sem:
            seat_models = schedule[gid]
            try:
                r = await play_game(client, cfg, seat_models, args.seed, gid,
                                    args.temperature, args.max_steps)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"[stratego] game {gid} FAILED: {e!r}", file=sys.stderr, flush=True)
                return None
            (out / f"game_{gid:02d}.json").write_text(json.dumps(r, indent=1))
            print(f"[stratego] g{gid:02d} {r['seat_models'][0].split('/')[-1]} vs "
                  f"{r['seat_models'][1].split('/')[-1]} steps={r['n_steps']} "
                  f"{'TIMEOUT ' if r['timed_out'] else ''}win={r['win_label'][:40]} "
                  f"reason={r['reason'][:60]}", flush=True)
            return r

    results = [r for r in await asyncio.gather(*(one(g) for g in range(args.games))) if r]
    summarize(args, results, models, out)


def summarize(args, results, models, out):
    per = {m: {"games": 0, "wins": 0, "material_led": 0} for m in models}
    for r in results:
        won = set(r["winners"])
        leader = r["meta"].get("movable_material_leader")
        for pid in range(2):
            m = r["seat_models"][pid]
            per[m]["games"] += 1
            per[m]["wins"] += int(pid in won)
            per[m]["material_led"] += int(leader == pid)
    rate = lambda a, b: round(a / b, 3) if b else None
    summary = {
        "env": "stratego", "players": 2, "games": len(results),
        "config": {"max_steps": args.max_steps},
        "draws": sum(1 for r in results if not r["winners"]),
        "timeouts": sum(1 for r in results if r["timed_out"]),
        "per_model": {m: {"games": d["games"], "wins": d["wins"],
                          "win_rate": rate(d["wins"], d["games"]),
                          "games_ended_as_material_leader": d["material_led"]}
                      for m, d in per.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--env-file", default="/workspace/allie/.env")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
