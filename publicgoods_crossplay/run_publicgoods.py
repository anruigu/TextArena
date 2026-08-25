#!/usr/bin/env python3
"""Frontier cross-play for TextArena Public Goods Game (N players, default 5).

Every game seats ALL models (one per seat); seats rotate by game id so no
model is pinned to one player number. Each round has simultaneous {message}
communication turns followed by a simultaneous [X] contribution decision.

Thinking traces are captured per turn FOR THE VIEWER ONLY and never fed back
into the game -- only each seat's spoken text reaches env.step() and its own
history, so reasoning never leaks to opponents. (The env additionally only
broadcasts the {curly-brace} portion of spoken text to other players.)

The summary reports per-model wins, average total score, and average
contribution -- the free-riding axis (low contribution, high score) is the
interesting behavioral readout.

Emits a per-game record (game_XX.json) + summary.json under --out.

Example:
  python3 run_publicgoods.py --games 10 --out results
"""
from __future__ import annotations
import argparse, asyncio, json, os, random, re, sys, time
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
    "You are playing a multi-player Public Goods Game. Your goal is to finish "
    "with the HIGHEST total score of any player -- score is tokens kept plus "
    "your equal share of the multiplied public pot. Note the tension: the "
    "group does best when everyone contributes, but each token you contribute "
    "returns you less than a token kept, so contributing less than others "
    "while they contribute a lot is individually profitable. Use the "
    "communication turns to coordinate, persuade, or posture -- ONLY text you "
    "wrap in curly braces {like this} is shown to other players; everything "
    "else you write stays private. During the decision phase, state your "
    "contribution as '[X]' (a number within the allowed range). Watch what "
    "others actually contributed versus what they promised, and remember the "
    "final round has no future to protect. Do your scheming in your reasoning; "
    "keep spoken output brief."
)


def _pgg_env(cfg):
    from textarena.envs.PublicGoodsGame.env import PublicGoodsGameEnv
    return PublicGoodsGameEnv(num_rounds=cfg["num_rounds"],
                              communication_turns=cfg["communication_turns"],
                              endowment=cfg["endowment"],
                              multiplication_factor=cfg["multiplication_factor"])


def _pgg_phase(env):
    gs = env.state.game_state
    if gs["phase"] == "conversation":
        return f"round {gs['round']}/{gs['num_rounds']} talk {gs['conversation_round'] + 1}"
    return f"round {gs['round']}/{gs['num_rounds']} decision"


def _pgg_meta(env):
    gs = env.state.game_state
    return {"total_scores": {str(p): s for p, s in gs["total_scores"].items()},
            "history": gs["history"],
            "eliminations": list(gs["eliminations"])}


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
    n = len(seat_models)
    env = _pgg_env(cfg)
    env.reset(num_players=n, seed=seed + gid)

    histories = {pid: [{"role": "system", "content": SYSTEM}] for pid in range(n)}
    transcript = []
    done = False
    t0 = time.time()
    steps = 0
    while not done and steps < max_steps:
        steps += 1
        pid, obs = env.get_observation()
        obs_text = render_obs(obs)
        phase = _pgg_phase(env)
        in_decision = env.state.game_state["phase"] == "decision"
        histories[pid].append({"role": "user", "content": obs_text})
        model = seat_models[pid]
        content, reasoning = await chat(client, model, histories[pid], temperature)
        action = content or ("[0]" if in_decision else "{...}")
        histories[pid].append({"role": "assistant", "content": action})
        transcript.append({
            "step": steps, "pid": pid, "model": model, "phase": phase,
            "obs": obs_text, "action": content, "reasoning": reasoning or "",
        })
        done, _ = env.step(action)

    rewards, game_info = env.close()
    rewards = rewards or {}
    meta = _safe(_pgg_meta(env))
    winners = [p for p, r in rewards.items() if r == 1]
    reason = ""
    if isinstance(game_info, dict):
        for v in game_info.values():
            if isinstance(v, dict) and v.get("reason"):
                reason = str(v["reason"]); break
    win_label = (", ".join(f"P{p}({seat_models[p].split('/')[-1]})" for p in winners)
                 if winners else "Draw / no winner")
    return {
        "game_id": gid, "env": "publicgoods", "players": n, "seed": seed + gid,
        "seat_models": seat_models,
        "rewards": {str(p): rewards.get(p) for p in range(n)},
        "total_scores": meta["total_scores"],
        "winners": winners,
        "winning_models": sorted({seat_models[p].split("/")[-1] for p in winners}),
        "win_label": win_label, "reason": reason,
        "meta": meta,
        "n_steps": steps, "timed_out": not done,
        "wall_seconds": round(time.time() - t0, 1),
        "transcript": transcript,
    }


def build_schedule(models, games):
    """All models in every game; seats rotate by game id."""
    n = len(models)
    return [[models[(i + g) % n] for i in range(n)] for g in range(games)]


async def main_async(args):
    load_env_file(args.env_file)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (see --env-file)")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                         timeout=240.0, max_retries=2)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cfg = {"num_rounds": args.num_rounds, "communication_turns": args.communication_turns,
           "endowment": args.endowment, "multiplication_factor": args.multiplication_factor}
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
                print(f"[pgg] game {gid} FAILED: {e!r}", file=sys.stderr, flush=True)
                return None
            (out / f"game_{gid:02d}.json").write_text(json.dumps(r, indent=1))
            print(f"[pgg] g{gid:02d} steps={r['n_steps']} "
                  f"{'TIMEOUT ' if r['timed_out'] else ''}scores={r['total_scores']} "
                  f"win={r['win_label'][:60]}", flush=True)
            return r

    results = [r for r in await asyncio.gather(*(one(g) for g in range(args.games))) if r]
    summarize(args, results, models, out)


def summarize(args, results, models, out):
    per = {m: {"games": 0, "wins": 0, "score": 0.0,
               "contrib": 0, "contrib_rounds": 0, "eliminated": 0} for m in models}
    for r in results:
        won = set(r["winners"])
        n = r["players"]
        for pid in range(n):
            m = r["seat_models"][pid]
            per[m]["games"] += 1
            per[m]["wins"] += int(pid in won)
            per[m]["score"] += r["total_scores"].get(str(pid), 0)
            per[m]["eliminated"] += int(pid in r["meta"]["eliminations"])
        for round_info in r["meta"]["history"]:
            for pid_s, c in round_info["contributions"].items():
                m = r["seat_models"][int(pid_s)]
                per[m]["contrib"] += c
                per[m]["contrib_rounds"] += 1
    rate = lambda a, b: round(a / b, 3) if b else None
    summary = {
        "env": "publicgoods", "players": len(models), "games": len(results),
        "config": {"num_rounds": args.num_rounds,
                   "communication_turns": args.communication_turns,
                   "endowment": args.endowment,
                   "multiplication_factor": args.multiplication_factor},
        "draws": sum(1 for r in results if not r["winners"]),
        "timeouts": sum(1 for r in results if r["timed_out"]),
        "per_model": {m: {"games": d["games"], "wins": d["wins"],
                          "win_rate": rate(d["wins"], d["games"]),
                          "avg_total_score": rate(d["score"], d["games"]),
                          "avg_contribution": rate(d["contrib"], d["contrib_rounds"]),
                          "times_eliminated": d["eliminated"]}
                      for m, d in per.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--num-rounds", type=int, default=5, help="PGG rounds per game (env default 5)")
    ap.add_argument("--communication-turns", type=int, default=3, help="talk turns before each decision")
    ap.add_argument("--endowment", type=int, default=20)
    ap.add_argument("--multiplication-factor", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--env-file", default="/workspace/allie/.env")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
