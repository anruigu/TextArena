#!/usr/bin/env python3
"""Frontier cross-play for TextArena Iterated Prisoner's Dilemma (2 players).

One OpenRouter model per seat. With 5 models and 10 games the schedule is a
full round-robin over all C(5,2)=10 unique pairs, so every model meets every
other exactly once; seat orientation flips on repeat passes.

Each round has free-form conversation turns followed by a simultaneous
[Cooperate]/[Defect] decision. The harness records every decision (parsed with
the SAME regex logic as the env: defect iff '[Defect]' appears, else
cooperate) so the summary can report cooperation rates and the CC/CD/DC/DD
outcome mix per model -- the interesting axis here is promise-keeping vs
betrayal after the talk phase.

Thinking traces are captured per turn FOR THE VIEWER ONLY and never fed back
into the game -- only each seat's spoken text reaches env.step() and its own
history, so reasoning never leaks to opponents.

Emits a per-game record (game_XX.json) + summary.json under --out.

Example:
  python3 run_ipd.py --games 10 --out results
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
    "You are playing an Iterated Prisoner's Dilemma against one opponent. Your "
    "goal is to MAXIMIZE YOUR OWN TOTAL SCORE across all rounds -- the winner "
    "is whoever has more points at the end. Before each decision there are "
    "free conversation turns: use them to negotiate, build trust, signal, or "
    "mislead as you see fit; everything you say is visible to the opponent. "
    "When the game announces the decision phase, reply with '[Cooperate]' or "
    "'[Defect]' (you may add brief text around the token, but include exactly "
    "one such token; if '[Defect]' appears anywhere it counts as defection). "
    "Think about round number, remaining rounds, and what your opponent said "
    "versus what they actually did in earlier rounds. Do your scheming in your "
    "reasoning; keep spoken output short and purposeful."
)

DEFECT_RE = re.compile(r"\[Defect\]", re.IGNORECASE)


def _ipd_env(cfg):
    from textarena.envs.IteratedPrisonersDilemma.env import IteratedPrisonersDilemmaEnv
    return IteratedPrisonersDilemmaEnv(
        num_rounds=cfg["num_rounds"], communication_turns=cfg["communication_turns"])


def _ipd_phase(env):
    gs = env.state.game_state
    if gs["phase"] == "conversation":
        return f"round {gs['round']}/{gs['num_rounds']} talk {gs['conversation_round'] + 1}"
    return f"round {gs['round']}/{gs['num_rounds']} decision"


def _ipd_meta(env, decisions):
    gs = env.state.game_state
    return {"final_scores": {str(p): s for p, s in gs["scores"].items()},
            "rounds_played": min(gs["round"], gs["num_rounds"]),
            "decisions": decisions}


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


async def chat(client, model, messages, temperature=0.7, max_tokens=8000, retries=4,
               reasoning=True):
    """max_tokens defaults to 8000 with retry-doubling on empty content:
    reasoning models can burn a small budget entirely on CoT and return an
    empty action (lesson from the blindauction runner). Pass reasoning=False to
    disable CoT (much faster; matches enable_thinking=false baselines)."""
    kw = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if reasoning:
        kw["extra_body"] = {"reasoning": {"enabled": True}}
    else:
        kw["extra_body"] = {"reasoning": {"enabled": False}}
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
    env = _ipd_env(cfg)
    env.reset(num_players=2, seed=seed + gid)

    histories = {pid: [{"role": "system", "content": SYSTEM}] for pid in range(2)}
    transcript = []
    decisions = []  # [{round, pid, decision}] parsed like the env parses them
    done = False
    t0 = time.time()
    steps = 0
    while not done and steps < max_steps:
        steps += 1
        pid, obs = env.get_observation()
        obs_text = render_obs(obs)
        gs = env.state.game_state
        phase = _ipd_phase(env)
        in_decision = gs["phase"] == "decision"
        rnd = gs["round"]
        histories[pid].append({"role": "user", "content": obs_text})
        model = seat_models[pid]
        content, reasoning = await chat(client, model, histories[pid], temperature,
                                        reasoning=cfg.get("reasoning", True))
        action = content or ("[Cooperate]" if in_decision else "(silence)")
        histories[pid].append({"role": "assistant", "content": action})
        if in_decision:
            decisions.append({"round": rnd, "pid": pid,
                              "decision": "defect" if DEFECT_RE.search(action) else "cooperate"})
        transcript.append({
            "step": steps, "pid": pid, "model": model, "phase": phase,
            "obs": obs_text, "action": content, "reasoning": reasoning or "",
        })
        done, _ = env.step(action)

    rewards, game_info = env.close()
    rewards = rewards or {}
    meta = _safe(_ipd_meta(env, decisions))
    winners = [p for p, r in rewards.items() if r == 1]
    reason = ""
    if isinstance(game_info, dict):
        for v in game_info.values():
            if isinstance(v, dict) and v.get("reason"):
                reason = str(v["reason"]); break
    win_label = (", ".join(f"P{p}({seat_models[p].split('/')[-1]})" for p in winners)
                 if winners else "Draw / no winner")
    return {
        "game_id": gid, "env": "ipd", "players": 2, "seed": seed + gid,
        "seat_models": seat_models,
        "rewards": {str(p): rewards.get(p) for p in range(2)},
        "final_scores": meta["final_scores"],
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
    cfg = {"num_rounds": args.num_rounds, "communication_turns": args.communication_turns,
           "reasoning": not args.no_reasoning}
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
                print(f"[ipd] game {gid} FAILED: {e!r}", file=sys.stderr, flush=True)
                return None
            (out / f"game_{gid:02d}.json").write_text(json.dumps(r, indent=1))
            print(f"[ipd] g{gid:02d} {r['seat_models'][0].split('/')[-1]} vs "
                  f"{r['seat_models'][1].split('/')[-1]} steps={r['n_steps']} "
                  f"{'TIMEOUT ' if r['timed_out'] else ''}scores={r['final_scores']} "
                  f"win={r['win_label'][:40]}", flush=True)
            return r

    results = [r for r in await asyncio.gather(*(one(g) for g in range(args.games))) if r]
    summarize(args, results, models, out)


def summarize(args, results, models, out):
    per = {m: {"games": 0, "wins": 0, "points": 0,
               "coop": 0, "defect": 0, "exploited": 0, "exploiter": 0} for m in models}
    outcome_mix = {"CC": 0, "CD_or_DC": 0, "DD": 0}
    for r in results:
        won = set(r["winners"])
        for pid in range(2):
            m = r["seat_models"][pid]
            per[m]["games"] += 1
            per[m]["wins"] += int(pid in won)
            per[m]["points"] += r["final_scores"].get(str(pid), 0)
        by_round = {}
        for d in r["meta"]["decisions"]:
            by_round.setdefault(d["round"], {})[d["pid"]] = d["decision"]
        for rnd, ds in by_round.items():
            if len(ds) < 2:
                continue
            d0, d1 = ds.get(0), ds.get(1)
            for pid, mine, theirs in ((0, d0, d1), (1, d1, d0)):
                m = r["seat_models"][pid]
                per[m]["coop" if mine == "cooperate" else "defect"] += 1
                if mine == "cooperate" and theirs == "defect":
                    per[m]["exploited"] += 1
                if mine == "defect" and theirs == "cooperate":
                    per[m]["exploiter"] += 1
            if d0 == d1 == "cooperate":
                outcome_mix["CC"] += 1
            elif d0 == d1 == "defect":
                outcome_mix["DD"] += 1
            else:
                outcome_mix["CD_or_DC"] += 1
    rate = lambda a, b: round(a / b, 3) if b else None
    summary = {
        "env": "ipd", "players": 2, "games": len(results),
        "config": {"num_rounds": args.num_rounds,
                   "communication_turns": args.communication_turns},
        "draws": sum(1 for r in results if not r["winners"]),
        "timeouts": sum(1 for r in results if r["timed_out"]),
        "round_outcome_mix": outcome_mix,
        "per_model": {m: {"games": d["games"], "wins": d["wins"],
                          "win_rate": rate(d["wins"], d["games"]),
                          "total_points": d["points"],
                          "avg_points_per_game": rate(d["points"], d["games"]),
                          "coop_rate": rate(d["coop"], d["coop"] + d["defect"]),
                          "times_exploited": d["exploited"],
                          "times_exploiter": d["exploiter"]}
                      for m, d in per.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--num-rounds", type=int, default=5, help="PD rounds per game (env default 5)")
    ap.add_argument("--communication-turns", type=int, default=3, help="talk turns before each decision")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-reasoning", action="store_true",
                    help="disable CoT/reasoning (faster; matches enable_thinking=false baseline)")
    ap.add_argument("--env-file", default="/workspace/allie/.env")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
