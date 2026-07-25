#!/usr/bin/env python3
"""Cross-play Secret Mafia on TextArena's SecretMafia env.

Drives each seat with a (possibly different) frontier model via OpenRouter.
Roles are randomized by the env every game; each model rotates through seat
positions across the batch so no model is pinned to one seat.

THINKING TRACES: each model's chain-of-thought (OpenRouter `reasoning` field)
is captured and stored per turn FOR THE VIEWER ONLY. It is never fed back into
the game: opponents only ever receive a seat's *spoken* action (the `content`
that the env broadcasts), never its reasoning. A seat's own chat history also
stores content only, so reasoning cannot leak across turns or players.

Outputs, per game, a raw transcript JSON under --out (default results/), with a
separate `reasoning` string on every turn.  Convert to agentviz with
to_agentviz.py.

Example:
  python3 run_mafia.py --games 10 --players 7 --out results
"""
from __future__ import annotations
import argparse, asyncio, json, os, random, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import textarena as ta  # noqa: E402
from textarena.envs.SecretMafia.env import SecretMafiaEnv  # noqa: E402

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: pip install openai", file=sys.stderr); raise


DEFAULT_MODELS = [
    "openai/gpt-5.6-sol-pro",
    "anthropic/claude-opus-4.8",
    "deepseek/deepseek-v4-pro",
    "meta-llama/llama-4-maverick",
    "qwen/qwen3.7-max",
    "google/gemini-3.6-flash",
    "moonshotai/kimi-k3",
]

SYSTEM = (
    "You are playing Secret Mafia, a social-deduction game. Read the game "
    "messages and the public discussion carefully, reason about who is lying, "
    "and act to make YOUR team win.\n"
    "- If you are Village-aligned (Villager/Doctor/Detective): find and vote out "
    "the Mafia.\n"
    "- If you are Mafia: blend in, misdirect, and eliminate villagers.\n"
    "During DAY discussion, say something useful and persuasive (your text is "
    "broadcast to everyone). When asked to vote or take a night action, you MUST "
    "include your target in square brackets, e.g. [3] to select Player 3. Choose "
    "only from the valid targets the game lists. Keep spoken output focused; do "
    "your private scheming in your reasoning, not out loud."
)


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
    """Render TextArena observation tuples [(from_id, msg, type)] to text."""
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


async def chat(client, model, messages, temperature=0.8, max_tokens=1000, retries=4):
    # Ask OpenRouter to surface chain-of-thought so we can save it for the viewer.
    kw = {"model": model, "messages": messages, "max_tokens": max_tokens,
          "temperature": temperature, "extra_body": {"reasoning": {"enabled": True}}}
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(**kw)
            if not getattr(resp, "choices", None):
                return "", ""
            m = resp.choices[0].message
            return (m.content or "").strip(), _extract_reasoning(m)
        except Exception as e:  # noqa: BLE001
            em = str(e).lower()
            if "reasoning" in em and "extra_body" in kw:
                kw.pop("extra_body")               # model rejects reasoning param
            elif "temperature" in em and "temperature" in kw:
                kw.pop("temperature")
            elif "max_tokens" in em and "max_tokens" in kw:
                kw["max_completion_tokens"] = kw.pop("max_tokens")
            if attempt == retries - 1:
                print(f"  chat FAILED [{model}]: {e!r}", file=sys.stderr, flush=True)
                return "", ""
            await asyncio.sleep(2.0 * (attempt + 1))
    return "", ""


async def play_game(client, seat_models, players, discussion_rounds, seed, gid,
                    temperature, max_steps):
    # Seed the module RNG the env uses for role assignment / speaking order, so
    # roles are randomized yet the game is reproducible from (seed, gid).
    random.seed(seed * 100003 + gid)
    env = SecretMafiaEnv(discussion_rounds=discussion_rounds)
    env.reset(num_players=players, seed=seed + gid)
    roles = dict(env.player_roles)  # pid -> role name (ground truth)

    # Per-seat chat history holds ONLY the seat's own spoken actions + the
    # observations the env sent it. Reasoning is deliberately excluded here so it
    # never re-enters any model's context (and thus never reaches opponents).
    histories = {pid: [{"role": "system", "content": SYSTEM}] for pid in range(players)}
    transcript = []
    done = False
    t0 = time.time()
    steps = 0
    while not done and steps < max_steps:
        steps += 1
        pid, obs = env.get_observation()
        obs_text = render_obs(obs)
        phase = env.phase.value
        histories[pid].append({"role": "user", "content": obs_text})
        model = seat_models[pid]
        content, reasoning = await chat(client, model, histories[pid], temperature)
        action = content or "[No action]"
        # store only the spoken action back into this seat's own history
        histories[pid].append({"role": "assistant", "content": action})
        transcript.append({
            "step": steps, "pid": pid, "model": model, "role": roles[pid],
            "phase": phase, "day": env.state.game_state.get("day_number"),
            "obs": obs_text, "action": content, "reasoning": reasoning or "",
        })
        done, _ = env.step(action)

    rewards, _game_info = env.close()
    rewards = rewards or {}
    alive = env.state.game_state["alive_players"]
    mafia_alive = [p for p in alive if roles[p] == "Mafia"]
    # rewards are +1 for winners, -1 losers (TeamMultiPlayerState.set_winners)
    winners = [p for p, r in rewards.items() if r == 1]
    win_team = None
    if winners:
        win_team = "Mafia" if roles[winners[0]] == "Mafia" else "Village"
    reason = env.state.game_info.get(0, {}).get("reason", "")
    return {
        "game_id": gid, "players": players, "seed": seed + gid,
        "discussion_rounds": discussion_rounds,
        "seat_models": seat_models,
        "roles": {str(p): roles[p] for p in range(players)},
        "rewards": {str(p): rewards.get(p) for p in range(players)},
        "winners": winners, "win_team": win_team, "reason": reason,
        "alive_at_end": alive, "mafia_alive_at_end": mafia_alive,
        "n_steps": steps, "timed_out": not done,
        "wall_seconds": round(time.time() - t0, 1),
        "transcript": transcript,
    }


def seats_for(gid, models, players):
    """Rotate the model list by gid, then tile to fill all seats."""
    rot = models[gid % len(models):] + models[:gid % len(models)]
    return [rot[i % len(rot)] for i in range(players)]


async def main_async(args):
    load_env_file(args.env_file)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (see --env-file)")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                         timeout=240.0, max_retries=2)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(gid):
        async with sem:
            seat_models = seats_for(gid, models, args.players)
            try:
                r = await play_game(client, seat_models, args.players,
                                    args.discussion_rounds, args.seed, gid,
                                    args.temperature, args.max_steps)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"game {gid} FAILED: {e!r}", file=sys.stderr, flush=True)
                return None
            (out / f"game_{gid:02d}.json").write_text(json.dumps(r, indent=1))
            print(f"[mafia] g{gid:02d} win={r['win_team']:>7} "
                  f"steps={r['n_steps']} {'TIMEOUT ' if r['timed_out'] else ''}"
                  f"reason={r['reason'][:50]!r}", flush=True)
            return r

    results = [r for r in await asyncio.gather(*(one(g) for g in range(args.games))) if r]
    summarize(results, models, out)


def summarize(results, models, out):
    per = {m: {"games": 0, "wins": 0, "mafia_games": 0, "mafia_wins": 0,
               "village_games": 0, "village_wins": 0, "invalid_ends": 0} for m in models}
    mafia_wins = 0
    for r in results:
        won = set(r["winners"])
        for pid_s, role in r["roles"].items():
            pid = int(pid_s)
            m = r["seat_models"][pid]
            d = per[m]
            d["games"] += 1
            is_maf = role == "Mafia"
            w = pid in won
            if w:
                d["wins"] += 1
            if is_maf:
                d["mafia_games"] += 1
                d["mafia_wins"] += int(w)
            else:
                d["village_games"] += 1
                d["village_wins"] += int(w)
        if r["win_team"] == "Mafia":
            mafia_wins += 1
    rate = lambda a, b: round(a / b, 3) if b else None
    summary = {
        "games": len(results),
        "mafia_win_rate": rate(mafia_wins, len(results)),
        "village_win_rate": rate(len(results) - mafia_wins, len(results)),
        "per_model": {
            m: {
                "seats_played": d["games"],
                "overall_win_rate": rate(d["wins"], d["games"]),
                "as_mafia": f"{d['mafia_wins']}/{d['mafia_games']}",
                "as_village": f"{d['village_wins']}/{d['village_games']}",
            } for m, d in per.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated OpenRouter model ids")
    ap.add_argument("--players", type=int, default=7)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--discussion-rounds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--env-file", default="~/.env")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
