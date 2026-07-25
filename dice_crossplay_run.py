#!/usr/bin/env python3
"""Frontier cross-play for two hidden-info card/dice games: KuhnPoker and LiarsDice.

Same harness contract as poker_crossplay/run_poker.py:
  * one OpenRouter model per seat, heads-up (2 players);
  * with 5 models and 10 games the schedule is a full round-robin over all
    C(5,2)=10 unique pairs, seats flipped on repeats so position is balanced;
  * thinking traces are captured per turn FOR THE VIEWER ONLY and never fed back
    into the game -- only each seat's spoken bracket action reaches env.step().

The one addition over the poker runner: each turn's record carries the acting
player's GROUND-TRUTH private state (Kuhn card / Liar's dice + the bid it faces),
read straight from env.state.game_state at decision time. That is what the
Term-2 leakage reader consumes downstream -- no regex on the obs, no LLM.

Emits per-game game_XX.json + summary.json under --out.

Example:
  python3 dice_crossplay_run.py --game kuhn  --games 10 --out kuhn_crossplay/results
  python3 dice_crossplay_run.py --game liars --games 10 --out liarsdice_crossplay/results
"""
from __future__ import annotations
import argparse, asyncio, itertools, json, os, random, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
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

SYSTEM = {
    "kuhn": (
        "You are a world-class poker player competing heads-up in Kuhn Poker, a "
        "minimal 3-card poker (deck J<Q<K, one card each, one ante). Read your card "
        "and the action so far, reason about your opponent's likely holding, and play "
        "to win the most chips over all rounds -- value-bet strong cards, bluff or "
        "check-call weak ones when the spot is right, fold when beaten. Respond with "
        "EXACTLY ONE of the bracketed tokens the game offers you this turn: '[Check]', "
        "'[Bet]', '[Call]', or '[Fold]'. Do all reasoning privately; your spoken "
        "output must be just the single action token."
    ),
    "liars": (
        "You are a world-class Liar's Dice player in a heads-up match. Each of you "
        "starts with five dice, hidden from the other. On your turn you either raise "
        "to a strictly higher bid -- '[Bid: <quantity>, <face>]', a claim about how "
        "many of that face are on the table across BOTH players' dice -- or challenge "
        "the last bid with '[Call]'. If a called bid's true count falls short the "
        "bidder loses a die, otherwise the caller does; lose all your dice and you are "
        "out. Use your own dice, the bid history, and the odds to bluff, trap, and "
        "call. Respond with EXACTLY ONE bracketed action: '[Bid: q, f]' or '[Call]'. "
        "Do all reasoning privately; your spoken output must be just the action token."
    ),
    "leduc": (
        "You are a world-class poker player in heads-up Leduc Hold'em (6-card deck: "
        "J J Q Q K K). You ante 1 and get one private card; there is a pre-flop betting "
        "round (bet size 2), then one public board card is revealed and a post-flop "
        "round (bet size 4), max 2 raises per round. At showdown a pair with the board "
        "beats a high card, else higher card wins. Read your card and the board, reason "
        "about your opponent, and play to win chips over the match -- value-bet strong "
        "cards, bluff or fold weak ones when the spot is right. Respond with EXACTLY ONE "
        "of the bracketed tokens offered this turn: '[check]', '[bet]', '[call]', "
        "'[raise]', or '[fold]'. Do all reasoning privately; speak only the action token."
    ),
}


def build_env(game, cfg):
    if game == "kuhn":
        from textarena.envs.KuhnPoker.env import KuhnPokerEnv
        return KuhnPokerEnv(max_rounds=cfg["rounds"])
    if game == "leduc":
        from textarena.envs.LeducHoldem.env import LeducHoldemEnv
        return LeducHoldemEnv(max_rounds=cfg["rounds"])
    from textarena.envs.LiarsDice.env import LiarsDiceEnv
    return LiarsDiceEnv(num_dice=cfg["num_dice"])


def private_state(game, env, pid):
    """Ground-truth hidden state of the player about to act (for the leakage reader)."""
    gs = env.state.game_state
    if game == "kuhn":
        return {"card": gs["player_cards"][pid]}  # 0=J 1=Q 2=K
    if game == "leduc":
        return {"card": gs["player_cards"][pid],           # 0=J 1=Q 2=K
                "board_card": gs["board_card"],             # None pre-flop
                "round": gs["round"], "pot": gs["pot"], "current_bet": gs["current_bet"]}
    return {"dice": list(gs["dice_rolls"][pid]),
            "facing_bid": dict(gs["current_bid"]),
            "remaining_dice": gs["remaining_dice"][pid]}


def phase_of(game, env):
    gs = env.state.game_state
    if game == "kuhn":
        return f"round {gs['current_round']}"
    if game == "leduc":
        return f"hand{gs.get('hands_dealt', 0)}/rd{gs['round']}"
    return f"dice {gs['remaining_dice']}"


def game_meta(game, env):
    gs = env.state.game_state
    if game == "kuhn":
        return {"final_chips": {str(p): c for p, c in gs["player_chips"].items()},
                "rounds": gs["current_round"]}
    if game == "leduc":
        return {"final_bank": {str(p): b for p, b in gs["player_bank"].items()},
                "hands_dealt": gs.get("hands_dealt", 0)}
    return {"final_remaining_dice": {str(p): c for p, c in gs["remaining_dice"].items()},
            "elimination_order": list(env.state.elimination_order)}


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
    """max_tokens defaults to 8000 with retry-doubling on empty content: reasoning
    models can burn a small budget entirely on CoT and return an empty action."""
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


async def play_game(client, game, cfg, seat_models, seed, gid, temperature, max_steps):
    random.seed(seed * 100003 + gid)
    env = build_env(game, cfg)
    env.reset(num_players=2, seed=seed + gid)

    histories = {pid: [{"role": "system", "content": SYSTEM[game]}] for pid in range(2)}
    transcript = []
    done = False
    t0 = time.time()
    steps = 0
    while not done and steps < max_steps:
        steps += 1
        pid, obs = env.get_observation()
        obs_text = render_obs(obs)
        priv = _safe(private_state(game, env, pid))
        phase = phase_of(game, env)
        histories[pid].append({"role": "user", "content": obs_text})
        model = seat_models[pid]
        content, reasoning = await chat(client, model, histories[pid], temperature)
        action = content or "[No action]"
        histories[pid].append({"role": "assistant", "content": action})
        transcript.append({
            "step": steps, "pid": pid, "model": model, "phase": phase,
            "private": priv, "obs": obs_text, "action": content,
            "reasoning": reasoning or "",
        })
        done, _ = env.step(action)

    rewards, game_info = env.close()
    rewards = rewards or {}
    meta = _safe(game_meta(game, env))
    winners = [p for p, r in rewards.items() if r == max(rewards.values())] if rewards else []
    reason = ""
    if isinstance(game_info, dict):
        reason = (game_info.get(0) or {}).get("reason", "") if 0 in game_info else ""
    win_label = (", ".join(f"P{p}({seat_models[p].split('/')[-1]})" for p in winners)
                 if winners else "Draw / no winner")
    return {
        "game_id": gid, "env": game, "players": 2, "seed": seed + gid,
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
            i, j = j, i
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
    cfg = {"rounds": args.rounds, "num_dice": args.num_dice}
    schedule = build_schedule(models, args.games)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(gid):
        async with sem:
            seat_models = schedule[gid]
            try:
                r = await play_game(client, args.game, cfg, seat_models, args.seed, gid,
                                    args.temperature, args.max_steps)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"[{args.game}] game {gid} FAILED: {e!r}", file=sys.stderr, flush=True)
                return None
            (out / f"game_{gid:02d}.json").write_text(json.dumps(r, indent=1))
            print(f"[{args.game}] g{gid:02d} {r['seat_models'][0].split('/')[-1]} vs "
                  f"{r['seat_models'][1].split('/')[-1]} steps={r['n_steps']} "
                  f"{'TIMEOUT ' if r['timed_out'] else ''}win={r['win_label'][:40]}", flush=True)
            return r

    results = [r for r in await asyncio.gather(*(one(g) for g in range(args.games))) if r]
    summarize(args, results, models, out)


def summarize(args, results, models, out):
    per = {m: {"games": 0, "wins": 0, "reward": 0.0} for m in models}
    for r in results:
        won = set(r["winners"])
        for pid in range(2):
            m = r["seat_models"][pid]
            per[m]["games"] += 1
            per[m]["wins"] += int(pid in won)
            per[m]["reward"] += r["rewards"].get(str(pid)) or 0.0
    rate = lambda a, b: round(a / b, 3) if b else None
    summary = {
        "env": args.game, "players": 2, "games": len(results),
        "config": {"rounds": args.rounds, "num_dice": args.num_dice},
        "draws": sum(1 for r in results if not r["winners"]),
        "timeouts": sum(1 for r in results if r["timed_out"]),
        "per_model": {m: {"games": d["games"], "wins": d["wins"],
                          "win_rate": rate(d["wins"], d["games"]),
                          "mean_reward": round(d["reward"] / d["games"], 3) if d["games"] else None}
                      for m, d in per.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", choices=["kuhn", "liars", "leduc"], required=True)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=20, help="KuhnPoker/Leduc: hands per match")
    ap.add_argument("--num-dice", type=int, default=5, help="LiarsDice: dice per player")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--env-file", default="/workspace/allie/.env")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
