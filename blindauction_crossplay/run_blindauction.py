#!/usr/bin/env python3
"""Cross-play LLM eval on TextArena's N-player BlindAuction env.

Drives each seat with a (possibly different) model via OpenRouter, logs full
transcripts + private valuations + bids, and computes per-seat auction metrics:

  - profit           : value of items won (own valuation) - amount paid
  - net_worth        : remaining capital + value of items won  (win = highest)
  - overbids         : bids strictly above own value for the item (winner's curse)
  - items_won, spent, n_bids, n_broadcasts, n_whispers, invalid_move
  - game-level: revenue, allocative efficiency (realized winner-value /
    max achievable if every item went to its highest-valuer)

Default game settings (capital 1000, 5 items, values ±20% around a shared
base). --conversation-rounds is sweepable to check the talk phase isn't
truncating discussion (default env setting is 3).

Seats rotate models per game so each model plays every seat position.

Example:
  python3 run_blindauction.py --models a,b,c,d,e,f,g --games 14 \
      --conversation-rounds 3 --out results/ba_r3
"""
from __future__ import annotations
import argparse, asyncio, copy, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import textarena as ta  # noqa: E402
from textarena.envs.BlindAuction.env import BlindAuctionEnv  # noqa: E402

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: pip install openai", file=sys.stderr); raise


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


SYSTEM = (
    "You are a skilled player in a multi-party blind (sealed-bid) auction game. "
    "Your goal is to MAXIMIZE your final net worth = remaining coins + the value "
    "TO YOU of items you win. Item values are private and differ across players "
    "(up to ±20%), so talk can reveal who wants what — reveal, withhold, or "
    "misrepresent strategically, and use what others say to decide where to bid "
    "and how much.\n"
    "Phases: first a conversation phase ('[Broadcast: msg]' to all, "
    "'[Whisper to X: msg]' privately), then ONE bidding turn where you must "
    "submit ALL your bids at once as '[Bid on Item X: amount]' tokens (several "
    "allowed; total must not exceed your capital). Highest bid wins each item; "
    "you only pay for items you win.\n"
    "IMPORTANT: only bracketed tokens count as actions. Winning an item above "
    "its value to you LOSES money — profit comes from winning items below your "
    "value. Keep any reasoning brief and put it before the bracket tokens."
)


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


async def chat(client, model, messages, temperature=0.7, max_tokens=8000, retries=4):
    kw = {"model": model, "messages": messages, "max_tokens": max_tokens}
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
                # Reasoning models can burn the whole completion budget thinking
                # and return empty content — retry with double the budget.
                if "max_tokens" in kw:
                    kw["max_tokens"] = min(kw["max_tokens"] * 2, 32000)
                elif "max_completion_tokens" in kw:
                    kw["max_completion_tokens"] = min(kw["max_completion_tokens"] * 2, 32000)
                continue
            return content, _extract_reasoning(m)
        except Exception as e:  # noqa: BLE001
            em = str(e).lower()
            if "temperature" in em and "temperature" in kw:
                kw.pop("temperature")
            elif "max_tokens" in em and "max_tokens" in kw:
                kw["max_completion_tokens"] = kw.pop("max_tokens")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2.0 * (attempt + 1))
    return "", ""


async def play_game(client, seat_models, players, conversation_rounds, seed, temperature, gid):
    env = BlindAuctionEnv(conversation_rounds=conversation_rounds)
    env.reset(num_players=players, seed=seed)
    gs = env.state.game_state
    vals = copy.deepcopy(gs["player_item_values"])
    item_names = list(gs["item_names"])
    n_items = len(item_names)
    max_turns = conversation_rounds * players + players

    histories = {pid: [{"role": "system", "content": SYSTEM}] for pid in range(players)}
    counts = {pid: {"broadcasts": 0, "whispers": 0, "bids": 0, "overbids": 0} for pid in range(players)}
    transcript = []
    done = False
    t0 = time.time()
    guard = 0
    while not done and guard < max_turns * 2 + 10:
        guard += 1
        pid, obs = env.get_observation()
        phase = gs["phase"]
        if phase == "conversation":
            state_line = (f"[STATE] Phase: conversation "
                          f"(turn {gs['conversations_completed'] + 1} of {conversation_rounds * players} total talk turns); "
                          f"your capital: {gs['remaining_capital'][pid]} coins.")
        else:
            state_line = ("[STATE] Phase: BIDDING — this is your ONLY bidding turn. Submit all "
                          f"'[Bid on Item X: amount]' bids now; your capital: {gs['remaining_capital'][pid]} coins.")
        text = state_line + "\n" + render_obs(obs)
        histories[pid].append({"role": "user", "content": text})
        model = seat_models[pid]
        content, reasoning = await chat(client, model, histories[pid], temperature)
        action = content or "[Broadcast: (pass)]"
        histories[pid].append({"role": "assistant", "content": action})
        transcript.append({
            "turn": guard, "pid": pid, "model": model, "phase": phase,
            "obs": text, "action": content, "reasoning": reasoning or "",
        })
        if phase == "conversation":
            counts[pid]["broadcasts"] += len(env._parse_broadcasts(action))
            counts[pid]["whispers"] += len(env._parse_whispers(action))
        else:
            for item_s, amt_s in env.bid_pattern.findall(action):
                counts[pid]["bids"] += 1
                try:
                    i, a = int(item_s), int(amt_s)
                    if 0 <= i < n_items and a > vals[pid][i]:
                        counts[pid]["overbids"] += 1
                except ValueError:
                    pass
        done, _ = env.step(action)

    # Safety net: never leave a game unresolved (e.g. a stuck seat drained the guard).
    truncated = False
    if gs["auction_results"] is None:
        truncated = True
        if gs["phase"] == "conversation":
            env._transition_to_bidding_phase()
        env._determine_auction_results()

    rewards, ginfo = env.close()
    res = gs["auction_results"]
    # allocative efficiency: realized winner-value / max achievable
    max_value = sum(max(vals[p][i] for p in range(players)) for i in range(n_items))
    realized = sum(vals[res["item_winners"][i]][i] for i in res["item_winners"])
    revenue = sum(res["winning_bids"].values())

    per_seat = {}
    for pid in range(players):
        per_seat[pid] = {
            "model": seat_models[pid],
            "profit": res["player_profit"].get(pid, 0),
            "net_worth": res["player_net_worth"].get(pid, 0),
            "items_won": res["player_wins"].get(pid, []),
            "spent": res["player_spent"].get(pid, 0),
            "win": (rewards or {}).get(pid, 0) == 1,
            "invalid_move": bool(ginfo[pid].get("invalid_move")),
            **counts[pid],
        }
    return {
        "game_id": gid, "players": players, "seed": seed,
        "conversation_rounds": conversation_rounds,
        "seat_models": seat_models,
        "item_names": item_names, "base_item_values": gs["base_item_values"],
        "player_item_values": {str(p): vals[p] for p in vals},
        "player_bids": {str(p): gs["player_bids"][p] for p in gs["player_bids"]},
        "item_winners": {str(k): v for k, v in res["item_winners"].items()},
        "winning_bids": {str(k): v for k, v in res["winning_bids"].items()},
        "per_seat": {str(p): per_seat[p] for p in per_seat},
        "revenue": revenue,
        "allocative_efficiency": round(realized / max_value, 4) if max_value else None,
        "items_unsold": n_items - len(res["item_winners"]),
        "truncated": truncated,
        "builtin_rewards": {str(k): v for k, v in (rewards or {}).items()},
        "wall_seconds": round(time.time() - t0, 1),
        "transcript": transcript,
    }


async def main_async(args):
    load_env_file(args.env_file)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (see --env-file)")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                         timeout=240.0, max_retries=2)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    players = args.players or len(models)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    # Seat assignments: rotate models across seats each game for balance.
    def seats_for(gid):
        off = gid % len(models)
        rot = models[off:] + models[:off]
        return [rot[s % len(rot)] for s in range(players)]

    async def one(gid):
        async with sem:
            seat_models = seats_for(gid)
            try:
                r = await play_game(client, seat_models, players,
                                    args.conversation_rounds, args.seed + gid,
                                    args.temperature, gid)
            except Exception as e:  # noqa: BLE001
                print(f"game {gid} FAILED: {e!r}", file=sys.stderr, flush=True)
                return None
            (out / f"game_{gid:03d}.json").write_text(json.dumps(r, indent=1))
            profits = {p: d["profit"] for p, d in r["per_seat"].items()}
            print(f"[{out.name}] g{gid:03d} r{args.conversation_rounds} "
                  f"eff={r['allocative_efficiency']} rev={r['revenue']} unsold={r['items_unsold']} "
                  f"profits={profits}{' TRUNCATED' if r['truncated'] else ''}", flush=True)
            return r

    results = [r for r in await asyncio.gather(*(one(g) for g in range(args.games))) if r]
    summarize(results, models, out)


def summarize(results, models, out):
    per_model = {m: {"profit": [], "net_worth": [], "win": [], "invalid": [],
                     "spent": [], "bids": [], "overbids": [], "broadcasts": [],
                     "whispers": [], "items_won": []} for m in models}
    eff, rev, unsold, truncated = [], [], [], 0
    for r in results:
        eff.append(r["allocative_efficiency"])
        rev.append(r["revenue"])
        unsold.append(r["items_unsold"])
        truncated += 1 if r["truncated"] else 0
        for pid, d in r["per_seat"].items():
            m = d["model"]
            per_model[m]["profit"].append(d["profit"])
            per_model[m]["net_worth"].append(d["net_worth"])
            per_model[m]["win"].append(1 if d["win"] else 0)
            per_model[m]["invalid"].append(1 if d["invalid_move"] else 0)
            per_model[m]["spent"].append(d["spent"])
            per_model[m]["bids"].append(d["bids"])
            per_model[m]["overbids"].append(d["overbids"])
            per_model[m]["broadcasts"].append(d["broadcasts"])
            per_model[m]["whispers"].append(d["whispers"])
            per_model[m]["items_won"].append(len(d["items_won"]))
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    summary = {
        "games": len(results),
        "conversation_rounds": results[0]["conversation_rounds"] if results else None,
        "truncated_games": truncated,
        "allocative_efficiency_mean": mean([e for e in eff if e is not None]),
        "revenue_mean": mean(rev),
        "items_unsold_mean": mean(unsold),
        "per_model": {
            m: {"n_seats": len(d["profit"]), "profit_mean": mean(d["profit"]),
                "net_worth_mean": mean(d["net_worth"]), "win_rate": mean(d["win"]),
                "items_won_mean": mean(d["items_won"]), "spent_mean": mean(d["spent"]),
                "bids_mean": mean(d["bids"]), "overbid_mean": mean(d["overbids"]),
                "broadcasts_mean": mean(d["broadcasts"]), "whispers_mean": mean(d["whispers"]),
                "invalid_rate": mean(d["invalid"])}
            for m, d in per_model.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma-separated OpenRouter model ids")
    ap.add_argument("--players", type=int, default=None, help="seats per game (default: len(models))")
    ap.add_argument("--games", type=int, default=14)
    ap.add_argument("--conversation-rounds", type=int, default=3, help="env default is 3")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-file", default="~/.env")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
