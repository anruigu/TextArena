#!/usr/bin/env python3
"""Cross-play LLM negotiation on TextArena's N-player Negotiation env.

Drives each seat with a (possibly different) model via OpenRouter, logs full
transcripts + initial/final inventories + private values, and computes
negotiation-quality metrics the built-in winner-take-all reward throws away:

  - value_gain per seat   : final − initial inventory value under OWN values
  - integrative_gain       : Δ joint value (sum over players of own-valued
                             inventory) from start to finish — did trades move
                             resources toward higher-valuers (expand the pie)?
  - integrative_ratio      : realized joint / max achievable joint (each unit
                             of each resource assigned to its highest-valuer)
  - nash_gain_product      : product of per-seat gains (fairness×efficiency)
  - trades_executed, invalid_moves, win (built-in reward)

Two valuation regimes:
  - stock       : env default (±20% around base — distributive-heavy)
  - integrative : independent uniform draws (big spread — strong gains from trade)

Seats are assigned models round-robin with a per-game rotation so each model
plays every seat position across the batch.

Example:
  python3 run_crossplay.py --models qwen/qwen3.6-27b,google/gemma-4-31b-it,openai/gpt-5.5 \
      --players 3 --games 24 --regime integrative --out results/xp_3p
"""
from __future__ import annotations
import argparse, asyncio, copy, itertools, json, os, random, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import textarena as ta  # noqa: E402
from textarena.envs.Negotiation.env import NegotiationEnv  # noqa: E402

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


class IntegrativeNegotiationEnv(NegotiationEnv):
    """Wide independent valuations -> large gains from trade (integrative)."""
    def reset(self, num_players: int, seed=None):
        super().reset(num_players=num_players, seed=seed)
        gs = self.state.game_state
        rng = random.Random((seed or 0) * 131 + 7)
        for pid in range(num_players):
            for r in self.resource_names:
                gs["player_values"][pid][r] = rng.randint(5, 40)
        # rebuild the seat prompts so players see their (new) values
        for pid in range(num_players):
            self.state.observations[pid] = []
            self.state.add_observation(
                to_id=pid,
                message=self._generate_player_prompt(pid, gs),
                observation_type=ta.ObservationType.PROMPT,
            )


SYSTEM = (
    "You are a skilled negotiator playing a multi-party resource trading game. "
    "Read the game state and all messages, then act to MAXIMIZE the total value "
    "of the resources you hold at the end, under your own private valuations. "
    "Model what others value from what they say and offer; reveal, withhold, or "
    "misrepresent strategically. Each turn, respond with the bracketed action "
    "tokens the game defines (you may combine several in one turn). Keep any "
    "reasoning brief and put it before the bracket tokens.\n"
    "IMPORTANT: talk changes nothing by itself. Resources move — and your score "
    "changes — ONLY when a concrete [Offer to X: ... -> ...] is made and the "
    "recipient replies [Accept #id], before the turn limit. Don't spend every "
    "turn talking; propose and close deals."
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


async def chat(client, model, messages, temperature=0.7, max_tokens=1200, retries=4):
    kw = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        kw["temperature"] = temperature
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(**kw)
            if not getattr(resp, "choices", None):
                return "", ""
            m = resp.choices[0].message
            return (m.content or "").strip(), _extract_reasoning(m)
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


def inv_value(res, vals, rn):
    return {p: sum(res[p][r] * vals[p][r] for r in rn) for p in res}


def max_joint(res, vals, rn):
    """Total units of each resource assigned to its highest-valuer."""
    n = len(res)
    total_units = {r: sum(res[p][r] for p in range(n)) for r in rn}
    return sum(total_units[r] * max(vals[p][r] for p in range(n)) for r in rn)


async def play_game(client, env_ctor, seat_models, players, turn_multiple, seed, temperature, gid):
    env = env_ctor(turn_multiple=turn_multiple)
    env.reset(num_players=players, seed=seed)
    gs = env.state.game_state
    rn = env.resource_names
    vals = copy.deepcopy(gs["player_values"])
    init_res = copy.deepcopy(gs["player_resources"])
    init_val = inv_value(init_res, vals, rn)
    mj = max_joint(init_res, vals, rn)

    histories = {pid: [{"role": "system", "content": SYSTEM}] for pid in range(players)}
    seat_prompted = {pid: False for pid in range(players)}
    transcript = []
    done = False
    t0 = time.time()
    guard = 0
    while not done and guard < players * turn_multiple + 5:
        guard += 1
        pid, obs = env.get_observation()
        # Inject the seat's CURRENT holdings+values each turn: the env's per-turn
        # observation only shows new messages, not the (post-trade) inventory, so
        # without this a trader loses track of what it holds — a memory handicap,
        # not a negotiation one.
        cur = env.state.game_state["player_resources"][pid]
        cv = env.state.game_state["player_values"][pid]
        state_line = "[YOUR CURRENT HOLDINGS] " + ", ".join(
            f"{cur[r]} {r} (@{cv[r]})" for r in rn
        ) + f"  | turn {guard} of ~{players*turn_multiple}"
        text = state_line + "\n" + render_obs(obs)
        histories[pid].append({"role": "user", "content": text})
        model = seat_models[pid]
        content, reasoning = await chat(client, model, histories[pid], temperature)
        histories[pid].append({"role": "assistant", "content": content or "[Broadcast: (pass)]"})
        transcript.append({
            "turn": guard, "pid": pid, "model": model,
            "obs": text, "action": content, "reasoning": reasoning or "",
        })
        done, _ = env.step(content or "[Broadcast: (pass)]")

    rewards, ginfo = env.close()
    final_res = env.state.game_state["player_resources"]
    final_val = inv_value(final_res, vals, rn)
    gains = {p: final_val[p] - init_val[p] for p in range(players)}
    joint_init, joint_final = sum(init_val.values()), sum(final_val.values())
    prod = 1.0
    for p in range(players):
        prod *= max(0, gains[p])
    return {
        "game_id": gid, "players": players, "seed": seed,
        "seat_models": seat_models,
        "regime": "integrative" if env_ctor is IntegrativeNegotiationEnv else "stock",
        "values": vals, "init_resources": init_res, "final_resources": final_res,
        "init_value": init_val, "final_value": final_val, "gains": gains,
        "joint_init": joint_init, "joint_final": joint_final,
        "integrative_gain": joint_final - joint_init,
        "integrative_ratio": (joint_final / mj) if mj else None,
        "max_joint": mj,
        "nash_gain_product": prod,
        "builtin_rewards": {str(k): v for k, v in (rewards or {}).items()},
        "invalid_moves": {str(p): bool(ginfo[p].get("invalid_move")) for p in range(players)},
        "wall_seconds": round(time.time() - t0, 1),
        "transcript": transcript,
    }


async def main_async(args):
    load_env_file(args.env_file)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (see --env-file)")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                         timeout=180.0, max_retries=2)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    env_ctor = IntegrativeNegotiationEnv if args.regime == "integrative" else NegotiationEnv
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    # Seat assignments: rotate models across seats each game for balance.
    def seats_for(gid):
        rot = itertools.cycle(models[gid % len(models):] + models[:gid % len(models)])
        return [next(rot) for _ in range(args.players)]

    async def one(gid):
        async with sem:
            seat_models = seats_for(gid)
            try:
                r = await play_game(client, env_ctor, seat_models, args.players,
                                    args.turn_multiple, args.seed + gid, args.temperature, gid)
            except Exception as e:  # noqa: BLE001
                print(f"game {gid} FAILED: {e!r}", file=sys.stderr, flush=True)
                return None
            (out / f"game_{gid:03d}.json").write_text(json.dumps(r, indent=1))
            g = r["gains"]
            print(f"[{out.name}] g{gid:03d} {r['regime']} seats={[m.split('/')[-1][:10] for m in seat_models]} "
                  f"intg_gain={r['integrative_gain']} ratio={r['integrative_ratio'] and round(r['integrative_ratio'],3)} "
                  f"gains={g}", flush=True)
            return r

    results = [r for r in await asyncio.gather(*(one(g) for g in range(args.games))) if r]
    summarize(results, models, out)


def summarize(results, models, out):
    rn = None
    per_model = {m: {"value_gain": [], "win": [], "invalid": [], "n": 0} for m in models}
    intg_gain, intg_ratio = [], []
    for r in results:
        intg_gain.append(r["integrative_gain"])
        if r["integrative_ratio"] is not None:
            intg_ratio.append(r["integrative_ratio"])
        rewards = r["builtin_rewards"]
        best = max(r["final_value"].values())
        for pid, m in enumerate(r["seat_models"]):
            per_model[m]["n"] += 1
            per_model[m]["value_gain"].append(r["gains"][pid])
            per_model[m]["win"].append(1 if r["final_value"][pid] == best else 0)
            per_model[m]["invalid"].append(1 if r["invalid_moves"].get(str(pid)) else 0)
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    summary = {
        "games": len(results),
        "integrative_gain_mean": mean(intg_gain),
        "integrative_ratio_mean": mean(intg_ratio),
        "per_model": {
            m: {"n_seats": d["n"], "value_gain_mean": mean(d["value_gain"]),
                "win_rate": mean(d["win"]), "invalid_rate": mean(d["invalid"])}
            for m, d in per_model.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma-separated OpenRouter model ids")
    ap.add_argument("--players", type=int, default=3)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--turn-multiple", type=int, default=4, help="turns per player")
    ap.add_argument("--regime", choices=["stock", "integrative"], default="integrative")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-file", default="~/.env")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
