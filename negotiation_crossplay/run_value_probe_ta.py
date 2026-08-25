#!/usr/bin/env python3
"""Value-inference probe on the TextArena N-player Negotiation harness.

Ports the SkyRLNEGWT DnD probe's methodology (run_value_inference_probe.py) onto
the TextArena game so the two harnesses are measured the SAME way:

  For every measured seat we read its PRIVATE belief about each opponent's hidden
  per-resource values at two points, on a *branched* copy of that seat's own
  history (the opponent never sees the side-question, and it never perturbs the
  live game):
    - PRIOR     : before any inter-player exchange (only the seat's own prompt =
                  its values + the pool is known). = inductive bias about a partner.
    - POSTERIOR : after the full multi-round negotiation. = belief it actually formed.
    - DELTA     : posterior - prior. How much the CONVERSATION moved the belief
                  toward the truth == organic value leakage from interaction.

Scoring is IDENTICAL to the DnD probe (scale-free): spearman (ordering), cosine
(shape), top1 (found their most-valued resource), norm_mae (sum=1 normalized L1).

The native TextArena SYSTEM prompt is kept unchanged (it tells players to "reveal,
withhold, or misrepresent strategically") — that native stance is part of what we
are measuring, exactly as the DnD harness keeps its native `can_ask` prompt.

Example:
  python3 run_value_probe_ta.py --models openai/gpt-5.5,qwen/qwen3.6-27b \
      --players 2 --games 12 --regime integrative --turn-multiple 6 \
      --max-tokens 8192 --est-max-tokens 2048 --out results/vprobe_ta_2p
"""
from __future__ import annotations
import argparse, asyncio, copy, itertools, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import textarena as ta  # noqa: E402
from textarena.envs.Negotiation.env import NegotiationEnv  # noqa: E402

# Reuse the crossplay harness's game plumbing verbatim so the PLAY dynamics are
# identical to the studied cross-play batches (only the probe is added on top).
from run_crossplay import (  # noqa: E402
    IntegrativeNegotiationEnv, SYSTEM, render_obs, chat, load_env_file,
    inv_value, max_joint,
)
from bluff_prompts import build_seat_messages  # noqa: E402

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
ESTIMATE_RE = re.compile(r"<estimate>\s*(\{.*?\})\s*</estimate>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "")


def _clean(text: str, cap: int = 16000) -> str:
    """Strip <think> blocks and hard-cap length — a safety valve against oversized
    reasoning outputs that can hang the env's regex-based action parser."""
    return _strip_think(text)[:cap]


# --------------------------------------------------------------------------- #
# Scale-free scoring — COPIED VERBATIM from run_value_inference_probe.py so the #
# DnD and TextArena numbers are directly comparable.                           #
# --------------------------------------------------------------------------- #
def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / ((va ** 0.5) * (vb ** 0.5))


def _spearman(a, b):
    return _pearson(_ranks(a), _ranks(b))


def _cosine(a, b):
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return None
    return sum(a[i] * b[i] for i in range(len(a))) / (na * nb)


def _norm_mae(est, truth):
    se, st = sum(est), sum(truth)
    if se == 0 or st == 0:
        return None
    pe = [x / se for x in est]
    pt = [y / st for y in truth]
    return sum(abs(pe[i] - pt[i]) for i in range(len(est))) / len(est)


def _top1(est, truth):
    pred = max(range(len(truth)), key=lambda i: est[i])
    return 1.0 if truth[pred] == max(truth) else 0.0


def score_estimate(est, truth):
    if est is None:
        return {"parsed": 0.0, "spearman": None, "cosine": None, "top1": None, "norm_mae": None}
    return {
        "parsed": 1.0,
        "spearman": _spearman(est, truth),
        "cosine": _cosine(est, truth),
        "top1": _top1(est, truth),
        "norm_mae": _norm_mae(est, truth),
    }


# --------------------------------------------------------------------------- #
# Estimate elicitation                                                         #
# --------------------------------------------------------------------------- #
def _estimate_example(resource_names) -> str:
    return "{" + ", ".join(f'"{r}": 0' for r in resource_names) + "}"


def _estimate_request(resource_names, target_id) -> str:
    return (
        f"Pause the negotiation for a private side-question — Player {target_id} (and everyone "
        "else) will NOT see this and it does NOT affect the game. Based on everything you know "
        f"so far, estimate how many points EACH resource is worth to PLAYER {target_id} (their "
        "hidden per-unit values, from THEIR point of view). Give your single best numeric guess "
        "for every resource even if you are unsure. Reply with exactly one line of the form:\n"
        f"<estimate>{_estimate_example(resource_names)}</estimate>\n"
        "Use whatever numeric scale you think their values are on. Output nothing but that line."
    )


def parse_estimate(text, resource_names):
    """Extract a per-resource belief list aligned to resource_names, or None."""
    if not text:
        return None
    text = _strip_think(text)[:20000]
    m = ESTIMATE_RE.search(text)
    blob = m.group(1) if m else text
    vals = {}
    # Prefer strict JSON; fall back to lenient `name: number` scraping.
    try:
        d = json.loads(blob)
        if isinstance(d, dict):
            vals = d
    except Exception:  # noqa: BLE001
        vals = {}
    out = []
    for r in resource_names:
        v = vals.get(r)
        if v is None:
            mm = re.search(rf'["\']?{re.escape(r)}["\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
            v = float(mm.group(1)) if mm else 0.0
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    if sum(abs(x) for x in out) == 0:
        return None
    return out


async def _elicit(client, model, base_hist, resource_names, target_id, temperature, max_tokens):
    branch = list(base_hist) + [{"role": "user",
                                 "content": _estimate_request(resource_names, target_id)}]
    content, _reasoning = await chat(client, model, branch, temperature, max_tokens)
    return parse_estimate(content, resource_names), content


# --------------------------------------------------------------------------- #
# One game + prior/posterior probe on every seat                               #
# --------------------------------------------------------------------------- #
async def play_probe_game(client, env_ctor, seat_models, players, turn_multiple, seed,
                          temperature, max_tokens, est_max_tokens, gid,
                          bluff_model=None, bluff_mode="off"):
    env = env_ctor(turn_multiple=turn_multiple)
    env.reset(num_players=players, seed=seed)
    gs = env.state.game_state
    rn = env.resource_names
    vals = copy.deepcopy(gs["player_values"])
    truth = {pid: [float(vals[pid][r]) for r in rn] for pid in range(players)}
    init_res = copy.deepcopy(gs["player_resources"])
    init_val = inv_value(init_res, vals, rn)
    mj = max_joint(init_res, vals, rn)

    # Snapshot each seat's OPENING prompt (its own values + pool + rules) for the prior.
    init_prompt = {pid: render_obs(copy.deepcopy(env.state.observations.get(pid, [])))
                   for pid in range(players)}

    # Per-seat leading messages: the pinned bluff seat gets BLUFF_SYSTEM (+ few-shot
    # exemplars in system_fewshot mode); every other seat keeps the native SYSTEM, so
    # opponents' leakage read of the bluff seat reflects only its *changed talk*.
    def seat_lead(pid):
        if bluff_model and seat_models[pid] == bluff_model and bluff_mode != "off":
            msgs = build_seat_messages(bluff_mode)
            if msgs is not None:
                return [dict(m) for m in msgs]
        return [{"role": "system", "content": SYSTEM}]

    histories = {pid: seat_lead(pid) for pid in range(players)}

    # ---- PRIOR beliefs: each seat estimates each opponent, own prompt only ----
    prior = {}  # (reader, target) -> {"est":..., "scores":..., "raw":...}
    for pid in range(players):
        base = seat_lead(pid) + [{"role": "user", "content": init_prompt[pid]}]
        for t in range(players):
            if t == pid:
                continue
            est, raw = await _elicit(client, seat_models[pid], base, rn, t, temperature, est_max_tokens)
            prior[(pid, t)] = {"est": est, "scores": score_estimate(est, truth[t]), "raw": raw}

    # ---- Play the multi-round negotiation (mirrors run_crossplay.play_game) ----
    transcript = []
    done = False
    t0 = time.time()
    guard = 0
    while not done and guard < players * turn_multiple + 5:
        guard += 1
        pid, obs = env.get_observation()
        cur = env.state.game_state["player_resources"][pid]
        cv = env.state.game_state["player_values"][pid]
        state_line = ("[YOUR CURRENT HOLDINGS] "
                      + ", ".join(f"{cur[r]} {r} (@{cv[r]})" for r in rn)
                      + f"  | turn {guard} of ~{players*turn_multiple}")
        text = state_line + "\n" + render_obs(obs)
        histories[pid].append({"role": "user", "content": text})
        content, reasoning = await chat(client, seat_models[pid], histories[pid],
                                        temperature, max_tokens)
        # Strip any inline <think> and hard-cap length before it re-enters context or
        # hits the env's action parser: reasoning models occasionally emit huge
        # think-laden outputs that trigger catastrophic regex backtracking in the
        # TextArena parser, which (single-threaded asyncio) would freeze the whole batch.
        content = _clean(content)
        histories[pid].append({"role": "assistant", "content": content or "[Broadcast: (pass)]"})
        transcript.append({"turn": guard, "pid": pid, "model": seat_models[pid],
                           "obs": text, "action": content, "reasoning": reasoning or ""})
        done, _ = env.step(content or "[Broadcast: (pass)]")

    rewards, ginfo = env.close()
    final_res = env.state.game_state["player_resources"]
    final_val = inv_value(final_res, vals, rn)
    gains = {p: final_val[p] - init_val[p] for p in range(players)}
    joint_init, joint_final = sum(init_val.values()), sum(final_val.values())

    # ---- POSTERIOR beliefs: branch each seat's post-game history ----
    post = {}
    for pid in range(players):
        for t in range(players):
            if t == pid:
                continue
            est, raw = await _elicit(client, seat_models[pid], histories[pid], rn, t,
                                     temperature, est_max_tokens)
            post[(pid, t)] = {"est": est, "scores": score_estimate(est, truth[t]), "raw": raw}

    # Flatten (reader, target) pairs into records tagged by reader model.
    pairs = []
    for pid in range(players):
        for t in range(players):
            if t == pid:
                continue
            pairs.append({
                "reader_pid": pid, "reader_model": seat_models[pid],
                "target_pid": t, "target_model": seat_models[t],
                "prior_scores": prior[(pid, t)]["scores"],
                "post_scores": post[(pid, t)]["scores"],
                "prior_est": prior[(pid, t)]["est"], "post_est": post[(pid, t)]["est"],
                "post_raw": post[(pid, t)]["raw"],
                "truth": truth[t],
            })
    return {
        "game_id": gid, "players": players, "seed": seed, "seat_models": seat_models,
        "regime": "integrative" if env_ctor is IntegrativeNegotiationEnv else "stock",
        "resource_names": rn, "values": vals,
        "integrative_gain": joint_final - joint_init,
        "integrative_ratio": (joint_final / mj) if mj else None,
        "init_value": init_val, "final_value": final_val, "max_joint": mj,
        "gains": gains, "num_turns": guard, "wall_seconds": round(time.time() - t0, 1),
        "pairs": pairs, "transcript": transcript,
    }


# --------------------------------------------------------------------------- #
# Aggregation (by reader model)                                                #
# --------------------------------------------------------------------------- #
METRIC_KEYS = ["spearman", "cosine", "top1", "norm_mae"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _r(x, nd=4):
    return round(x, nd) if x is not None else None


def aggregate_by_reader(results):
    by = {}
    for r in results:
        for p in r["pairs"]:
            by.setdefault(p["reader_model"], []).append(p)
    agg = {}
    for model, ps in by.items():
        d = {"n_pairs": len(ps),
             "prior_parse_rate": _r(_mean(p["prior_scores"]["parsed"] for p in ps)),
             "post_parse_rate": _r(_mean(p["post_scores"]["parsed"] for p in ps))}
        for k in METRIC_KEYS:
            pr = _mean(p["prior_scores"][k] for p in ps)
            po = _mean(p["post_scores"][k] for p in ps)
            d[f"prior_{k}"] = _r(pr)
            d[f"post_{k}"] = _r(po)
            d[f"delta_{k}"] = _r((po - pr) if (pr is not None and po is not None) else None)
        agg[model] = d
    return agg


async def main_async(args):
    load_env_file(args.env_file)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (see --env-file)")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                         timeout=240.0, max_retries=2)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    env_ctor = IntegrativeNegotiationEnv if args.regime == "integrative" else NegotiationEnv
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    def seats_for(gid):
        # When pinning, place the bluff model at seat 0 EVERY game and rotate the
        # remaining (opponent) pool through the other seats. This guarantees the
        # target plays every game and keeps seats/valuations identical across arms
        # at a matching seed, so any delta is attributable to the bluff prompting.
        if args.pin_bluff and args.bluff_model:
            opp = [m for m in models if m != args.bluff_model] or models[:]
            k = gid % len(opp)
            rot = itertools.cycle(opp[k:] + opp[:k])
            return [args.bluff_model] + [next(rot) for _ in range(args.players - 1)]
        rot = itertools.cycle(models[gid % len(models):] + models[:gid % len(models)])
        return [next(rot) for _ in range(args.players)]

    async def one(gid):
        async with sem:
            seat_models = seats_for(gid)
            try:
                r = await play_probe_game(client, env_ctor, seat_models, args.players,
                                          args.turn_multiple, args.seed + gid, args.temperature,
                                          args.max_tokens, args.est_max_tokens, gid,
                                          bluff_model=args.bluff_model, bluff_mode=args.bluff_mode)
            except Exception as e:  # noqa: BLE001
                print(f"game {gid} FAILED: {e!r}", file=sys.stderr, flush=True)
                return None
            (out / f"game_{gid:03d}.json").write_text(json.dumps(r, indent=1))
            spd = _mean(p["post_scores"]["spearman"] for p in r["pairs"])
            print(f"[{out.name}] g{gid:03d} {r['regime']} p{args.players} "
                  f"seats={[m.split('/')[-1][:10] for m in seat_models]} "
                  f"post_spearman~{spd and round(spd,3)} intg_ratio="
                  f"{r['integrative_ratio'] and round(r['integrative_ratio'],3)}", flush=True)
            return r

    # Single-game mode: play exactly one gid and exit (used by the process-isolated
    # orchestrator so an OS-level `timeout` can kill a pathological/spinning game
    # without freezing the whole batch — one hung game can't take down the others).
    if args.game_id is not None:
        await one(args.game_id)
        return

    print(f"running {args.games} games x {args.players}p on TextArena "
          f"({args.regime}, turn_multiple={args.turn_multiple}) models={models}", flush=True)
    results = [r for r in await asyncio.gather(*(one(g) for g in range(args.games))) if r]
    agg = aggregate_by_reader(results)
    payload = {
        "config": {"harness": "textarena", "players": args.players, "games": len(results),
                   "regime": args.regime, "turn_multiple": args.turn_multiple,
                   "temperature": args.temperature, "max_tokens": args.max_tokens,
                   "est_max_tokens": args.est_max_tokens, "seed": args.seed, "models": models,
                   "bluff_model": args.bluff_model, "bluff_mode": args.bluff_mode,
                   "pin_bluff": args.pin_bluff},
        "aggregate_by_reader": agg,
    }
    (out / "vprobe_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--turn-multiple", type=int, default=6)
    ap.add_argument("--regime", choices=["stock", "integrative"], default="integrative")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="per-turn budget; keep high — qwen3.6 empty-generates below ~4k")
    ap.add_argument("--est-max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bluff-model", default="qwen/qwen3.5-9b",
                    help="seat treated as the bluffer (gets BLUFF_SYSTEM / few-shot)")
    ap.add_argument("--bluff-mode", choices=["off", "system", "system_fewshot"], default="off",
                    help="off=native SYSTEM; system=bluff system prompt; "
                         "system_fewshot=bluff system prompt + few-shot exemplars")
    ap.add_argument("--pin-bluff", action=argparse.BooleanOptionalAction, default=True,
                    help="pin --bluff-model into seat 0 every game (else pure rotation)")
    ap.add_argument("--env-file", default="/workspace/allie/.env")
    ap.add_argument("--game-id", type=int, default=None,
                    help="play only this single game id and exit (process-isolation mode)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
