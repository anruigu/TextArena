#!/usr/bin/env python3
"""Generalized frontier cross-play for TextArena deception/negotiation envs.

Supports: coup, newrecruit, scorablegames (and mafia, for parity). One
OpenRouter model per seat; models rotate across seats per game. Thinking traces
are captured per turn FOR THE VIEWER ONLY and never fed back into the game — only
each seat's spoken action reaches `env.step()` and its own history, so reasoning
never leaks to opponents.

Emits a uniform per-game record (game_XX.json) + summary.json under --out.
Convert to agentviz with to_agentviz_general.py.

Example:
  python3 run_xplay.py --env coup --games 10 --out results_coup
  python3 run_xplay.py --env scorablegames --game-config vendor_retailer --out results_sg2p
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
    "openai/gpt-5.6-sol-pro",
    "anthropic/claude-opus-4.8",
    "deepseek/deepseek-v4-pro",
    "meta-llama/llama-4-maverick",
    "qwen/qwen3.7-max",
    "google/gemini-3.6-flash",
    "moonshotai/kimi-k3",
]

SYSTEM = (
    "You are competing in a multi-player strategy game that rewards social "
    "deduction and negotiation. Read the game messages and everything other "
    "players have said, reason carefully about their hidden information and "
    "intentions, then act to WIN for yourself. Bluffing, strategic concealment, "
    "and misrepresentation are allowed within the rules of the game — read the "
    "rules the game gives you. Always respond using the exact bracketed action "
    "tokens the game specifies (e.g. [income], [coup 2], [Propose] ABCDEFGH, "
    "[Accept]). Keep spoken output concise; do your private strategizing in your "
    "reasoning, not out loud."
)


# ---------------------------------------------------------------------------
# Per-env adapters: build the env, its player count, and extract phase / role /
# per-game meta.  Keeps the driver loop identical across games.
# ---------------------------------------------------------------------------
def _coup_env(cfg):
    from textarena.envs.Coup.env import CoupEnv
    return CoupEnv()

def _coup_phase(env):
    ph = env.state.game_state.get("phase")
    return getattr(ph, "name", str(ph))

def _coup_meta(env):
    gs = env.state.game_state
    return {"final_coins": dict(gs["coins"]),
            "survivors": [p for p in range(env.state.num_players)
                          if len(gs["hidden_hand"][p]) > 0]}

def _newrecruit_env(cfg):
    from textarena.envs.NewRecruit.env import NewRecruitEnv
    return NewRecruitEnv()

def _newrecruit_role(env, pid):
    return {0: "Recruiter", 1: "Candidate"}.get(pid, f"Player {pid}")

def _newrecruit_meta(env):
    gs = env.state.game_state
    prop = gs.get("current_proposal")
    return {"final_proposal": prop, "turn": gs.get("current_turn")}

def _scorable_env(cfg):
    from textarena.envs.ScorableGames.env import ScorableGamesEnv
    return ScorableGamesEnv(game_config=cfg["game_config"], max_rounds=cfg.get("max_rounds", 40))

def _scorable_role(env, pid):
    pc = getattr(env, "player_configs", {}).get(pid)
    if isinstance(pc, dict):
        return pc.get("role") or pc.get("name") or f"Player {pid}"
    return f"Player {pid}"

def _scorable_meta(env):
    gs = env.state.game_state
    return {k: gs.get(k) for k in ("current_deal", "player_votes") if k in gs}

def _diplomacy_env(cfg):
    from textarena.envs.Diplomacy.env import DiplomacyEnv
    return DiplomacyEnv(max_turns=cfg.get("dip_max_turns", 2),
                        negotiations_per_phase=cfg.get("neg_per_phase", 2))

def _diplomacy_phase(env):
    e = env.engine
    s = getattr(e.season, "value", e.season)
    p = getattr(e.phase, "value", e.phase)
    return f"{s} {e.year} {p}"

def _diplomacy_role(env, pid):
    return env.player_power_map.get(pid, f"Player {pid}")

def _diplomacy_meta(env):
    e = env.engine
    return {
        "final_sc": {str(pid): (len(e.powers[pw].controlled_centers) if pw in e.powers else None)
                     for pid, pw in env.player_power_map.items() if isinstance(pid, int) and pid >= 0},
        "winners_powers": list(e.winners or []),
    }

def _mafia_env(cfg):
    from textarena.envs.SecretMafia.env import SecretMafiaEnv
    return SecretMafiaEnv(discussion_rounds=cfg.get("discussion_rounds", 3))

def _mafia_phase(env):
    return env.phase.value

def _mafia_role(env, pid):
    return env.player_roles[pid]


ADAPTERS = {
    "coup":          {"build": _coup_env,       "players": 5, "phase": _coup_phase,
                      "role": None,            "meta": _coup_meta,      "env_key": "coup"},
    "newrecruit":    {"build": _newrecruit_env, "players": 2, "phase": None,
                      "role": _newrecruit_role, "meta": _newrecruit_meta, "env_key": "newrecruit"},
    "scorablegames": {"build": _scorable_env,   "players": None, "phase": None,
                      "role": _scorable_role,  "meta": _scorable_meta,  "env_key": "scorablegames"},
    "mafia":         {"build": _mafia_env,      "players": 7, "phase": _mafia_phase,
                      "role": _mafia_role,     "meta": lambda e: {},    "env_key": "secret_mafia"},
    "diplomacy":     {"build": _diplomacy_env,  "players": 7, "phase": _diplomacy_phase,
                      "role": _diplomacy_role, "meta": _diplomacy_meta, "env_key": "diplomacy"},
}


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


async def chat(client, model, messages, temperature=0.8, max_tokens=1200, retries=4):
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
    """Best-effort JSON-serializable coercion for env meta."""
    try:
        json.dumps(obj); return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        return str(obj)


async def play_game(client, adapter, cfg, seat_models, players, seed, gid,
                    temperature, max_steps):
    random.seed(seed * 100003 + gid)
    env = adapter["build"](cfg)
    env.reset(num_players=players, seed=seed + gid)

    role_fn = adapter["role"]
    phase_fn = adapter["phase"]
    roles = {pid: (role_fn(env, pid) if role_fn else None) for pid in range(players)}

    histories = {pid: [{"role": "system", "content": SYSTEM}] for pid in range(players)}
    transcript = []
    done = False
    t0 = time.time()
    steps = 0
    while not done and steps < max_steps:
        steps += 1
        pid, obs = env.get_observation()
        obs_text = render_obs(obs)
        phase = phase_fn(env) if phase_fn else ""
        histories[pid].append({"role": "user", "content": obs_text})
        model = seat_models[pid]
        content, reasoning = await chat(client, model, histories[pid], temperature)
        action = content or "[No action]"
        histories[pid].append({"role": "assistant", "content": action})
        transcript.append({
            "step": steps, "pid": pid, "model": model, "role": roles[pid],
            "phase": phase, "obs": obs_text, "action": content,
            "reasoning": reasoning or "",
        })
        done, _ = env.step(action)

    rewards, game_info = env.close()
    rewards = rewards or {}
    winners = [p for p, r in rewards.items() if r == 1]
    reason = ""
    if isinstance(game_info, dict):
        reason = (game_info.get(0) or {}).get("reason", "") if 0 in game_info else ""
    win_models = sorted({seat_models[p].split("/")[-1] for p in winners})
    win_label = (", ".join(f"P{p}({seat_models[p].split('/')[-1]})" for p in winners)
                 if winners else "Draw / no winner")
    return {
        "game_id": gid, "env": adapter["env_key"], "players": players,
        "seed": seed + gid, "game_config": cfg.get("game_config"),
        "seat_models": seat_models,
        "seat_roles": {str(p): roles[p] for p in range(players)},
        "rewards": {str(p): rewards.get(p) for p in range(players)},
        "winners": winners, "winning_models": win_models, "win_label": win_label,
        "reason": reason,
        "meta": _safe(adapter["meta"](env)),
        "n_steps": steps, "timed_out": not done,
        "wall_seconds": round(time.time() - t0, 1),
        "transcript": transcript,
    }


def seats_for(gid, models, players):
    rot = models[gid % len(models):] + models[:gid % len(models)]
    return [rot[i % len(rot)] for i in range(players)]


async def main_async(args):
    load_env_file(args.env_file)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (see --env-file)")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                         timeout=240.0, max_retries=2)
    adapter = ADAPTERS[args.env]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cfg = {"game_config": args.game_config, "max_rounds": args.max_rounds,
           "discussion_rounds": args.discussion_rounds,
           "dip_max_turns": args.dip_max_turns, "neg_per_phase": args.neg_per_phase}

    players = args.players or adapter["players"]
    if players is None:  # scorablegames: infer from config
        tmp = adapter["build"](cfg)
        tmp._load_game_configuration()
        players = len(tmp.player_configs)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(gid):
        async with sem:
            seat_models = seats_for(gid, models, players)
            try:
                r = await play_game(client, adapter, cfg, seat_models, players,
                                    args.seed, gid, args.temperature, args.max_steps)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"[{args.env}] game {gid} FAILED: {e!r}", file=sys.stderr, flush=True)
                return None
            (out / f"game_{gid:02d}.json").write_text(json.dumps(r, indent=1))
            print(f"[{args.env}] g{gid:02d} steps={r['n_steps']} "
                  f"{'TIMEOUT ' if r['timed_out'] else ''}win={r['win_label'][:48]}",
                  flush=True)
            return r

    results = [r for r in await asyncio.gather(*(one(g) for g in range(args.games))) if r]
    summarize(args, results, models, players, out)


def summarize(args, results, models, players, out):
    per = {m: {"seats": 0, "wins": 0} for m in models}
    for r in results:
        won = set(r["winners"])
        for pid in range(players):
            m = r["seat_models"][pid]
            per[m]["seats"] += 1
            per[m]["wins"] += int(pid in won)
    rate = lambda a, b: round(a / b, 3) if b else None
    summary = {
        "env": args.env, "game_config": args.game_config, "players": players,
        "games": len(results),
        "draws": sum(1 for r in results if not r["winners"]),
        "timeouts": sum(1 for r in results if r["timed_out"]),
        "per_model": {m: {"seats_played": d["seats"], "win_rate": rate(d["wins"], d["seats"]),
                          "wins": d["wins"]} for m, d in per.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=list(ADAPTERS))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--players", type=int, default=None, help="override adapter default")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--game-config", default="base", help="ScorableGames config folder")
    ap.add_argument("--max-rounds", type=int, default=40, help="ScorableGames round cap")
    ap.add_argument("--discussion-rounds", type=int, default=3, help="mafia")
    ap.add_argument("--dip-max-turns", type=int, default=2, help="diplomacy game-years")
    ap.add_argument("--neg-per-phase", type=int, default=2, help="diplomacy negotiation rounds/phase")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--env-file", default="~/.env")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
