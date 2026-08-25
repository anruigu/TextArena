#!/usr/bin/env python3
"""Frontier cross-play for The Resistance: Avalon (hidden-role social deduction).

Avalon is not a TextArena env; we drive the vendored AvalonBench state machine
(avalon_engine.py, from jonathanmli/Avalon-LLM) and wrap one OpenRouter frontier
model per seat. Models rotate across seats per game. Each seat gets ONLY its own
private role knowledge (Merlin sees evil; Percival sees Merlin+Morgana ambiguously;
evil see each other); public events (proposals, team-vote tallies, quest fail
counts, discussion) are broadcast to all. Reasoning traces are captured per turn
FOR THE VIEWER ONLY and never fed back into any seat's history — only spoken
actions and public events propagate, so reasoning never leaks to opponents.

Default table: 5p classic (Merlin, Percival, Servant vs Morgana, Assassin).

Emits a per-game record (game_XX.json) + summary.json under --out, in the same
schema as mafia_crossplay/run_xplay.py, so to_agentviz_general.py converts it.

Example:
  python3 run_avalon.py --games 2 --out results_smoke
  python3 run_avalon.py --games 10 --out results
"""
from __future__ import annotations
import argparse, asyncio, json, os, random, re, sys, time
from pathlib import Path

from avalon_engine import AvalonGameEnvironment, AvalonBasicConfig
import numpy as np

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

RULES = (
    "You are playing The Resistance: AVALON, a hidden-role social-deduction game.\n"
    "- 5 players. GOOD side (Loyal Servants of Arthur): Merlin, Percival, Servant. "
    "EVIL side (Minions of Mordred): Morgana, Assassin.\n"
    "- The game runs 5 Quests. Each Quest: the current Leader proposes a team of a "
    "fixed size; ALL players vote publicly to Approve or Reject the team. If a strict "
    "majority approve, the team goes on the Quest; if rejected, leadership passes to "
    "the next player and a new team is proposed (the 5th proposal auto-passes).\n"
    "- On a Quest, each team member secretly plays a Success or Fail card. GOOD "
    "players MUST play Success. EVIL players MAY play Fail. A Quest fails if it gets "
    "enough Fail cards (usually 1). Only the NUMBER of Fail cards is revealed, not who "
    "played them.\n"
    "- GOOD wins by succeeding 3 Quests; EVIL wins by failing 3 Quests.\n"
    "- If GOOD succeeds 3 Quests, the Assassin gets one guess at who Merlin is. If the "
    "Assassin identifies Merlin, EVIL wins instead. So Merlin must help Good win "
    "without revealing themselves.\n"
    "- Team votes are PUBLIC (everyone sees who approved/rejected). Quest cards are "
    "SECRET. Deception, bluffing, and role-claiming are core to the game.\n"
    "Always act using the exact bracketed tokens requested (e.g. [Propose] 0 2, "
    "[Approve], [Reject], [Pass], [Fail], [Assassinate] 3). Keep spoken output concise; "
    "do private strategizing in your reasoning, not out loud."
)


def role_knowledge(pid, roles):
    """Private knowledge string for one seat. roles: list of (rid,name,is_good)."""
    name = roles[pid][1]
    evil = [p for p in range(len(roles)) if not bool(roles[p][2])]
    merlin = next(p for p, r in enumerate(roles) if r[1] == "Merlin")
    morgana = next((p for p, r in enumerate(roles) if r[1] == "Morgana"), None)
    if name == "Merlin":
        return (f"You are Player {pid}. Your role: MERLIN (GOOD). You secretly know the "
                f"EVIL players are: {evil} (you do not know which evil role each holds). "
                "Guide Good to win 3 Quests, but stay hidden — if Good wins, the Assassin "
                "will try to name you.")
    if name == "Percival":
        pair = sorted([merlin, morgana]) if morgana is not None else [merlin]
        return (f"You are Player {pid}. Your role: PERCIVAL (GOOD). You see that players "
                f"{pair} are special: one is Merlin (GOOD) and the other is Morgana (EVIL, "
                "disguised to look like Merlin to you). You do NOT know which is which. "
                "Protect the real Merlin.")
    if name == "Servant":
        return (f"You are Player {pid}. Your role: LOYAL SERVANT (GOOD). You have no secret "
                "knowledge. Deduce who is Evil from proposals, votes, and Quest outcomes.")
    # Evil roles
    mates = [p for p in evil if p != pid]
    extra = ""
    if name == "Assassin":
        extra = (" As the Assassin, if Good succeeds 3 Quests you get one guess at Merlin's "
                 "identity to steal the win — pay attention to who seems to know too much.")
    if name == "Morgana":
        extra = (" As Morgana, you appear as a possible Merlin to Percival, which you can "
                 "exploit to sow confusion.")
    return (f"You are Player {pid}. Your role: {name.upper()} (EVIL). Your fellow Evil "
            f"player(s): {mates}. Win by failing 3 Quests or by assassinating Merlin. On "
            "Quests you may secretly play Fail, but be subtle — Good is counting fails."
            + extra)


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


def _extract_reasoning(msg):
    r = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if not r:
        r = (getattr(msg, "model_extra", None) or {}).get("reasoning")
    return (r or "").strip()


async def chat(client, model, messages, temperature=0.8, max_tokens=2200, retries=4):
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


# ---- action parsing (robust; fall back to a legal random move) ------------
def parse_team(text, size, n, leader, rng):
    ids = [int(x) for x in re.findall(r"\d+", text or "")]
    seen, team = set(), []
    for i in ids:
        if 0 <= i < n and i not in seen:
            seen.add(i); team.append(i)
        if len(team) == size:
            break
    if len(team) != size:  # fallback: leader + random others
        pool = [p for p in range(n) if p != leader]
        rng.shuffle(pool)
        team = ([leader] + pool)[:size] if leader < n else pool[:size]
        team = list(dict.fromkeys(team))[:size]
    return sorted(team)


def parse_vote(text):
    """Approve=1, Reject=0. Prefer the explicit bracket token the game asks for;
    only fall back to bare prose words when NO bracket token is present, and even
    then require them to be unambiguous. A bare 'approve' inside a '[Reject] ...'
    rationale must NOT flip the vote to approve."""
    t = (text or "").lower()
    ai, ri = t.rfind("[approve]"), t.rfind("[reject]")
    if ai != -1 or ri != -1:
        return 1 if ai > ri else 0          # if both appear, the last-stated wins
    has_app = bool(re.search(r"\bapprove\b", t) or re.search(r"\byes\b", t))
    has_rej = bool(re.search(r"\breject\b", t) or re.search(r"\bno\b", t))
    if has_app and not has_rej:
        return 1
    if has_rej and not has_app:
        return 0
    return 1  # empty / contradictory / unparseable -> default approve


def parse_quest(text):
    """Pass(success)=1, Fail=0. Bracket token wins; a bare 'fail' only counts when
    there is no bracket token and no competing pass/success word (so 'I won't fail
    the mission' does not read as a Fail card)."""
    t = (text or "").lower()
    fi, pi = t.rfind("[fail]"), t.rfind("[pass]")
    if fi != -1 or pi != -1:
        return 0 if fi > pi else 1
    if re.search(r"\bfail\b", t) and not re.search(r"\b(pass|success|succeed)\b", t):
        return 0
    return 1  # default / ambiguous -> pass (success)


def parse_target(text, n, exclude, rng):
    ids = [int(x) for x in re.findall(r"\d+", text or "")]
    for i in ids:
        if 0 <= i < n:
            return i
    pool = [p for p in range(n) if p not in exclude]
    return rng.choice(pool) if pool else 0


def seats_for(gid, models, players):
    rot = models[gid % len(models):] + models[:gid % len(models)]
    return [rot[i % len(rot)] for i in range(players)]


async def play_game(client, seat_models, players, seed, gid, temperature,
                    max_steps, discussion):
    rng = random.Random(seed * 100003 + gid)
    np.random.seed((seed * 100003 + gid) % (2**31 - 1))
    cfg = AvalonBasicConfig.from_num_players(players, merlin=True, percival=True,
                                             morgana=True)
    env = AvalonGameEnvironment(cfg)
    roles = [(int(r[0]), r[1], bool(r[2])) for r in env.get_roles()]
    role_name = {p: roles[p][1] for p in range(players)}
    merlin = next(p for p, r in enumerate(roles) if r[1] == "Merlin")

    histories = {p: [{"role": "system", "content": RULES + "\n\n" + role_knowledge(p, roles)}]
                 for p in range(players)}
    public_log, seen = [], {p: 0 for p in range(players)}
    transcript = []
    t0 = time.time()
    steps = 0
    ended_reason = ""
    assassin_target = None

    def broadcast(msg):
        public_log.append(msg)

    def obs_for(pid, ask):
        new = public_log[seen[pid]:]
        seen[pid] = len(public_log)
        parts = new + (["", ask] if ask else [])
        return "\n".join(parts).strip()

    async def act(pid, ask, phase):
        nonlocal steps
        steps += 1
        obs = obs_for(pid, ask)
        histories[pid].append({"role": "user", "content": obs})
        content, reasoning = await chat(client, seat_models[pid], histories[pid], temperature)
        histories[pid].append({"role": "assistant", "content": content or "[No action]"})
        transcript.append({"step": steps, "pid": pid, "model": seat_models[pid],
                           "role": role_name[pid], "phase": phase, "obs": obs,
                           "action": content, "reasoning": reasoning or ""})
        return content

    broadcast(f"=== AVALON · 5 players · Good: Merlin, Percival, Servant | "
              f"Evil: Morgana, Assassin. Team sizes per quest: "
              f"{list(env.num_players_for_quest)}. ===")

    current_team = []
    while not env.done and steps < max_steps:
        phase = env.phase
        if phase == 0:  # Team Selection
            leader = int(env.get_quest_leader())
            tsize = int(env.get_team_size())
            qi = int(env.turn)
            nfail = int(env.num_fails_for_quest[qi])
            broadcast(f"--- Quest {qi+1}: team of {tsize}, {nfail} Fail card(s) sink it. "
                      f"Leader is Player {leader}. Proposal round {env.round+1}/5. ---")
            if discussion:
                for k in range(players):
                    pid = (leader + k) % players
                    await act(pid, "Open discussion before the Leader proposes the team. "
                              "Share your read, claim a role if useful, or press others. "
                              "1-2 sentences.", "Discussion")
                    last = transcript[-1]["action"] or "(silent)"
                    broadcast(f"[Player {pid}]: {last}")
            c = await act(leader, f"You are the Leader. Propose exactly {tsize} players for "
                          f"Quest {qi+1}: e.g. [Propose] {' '.join(str(x) for x in range(tsize))}. "
                          "You may include yourself.", "Team Selection")
            team = parse_team(c, tsize, players, leader, rng)
            current_team = team
            broadcast(f"[Player {leader}] proposes team {team} for Quest {qi+1}.")
            env.choose_quest_team(frozenset(team), leader)

        elif phase == 1:  # Team Voting (public)
            votes = [1] * players
            for pid in range(players):
                c = await act(pid, f"Vote on the proposed team {current_team} for the current "
                              "Quest. Reply [Approve] or [Reject] (with a brief reason).",
                              "Team Voting")
                votes[pid] = parse_vote(c)
            tally = ", ".join(f"P{p}={'Approve' if votes[p] else 'Reject'}"
                              for p in range(players))
            broadcast(f"Team vote — {tally}  ({sum(votes)}/{players} approve)")
            _, _, accepted = env.gather_team_votes(votes)
            broadcast("Result: team APPROVED, going on the Quest."
                      if accepted else "Result: team REJECTED, leadership passes.")

        elif phase == 2:  # Quest Voting (secret)
            team = list(current_team)
            qvotes = []
            for pid in team:
                is_good = roles[pid][2]
                if is_good:
                    await act(pid, "You are on the Quest. Good must play [Pass] (Success). "
                              "Confirm [Pass].", "Quest Voting")
                    qvotes.append(1)
                else:
                    c = await act(pid, "You are on the Quest. Secretly play [Pass] (Success) "
                                  "or [Fail]. Only the total number of Fails is revealed.",
                                  "Quest Voting")
                    qvotes.append(parse_quest(c))
            _, _, succeeded, num_fails = env.gather_quest_votes(qvotes)
            results = [bool(x) for x in env.quest_results]
            broadcast(f"Quest {int(env.turn)} outcome: {'SUCCESS' if succeeded else 'FAIL'} "
                      f"({num_fails} Fail card(s)). Quest record so far "
                      f"(True=success): {results}.")
            if env.done:
                ended_reason = "Evil failed 3 Quests."

        elif phase == 3:  # Assassination
            assassin = int(env.get_assassin())
            goodies = [p for p in range(players) if roles[p][2]]
            c = await act(assassin, "Good has succeeded 3 Quests. You are the ASSASSIN. Name the "
                          "player you believe is MERLIN to steal the win for Evil: "
                          f"[Assassinate] <player>. Good-aligned candidates: {goodies}.",
                          "Assassination")
            assassin_target = parse_target(c, players, exclude={assassin}, rng=rng)
            broadcast(f"[Player {assassin}] (Assassin) assassinates Player {assassin_target}.")
            env.choose_assassination_target(assassin, assassin_target)
            if roles[assassin_target][1] == "Merlin":
                ended_reason = (f"Evil wins: Assassin identified Merlin (Player "
                                f"{assassin_target}).")
            else:
                ended_reason = (f"Good wins: 3 Quests succeeded and Merlin survived "
                                f"(Assassin missed, hit Player {assassin_target}).")

    good_win = bool(env.good_victory)
    if not ended_reason:
        ended_reason = ("Good prevailed." if good_win else "Evil prevailed.") + \
                       (" (reached step cap)" if steps >= max_steps else "")
    winners = [p for p in range(players) if roles[p][2] == good_win]
    rewards = {p: (1 if p in winners else 0) for p in range(players)}
    win_models = sorted({seat_models[p].split("/")[-1] for p in winners})
    side = "GOOD" if good_win else "EVIL"
    win_label = f"{side} — " + ", ".join(
        f"P{p}({role_name[p]},{seat_models[p].split('/')[-1]})" for p in winners)

    return {
        "game_id": gid, "env": "avalon", "players": players, "seed": seed + gid,
        "game_config": "5p_classic",
        "seat_models": seat_models,
        "seat_roles": {str(p): role_name[p] for p in range(players)},
        "rewards": {str(p): rewards[p] for p in range(players)},
        "winners": winners, "winning_models": win_models, "win_label": win_label,
        "reason": ended_reason,
        "meta": {"good_victory": good_win,
                 "quest_results": [bool(x) for x in env.quest_results],
                 "merlin": merlin, "assassin": int(env.get_assassin()),
                 "assassin_target": assassin_target,
                 "sides": {str(p): ("Good" if roles[p][2] else "Evil") for p in range(players)}},
        "n_steps": steps, "timed_out": steps >= max_steps and not env.done,
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
    players = args.players
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(gid):
        async with sem:
            seat_models = seats_for(gid, models, players)
            try:
                r = await play_game(client, seat_models, players, args.seed, gid,
                                    args.temperature, args.max_steps, not args.no_discussion)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"[avalon] game {gid} FAILED: {e!r}", file=sys.stderr, flush=True)
                return None
            (out / f"game_{gid:02d}.json").write_text(json.dumps(r, indent=1))
            print(f"[avalon] g{gid:02d} steps={r['n_steps']} "
                  f"{'TIMEOUT ' if r['timed_out'] else ''}quests={r['meta']['quest_results']} "
                  f"win={r['win_label'][:60]}", flush=True)
            return r

    results = [r for r in await asyncio.gather(*(one(g) for g in range(args.games))) if r]
    summarize(args, results, models, players, out)


def summarize(args, results, models, players, out):
    per = {m: {"seats": 0, "wins": 0, "good_seats": 0, "good_wins": 0,
               "evil_seats": 0, "evil_wins": 0} for m in models}
    for r in results:
        won = set(r["winners"])
        for pid in range(players):
            m = r["seat_models"][pid]
            good = r["meta"]["sides"][str(pid)] == "Good"
            per[m]["seats"] += 1
            per[m]["wins"] += int(pid in won)
            per[m]["good_seats" if good else "evil_seats"] += 1
            per[m]["good_wins" if good else "evil_wins"] += int(pid in won)
    rate = lambda a, b: round(a / b, 3) if b else None
    summary = {
        "env": "avalon", "game_config": "5p_classic", "players": players,
        "games": len(results),
        "good_wins": sum(1 for r in results if r["meta"]["good_victory"]),
        "evil_wins": sum(1 for r in results if not r["meta"]["good_victory"]),
        "timeouts": sum(1 for r in results if r["timed_out"]),
        "per_model": {m: {"seats": d["seats"], "win_rate": rate(d["wins"], d["seats"]),
                          "good_win_rate": rate(d["good_wins"], d["good_seats"]),
                          "evil_win_rate": rate(d["evil_wins"], d["evil_seats"])}
                      for m, d in per.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--players", type=int, default=5)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-discussion", action="store_true",
                    help="skip the per-quest open discussion round")
    ap.add_argument("--env-file", default="~/.env")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
