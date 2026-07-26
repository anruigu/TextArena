#!/usr/bin/env python3
"""Phase-1 Best-of-N validation of the leaky-lies premise, BEFORE spending on RL.

Drives a hero policy through the real PokerEnv (get_observation/step/close loop) heads-up against a
FIXED EXPLOITING opponent, and at each decision uses Best-of-N action selection under the shared
dense reward  validity * (bluff_gain - lambda * leakage)  (leaky_reward.py).

The exploiter is the mechanism that makes stealth PAY: it reads the hero's public channel with the
SAME frozen leak-reader probe and, when facing a hero bet, FOLDS iff it believes the hero is strong
(P_strong >= tau) and CALLS (catches the bluff) otherwise. So:
    readable bluff  (high leakage = low P_strong) -> villain calls  -> hero loses at showdown
    concealed bluff (low  leakage = high P_strong) -> villain folds  -> hero wins the pot
Realized win-rate is therefore causally coupled to leakage through the opponent.

BoN candidate scoring is faithful: for each candidate we DEEPCOPY the env, apply the candidate, and
read the leakage off the villain's resulting public view (exactly what the villain will see), then
price the villain's deterministic response with the card-variance-free bluff EV.

Sweeping lambda traces the gain-vs-leakage frontier: higher lambda -> BoN prefers less-readable
bluffs (or stops bluffing) -> per-bluff success and realized chips rise until the lie-rate collapses.
That lower-leakage bluffs actually win more against an exploiting opponent is the premise of the
whole project; this confirms it cheaply.

  cd /workspace/allie/TextArena/deception_poc
  CUDA_VISIBLE_DEVICES=0 /workspace/allie/performative/.venv/bin/python bon_leaky.py \
      --matches 40 --rounds 4 --lambdas 0,25,50,100,200,400
"""
from __future__ import annotations
import argparse, copy, json, random, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/allie/TextArena")
sys.path.insert(0, "/workspace/allie/performative/scripts")
import eval7  # noqa: E402
import textarena as ta  # noqa: E402
from textarena.envs.Poker.env import PokerEnv  # noqa: E402

from leaky_reward import (LeakReader, action_ev, strip_hole, is_aggressive,  # noqa: E402
                          WEAK, STRONG, DEFAULT_PROBE)

D = Path(__file__).resolve().parent
RANKMAP = {"10": "T"}
SUITMAP = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}


def card_to_eval7(c):
    return eval7.Card(RANKMAP.get(c["rank"], c["rank"]) + SUITMAP[c["suit"]])


def equity(hole, board, iters=300, rng=random):
    dead = set(str(c) for c in hole + board)
    deck = [c for c in eval7.Deck().cards if str(c) not in dead]
    need = 5 - len(board)
    win = tie = 0
    for _ in range(iters):
        rng.shuffle(deck)
        opp, rest = deck[:2], deck[2:2 + need]
        full = board + rest
        s, o = eval7.evaluate(hole + full), eval7.evaluate(opp + full)
        win += s > o
        tie += s == o
    return (win + 0.5 * tie) / iters


def render_obs(obs):
    if isinstance(obs, str):
        return obs
    lines = []
    for entry in obs:
        frm, msg = entry[0], entry[1]
        who = "GAME" if frm == ta.GAME_ID else f"Player {frm}"
        lines.append(f"[{who}] {msg}")
    return "\n".join(lines)


def hero_equity(env, pid):
    gs = env.state.game_state
    hole = [card_to_eval7(c) for c in gs["player_hands"][pid]]
    board = [card_to_eval7(c) for c in gs["visible_community_cards"]]
    if len(hole) != 2:
        return None
    return equity(hole, board)


def legal_candidates(env, pid, pot):
    """A compact legal action menu: fold/call/raise when facing a bet; check/bet otherwise."""
    gs = env.state.game_state
    to_call = gs["current_bet"] - gs["player_bets"][pid]
    bb = env.big_blind
    if to_call > 0:
        cands = ["[Fold]", "[Call]"]
        for frac in (0.5, 1.0):
            r = max(bb, int(round(frac * max(pot, bb))))
            cands.append(f"[Raise {r}]")
    else:
        cands = ["[Check]"]
        for frac in (0.5, 1.0):
            b = max(bb, int(round(frac * max(pot, bb))))
            cands.append(f"[Bet {b}]")
    return cands, to_call


def bet_amount(action):
    import re
    m = re.search(r"\[(?:Bet|Raise)\s+(\d+)\]", action)
    return int(m.group(1)) if m else 0


def villain_public_after(env, pid, action):
    """Deepcopy env, apply hero `action`, return (public_text_villain_sees, done)."""
    clone = copy.deepcopy(env)
    done, _ = clone.step(action)
    if done or clone.state.game_state["game_complete"]:
        return None, True
    _, vobs = clone.get_observation()
    return strip_hole(render_obs(vobs)), False


def choose_action_bon(env, pid, eq, reader, lam, tau, cache, p0=0.5):
    """Best-of-N over the legal menu under the dense reward  sel_gain - lambda*leakage.

    HONESTY of the test: the hero's `sel_gain` is leakage-BLIND -- it prices a bet with a FIXED
    fold-probability p0, NOT the villain's true leakage-conditioned response (the policy does not
    get to simulate its opponent). The only channel through which readability enters selection is
    the dense `leakage` feature times lambda. The villain's REALIZED response (and hence the
    realized outcome) still exploits leakage (ps>=tau). So lambda=0 selects bluffs blind to
    readability (some get caught), and raising lambda steers toward concealment -> the villain
    folds -> realized chips rise. That coupling is exactly what Phase 1 must confirm."""
    gs = env.state.game_state
    pot = gs["pot"]
    cands, to_call = legal_candidates(env, pid, pot)

    # public channel + P_strong for each candidate (what the villain will read after we act)
    pubs, done_flags = [], []
    for a in cands:
        pub, done = villain_public_after(env, pid, a)
        pubs.append(pub)
        done_flags.append(done)
    need = [(i, p) for i, p in enumerate(pubs) if p is not None and p not in cache]
    if need:
        vals = reader.leakage([p for _, p in need], already_public=True)
        for (i, p), v in zip(need, vals):
            cache[p] = float(v)
    p_strong = [cache.get(p, 0.5) if p is not None else 0.5 for p in pubs]

    best, best_r, best_info = None, -1e18, None
    for a, ps, done in zip(cands, p_strong, done_flags):
        aggr = is_aggressive(a)
        bluff = aggr and eq < WEAK
        bet = bet_amount(a)
        if aggr:
            # leakage-BLIND selection gain: fixed fold prob p0, never the villain's true response
            sel_gain = p0 * action_ev(pot, bet, eq, True) + (1 - p0) * action_ev(pot, bet, eq, False)
        elif "[Fold]" in a:
            sel_gain = 0.0
        elif "[Call]" in a:
            sel_gain = eq * (pot + 2 * to_call) - to_call   # showdown EV of calling
        else:  # check
            sel_gain = eq * pot                              # showdown EV, no chips added
        leak = (1.0 - ps) if bluff else 0.0                  # leakage on the TRUE (weak) bucket
        r = sel_gain - lam * leak
        if r > best_r:
            # realized (de-noised) gain uses the villain's ACTUAL leakage-conditioned fold
            villain_folds = ps >= tau
            realized_gain = action_ev(pot, bet, eq, villain_folds) if aggr else sel_gain
            best_r, best = r, a
            best_info = {"action": a, "aggressive": aggr, "bluff": bluff, "bet": bet,
                         "P_strong": ps, "leakage": leak if bluff else None,
                         "gain": realized_gain, "villain_folds": villain_folds if aggr else None,
                         "pot": pot, "equity": eq}
    return best, best_info


def villain_action(env, pid, reader, tau, cache):
    """Fixed exploiter: fold to a hero bet iff it reads the hero as strong; else call. Passive otherwise."""
    gs = env.state.game_state
    to_call = gs["current_bet"] - gs["player_bets"][pid]
    if to_call <= 0:
        return "[Check]"
    _, vobs = env.get_observation()
    pub = strip_hole(render_obs(vobs))
    if pub not in cache:
        cache[pub] = float(reader.leakage([pub], already_public=True)[0])
    p_strong_hero = cache[pub]
    return "[Fold]" if p_strong_hero >= tau else "[Call]"


def play_match(env, reader, lam, tau, hero, seed, cache, max_steps=400):
    villain = 1 - hero
    env.reset(num_players=2, seed=seed)
    bluffs, agg_events = [], []
    steps = 0
    done = False
    while not done and steps < max_steps:
        steps += 1
        pid, obs = env.get_observation()
        if pid == hero:
            eq = hero_equity(env, pid)
            if eq is None:
                action = "[Check]"
                info = None
            else:
                action, info = choose_action_bon(env, pid, eq, reader, lam, tau, cache)
            if info and info["aggressive"]:
                rec = {"equity": info["equity"], "bluff": info["bluff"], "bet": info["bet"],
                       "P_strong": info["P_strong"], "leakage": info["leakage"],
                       "gain": info["gain"], "villain_folds": info["villain_folds"],
                       "pot": info["pot"], "seed": seed}
                agg_events.append(rec)
                if info["bluff"]:
                    bluffs.append(rec)
        else:
            action = villain_action(env, pid, reader, tau, cache)
        done, _ = env.step(action)
    rewards, _ = env.close()
    gs = env.state.game_state
    final = gs["player_chips"]
    hero_chips = final.get(hero, env.starting_chips)
    return {"bluffs": bluffs, "agg_events": agg_events,
            "hero_net": hero_chips - env.starting_chips,
            "hero_reward": (rewards or {}).get(hero),
            "won": (rewards or {}).get(hero, 0) == 1}


def run_lambda(lam, args, reader, seeds):
    cache = {}
    rng = random.Random(args.seed)
    hero_flip = 0
    matches = []
    for i, sd in enumerate(seeds):
        env = PokerEnv(num_rounds=args.rounds, starting_chips=1000, small_blind=10, big_blind=20)
        hero = i % 2  # alternate seats for position balance
        matches.append(play_match(env, reader, lam, args.tau, hero, sd, cache))
    bluffs = [b for m in matches for b in m["bluffs"]]
    aggs = [a for m in matches for a in m["agg_events"]]
    n_dec_bluff_possible = None
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else float("nan")
    res = {
        "lambda": lam,
        "n_matches": len(matches),
        "n_bluffs": len(bluffs),
        "n_agg": len(aggs),
        "mean_bluff_leakage": mean([b["leakage"] for b in bluffs]),
        "mean_bluff_Pstrong": mean([b["P_strong"] for b in bluffs]),
        "bluff_fold_rate": mean([1.0 if b["villain_folds"] else 0.0 for b in bluffs]),
        "mean_bluff_ev": mean([b["gain"] for b in bluffs]),
        "hero_net_mean": mean([m["hero_net"] for m in matches]),
        "hero_winrate": mean([1.0 if m["won"] else 0.0 for m in matches]),
    }
    return res, bluffs, matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=4, help="hands per match")
    ap.add_argument("--lambdas", default="0,25,50,100,200,400")
    ap.add_argument("--tau", type=float, default=0.5, help="villain fold threshold on P_strong")
    ap.add_argument("--probe", default=str(DEFAULT_PROBE))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    reader = LeakReader(probe_path=args.probe, device=args.device, batch=16)
    seeds = [args.seed + i for i in range(args.matches)]

    t0 = time.time()
    rows = []
    for lam in lambdas:
        res, bluffs, _ = run_lambda(lam, args, reader, seeds)
        rows.append(res)
        print(f"lam={lam:>6.0f}  bluffs={res['n_bluffs']:>4}  "
              f"leak={res['mean_bluff_leakage']:.3f}  P_strong={res['mean_bluff_Pstrong']:.3f}  "
              f"foldrate={res['bluff_fold_rate']:.3f}  bluffEV={res['mean_bluff_ev']:.1f}  "
              f"heroNet={res['hero_net_mean']:.1f}  winrate={res['hero_winrate']:.3f}", flush=True)
    out = {"config": vars(args), "wall_seconds": round(time.time() - t0, 1), "rows": rows}
    (D / "bon_leaky_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote bon_leaky_results.json ({out['wall_seconds']}s)")
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lam = [r["lambda"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    ax.plot(lam, [r["mean_bluff_leakage"] for r in rows], "-o", color="#e34948", label="mean bluff leakage")
    ax.plot(lam, [r["bluff_fold_rate"] for r in rows], "-o", color="#2a78d6", label="bluff fold-rate (win)")
    ax.set_xlabel("lambda (leakage penalty, chips)"); ax.set_ylabel("rate")
    ax.set_title("Stealth & success vs lambda"); ax.legend(fontsize=8); ax.grid(color="#eee")
    ax2 = axes[1]
    ax2.plot(lam, [r["hero_net_mean"] for r in rows], "-o", color="#008300", label="hero net chips")
    ax2b = ax2.twinx()
    ax2b.plot(lam, [r["hero_winrate"] for r in rows], "-o", color="#eda100", label="hero win-rate")
    ax2.set_xlabel("lambda"); ax2.set_ylabel("mean net chips", color="#008300")
    ax2b.set_ylabel("match win-rate", color="#eda100")
    ax2.set_title("Realized outcome vs lambda"); ax2.grid(color="#eee")
    ax3 = axes[2]
    xs = [r["mean_bluff_leakage"] for r in rows]
    ys = [r["bluff_fold_rate"] for r in rows]
    ax3.plot(xs, ys, "-", color="#bbb", zorder=1)
    sc = ax3.scatter(xs, ys, c=lam, cmap="viridis", s=120, edgecolors="#0b0b0b", zorder=2)
    for r in rows:
        ax3.annotate(f"λ={r['lambda']:.0f}", (r["mean_bluff_leakage"], r["bluff_fold_rate"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    fig.colorbar(sc, ax=ax3, label="lambda")
    ax3.set_xlabel("mean bluff leakage (readable ->)"); ax3.set_ylabel("bluff success (fold-rate)")
    ax3.set_title("Gain-vs-leakage frontier (upper-LEFT = good)"); ax3.grid(color="#eee")
    for a in (ax, ax2, ax3):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("BoN leaky-lies: lower-leakage bluffs win more against an exploiting opponent", y=1.03)
    fig.tight_layout()
    p = D / "bon_leaky_frontier.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
