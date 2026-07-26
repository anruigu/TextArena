#!/usr/bin/env python3
"""Build OPPONENT-VIEW (public-channel) poker decision spots for the leak-READER probe
(runs in the TextArena venv, which has eval7). Emits reader_spots.jsonl: one poker decision
per line where `obs` is the view the OPPONENT sees -- the running betting history, board, pot
and stacks, with the actor's `Your hole:` line STRIPPED -- labeled with the actor's true
hand-strength (eval7 equity) bucket.

This is the detector that DEFINES leakage in the arms race: a frozen reader model ingests only
the public channel; a linear probe on its residual (train_reader_probe.py) recovers the actor's
hidden strength. Unlike activation_probe.py / build_probe_spots.py (which keep the hole cards, so
the probe re-decodes visible information -- see ACTIVATION_PROBE_FINDINGS.md), here the strength is
GENUINELY HIDDEN from the reader; any recovery is real type-leakage through the betting behavior.

Two sources, combined for n:
  (a) REAL spots from poker_crossplay transcripts: faithful public game states + the actor's real
      prior betting line (the leakage signal), hole line removed. These are the gold spots.
  (b) SYNTHETIC public sequences: random hole+board dealt fresh (equity by eval7); a compact,
      strength-correlated prior betting line is simulated for the actor and shown in the public
      history (hole never shown). Plentiful and balanced for a linear probe; the tell strength is
      heuristic (documented), so treat AUROC on synth-only as an upper-scaffold, not the headline.

  /workspace/allie/TextArena/.venv/bin/python build_reader_spots.py [n_synth]
"""
import json, random, re, sys
from pathlib import Path
import eval7

random.seed(1)
RESULT_DIRS = [Path("/workspace/allie/TextArena/poker_crossplay/results"),
               Path("/workspace/allie/TextArena/poker_crossplay/results_more")]
OUT = Path("/workspace/allie/TextArena/deception_poc/reader_spots.jsonl")

RANKMAP = {"10": "T"}
SUITMAP = {"♥": "h", "♠": "s", "♦": "d", "♣": "c"}
HOLE_RE = re.compile(r"^.*Your hole:.*$\n?", re.MULTILINE)
HOLE_CARDS_RE = re.compile(r"Your hole:\s*([^\n]+)")
BOARD_RE = re.compile(r"Visible board:\s*\[([^\]]*)\]")
POT_RE = re.compile(r"Pot:\s*(\d+)")
AGGR_RE = re.compile(r"\[(Bet|Raise)\b", re.I)


def to_card(tok):
    tok = tok.strip()
    if not tok:
        return None
    suit = SUITMAP.get(tok[-1])
    rank = RANKMAP.get(tok[:-1], tok[:-1])
    if suit is None or not rank:
        return None
    try:
        return eval7.Card(rank + suit)
    except Exception:
        return None


def parse_cards(text):
    return [c for c in (to_card(t) for t in text.split(",")) if c is not None]


def equity(hole, board, iters=400):
    if len(hole) != 2:
        return None
    dead = set(str(c) for c in hole + board)
    deck = [c for c in eval7.Deck().cards if str(c) not in dead]
    need = 5 - len(board)
    win = tie = 0
    for _ in range(iters):
        random.shuffle(deck)
        opp, rest = deck[:2], deck[2:2 + need]
        full = board + rest
        s, o = eval7.evaluate(hole + full), eval7.evaluate(opp + full)
        win += s > o
        tie += s == o
    return (win + 0.5 * tie) / iters


def strip_hole(obs):
    """Remove the actor's private `Your hole: ...` line -> the opponent's public view."""
    return HOLE_RE.sub("", obs).strip()


def street_of(board):
    return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(len(board), f"b{len(board)}")


def real_spots():
    spots = []
    fps = [(rd, fp) for rd in RESULT_DIRS if rd.exists()
           for fp in sorted(rd.glob("game_*.json"))]
    for rd, fp in fps:
        g = json.load(open(fp))
        gid = f"{rd.name}:{g['game_id']}"
        for t in g["transcript"]:
            obs = t.get("obs", "")
            hm = HOLE_CARDS_RE.search(obs)
            if not hm:
                continue
            hole = parse_cards(hm.group(1))
            if len(hole) != 2:
                continue
            bm = BOARD_RE.search(obs)
            board = parse_cards(bm.group(1)) if bm else []
            eq = equity(hole, board)
            if eq is None:
                continue
            pub = strip_hole(obs)
            spots.append({
                "source": "real", "game": gid, "pid": t["pid"],
                "obs": pub, "equity": round(eq, 4), "street": street_of(board),
                "actor_action": (t.get("action", "") or "").strip()[:16],
                "actor_aggressive": bool(AGGR_RE.search(t.get("action", "") or "")),
            })
    return spots


# ---------- synthetic public sequences (heuristic, strength-correlated tell) ----------
CARD_STR = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
            10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
SUIT_STR = {"h": "♥", "s": "♠", "d": "♦", "c": "♣"}
STREET_NAME = {3: "Flop", 4: "Turn", 5: "River"}
PRIOR_STREETS = {3: [("Pre‑flop", 0)], 4: [("Pre‑flop", 0), ("Flop", 3)],
                 5: [("Pre‑flop", 0), ("Flop", 3), ("Turn", 4)]}


def _fmt(card):
    s = str(card)
    r = {"T": "10"}.get(s[:-1], s[:-1])
    return f"{r}{SUIT_STR.get(s[-1], s[-1])}"


def _heuristic_action(eq, pot, tell=0.7):
    """Actor's public action, strength-correlated with probability `tell` (else uniform noise).
    Returns (verb_line, chips_added, new_current_bet). Strong -> bet, medium -> check/call, weak -> check."""
    r = random.random()
    if r < tell:
        if eq >= 0.6:
            bet = random.choice([40, 60, 80])
            return f"bets to {bet}", bet, bet
        if eq < 0.4:
            return "checks", 0, 0
        return ("calls" if random.random() < 0.5 else "checks"), 0, 0
    # noise: act against type sometimes (this is exactly the stealth an arms race would learn)
    if random.random() < 0.5:
        bet = random.choice([40, 60])
        return f"bets to {bet}", bet, bet
    return "checks", 0, 0


def synth_spots(n=600, tell=0.7):
    spots = []
    deck0 = eval7.Deck().cards
    tries = 0
    while len(spots) < n and tries < n * 4:
        tries += 1
        deck = list(deck0)
        random.shuffle(deck)
        target_len = random.choice([3, 4, 5])  # flop/turn/river (need a prior betting line)
        hole = deck[:2]
        board = deck[2:2 + target_len]
        eq = equity(hole, board)
        if eq is None:
            continue
        actor = random.choice([0, 1])
        opp = 1 - actor
        pot = 40
        lines = [f"[GAME] Starting a new 10-round Texas Hold'em game with 2 players."]
        for sname, slen in PRIOR_STREETS[target_len]:
            comm = ", ".join(_fmt(c) for c in board[:slen])
            lines.append(f"[GAME] ===== Hand 1 / 10 - {sname} =====")
            lines.append(f"[GAME] Visible board: [{comm}]")
            # actor acts first this street; its action carries the leakage
            verb, add, _ = _heuristic_action(eq, pot, tell)
            pot += add
            lines.append(f"[Player {actor}] {'[Bet %d]' % add if add else '[Check]'}")
            lines.append(f"[GAME] Player {actor} {verb}.")
            # opponent calls any bet (neutral responder), else checks
            if add:
                pot += add
                lines.append(f"[Player {opp}] [Call]")
                lines.append(f"[GAME] Player {opp} calls {add}.")
            else:
                lines.append(f"[Player {opp}] [Check]")
                lines.append(f"[GAME] Player {opp} checks.")
        # current decision spot (actor to act), public board shown, NO hole
        comm = ", ".join(_fmt(c) for c in board)
        cur = 0
        obs = ("\n".join(lines) + "\n"
               f"[GAME] ===== Hand 1 / 10 - {STREET_NAME[target_len]} =====\n"
               f"Pot: {pot} | Current bet: {cur}\nVisible board: [{comm}]\n"
               f"P{actor}: 960.00 chips | bet 0.00 | active\n"
               f"P{opp}: 960.00 chips | bet 0.00 | active\n"
               f"It is Player {actor}'s turn to act.\n"
               "==============================================")
        spots.append({"source": "synth", "game": f"synth_{len(spots)}", "pid": actor,
                      "obs": obs, "equity": round(eq, 4), "street": street_of(board),
                      "actor_action": None, "actor_aggressive": None})
    return spots


def main():
    n_synth = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    tell = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    spots = real_spots() + synth_spots(n_synth, tell)
    random.shuffle(spots)
    OUT.write_text("\n".join(json.dumps(s) for s in spots) + "\n")
    eqs = [s["equity"] for s in spots]
    strong = sum(1 for e in eqs if e >= 0.6)
    weak = sum(1 for e in eqs if e < 0.4)
    nreal = sum(1 for s in spots if s["source"] == "real")
    print(f"wrote {len(spots)} reader spots -> {OUT}  (real {nreal}, synth {len(spots)-nreal}) | "
          f"strong>=0.6: {strong}, weak<0.4: {weak}, mid: {len(spots)-strong-weak}")
    # sanity: no hole cards leaked into the public channel
    leaked = sum(1 for s in spots if "Your hole" in s["obs"])
    print(f"hole-card leaks into public obs: {leaked} (must be 0)")


if __name__ == "__main__":
    main()
