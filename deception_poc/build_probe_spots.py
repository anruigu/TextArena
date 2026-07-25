#!/usr/bin/env python3
"""Build hidden-type decision spots for the Term-2 ACTIVATION probe (runs in the TextArena venv,
which has eval7). Emits spots.jsonl: one poker decision per line with the exact private type
(hand equity) as ground truth, the observation the acting player saw, and the frontier model's
realized action. The activation-probe script (performative venv, torch) consumes this.

Two sources, combined for n:
  (a) REAL spots from poker_crossplay transcripts (faithful game states + real actions).
  (b) SYNTHETIC spots: random hole+board dealt fresh, equity by eval7, templated into the same
      observation format — balanced and plentiful, ideal for a linear probe.
"""
import json, random, re, sys
from pathlib import Path
import eval7

random.seed(1)
RESULTS = Path("/workspace/allie/TextArena/poker_crossplay/results")
OUT = Path("/workspace/allie/TextArena/deception_poc/spots.jsonl")

RANKMAP = {"10": "T"}
SUITMAP = {"♥": "h", "♠": "s", "♦": "d", "♣": "c"}
HOLE_RE = re.compile(r"Your hole:\s*([^\n]+)")
BOARD_RE = re.compile(r"Visible board:\s*\[([^\]]*)\]")
POT_RE = re.compile(r"Pot:\s*(\d+)")


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
        win += s > o; tie += s == o
    return (win + 0.5 * tie) / iters


def is_aggr(a):
    return bool(re.search(r"\[(Bet|Raise)\b", a or "", re.I))


def real_spots():
    spots = []
    for fp in sorted(RESULTS.glob("game_*.json")):
        g = json.load(open(fp))
        for t in g["transcript"]:
            obs = t.get("obs", "")
            hm = HOLE_RE.search(obs)
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
            spots.append({"source": "real", "obs": obs.strip(), "equity": round(eq, 4),
                          "aggressive_frontier": is_aggr(t.get("action", ""))})
    return spots


CARD_STR = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
            10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
SUIT_STR = {"h": "♥", "s": "♠", "d": "♦", "c": "♣"}


def _fmt(card):
    s = str(card)  # e.g. 'Ah'
    r = s[:-1]; su = s[-1]
    r = {"T": "10"}.get(r, r)
    return f"{r}{SUIT_STR.get(su, su)}"


def synth_spots(n=700):
    spots = []
    deck0 = eval7.Deck().cards
    for _ in range(n):
        deck = list(deck0)
        random.shuffle(deck)
        street = random.choice([0, 3, 4, 5])
        hole = deck[:2]
        board = deck[2:2 + street]
        eq = equity(hole, board)
        if eq is None:
            continue
        pot = random.choice([40, 60, 80, 120, 160, 240, 320])
        cur = random.choice([0, 20, 40, 60])
        hole_s = ", ".join(_fmt(c) for c in hole)
        board_s = ", ".join(_fmt(c) for c in board)
        obs = (f"===== Hand X - {'Pre-flop' if street==0 else 'Flop' if street==3 else 'Turn' if street==4 else 'River'} =====\n"
               f"Pot: {pot} | Current bet: {cur}\nVisible board: [{board_s}]\n"
               f"Your hole: {hole_s}\nYour stack: 1000. It is your turn to act.")
        spots.append({"source": "synth", "obs": obs, "equity": round(eq, 4),
                      "aggressive_frontier": None})
    return spots


def main():
    n_synth = int(sys.argv[1]) if len(sys.argv) > 1 else 700
    spots = real_spots() + synth_spots(n_synth)
    random.shuffle(spots)
    OUT.write_text("\n".join(json.dumps(s) for s in spots) + "\n")
    eqs = [s["equity"] for s in spots]
    strong = sum(1 for e in eqs if e >= 0.6); weak = sum(1 for e in eqs if e < 0.4)
    print(f"wrote {len(spots)} spots -> {OUT}  (real {sum(1 for s in spots if s['source']=='real')}, "
          f"synth {sum(1 for s in spots if s['source']=='synth')}) | strong≥0.6: {strong}, weak<0.4: {weak}")


if __name__ == "__main__":
    main()
