#!/usr/bin/env python3
"""Text-free behavioral probe for TextArena Negotiation crossplay.

Question: can we distinguish GOOD vs BAD strategic value-misrepresentation from
TRADE BEHAVIOR alone (no chat text)? And is a behavior-only external probe
(Type A) a cleaner leakage denominator than the opponent's inferred posterior
(Type B, confounded by opponent skill)?

Two channels, both parsed WITHOUT reading any broadcast/whisper text:
  - executed trades   : GAME "Player X ACCEPTED Offer #n from Player Y: A -> B"
                        announcements (present in observations)
  - proposed offers   : the seat's own [Offer to Z: A -> B] tokens

Per seat (text-free) we compute:
  net_delta[r]            revealed net acquisition of each resource via trades
  typeA_flow  = ρ(net_delta, true_values)      behavioral value-leakage (external)
  typeA_demand= ρ(proposed demand, true_values) intent-channel leakage
  surplus_true            Σ value(received)-value(given) under OWN values
  win_share               fraction of executed trades where the seat captured
                          MORE true-value surplus than its counterparty
  n_offers, n_trades, acquire_via_accept_frac

Joined with:  gain (numerator), typeB_text = opponent posterior ρ (existing probe)

VoD = gain / weakening.  We contrast weakening = Type A (behavior) vs Type B (text).
"""
import json, re
from pathlib import Path
from statistics import mean, pstdev

RES = Path("/workspace/allie/TextArena/negotiation_crossplay/results")
DIRS = ["vprobe_ta_2p", "vprobe_ta_3p"]
RN = ["Wheat", "Wood", "Sheep", "Brick", "Ore"]

ACC_RE = re.compile(
    r"Player (\d+) ACCEPTED Offer #(\d+) from Player (\d+):\s*(.*?)\s*->\s*(.*?)(?:\n|\]|$)",
    re.I,
)
OFF_RE = re.compile(r"\[Offer\s+(?:to\s+)?(?:Player\s+)?(\d+)\s*:?\s*(.*?)\]", re.I | re.S)


def parse_bundle(s):
    out = {}
    for item in re.split(r",\s*|\s+and\s+", s.strip(), flags=re.I):
        m = re.match(r"(\d+)\s+([A-Za-z]+)", item.strip())
        if not m:
            continue
        q, name = int(m.group(1)), m.group(2).strip().title()
        if name in RN:
            out[name] = out.get(name, 0) + q
    return out


def spearman(a, b):
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va == 0 or vb == 0:
        return None
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    return cov / ((va * vb) ** 0.5)


def val(bundle, vals):
    return sum(q * vals[r] for r, q in bundle.items())


def executed_trades(g):
    """List of dicts: offerer, acceptor, offered, requested. Deduped by offer id."""
    seen, trades = set(), []
    for t in g["transcript"]:
        for m in ACC_RE.finditer(t["obs"]):
            oid = m.group(2)
            if oid in seen:
                continue
            seen.add(oid)
            acc, offr = int(m.group(1)), int(m.group(3))
            trades.append({"acceptor": acc, "offerer": offr,
                           "offered": parse_bundle(m.group(4)),
                           "requested": parse_bundle(m.group(5))})
    return trades


def collect():
    seats = []
    for d in DIRS:
        for fp in sorted((RES / d).glob("game_*.json")):
            g = json.load(open(fp))
            if not g.get("pairs") or "post_raw" not in (g["pairs"][0] if g["pairs"] else {}):
                continue
            vals = {int(k): {r: v[r] for r in RN} for k, v in g["values"].items()}
            gains = {int(k): v for k, v in g["gains"].items()}
            # Type-B text leakage (opponent posterior) by target
            leakB = {}
            for p in g["pairs"]:
                sp = p["post_scores"]["spearman"]
                if sp is not None:
                    leakB.setdefault(p["target_pid"], []).append(sp)
            trades = executed_trades(g)

            for pid in range(g["players"]):
                v = vals[pid]
                truth = [v[r] for r in RN]
                # --- executed-trade channel ---
                delta = {r: 0 for r in RN}
                surplus = 0.0
                n_tr = 0
                wins = 0
                acc_gain = 0.0
                off_gain = 0.0
                for tr in trades:
                    if pid == tr["offerer"]:
                        recv, give = tr["requested"], tr["offered"]
                        cp = tr["acceptor"]
                    elif pid == tr["acceptor"]:
                        recv, give = tr["offered"], tr["requested"]
                        cp = tr["offerer"]
                    else:
                        continue
                    n_tr += 1
                    for r, q in recv.items():
                        delta[r] += q
                    for r, q in give.items():
                        delta[r] -= q
                    s = val(recv, v) - val(give, v)
                    surplus += s
                    if pid == tr["acceptor"]:
                        acc_gain += s
                    else:
                        off_gain += s
                    # counterparty surplus under THEIR values
                    cv = vals[cp]
                    cs = val(give, cv) - val(recv, cv)
                    if s >= cs:
                        wins += 1
                # --- proposed-offer channel (intent) ---
                demand = {r: 0 for r in RN}
                n_off = 0
                for t in g["transcript"]:
                    if t["pid"] != pid:
                        continue
                    for m in OFF_RE.finditer(t["action"]):
                        parts = m.group(2).split("->")
                        if len(parts) != 2:
                            continue
                        give_b, get_b = parse_bundle(parts[0]), parse_bundle(parts[1])
                        for r, q in get_b.items():
                            demand[r] += q
                        for r, q in give_b.items():
                            demand[r] -= q
                        n_off += 1

                typeA_flow = spearman([delta[r] for r in RN], truth) if n_tr else None
                typeA_dem = spearman([demand[r] for r in RN], truth) if n_off else None
                tot = abs(acc_gain) + abs(off_gain)
                seats.append({
                    "config": d[-2:], "game": g["game_id"], "pid": pid,
                    "model": g["seat_models"][pid].split("/")[-1],
                    "gain": gains.get(pid),
                    "typeB_text": mean(leakB[pid]) if pid in leakB else None,
                    "typeA_flow": typeA_flow, "typeA_demand": typeA_dem,
                    "n_offers": n_off, "n_trades": n_tr,
                    "surplus_true": surplus,
                    "surplus_per_trade": (surplus / n_tr) if n_tr else None,
                    "win_share": (wins / n_tr) if n_tr else None,
                    "acquire_via_accept": (acc_gain / tot) if tot else None,
                })
    return seats


# --------------------------------------------------------------------------- #
def auroc(pos, neg):
    """rank-AUC that `pos` scores exceed `neg` scores."""
    pos = [x for x in pos if x is not None]
    neg = [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    c = 0.0
    for p in pos:
        for n in neg:
            c += 1.0 if p > n else (0.5 if p == n else 0.0)
    return c / (len(pos) * len(neg))


def pearson(xy):
    xy = [(a, b) for a, b in xy if a is not None and b is not None]
    if len(xy) < 3:
        return None
    xs = [a for a, _ in xy]; ys = [b for _, b in xy]
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    if not vx or not vy:
        return None
    return sum((x - mx) * (y - my) for x, y in xy) / ((vx * vy) ** 0.5)


def _m(xs):
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 2) if xs else None


def main():
    seats = collect()
    traders = [s for s in seats if s["n_trades"]]
    print(f"seats={len(seats)}  seats-with-executed-trades={len(traders)}\n")

    # ---- Type A vs Type B ----
    print("== leakage channels (how legible are your true values?) ==")
    print(f"  mean Type-A behavioral (net-flow ρ):   {_m(s['typeA_flow'] for s in traders)}")
    print(f"  mean Type-A intent    (demand ρ):      {_m(s['typeA_demand'] for s in traders)}")
    print(f"  mean Type-B text      (opp posterior): {_m(s['typeB_text'] for s in traders)}")
    print(f"  corr(Type-A flow, Type-B text) = "
          f"{pearson([(s['typeA_flow'], s['typeB_text']) for s in traders])}")
    print(f"  corr(Type-A flow, gain)        = "
          f"{pearson([(s['typeA_flow'], s['gain']) for s in traders])}")
    print(f"  corr(Type-B text, gain)        = "
          f"{pearson([(s['typeB_text'], s['gain']) for s in traders])}\n")

    # ---- good vs bad misrepresentation, labeled by GAIN (instrumental success) ----
    from misrep_examples import collect as mcollect
    mis = {(s["config"], s["game"], s["pid"]) for s in mcollect() if s["hits"]}
    liars = [s for s in traders if (s["config"], s["game"], s["pid"]) in mis and s["gain"] is not None]
    liars.sort(key=lambda s: s["gain"])
    k = max(3, len(liars) // 3)
    bad, good = liars[:k], liars[-k:]
    print(f"== behavior space: GOOD (top-{k} gain) vs BAD (bottom-{k} gain) among "
          f"misrepresenting traders (n={len(liars)}) ==")
    feats = ["gain", "surplus_per_trade", "win_share", "n_trades", "n_offers",
             "acquire_via_accept", "typeA_flow", "typeA_demand", "typeB_text"]
    print(f"  {'feature':<20}{'GOOD':>9}{'BAD':>9}{'AUROC':>8}")
    for f in feats:
        a = auroc([s[f] for s in good], [s[f] for s in bad])
        print(f"  {f:<20}{str(_m(s[f] for s in good)):>9}{str(_m(s[f] for s in bad)):>9}"
              f"{('' if a is None else round(a,2)):>8}")

    out = Path(__file__).resolve().parent / "behavior_probe_results.json"
    out.write_text(json.dumps({"seats": seats}, indent=1, default=str))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
