#!/usr/bin/env python3
"""Does lie-consistency explain the small-model bad-liar result in negotiation?

Earlier result (prompt-arm negotiation evals): small models, when licensed to
misrepresent, lie MORE but leak MORE of their private values and do NOT gain more.
Hypothesis tested here: the mechanism is SUSTAINABILITY — small models cannot keep a
false story internally consistent, and every self-contradiction is a tell that lets
readers recover the true values (leakage) and wipes out the lie's payoff.

Data: vprobe_ta_2p / vprobe_ta_3p (the runs that carry an actual LLM value-reader:
`pairs` post_scores.spearman = how well opponents recovered the target's true value
ranking after the game) + the lie episodes from negotiation_sustain.json (restricted
to those dirs). Per model:
  talk        = talk messages per seat (exposure)
  lie_rate    = lie episodes per 10 talk messages (frequency, exposure-normalized)
  flip_rate   = P(lie story self-contradicts) — the consistency failure
  sustain     = mean story lifetime in talk messages
  leakage     = mean spearman of opponents' post-game read of this model's values
  gain        = mean end-of-game value gain
Correlations reported over models, and over individual player-games (flip vs leakage).

  python3 smallmodel_negotiation.py
"""
from __future__ import annotations
import json, re
from pathlib import Path

D = Path(__file__).resolve().parent
BASE = Path("/workspace/allie/TextArena/negotiation_crossplay/results")
DIRS = ["vprobe_ta_2p", "vprobe_ta_3p"]
BCAST_RE = re.compile(r"\[Broadcast(?:\s*:)?\s*(.*?)\]", re.DOTALL)
WHISP_RE = re.compile(r"\[Whisper to (\d+)\s*:?\s*(.*?)\]", re.DOTALL)
SMALL = {"qwen3.5-9b", "qwen3.5-27b", "qwen3.6-27b", "gemma-4-31b-it"}


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def seats():
    """One record per (dir, game, pid): model, talk msgs, leakage-of-self, gain."""
    out = {}
    for dname in DIRS:
        for fp in sorted((BASE / dname).glob("game_*.json")):
            g = json.load(open(fp))
            gains = {int(k): v for k, v in (g.get("gains") or {}).items()}
            n_talk = {int(k): 0 for k in g["values"]}
            for t in g["transcript"]:
                a = t.get("action") or ""
                n = len([x for x in BCAST_RE.findall(a) if x.strip()])
                n += len([x for _t, x in WHISP_RE.findall(a) if x.strip()])
                n_talk[t["pid"]] = n_talk.get(t["pid"], 0) + n
            leak = {}
            for p in g.get("pairs", []):
                sp = (p.get("post_scores") or {}).get("spearman")
                if sp is not None:
                    leak.setdefault(p["target_pid"], []).append(sp)
            for pid in n_talk:
                model = g["seat_models"][pid].split("/")[-1]
                out[f"{dname}/{g['game_id']}:{pid}"] = {
                    "model": model, "talk": n_talk[pid],
                    "leakage": (sum(leak[pid]) / len(leak[pid])) if pid in leak else None,
                    "gain": gains.get(pid)}
    return out


def main():
    st = seats()
    eps = [e for e in json.loads((D / "negotiation_sustain.json").read_text())["episodes"]
           if e["game"].split("/")[0] in DIRS]
    for e in eps:
        key = f"{e['game']}:{e['pid']}"
        if key in st:
            st[key].setdefault("eps", []).append(e)

    models = sorted({s["model"] for s in st.values()})
    print(f"{'model':<18}{'seats':>6}{'talk/seat':>10}{'lies/10msg':>11}{'flip%':>7}"
          f"{'sustain':>9}{'leakage':>9}{'gain':>8}")
    rows = []
    for m in models:
        ss = [s for s in st.values() if s["model"] == m]
        talk = sum(s["talk"] for s in ss)
        es = [e for s in ss for e in s.get("eps", [])]
        flips = [e for e in es if e["fate"] == "contradicted_internal"]
        leak = [s["leakage"] for s in ss if s["leakage"] is not None]
        gain = [s["gain"] for s in ss if s["gain"] is not None]
        r = {"model": m, "n_seats": len(ss), "talk_per_seat": talk / len(ss),
             "lie_per_10msg": 10 * len(es) / talk if talk else None,
             "flip_rate": len(flips) / len(es) if es else None,
             "sustain": (sum(e["t_end"] - e["t0"] for e in es) / len(es)) if es else None,
             "leakage": sum(leak) / len(leak) if leak else None,
             "gain": sum(gain) / len(gain) if gain else None,
             "small": m in SMALL}
        rows.append(r)
        f = lambda v, s="{:.2f}": s.format(v) if v is not None else "    -"
        print(f"{m:<18}{len(ss):>6}{talk/len(ss):>10.1f}{f(r['lie_per_10msg']):>11}"
              f"{f(r['flip_rate'], '{:.0%}'):>7}{f(r['sustain']):>9}"
              f"{f(r['leakage']):>9}{f(r['gain'], '{:.0f}'):>8}")

    # model-level correlations
    print("\nmodel-level Pearson r:")
    for xk, yk in (("flip_rate", "leakage"), ("flip_rate", "gain"),
                   ("lie_per_10msg", "leakage"), ("lie_per_10msg", "gain"),
                   ("sustain", "gain"), ("sustain", "leakage")):
        r = pearson([r[xk] for r in rows], [r[yk] for r in rows])
        print(f"  {xk:>13} vs {yk:<8} r={r:+.2f}")

    # seat-level: does a seat WITH a self-contradiction leak more / gain less?
    with_flip, no_flip = [], []
    for s in st.values():
        es = s.get("eps", [])
        if not es:
            continue
        bucket = with_flip if any(e["fate"] == "contradicted_internal" for e in es) \
            else no_flip
        bucket.append(s)
    for name, b in (("liar seats WITH self-contradiction", with_flip),
                    ("liar seats, story held", no_flip)):
        lk = [s["leakage"] for s in b if s["leakage"] is not None]
        gn = [s["gain"] for s in b if s["gain"] is not None]
        print(f"\n{name}: n={len(b)}  leakage={sum(lk)/len(lk):.3f} (n={len(lk)})"
              f"  gain={sum(gn)/len(gn):.0f}" if lk and gn else f"\n{name}: n={len(b)}")

    (D / "smallmodel_negotiation.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {D/'smallmodel_negotiation.json'}")


if __name__ == "__main__":
    main()
