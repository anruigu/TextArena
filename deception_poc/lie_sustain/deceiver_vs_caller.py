#!/usr/bin/env python3
"""Is a good deceiver (sustain side) also a good BS-caller (exploit/reward side)?

Joins, per (model, env):
  deceiver skill  = 1 - caught_rate           (lie survival; from <env>_sustain.json)
                    cashed_rate               (value banked before dying)
  caller skill    = youden_j                  (discrimination; from <env>_exploit.json)
                    mean_G_norm               (discounted reward; from reward.json)
Then Spearman across cells and across per-model aggregates.
"""
from __future__ import annotations
import json
from pathlib import Path

D = Path(__file__).resolve().parent
ENVS = ["liarsdice", "ipd", "pgg", "poker", "coup", "avalon", "mafia",
        "newrecruit", "negotiation"]
# envs where deception has headroom (fixed-fuse envs catch ~everyone, no variance)
HEADROOM = {"poker", "coup", "mafia", "newrecruit", "negotiation", "avalon"}


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None, n
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return (num / (dx * dy) if dx and dy else None), n


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else None


# ---- caller side: reward.json (mean_G_norm) + <env>_exploit.json (youden_j)
reward = json.loads((D / "reward.json").read_text())
Gcell = {}   # (model, env) -> mean_G_norm
for k, c in reward["per_model_env"].items():
    Gcell[(c["model"], c["env"])] = c["mean_G_norm"]

Jcell = {}   # (model, env) -> defender youden_j
for env in ENVS:
    fp = D / f"{env}_exploit.json"
    if not fp.exists():
        continue
    pm = json.loads(fp.read_text()).get("per_model", {})
    for m, d in pm.items():
        if d.get("youden_j") is not None:
            Jcell[(m, env)] = d["youden_j"]

# ---- deceiver side: <env>_sustain.json per_model
Dcell = {}   # (model, env) -> {surv, cashed, caught, n_lies}
for env in ENVS:
    fp = D / f"{env}_sustain.json"
    if not fp.exists():
        continue
    pm = json.loads(fp.read_text()).get("per_model", {})
    for m, d in pm.items():
        cr = d.get("caught_rate")
        Dcell[(m, env)] = {
            "surv": (1 - cr) if cr is not None else None,
            "cashed": d.get("cashed_rate"),
            "caught": cr,
            "n_lies": d.get("n_lies", 0),
        }

# ---- cell-level join
print("=== per (model, env) cells ===")
print(f"{'model':<20}{'env':<12}{'n_lie':>6}{'surv':>7}{'cashed':>8}{'callJ':>7}{'Gnorm':>8}")
rows = []
for key in sorted(set(Dcell) & (set(Jcell) | set(Gcell))):
    m, env = key
    dd = Dcell[key]
    j = Jcell.get(key)
    g = Gcell.get(key)
    if dd["surv"] is None or dd["n_lies"] < 3:
        continue
    rows.append((m, env, dd, j, g))
    print(f"{m:<20}{env:<12}{dd['n_lies']:>6}{dd['surv']:>7.2f}"
          f"{(dd['cashed'] if dd['cashed'] is not None else float('nan')):>8.2f}"
          f"{(j if j is not None else float('nan')):>7.2f}"
          f"{(g if g is not None else float('nan')):>8.2f}")


def corr_block(title, pairs):
    xs = [x for x, y in pairs]
    ys = [y for x, y in pairs]
    rs, n = spearman(xs, ys)
    rp = pearson(xs, ys)
    print(f"  {title:<42} n={n:<3} spearman={rs:+.3f}  pearson={rp:+.3f}"
          if rs is not None else f"  {title:<42} n={len(xs)} (too few)")


print("\n=== correlations across cells (deceiver vs caller) ===")
# survival vs caller-J
p = [(d["surv"], j) for m, e, d, j, g in rows if j is not None]
corr_block("lie-survival  vs  caller Youden J  (all envs)", p)
p = [(d["surv"], j) for m, e, d, j, g in rows if j is not None and e in HEADROOM]
corr_block("lie-survival  vs  caller Youden J  (headroom)", p)
p = [(d["surv"], g) for m, e, d, j, g in rows if g is not None]
corr_block("lie-survival  vs  caller G_norm    (all envs)", p)
p = [(d["cashed"], j) for m, e, d, j, g in rows if j is not None and d["cashed"] is not None]
corr_block("lie-cashed    vs  caller Youden J  (all envs)", p)

# ---- per-model aggregate (weighted by n_lies), across the shared frontier envs
print("\n=== per-model aggregate (n_lie-weighted mean over envs) ===")
bym = {}
for m, e, d, j, g in rows:
    bym.setdefault(m, []).append((d, j, g))
print(f"{'model':<20}{'envs':>5}{'lies':>6}{'surv':>7}{'callJ':>7}{'Gnorm':>8}")
agg = []
for m, lst in sorted(bym.items()):
    w = sum(d["n_lies"] for d, j, g in lst)
    surv = sum(d["surv"] * d["n_lies"] for d, j, g in lst) / w
    js = [j for d, j, g in lst if j is not None]
    gs = [g for d, j, g in lst if g is not None]
    callj = sum(js) / len(js) if js else None
    gnorm = sum(gs) / len(gs) if gs else None
    agg.append((m, surv, callj, gnorm))
    print(f"{m:<20}{len(lst):>5}{w:>6}{surv:>7.2f}"
          f"{(callj if callj is not None else float('nan')):>7.2f}"
          f"{(gnorm if gnorm is not None else float('nan')):>8.2f}")

print()
p = [(s, j) for m, s, j, g in agg if j is not None]
corr_block("model deceiver(surv) vs caller Youden J", p)
p = [(s, g) for m, s, j, g in agg if g is not None]
corr_block("model deceiver(surv) vs caller G_norm", p)
