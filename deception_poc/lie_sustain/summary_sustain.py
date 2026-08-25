#!/usr/bin/env python3
"""Cross-game synthesis of the lie-sustainability metric (lie-scaling-0801.md).

Loads every <env>_sustain.json produced by the per-game scripts and reports the
verifiability taxonomy: how a lie dies depends on the env's evidence channel.

  fixed fuse        : liarsdice (call ends every round), ipd / pgg (actions revealed
                      every round) — evidence arrives unconditionally and fast.
  defusable fuse    : poker (showdown only if nobody folds), coup (challenge only if
                      an opponent risks calling) — the liar/table controls T_v.
  consistency-only  : mafia (this env never flips roles), newrecruit / negotiation
                      (values never revealed) — E_t vacuous, only internal
                      self-contradiction can kill a story.

  python3 summary_sustain.py
"""
from __future__ import annotations
import json
from pathlib import Path

D = Path(__file__).resolve().parent
ENVS = [  # (env, class, fuse note)
    ("liarsdice", "fixed", "call ends round"),
    ("ipd", "fixed", "round reveal"),
    ("pgg", "fixed", "round reveal"),
    ("poker", "defusable", "showdown iff no fold"),
    ("coup", "defusable", "challenge optional"),
    ("mafia", "consistency", "no role flips"),
    ("newrecruit", "consistency", "values never shown"),
    ("negotiation", "consistency", "values never shown"),
]
CLS_COLOR = {"fixed": "#e34948", "defusable": "#eda100", "consistency": "#1baf7a"}
FATE_COLOR = {"contradicted_evidence": "#e34948", "contradicted_internal": "#8e44ad",
              "resolved_unrefuted": "#eda100", "survived_censored": "#1baf7a"}
FATE_LABEL = {"contradicted_evidence": "evidence-contradicted",
              "contradicted_internal": "self-contradicted",
              "resolved_unrefuted": "resolved safe (defused)",
              "survived_censored": "survived to game end"}


def load():
    out = {}
    for env, cls, note in ENVS:
        fp = D / f"{env}_sustain.json"
        if fp.exists():
            d = json.loads(fp.read_text())
            out[env] = {"cls": cls, "note": note, "eps": d["episodes"],
                        "per_model": d["per_model"]}
    return out


def table(data):
    print(f"{'env':<12}{'class':<13}{'lies':>6}{'evid%':>7}{'self%':>7}{'safe%':>7}"
          f"{'surv%':>7}{'med sustain':>13}{'cash<T':>8}  unit")
    rows = []
    for env, cls, note in ENVS:
        if env not in data:
            continue
        eps = data[env]["eps"]
        n = len(eps)
        f = lambda k: sum(1 for e in eps if e["fate"] == k) / n
        caught = [e for e in eps if e["fate"].startswith("contradicted")]
        cash_pre = (sum(1 for e in caught if e["cashed"]) / len(caught)) if caught else None
        sus = sorted(e["t_end"] - e["t0"] for e in eps)
        med = sus[n // 2]
        unit = eps[0]["unit"]
        rows.append({"env": env, "cls": cls, "n": n,
                     "evid": f("contradicted_evidence"), "internal": f("contradicted_internal"),
                     "safe": f("resolved_unrefuted"), "cens": f("survived_censored"),
                     "med": med, "unit": unit, "cash_pre": cash_pre})
        print(f"{env:<12}{cls:<13}{n:>6}{f('contradicted_evidence')*100:>6.0f}%"
              f"{f('contradicted_internal')*100:>6.0f}%{f('resolved_unrefuted')*100:>6.0f}%"
              f"{f('survived_censored')*100:>6.0f}%{med:>10} {unit:<4}"
              f"{('%.2f' % cash_pre) if cash_pre is not None else '   -':>8}  {note}")
    return rows


def model_grid(data):
    """model x env: P(story never contradicted) = 1 - caught_rate."""
    grid = {}
    for env in data:
        for m, d in data[env]["per_model"].items():
            grid.setdefault(m, {})[env] = 1 - d["caught_rate"]
    return grid


def plot(rows, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.6),
                                        gridspec_kw={"width_ratios": [1.2, 0.8, 1.3]})

    # P1: fate composition per env, ordered by fuse class
    envs = [r["env"] for r in rows]
    bottoms = [0.0] * len(rows)
    for fate in ("contradicted_evidence", "contradicted_internal",
                 "resolved_unrefuted", "survived_censored"):
        key = {"contradicted_evidence": "evid", "contradicted_internal": "internal",
               "resolved_unrefuted": "safe", "survived_censored": "cens"}[fate]
        vals = [r[key] for r in rows]
        ax1.bar(range(len(rows)), vals, bottom=bottoms, color=FATE_COLOR[fate],
                edgecolor="#0b0b0b", lw=0.5, label=FATE_LABEL[fate])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, r in enumerate(rows):
        ax1.annotate(r["cls"], (i, 1.02), ha="center", fontsize=7,
                     color=CLS_COLOR[r["cls"]])
        ax1.annotate(f"n={r['n']}", (i, -0.07), ha="center", fontsize=7)
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels(envs, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("share of lie episodes")
    ax1.set_ylim(0, 1.12)
    ax1.legend(fontsize=7, loc="center left", bbox_to_anchor=(0, -0.32), ncol=2)
    ax1.set_title("How lies die, by evidence channel\n(evidence needs a reveal; "
                  "self-contradiction needs none)", fontsize=10)

    # P2: P(story never provably contradicted) per env, class-colored
    surv = [r["safe"] + r["cens"] for r in rows]
    ax2.bar(range(len(rows)), surv, color=[CLS_COLOR[r["cls"]] for r in rows],
            edgecolor="#0b0b0b")
    for i, v in enumerate(surv):
        ax2.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels(envs, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("P(lie never contradicted)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Story survival by fuse type\n(fixed fuse kills all; poker escapes "
                  "by ending the hand first)", fontsize=10)

    # P3: model x env story-survival heatmap (shared frontier envs)
    grid = model_grid(data)
    envs3 = [e for e in ("liarsdice", "ipd", "pgg", "poker", "coup", "mafia",
                         "newrecruit") if e in data]
    models = sorted({m for m in grid if sum(1 for e in envs3 if e in grid[m]) >= 3},
                    key=lambda m: -sum(grid[m].get(e, 0) for e in envs3)
                    / max(1, sum(1 for e in envs3 if e in grid[m])))
    import numpy as np
    M = np.full((len(models), len(envs3)), np.nan)
    for i, m in enumerate(models):
        for j, e in enumerate(envs3):
            if e in grid[m]:
                M[i, j] = grid[m][e]
    im = ax3.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax3.set_xticks(range(len(envs3)))
    ax3.set_xticklabels(envs3, rotation=20, ha="right", fontsize=9)
    ax3.set_yticks(range(len(models)))
    ax3.set_yticklabels([m.replace("-sol-pro", "").replace("-maverick", "")
                         for m in models], fontsize=8)
    for i in range(len(models)):
        for j in range(len(envs3)):
            if M[i, j] == M[i, j]:
                ax3.annotate(f"{M[i, j]:.2f}", (j, i), ha="center", va="center",
                             fontsize=7)
    ax3.set_title("P(lie never contradicted), model x env\n(negotiation pool disjoint "
                  "-> excluded)", fontsize=10)
    fig.colorbar(im, ax=ax3, shrink=0.8)
    for a in (ax1, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(D / "lie_sustain_summary.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {D/'lie_sustain_summary.png'}")


def main():
    data = load()
    rows = table(data)
    (D / "lie_sustain_summary.json").write_text(json.dumps(rows, indent=1))
    plot(rows, data)


if __name__ == "__main__":
    main()
