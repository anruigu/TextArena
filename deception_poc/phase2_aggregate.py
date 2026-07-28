#!/usr/bin/env python3
"""Phase-2 aggregation: turn the per-fork counterfactual JSONL into per-model / per-env
summaries, a true-VoD-vs-Phase-1-leakage frontier figure, and a Term-2b leaked-type dollar-cost
table. Reads results/<env>_forks.jsonl (written incrementally by counterfactual_replay.py).

Outputs:
  deception_poc/phase2_summary.json          per-env, per-model means (VoD gain, leak cost, n)
  deception_poc/deception_frontier_phase2.png x = Phase-1 behavioral leakage, y = TRUE VoD gain
"""
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
RES = D / "results"
ENVS = ["poker", "blindauction", "newrecruit"]
COLOR = {"gpt-5.6-sol-pro": "#2a78d6", "claude-opus-4.8": "#008300", "kimi-k3": "#eda100",
         "deepseek-v4-pro": "#e87ba4", "glm-5.2": "#1baf7a", "qwen3.7-max": "#eb6834",
         "gemini-3.6-flash": "#4a3aa7", "llama-4-maverick": "#e34948"}

# Phase-1 behavioral-leakage source per env: (results file, container key, leakage key, x-label)
P1 = {
    "poker": ("poker_bluff_results.json", "models", "tell", "Phase-1 leakage: hand tell (AUROC)"),
    "blindauction": ("blindauction_vod_results.json", "by_model", "leakage",
                     "Phase-1 leakage: ρ(stated, true value)"),
    "newrecruit": ("newrecruit_vod_results.json", "by_model", "leakage",
                   "Phase-1 leakage: ρ(stated, true importance)"),
}


def load_forks(env):
    fp = RES / f"{env}_forks.jsonl"
    if not fp.exists():
        return []
    rows = []
    for line in fp.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def phase1_leakage(env):
    fn, ck, lk, _ = P1[env]
    fp = D / fn
    if not fp.exists():
        return {}
    data = json.load(open(fp)).get(ck, {})
    return {m: d.get(lk) for m, d in data.items() if d.get(lk) is not None}


def summarize():
    summary = {}
    for env in ENVS:
        rows = load_forks(env)
        if not rows:
            continue
        t1 = [r for r in rows if r.get("term") == "term1"]
        t2 = [r for r in rows if r.get("term") == "term2b"]
        models = sorted({r["model"] for r in rows})
        per_model = {}
        for m in models:
            g = [r["vod_gain"] for r in t1 if r["model"] == m]
            costkey = "leak_cost"
            c = [r[costkey] for r in t2 if r["model"] == m]
            per_model[m] = {
                "n_term1_forks": len(g),
                "vod_gain_mean": round(mean(g), 2) if g else None,
                "n_term2b_forks": len(c),
                "leak_cost_mean": round(mean(c), 2) if c else None,
            }
        allg = [r["vod_gain"] for r in t1]
        allc = [r["leak_cost"] for r in t2]
        summary[env] = {
            "n_term1_forks": len(t1), "n_term2b_forks": len(t2),
            "vod_gain_mean_all": round(mean(allg), 2) if allg else None,
            "vod_gain_positive_frac": round(sum(1 for x in allg if x > 0) / len(allg), 2) if allg else None,
            "leak_cost_mean_all": round(mean(allc), 2) if allc else None,
            "leak_cost_negative_frac": round(sum(1 for x in allc if x < 0) / len(allc), 2) if allc else None,
            "per_model": per_model,
        }
    (D / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot(summary):
    envs = [e for e in ENVS if e in summary]
    if not envs:
        print("no env data yet — nothing to plot")
        return
    fig, axes = plt.subplots(1, len(envs), figsize=(6.2 * len(envs), 5.4), squeeze=False)
    axes = axes[0]
    for ax, env in zip(axes, envs):
        leak = phase1_leakage(env)
        pm = summary[env]["per_model"]
        for m, d in pm.items():
            y = d["vod_gain_mean"]
            x = leak.get(m)
            if y is None or x is None:
                continue
            ax.scatter(x, y, s=140, c=COLOR.get(m, "#888"), edgecolors="#0b0b0b", lw=0.9, zorder=3)
            ax.annotate(m.replace("-sol-pro", "").replace("-maverick", ""), (x, y),
                        textcoords="offset points", xytext=(7, 3), fontsize=7.5, color="#0b0b0b")
        ax.axhline(0, color="#bbb", lw=1)
        ax.set_title(f"{env}  (n={summary[env]['n_term1_forks']} forks)", fontsize=11)
        ax.set_xlabel(P1[env][3], fontsize=8.5, color="#52514e")
        ax.set_ylabel("TRUE VoD gain  (deceptive − honest, live rollouts)", fontsize=8.5, color="#52514e")
        ax.grid(color="#dcdcd7", lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Phase-2 true-counterfactual VoD gain vs Phase-1 behavioral leakage  ·  "
                 "y>0 = the deceptive line beat honest disclosure against live opponents",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    p = D / "deception_frontier_phase2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")


def plot_term2b(summary):
    """Per-model Term-2b leaked-type cost, grouped by env — the headline sign-flip: being read
    HURTS in adversarial poker (bars < 0) but HELPS in integrative NewRecruit (bars > 0)."""
    envs = [e for e in ENVS if e in summary and summary[e]["n_term2b_forks"] > 0]
    if not envs:
        return
    fig, axes = plt.subplots(1, len(envs), figsize=(5.6 * len(envs), 4.6), squeeze=False)
    axes = axes[0]
    for ax, env in zip(axes, envs):
        pm = {m: d for m, d in summary[env]["per_model"].items() if d["leak_cost_mean"] is not None}
        ms = list(pm.keys())
        vals = [pm[m]["leak_cost_mean"] for m in ms]
        ax.bar(range(len(ms)), vals, color=[COLOR.get(m, "#888") for m in ms],
               edgecolor="#0b0b0b", lw=0.8)
        ax.axhline(0, color="#333", lw=1.2)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([m.replace("-sol-pro", "").replace("-v4-pro", "") for m in ms],
                           rotation=30, ha="right", fontsize=8)
        allc = summary[env]["leak_cost_mean_all"]
        ax.set_title(f"{env}  (mean {allc:+.0f}, n={summary[env]['n_term2b_forks']})", fontsize=11)
        ax.set_ylabel("Term-2b leaked-type cost\n(revealed − hidden payoff)", fontsize=8.5, color="#52514e")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(axis="y", color="#dcdcd7", lw=0.6)
    fig.suptitle("Term-2b: dollar-cost of your type leaking  ·  < 0 = being read HURTS (adversarial) · "
                 "> 0 = being read HELPS (integrative logrolling)", fontsize=11.5, y=1.02)
    fig.tight_layout()
    p = D / "deception_term2b_phase2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")


def main():
    s = summarize()
    print(json.dumps(s, indent=2))
    plot(s)
    plot_term2b(s)


if __name__ == "__main__":
    main()
