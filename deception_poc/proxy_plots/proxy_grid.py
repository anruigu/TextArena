#!/usr/bin/env python3
"""Cross EVERY Term-2 (leakage) proxy with EVERY Term-1 (gain) proxy, for every deception-PoC game,
to see which measurement is the most discerning (largest |model-mean Pearson r|).

Pure re-analysis of the per-model numbers already on disk in ../*_results.json — no rollouts, no
LLM. For each game we build a grid of frontier scatters (rows = gain proxy, cols = leakage proxy),
annotate each cell with the model-mean Pearson r, save <game>_grid.png, and write a single markdown
file (proxy_combinations.md) tabulating every (leakage x gain) pair per game, ranked by |r|.

Sign convention: leakage axes are "higher = more readable/transparent" unless tagged (inv=more
deceptive); gain axes are "higher = better". A NEGATIVE r on a readable-leakage axis = the
concealment-pays direction (good deceiver = low leakage + high gain).

  python3 proxy_grid.py
"""
from __future__ import annotations
import json
from pathlib import Path

D = Path(__file__).resolve().parent
SRC = D.parent  # deception_poc/

# game -> (results file, per-model dict key, leakage proxies, gain proxies)
# each proxy = (json_field, short_label, is_inverse_leakage_flag_for_display_only)
GAMES = {
    "poker": ("poker_bluff_results.json", "models",
              [("bluff_caught_rate", "bluff caught rate", False),
               ("bluff_detect_auroc", "bluff-detect AUROC", False),
               ("tell", "tell AUROC", False), ("tell_size", "bet-size tell", False),
               ("bluff_rate", "bluff rate", True)],
              [("bluff_ev", "bluff EV (de-noised)"),
               ("bluff_chips_won", "bluff chips won"), ("meanchips", "mean end chips")]),
    "kuhn": ("kuhn_bluff_results.json", "models",
             [("tell", "tell AUROC", False), ("bluff_rate", "bluff rate", True)],
             [("bluff_chips_won", "bluff chips won"),
              ("meanchips", "mean end chips")]),
    "leduc": ("leduc_vod_results.json", "models",
              [("bluff_caught_rate", "bluff caught rate", False),
               ("bluff_detect_auroc", "bluff-detect AUROC", False),
               ("tell", "tell AUROC", False), ("bluff_rate", "bluff rate", True)],
              [("bluff_ev", "bluff EV (de-noised)"),
               ("meanbank", "mean end bank")]),
    "liarsdice": ("liarsdice_vod_results.json", "models",
                  [("bluff_caught_rate", "bluff caught rate", False),
                   ("bluff_detect_auroc", "bluff-detect AUROC", False),
                   ("leakage", "leakage own_frac-1/6", False), ("mean_own_frac", "mean own_frac", False),
                   ("bluff_rate", "bluff rate", True), ("bluff_stick_rate", "bluff-stick rate", True)],
                  [("gain", "gain (rank reward)"), ("mean_reward", "mean reward")]),
    "blindauction": ("blindauction_vod_results.json", "by_model",
                     [("reader_leak_on_lies", "reader recovery on lies", False),
                      ("leakage", "leakage rho", False), ("misrep", "misrep rate", True)],
                     [("profit", "profit"), ("concealed_wins", "concealed wins")]),
    "newrecruit": ("newrecruit_vod_results.json", "by_model",
                   [("reader_leak_on_lies", "reader recovery on lies", False),
                    ("leakage", "leakage rho", False), ("misrep", "misrep rate", True)],
                   [("surplus", "surplus")]),
    "scorablegames": ("scorablegames_vod_results.json", "by_model",
                      [("reader_leak_on_lies", "reader recovery on lies", False),
                       ("leakage", "leakage rho", False), ("misrep", "misrep rate", True)],
                      [("surplus", "surplus")]),
    "negotiation": ("negotiation_vod_results.json", "by_model",
                    [("leakage", "leakage (reader->you)", False)],
                    [("gain", "value captured")]),
    "mafia": ("mafia_leakage_results.json", "per_model_role_leakage",
              [("role_leakage", "role leakage", False)],
              [("mafia_win_rate", "mafia win rate"), ("mafia_survival_rate", "mafia survival")]),
    "coup": ("coup_leakage_results.json", "per_model",
             [("claim_honesty", "claim honesty", False), ("bluff_caught_rate", "bluff caught rate", False),
              ("table_read_auroc", "table-read AUROC", False), ("bluff_rate", "bluff rate", True)],
             [("win_rate", "win rate"), ("survival_rate", "survival rate")]),
}

# Conditioning class of each LEAKAGE (Term-2) proxy, from a source-code audit of how the field is
# computed (see best_conditioned.py header). Key = (game, json_field). Values:
#   "A" = lie-CONDITIONED readability: denominator is lie events, OR a class-separability/posterior
#         measure that is base-rate invariant (not a frequency). NOT confounded with lie rate.
#   "B" = RAW RATE: literally the frequency of lying (e.g. bluff_rate, misrep, claim_honesty).
#   "C" = UNCONDITIONED-MIX: a readability/correlation computed over ALL actions (honest+deceptive),
#         so it moves mechanically with the lie rate even at fixed per-lie behavior.
# Only "A" is safe from the raw-deception-rate confound.
COND = {
    # class-A conditioned leakage built from traces (P(opp calls|bluff) + bluff-vs-value AUROC):
    ("poker", "bluff_caught_rate"): "A", ("poker", "bluff_detect_auroc"): "A",
    ("leduc", "bluff_caught_rate"): "A", ("leduc", "bluff_detect_auroc"): "A",
    ("liarsdice", "bluff_caught_rate"): "A", ("liarsdice", "bluff_detect_auroc"): "A",
    ("poker", "tell"): "C", ("poker", "tell_size"): "C", ("poker", "bluff_rate"): "B",
    ("kuhn", "tell"): "C", ("kuhn", "bluff_rate"): "B",
    ("leduc", "tell"): "C", ("leduc", "bluff_rate"): "B",
    ("liarsdice", "leakage"): "C", ("liarsdice", "mean_own_frac"): "C",
    ("liarsdice", "bluff_rate"): "B", ("liarsdice", "bluff_stick_rate"): "A",
    # class-A conditioned leakage from the LLM reader pass (recovery-of-truth on lied items):
    ("blindauction", "reader_leak_on_lies"): "A", ("newrecruit", "reader_leak_on_lies"): "A",
    ("scorablegames", "reader_leak_on_lies"): "A",
    ("blindauction", "leakage"): "C", ("blindauction", "misrep"): "B",
    ("newrecruit", "leakage"): "C", ("newrecruit", "misrep"): "B",
    ("scorablegames", "leakage"): "C", ("scorablegames", "misrep"): "B",
    ("negotiation", "leakage"): "A",
    ("mafia", "role_leakage"): "A",
    ("coup", "claim_honesty"): "B", ("coup", "bluff_caught_rate"): "A",
    ("coup", "table_read_auroc"): "A", ("coup", "bluff_rate"): "B",
}

COLOR = {"gpt-5.6-sol-pro": "#2a78d6", "claude-opus-4.8": "#008300", "kimi-k3": "#eda100",
         "deepseek-v4-pro": "#e87ba4", "qwen3.7-max": "#eb6834", "gemini-3.6-flash": "#4a3aa7",
         "llama-4-maverick": "#e34948", "glm-5.2": "#7a3b12", "z-ai/glm-5.2": "#7a3b12",
         "claude-sonnet-5": "#008300", "gpt-5.5": "#2a78d6", "qwen3.5-9b": "#eb6834",
         "qwen3.6-27b": "#00a0a0", "gemma-4-31b-it": "#4a3aa7"}
_FALLBACK = ["#555", "#999", "#c33", "#3c3", "#33c", "#cc3", "#c3c"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pearson(xy):
    """xy: list of (x,y) with no Nones. Returns (r, n) or (nan, n)."""
    n = len(xy)
    if n < 3:
        return float("nan"), n
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in xy)
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return (num / den if den else float("nan")), n


def get(dct, key):
    """Extract a numeric metric from a per-model record; unwrap {'mean':..} CI dicts."""
    v = dct.get(key)
    if isinstance(v, dict):
        v = v.get("mean")
    return v if isinstance(v, (int, float)) else None


def load_models(game):
    fp = SRC / GAMES[game][0]
    o = json.load(open(fp))
    key = GAMES[game][1]
    md = o[key]
    # merge in the LLM reader-pass conditioned leakage (separate file, run once) if present
    rl = SRC / "reader_leakage_results.json"
    if rl.exists():
        rj = json.load(open(rl))
        for m, v in rj.get(game, {}).items():
            if m in md and isinstance(md[m], dict) and v.get("leakage_on_lies") is not None:
                md[m] = {**md[m], "reader_leak_on_lies": v["leakage_on_lies"]}
    return md


def build(game):
    """Return (models list, {field: {model: value}}, leak_proxies, gain_proxies)."""
    md = load_models(game)
    _, _, leaks, gains = GAMES[game]
    models = list(md)
    vals = {}
    for f, *_ in leaks:
        vals[f] = {m: get(md[m], f) for m in models}
    for f, _ in gains:
        vals[f] = {m: get(md[m], f) for m in models}
    return models, vals, leaks, gains


def color_for(m, i):
    return COLOR.get(m, COLOR.get(m.split("/")[-1], _FALLBACK[i % len(_FALLBACK)]))


def plot_game(game):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models, vals, leaks, gains = build(game)
    rows = []  # (game, leak_label, gain_label, r, n, absr)
    nx, ny = len(leaks), len(gains)
    fig, axes = plt.subplots(ny, nx, figsize=(4.7 * nx, 4.3 * ny), squeeze=False)
    for iy, (gf, gl) in enumerate(gains):
        for ix, (lf, ll, inv) in enumerate(leaks):
            ax = axes[iy][ix]
            pts = [(vals[lf][m], vals[gf][m], m) for m in models
                   if vals[lf][m] is not None and vals[gf][m] is not None]
            for i, (x, y, m) in enumerate(pts):
                ax.scatter(x, y, s=110, c=color_for(m, i), edgecolors="#0b0b0b", lw=1, zorder=3)
                ax.annotate(m.split("/")[-1].replace("-sol-pro", "").replace("-maverick", "")
                            .replace("-3.6-flash", "").replace("-opus-4.8", "").replace("-v4-pro", ""),
                            (x, y), textcoords="offset points", xytext=(5, 2), fontsize=6.5)
            r, n = pearson([(p[0], p[1]) for p in pts])
            cls = COND.get((game, lf), "?")
            rows.append((game, ll + (" (inv)" if inv else ""), gl, r, n,
                         abs(r) if r == r else -1, cls))
            ax.set_title(f"r={r:+.2f} (n={n})", fontsize=11,
                         color="#b00" if (r == r and abs(r) >= 0.5) else "#333")
            ax.set_xlabel(ll + (" [inv]" if inv else ""), fontsize=8)
            ax.set_ylabel(gl, fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle(f"{game}: leakage (x) x gain (y) — model-mean Pearson r", fontsize=13, y=1.004)
    fig.tight_layout()
    out = D / f"{game}_grid.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return rows


def main():
    all_rows = []
    for game in GAMES:
        try:
            all_rows += plot_game(game)
            print(f"wrote {game}_grid.png")
        except Exception as e:  # noqa: BLE001
            print(f"  {game} FAILED: {e!r}")

    # ---- markdown ----
    lines = ["# Proxy combinations — which Term-2 x Term-1 pairing is most discerning",
             "",
             "Model-mean **Pearson r** for every leakage (Term-2) x gain (Term-1) proxy pairing, per "
             "game, from the per-model numbers in `../*_results.json` (no rollouts/LLM). Sign: leakage "
             "axes are *higher = more readable* unless tagged **(inv)** = higher = more deceptive; gain "
             "axes are *higher = better*. So a **negative r** on a readable-leakage axis = the "
             "**concealment-pays** direction. `|r|` ranks discernment; **n** = models (small — treat as "
             "directional). Bold = |r| >= 0.5.",
             "",
             "**cond** = whether the leakage proxy is conditioned on a lie event (see `best_conditioned.py`): "
             "**A** = lie-conditioned readability (denominator = lie events, or a base-rate-invariant "
             "separability/posterior measure) — *safe*; **B** = raw lie *rate* (a frequency of lying); "
             "**C** = unconditioned mix (readability over ALL actions, so it moves with the lie rate). "
             "**Only class A is free of the raw-deception-rate confound** — B and C rows are confounded and "
             "should not be read as pure leakage. Of every leakage proxy here, only mafia `role_leakage`, "
             "coup `bluff_caught_rate`, coup `table_read_auroc`, negotiation `leakage`, and liarsdice "
             "`bluff_stick_rate` (empty in data) are class A.",
             ""]
    # per-game sections
    for game in GAMES:
        grows = [r for r in all_rows if r[0] == game]
        grows.sort(key=lambda x: -x[5])
        lines += [f"## {game}", "", f"![{game} grid]({game}_grid.png)", "",
                  "| Term-2 leakage (x) | cond | Term-1 gain (y) | Pearson r | n |",
                  "|---|:--:|---|---:|---:|"]
        for _g, ll, gl, r, n, ar, cls in grows:
            rstr = "n/a" if r != r else f"{r:+.2f}"
            if ar >= 0.5:
                rstr = f"**{rstr}**"
            tag = {"A": "A", "B": "B⚠", "C": "C⚠"}.get(cls, cls)
            lines.append(f"| {ll} | {tag} | {gl} | {rstr} | {n} |")
        lines.append("")
    # global leaderboard
    ranked = sorted([r for r in all_rows if r[5] >= 0], key=lambda x: -x[5])
    lines += ["## Most discerning pairings across ALL games", "",
              "`cond`=A rows are the only ones free of the raw-lie-rate confound; **B⚠/C⚠ are "
              "confounded** with how often the model lies.", "",
              "| game | leakage | cond | gain | r | n |", "|---|---|:--:|---|---:|---:|"]
    for g, ll, gl, r, n, ar, cls in ranked[:20]:
        tag = {"A": "A", "B": "B⚠", "C": "C⚠"}.get(cls, cls)
        lines.append(f"| {g} | {ll} | {tag} | {gl} | {r:+.2f} | {n} |")
    lines.append("")
    # conditioned-only leaderboard
    cond_ranked = [r for r in ranked if r[6] == "A"]
    lines += ["## Lie-CONDITIONED leakage proxies only (class A — no raw-rate confound)", "",
              "| game | leakage | gain | r | n |", "|---|---|---|---:|---:|"]
    for g, ll, gl, r, n, ar, cls in cond_ranked:
        lines.append(f"| {g} | {ll} | {gl} | {r:+.2f} | {n} |")
    lines.append("")

    (D / "proxy_combinations.md").write_text("\n".join(lines))
    print(f"\nwrote {D/'proxy_combinations.md'} ({len(all_rows)} combinations across {len(GAMES)} games)")
    print("\nTop 12 most discerning (|r|):")
    for g, ll, gl, r, n, ar, cls in ranked[:12]:
        print(f"  |r|={ar:.2f} r={r:+.2f} n={n}  cond={cls}  {g}: {ll} vs {gl}")


if __name__ == "__main__":
    main()
