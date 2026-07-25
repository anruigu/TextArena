#!/usr/bin/env python3
"""Quantitative cross-play reports for the TextArena frontier sweep.

Reads the per-game records (game_*.json) written by run_xplay.py / run_mafia.py
and emits, per result dir:
  - analysis.json  : machine-readable aggregate stats
  - REPORT.md      : human-readable tables (who won, win rates, outcome + value stats)
  - *.png          : win-rate (and, for negotiation, self-score) bar charts (if matplotlib)
and a combined REPORT.md + analysis_all.json across all dirs.

Focus is quantitative: who won across games, win/agreement rates, and — for the
negotiation envs (New Recruit, ScorableGames) — raw self-score, value-extraction
(self / self-optimal) and joint efficiency (realized joint / optimal joint),
computed by replaying each game's final deal through the env's own additive
scoring tables. No LLM/qualitative judging.

Run with the TextArena venv so the envs import cleanly:
  ../.venv/bin/python analyze_xplay.py \
      --inputs results_coup,results_nr,results_sg2p,results_sg6p,results \
      --out .
"""
from __future__ import annotations
import argparse, glob, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def short(m):
    return (m or "").split("/")[-1]


def rate(a, b):
    return round(a / b, 3) if b else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 2) if xs else None


# --------------------------------------------------------------------------- #
# Negotiation scoring: additive per-issue value tables pulled from the envs.
# Cached per (env, config) so we build each env at most once.
# --------------------------------------------------------------------------- #
_TABLE_CACHE = {}


def _nr_tables():
    from textarena.envs.NewRecruit.env import NewRecruitEnv
    pvd = NewRecruitEnv().point_value_dict          # pvd[issue][choice][pid], pid in {0,1}
    issues = list(pvd.keys())
    # value[pid][issue][option]
    tables = {pid: {i: {c: pvd[i][c][pid] for c in pvd[i]} for i in issues} for pid in (0, 1)}
    return {"issues": issues, "tables": tables, "players": 2}


def _sg_tables(cfg, players):
    from textarena.envs.ScorableGames.env import ScorableGamesEnv
    e = ScorableGamesEnv(game_config=cfg)
    e.reset(num_players=players)
    issues = list(e.issues.keys())
    tables = {}
    for pid, sc in e.player_scores.items():
        tables[pid] = {i: dict(sc[i]) for i in issues if isinstance(sc.get(i), dict)}
    return {"issues": issues, "tables": tables, "players": players}


def get_tables(env, cfg, players):
    key = (env, cfg, players)
    if key not in _TABLE_CACHE:
        if env == "newrecruit":
            _TABLE_CACHE[key] = _nr_tables()
        elif env == "scorablegames":
            _TABLE_CACHE[key] = _sg_tables(cfg, players)
        else:
            _TABLE_CACHE[key] = None
    return _TABLE_CACHE[key]


def deal_of(game):
    """Final agreed deal/proposal as {issue: option}, or None if no agreement."""
    meta = game.get("meta") or {}
    if "final_proposal" in meta and meta["final_proposal"]:
        return (meta["final_proposal"] or {}).get("choices")
    d = meta.get("current_deal")
    return d or None


def value_metrics(game, tab):
    """Per-seat self-score / value-extraction + joint efficiency for one deal."""
    deal = deal_of(game)
    if not deal or not tab:
        return None
    issues, tables = tab["issues"], tab["tables"]
    pids = sorted(tables.keys())

    def self_score(pid):
        t = tables[pid]
        return sum(t[i].get(deal.get(i), 0) for i in issues if i in t)

    def self_max(pid):
        t = tables[pid]
        return sum(max(t[i].values()) for i in issues if t.get(i))

    seat = {pid: {"self_score": self_score(pid),
                  "self_max": self_max(pid),
                  "value_extraction": rate(self_score(pid), self_max(pid))}
            for pid in pids}
    joint_realized = sum(s["self_score"] for s in seat.values())
    # optimal joint: per issue pick option maximizing summed value across seats
    joint_max = 0
    for i in issues:
        opts = set()
        for pid in pids:
            opts |= set(tables[pid].get(i, {}).keys())
        if opts:
            joint_max += max(sum(tables[pid].get(i, {}).get(o, 0) for pid in pids) for o in opts)
    return {"seat": seat, "joint_realized": joint_realized,
            "joint_max": joint_max, "efficiency_ratio": rate(joint_realized, joint_max)}


# --------------------------------------------------------------------------- #
# Per-dir analysis
# --------------------------------------------------------------------------- #
def analyze_dir(d):
    files = sorted(glob.glob(str(Path(d) / "game_*.json")))
    if not files:
        return None
    games = [json.loads(Path(f).read_text()) for f in files]
    g0 = games[0]
    env = g0.get("env") or "secret_mafia"
    cfg = g0.get("game_config")
    players = g0.get("players")
    is_neg = env in ("newrecruit", "scorablegames")

    tab = get_tables(env, cfg, players) if is_neg else None

    per = {}  # model -> counters

    def pm(m):
        return per.setdefault(m, {"seats": 0, "wins": 0, "self_scores": [], "value_extractions": []})

    draws = timeouts = 0
    steps, walls, effs, agreements = [], [], [], []
    mafia_wins = 0

    for g in games:
        winners = set(g.get("winners") or [])
        roles = g.get("seat_roles") or g.get("roles") or {}
        seat_models = g.get("seat_models") or []
        np_ = g.get("players", len(seat_models))
        if not winners:
            draws += 1
        if g.get("timed_out"):
            timeouts += 1
        if g.get("n_steps") is not None:
            steps.append(g["n_steps"])
        if g.get("wall_seconds") is not None:
            walls.append(g["wall_seconds"])
        if env == "secret_mafia" and g.get("win_team") == "Mafia":
            mafia_wins += 1

        vm = value_metrics(g, tab) if is_neg else None
        if is_neg:
            agreements.append(1 if deal_of(g) else 0)
            if vm:
                effs.append(vm["efficiency_ratio"])

        for pid in range(np_):
            m = short(seat_models[pid]) if pid < len(seat_models) else f"seat{pid}"
            d_ = pm(m)
            d_["seats"] += 1
            d_["wins"] += int(pid in winners)
            if vm and pid in vm["seat"]:
                d_["self_scores"].append(vm["seat"][pid]["self_score"])
                d_["value_extractions"].append(vm["seat"][pid]["value_extraction"])

    per_model = {}
    for m, d_ in sorted(per.items(), key=lambda kv: (-rate(kv[1]["wins"], kv[1]["seats"]) or 0)):
        row = {"seats_played": d_["seats"], "wins": d_["wins"],
               "win_rate": rate(d_["wins"], d_["seats"])}
        if is_neg:
            row["mean_self_score"] = mean(d_["self_scores"])
            row["mean_value_extraction"] = mean(d_["value_extractions"])
        per_model[m] = row

    out = {
        "dir": str(d), "env": env, "game_config": cfg, "players": players,
        "games": len(games), "draws": draws, "timeouts": timeouts,
        "avg_steps": mean(steps), "avg_wall_seconds": mean(walls),
        "per_model": per_model,
    }
    if env == "secret_mafia":
        out["mafia_win_rate"] = rate(mafia_wins, len(games))
        out["village_win_rate"] = rate(len(games) - mafia_wins, len(games))
    if is_neg:
        out["agreement_rate"] = rate(sum(agreements), len(agreements))
        out["mean_efficiency_ratio"] = mean(effs)
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
ENV_TITLE = {"coup": "Coup", "newrecruit": "New Recruit",
             "scorablegames": "ScorableGames", "secret_mafia": "Secret Mafia"}


def md_table(headers, rows):
    line = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    out = [line(headers), "|" + "|".join("---" for _ in headers) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def render_md(a, plots=None):
    title = ENV_TITLE.get(a["env"], a["env"])
    cfg = f" · `{a['game_config']}`" if a.get("game_config") else ""
    L = [f"## {title}{cfg}", ""]
    L.append(f"- **games**: {a['games']}  ·  **players**: {a['players']}  ·  "
             f"**draws**: {a['draws']}  ·  **timeouts**: {a['timeouts']}")
    line2 = f"- **avg steps/game**: {a['avg_steps']}  ·  **avg wall (s)**: {a['avg_wall_seconds']}"
    if a["env"] == "secret_mafia":
        line2 += f"  ·  **mafia win rate**: {a['mafia_win_rate']}  ·  **village win rate**: {a['village_win_rate']}"
    if "agreement_rate" in a:
        line2 += f"  ·  **agreement rate**: {a['agreement_rate']}  ·  **mean efficiency**: {a['mean_efficiency_ratio']}"
    L.append(line2)
    L.append("")

    neg = "mean_self_score" in next(iter(a["per_model"].values()), {})
    if neg:
        headers = ["model", "seats", "wins", "win rate", "mean self-score", "mean value-extraction"]
        rows = [[m, r["seats_played"], r["wins"], r["win_rate"],
                 r["mean_self_score"], r["mean_value_extraction"]]
                for m, r in a["per_model"].items()]
    else:
        headers = ["model", "seats", "wins", "win rate"]
        rows = [[m, r["seats_played"], r["wins"], r["win_rate"]] for m, r in a["per_model"].items()]
    L.append(md_table(headers, rows))
    L.append("")
    for p in (plots or []):
        L.append(f"![{p}]({p})")
    L.append("")
    return "\n".join(L)


def make_plots(a, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    tag = a["env"] + (f"_{a['game_config']}" if a.get("game_config") else "")
    models = list(a["per_model"].keys())
    wr = [a["per_model"][m]["win_rate"] or 0 for m in models]
    made = []
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(models)), wr, color="#4C78A8")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("win rate"); ax.set_title(f"{ENV_TITLE.get(a['env'], a['env'])} — win rate by model")
    fig.tight_layout(); fn = f"winrate_{tag}.png"; fig.savefig(outdir / fn, dpi=120); plt.close(fig)
    made.append(fn)
    if "mean_self_score" in next(iter(a["per_model"].values()), {}):
        ss = [a["per_model"][m]["mean_self_score"] or 0 for m in models]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(range(len(models)), ss, color="#F58518")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("mean self-score"); ax.set_title(f"{ENV_TITLE.get(a['env'], a['env'])} — mean self-score by model")
        fig.tight_layout(); fn = f"selfscore_{tag}.png"; fig.savefig(outdir / fn, dpi=120); plt.close(fig)
        made.append(fn)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="results_coup,results_nr,results_sg2p,results_sg6p,results")
    ap.add_argument("--out", default=".")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    outroot = Path(args.out)
    if not outroot.is_absolute():
        outroot = base / outroot
    dirs = [x.strip() for x in args.inputs.split(",") if x.strip()]

    analyses = []
    for d in dirs:
        dp = (base / d) if not Path(d).is_absolute() else Path(d)
        if not dp.exists():
            continue
        a = analyze_dir(dp)
        if not a:
            continue
        plots = [] if args.no_plots else make_plots(a, dp)
        (dp / "analysis.json").write_text(json.dumps(a, indent=2))
        (dp / "REPORT.md").write_text("# Cross-play quantitative report\n\n" + render_md(a, plots))
        # rewrite plot refs to be relative to the per-dir report (same dir)
        analyses.append((a, dp, plots))
        print(f"[{a['env']}{('/'+a['game_config']) if a.get('game_config') else ''}] "
              f"{a['games']} games -> {dp/'REPORT.md'}")

    # combined report
    order = {"coup": 0, "newrecruit": 1, "scorablegames": 2, "secret_mafia": 3}
    analyses.sort(key=lambda t: (order.get(t[0]["env"], 9), t[0].get("game_config") or ""))
    combined = ["# TextArena frontier cross-play — quantitative report", "",
                "_Auto-generated by `analyze_xplay.py`. Outcome stats from game records; "
                "negotiation value metrics replay each final deal through the env's additive "
                "scoring tables._", ""]
    rel = outroot
    for a, dp, plots in analyses:
        try:
            prefix = str(dp.relative_to(rel)) + "/"
        except ValueError:
            prefix = str(dp) + "/"
        combined.append(render_md(a, [prefix + p for p in plots]))
    (outroot / "REPORT.md").write_text("\n".join(combined))
    (outroot / "analysis_all.json").write_text(json.dumps([a for a, _, _ in analyses], indent=2))
    print(f"\nwrote combined {outroot/'REPORT.md'} + analysis_all.json ({len(analyses)} envs)")


if __name__ == "__main__":
    main()
