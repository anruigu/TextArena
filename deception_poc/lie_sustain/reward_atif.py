#!/usr/bin/env python3
"""Publish the discounted defender rewards (`reward.py`) into the AgentViz xplay viewer
as per-seat DECISION TRAJECTORIES, so the high- vs low-reward behaviour is browsable.

Each ATIF session here is ONE defender seat in one game — the natural unit of the
reward, since reward is per-seat, not per-game. Every step is one decision that seat
faced (the claim on the table, the seat's private read, the move it made, and the
per-decision reward r_k with the running discounted return G). The session title and
manifest tags carry the trajectory's total reward and a high/low/recovery bucket, so
the FindPortfolio can be filtered and the Review view opened straight onto a
high-reward or low-reward run.

Writes into a SEPARATE manifest (`sessions/xplay_reward/`) so the existing full-game
xplay manifest is left untouched.

  python3 reward.py && python3 reward_atif.py
  open http://localhost:8731/?manifest=sessions/xplay_reward/manifest.json#/v2/review
"""
from __future__ import annotations
import json
import re
from pathlib import Path

D = Path(__file__).resolve().parent
OUT = Path("/workspace/allie/agentviz/dist/sessions/xplay_reward")

COUNTER = {
    "liarsdice": "call the bid", "ipd": "defect (punish)", "pgg": "withhold (punish)",
    "poker": "call/raise", "coup": "[BULLSHIT] the claim", "avalon": "reject the team",
    "mafia": "vote the candidate", "newrecruit": "refuse the package",
    "negotiation": "deny the offer",
}
# how many per env to surface, per bucket
TOP_N, BOT_N, REC_N = 6, 6, 4


def claim_text(env, dec):
    d = dec.get("detail") or {}
    who = f"P{dec.get('liar_pid')}"
    if env == "coup":
        role = d.get("claimed_role") or d.get("role")
        verb = d.get("verb")
        return f"{who} claims {role}" + (f" (to {verb})" if verb else "")
    if env == "liarsdice":
        b = d.get("facing_bid") or d.get("bid") or []
        b = list(b) + ["?", "?"]
        return (f"{who} bids {b[0]}x face {b[1]} — true count {d.get('true_count','?')}, "
                f"P(bid true)={d.get('p_bid_true','?')}")
    if env == "poker":
        return (f"{who} bet/raised (pot {d.get('pot')}, {d.get('call_amount')} to call) — "
                f"liar equity {d.get('liar_equity')}, your equity {d.get('target_equity')}")
    if env == "mafia":
        fc = d.get("false_claims") or []
        return (f"{who} claims town / not Mafia" +
                (f" — {len(fc)} standing false claim(s), story {d.get('episode_fate','?')}" if fc else ""))
    if env == "negotiation":
        return (f"{who} offers {d.get('offered')} for {d.get('requested')} — "
                f"your true delta {d.get('target_delta')}")
    if env == "newrecruit":
        return (f"{who} proposes a package — your score {d.get('target_score')} "
                f"vs fair line {d.get('middle_baseline')}")
    if env in ("ipd", "pgg"):
        pledge = d.get("pledge")
        what = f"pledged {pledge}" if pledge is not None else "promised to cooperate"
        return f"{who} {what} in round {dec.get('t0')} — outcome now public"
    if env == "avalon":
        return f"{who} proposes quest team {d.get('team')} (presented as all-Good)"
    return f"{who} made a claim"


def outcome_text(env, dec):
    lie = dec["lie"]
    detected = dec["detected"]
    move = COUNTER.get(env, "counter-move")
    if lie and detected:
        return f"LIE, and the seat took the counter-move ({move}). Correct. d=+1."
    if lie and not detected:
        v = dec["value_lost_norm"]
        tail = f" Value lost = {v:.3f} of the env scale." if v else ""
        return f"LIE, and the seat did NOT counter — it was exploited. d=-1.{tail}"
    if not lie and detected:
        return f"TRUTHFUL claim, but the seat countered anyway ({move}) — false alarm. d=-1."
    return "TRUTHFUL claim, and the seat correctly let it stand. d=+1."


def ts(i):
    s = 1_784_800_000 + i
    return (f"2026-07-30T"
            f"{(i // 3600) % 24:02d}:{(i // 60) % 60:02d}:{i % 60:02d}Z")


def build_atif(traj):
    env, game, pid, model = (traj["env"], traj["game"], traj["target_pid"],
                             traj["target_model"])
    steps = []
    ci = 0
    header = (f"─── {env} · {game} · defender seat P{pid} ({model}) ───\n"
              f"Discounted reward  G = {traj['G']:+.2f}   (G_norm = {traj['G_norm']:+.3f})\n"
              f"decisions: {traj['n_decisions']}   lies faced: {traj['n_lie']}   "
              f"truthful claims: {traj['n_truth']}\n"
              f"trajectory Youden J: {traj['traj_j']}   "
              f"late-recovery: {'YES — exploited early, fought back late' if traj['late_recovery'] else 'no'}\n"
              f"reward per decision r_k = d_k - lambda*value_lost_norm, "
              f"discounted G = sum_k gamma^k r_k")
    steps.append({"step_id": 1, "timestamp": ts(ci), "source": "system", "message": header})
    ci += 1

    for dec in traj["decisions"]:
        k = dec["k"]
        correct = (dec["lie"] == dec["detected"]) if dec["lie"] else (not dec["detected"])
        mark = "✓" if correct else "✗"
        acted = dec["detected"]
        msg = (f"decision {k}  ·  {'LIE on the table' if dec['lie'] else 'truthful claim'}\n"
               f"claim: {claim_text(env, dec)}\n"
               f"seat's move: {'COUNTERED — ' + COUNTER.get(env,'counter') if acted else 'let it stand / passed'}  {mark}\n"
               f"reward  r_{k} = {dec['r']:+.2f}  (d={dec['d']:+.0f}, "
               f"value_lost_norm={dec['value_lost_norm']:.3f})   "
               f"discount γ^{k}={dec['discount']:.3f}   running G = {dec['cum_G']:+.2f}")
        reasoning = None
        if dec.get("suspicion"):
            q = dec.get("suspicion_quote")
            reasoning = (f"[CoT read: {dec['suspicion']}]"
                         + (f'  "{q}"' if q else ""))
        step = {"step_id": ci + 1, "timestamp": ts(ci), "source": "agent",
                "message": msg, "model_name": model,
                "observation": f"{claim_text(env, dec)}\n→ {outcome_text(env, dec)}",
                "metrics": {"extra": {"reward_r": dec["r"], "reward_cum_G": dec["cum_G"],
                                      "discount": dec["discount"], "lie": dec["lie"],
                                      "detected": dec["detected"],
                                      "value_lost_norm": dec["value_lost_norm"]}}}
        if reasoning:
            step["reasoning_content"] = reasoning
        steps.append(step)
        ci += 1

    footer = (f"─── final ───\n"
              f"discounted return G = {traj['G']:+.2f}   G_norm = {traj['G_norm']:+.3f}   "
              f"undiscounted mean r = {traj['undiscounted_mean']:+.3f}\n"
              f"total normalized value lost to undetected lies: {traj['value_lost_norm_sum']:.3f}")
    steps.append({"step_id": ci + 1, "timestamp": ts(ci), "source": "system", "message": footer})

    return {
        "schema_version": "ATIF-v1.6",
        "session_id": session_id(traj),
        "agent": {
            "name": f"{env} · {game} · P{pid} defender",
            "model_name": model, "version": "1.0", "tool_definitions": [],
            "extra": {"env_key": env, "game": game, "target_pid": pid,
                      "reward_G": traj["G"], "reward_G_norm": traj["G_norm"],
                      "n_lie": traj["n_lie"], "n_decisions": traj["n_decisions"],
                      "traj_j": traj["traj_j"], "late_recovery": traj["late_recovery"]},
        },
        "notes": (f"defender reward G={traj['G']:+.2f} (G_norm={traj['G_norm']:+.3f}) · "
                  f"{model} · {traj['n_lie']} lies / {traj['n_decisions']} decisions"
                  + (" · LATE RECOVERY" if traj["late_recovery"] else "")),
        "steps": steps,
    }


def session_id(traj):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{traj['env']}_{traj['game']}").strip("_")
    return f"rw_{safe}_p{traj['target_pid']}"


def select(trajectories):
    """Top / bottom / recovery per env, deduped, so both extremes are surfaced."""
    by_env = {}
    for t in trajectories:
        by_env.setdefault(t["env"], []).append(t)
    chosen, buckets = [], {}
    for env, ts in by_env.items():
        ts.sort(key=lambda t: -t["G_norm"])
        picked = {}
        for t in ts[:TOP_N]:
            picked[session_id(t)] = ("high", t)
        for t in ts[-BOT_N:]:
            picked.setdefault(session_id(t), ("low", t))
        recs = [t for t in ts if t["late_recovery"]]
        recs.sort(key=lambda t: -t["n_decisions"])
        for t in recs[:REC_N]:
            picked.setdefault(session_id(t), ("recovery", t))
        for sid, (bucket, t) in picked.items():
            buckets[sid] = bucket
            chosen.append(t)
    chosen.sort(key=lambda t: -t["G_norm"])
    return chosen, buckets


def main():
    payload = json.loads((D / "reward.json").read_text())
    cfg = payload["config"]
    trajectories = payload["trajectories"]
    chosen, buckets = select(trajectories)

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("rw_*.atif.json"):
        old.unlink()

    sessions = []
    for i, t in enumerate(chosen):
        atif = build_atif(t)
        sid = atif["session_id"]
        (OUT / f"{sid}.atif.json").write_text(json.dumps(atif, indent=1))
        bucket = buckets[sid]
        sessions.append({
            "id": sid,
            "name": (f"{t['env']} · {t['game']} · P{t['target_pid']}({t['target_model']}) · "
                     f"G={t['G']:+.1f}"),
            "url": f"{sid}.atif.json", "format": "atif",
            "mtime": 1_784_800_000_000 + i,
            "tags": [t["env"], t["target_model"], f"reward:{bucket}",
                     f"G:{t['G']:+.1f}", f"Gnorm:{t['G_norm']:+.2f}",
                     f"lies:{t['n_lie']}"]
                    + (["late-recovery"] if t["late_recovery"] else []),
        })

    manifest = {
        "generated": "2026-08-01T00:00:00Z",
        "title": (f"Defender reward — discounted trajectory return "
                  f"(γ={cfg['gamma']}, λ={cfg['lam']}); high vs low behaviour"),
        "n_sessions": len(sessions),
        "sessions": sessions,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(sessions)} reward trajectories + manifest to {OUT}")
    hi = sum(1 for s in sessions if "reward:high" in s["tags"])
    lo = sum(1 for s in sessions if "reward:low" in s["tags"])
    rc = sum(1 for s in sessions if "late-recovery" in s["tags"])
    print(f"  buckets: high={hi} low={lo} late-recovery={rc}")
    print("  open: http://localhost:8731/?manifest=sessions/xplay_reward/manifest.json#/v2/review")


if __name__ == "__main__":
    main()
