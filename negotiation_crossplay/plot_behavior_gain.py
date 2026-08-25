#!/usr/bin/env python3
"""Predictive power of each text-free behavioral feature vs instrumental gain.

Clarifies what the 'behavior probe' actually measures by plotting, for every
seat with executed trades, each behavioral feature against realized gain and
annotating the Pearson r. Splits features into three families:
  competence   (surplus, win-share, acquire-via-accept)  -> should track gain
  effort       (n_trades, n_offers)
  legibility   (Type-A behavioral rho, Type-B text rho)   -> the 'leakage' channels

Key message: there is NO purely-behavioral 'is-lying' feature. A value-lie only
leaves a behavioral trace when it MOVES a trade in your favor -> that trace IS
the surplus/win-share, i.e. outcome. Legibility is a separate axis.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from behavior_probe import collect

FEATS = [
    ("surplus_true", "Σ surplus (own values)", "competence"),
    ("surplus_per_trade", "surplus / trade", "competence"),
    ("win_share", "win-share vs counterparty", "competence"),
    ("acquire_via_accept", "acquire-via-accept", "competence"),
    ("n_trades", "# trades executed", "effort"),
    ("n_offers", "# offers proposed", "effort"),
    ("typeA_flow", "Type-A behavioral ρ (net-flow)", "legibility"),
    ("typeA_demand", "Type-A intent ρ (demand)", "legibility"),
    ("typeB_text", "Type-B text ρ (opponent)", "legibility"),
]
COLORS = {"competence": "#1F8A65", "effort": "#7B64B8", "legibility": "#C06028"}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main():
    seats = [s for s in collect() if s["n_trades"] and s["gain"] is not None]
    gain = np.array([s["gain"] for s in seats], float)

    # correlations
    corr = {}
    for key, _, _ in FEATS:
        xy = [(s[key], s["gain"]) for s in seats if s[key] is not None]
        if len(xy) >= 3:
            corr[key] = (pearson([a for a, _ in xy], [b for _, b in xy]), len(xy))
        else:
            corr[key] = (np.nan, len(xy))

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1, 1], hspace=0.55, wspace=0.32)

    # ---- Panel A: bar of Pearson r vs gain, sorted ----
    axA = fig.add_subplot(gs[0, :])
    order = sorted(FEATS, key=lambda f: -(corr[f[0]][0] if not np.isnan(corr[f[0]][0]) else -9))
    names = [lbl for _, lbl, _ in order]
    rs = [corr[k][0] for k, _, _ in order]
    cols = [COLORS[fam] for _, _, fam in order]
    y = np.arange(len(order))[::-1]
    axA.barh(y, rs, color=cols, edgecolor="black", linewidth=0.5, height=0.66)
    axA.set_yticks(y); axA.set_yticklabels(names, fontsize=10)
    axA.axvline(0, color="black", lw=0.8)
    for yi, (k, _, _) in zip(y, order):
        r, n = corr[k]
        axA.text(r + (0.02 if r >= 0 else -0.02), yi, f"{r:+.2f}",
                 va="center", ha="left" if r >= 0 else "right", fontsize=9)
    axA.set_xlim(-0.5, 1.05)
    axA.set_xlabel("Pearson r  with instrumental gain  (per seat, n=%d trading seats)" % len(seats))
    axA.set_title("Predictive power of each text-free behavioral feature for realized gain",
                  fontsize=13, weight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[f]) for f in ("competence", "effort", "legibility")]
    axA.legend(handles, ["competence (outcome)", "effort", "legibility (leakage channels)"],
               loc="lower right", fontsize=9, frameon=False)

    # ---- Panels B..: scatter of the 4 non-tautological features vs gain ----
    focus = ["win_share", "acquire_via_accept", "typeA_flow", "typeB_text"]
    focus_lbl = {k: lbl for k, lbl, _ in FEATS}
    focus_fam = {k: fam for k, _, fam in FEATS}
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])]
    for ax, key in zip(axes, focus):
        pts = [(s[key], s["gain"]) for s in seats if s[key] is not None]
        xs = np.array([a for a, _ in pts], float); ys = np.array([b for _, b in pts], float)
        c = COLORS[focus_fam[key]]
        ax.scatter(xs, ys, s=22, c=c, alpha=0.6, edgecolor="none")
        if xs.std() > 0:
            m, b = np.polyfit(xs, ys, 1)
            xr = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xr, m * xr + b, color="black", lw=1.3)
        r, _ = corr[key]
        ax.set_title(f"{focus_lbl[key]}\nr = {r:+.2f}", fontsize=10)
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_ylabel("gain")
        ax.tick_params(labelsize=8)

    # ---- Panel C: gain vs surplus (the near-tautology) + Type-A vs gain colored ----
    axT = fig.add_subplot(gs[2, 0:2])
    xs = np.array([s["surplus_true"] for s in seats], float)
    axT.scatter(xs, gain, s=22, c=COLORS["competence"], alpha=0.6, edgecolor="none")
    m, b = np.polyfit(xs, gain, 1); xr = np.linspace(xs.min(), xs.max(), 50)
    axT.plot(xr, m * xr + b, color="black", lw=1.3)
    axT.set_title("Why 'competence' features win: Σ surplus ≈ gain by construction (r=%.2f)"
                  % corr["surplus_true"][0], fontsize=10)
    axT.set_xlabel("Σ trade surplus under own values"); axT.set_ylabel("gain")
    axT.tick_params(labelsize=8)

    axL = fig.add_subplot(gs[2, 2:4])
    a = np.array([s["typeA_flow"] if s["typeA_flow"] is not None else np.nan for s in seats])
    bt = np.array([s["typeB_text"] if s["typeB_text"] is not None else np.nan for s in seats])
    mask = ~np.isnan(a) & ~np.isnan(bt)
    sc = axL.scatter(a[mask], bt[mask], s=28, c=gain[mask], cmap="viridis", alpha=0.85, edgecolor="none")
    axL.set_xlabel("Type-A behavioral leakage ρ"); axL.set_ylabel("Type-B text leakage ρ")
    axL.set_title("The two leakage channels barely agree (r=%.2f); color = gain" % pearson(a[mask], bt[mask]),
                  fontsize=10)
    axL.axhline(0, color="gray", lw=0.5, ls=":"); axL.axvline(0, color="gray", lw=0.5, ls=":")
    axL.tick_params(labelsize=8)
    cb = fig.colorbar(sc, ax=axL, fraction=0.046, pad=0.04); cb.set_label("gain", fontsize=8)

    fig.suptitle("Behavior probe vs instrumental gain — TextArena Negotiation crossplay (vprobe_ta_2p+3p)",
                 fontsize=14, weight="bold", y=0.995)
    out = Path(__file__).resolve().parent / "behavior_gain.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)
    print("\nPearson r with gain:")
    for k, lbl, fam in FEATS:
        r, n = corr[k]
        print(f"  {lbl:<34} r={r:+.2f}  (n={n}, {fam})")


if __name__ == "__main__":
    main()
