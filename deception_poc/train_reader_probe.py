#!/usr/bin/env python3
"""Phase-0 leak READER probe: the detector that DEFINES leakage in the arms race.

A frozen open reader model ingests ONLY the public channel (betting history, board, pot -- the
opponent's view, NO hole cards; from build_reader_spots.py) and a standardized logistic probe on
its residual stream predicts the actor's HIDDEN hand-strength bucket (strong equity>=0.6 vs
weak <0.4). Because the hole cards are stripped, any recovery is genuine type-LEAKAGE through the
betting behavior -- not the visible-card re-decoding activation_probe.py measures.

  leakage(spot) = probe posterior P(strong | public-channel residual).

We report per-layer 5-fold CV AUROC + a continuous-equity Pearson r, and at the best layer the
CALIBRATION (Brier score + expected calibration error) -- the paper's probe-health criteria
(Fig 3). We save the fitted direction to probes/probe_leakreader_<tag>.npz in the performative npz
format (w,b,mu,sd,layer,diffmean) so leaky_reward.py can score a live public channel and the outer
arms-race loop can REFIT this same probe on stealthier bluffs each round.

Run in the performative venv (torch + sklearn):
  cd /workspace/allie/TextArena/deception_poc
  /workspace/allie/performative/.venv/bin/python train_reader_probe.py --model Qwen/Qwen3-8B --batch 16
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/workspace/allie/performative/scripts")
from model_utils import load_tok, load_model, text_dims, apply_template  # noqa: E402

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss

D = Path(__file__).resolve().parent
SPOTS = D / "reader_spots.jsonl"

# The reader is an OBSERVER, not the actor: it never sees hole cards and is asked to READ strength.
SYSTEM = (
    "You are an expert poker analyst observing ONE player in a heads-up Texas Hold'em hand. You see "
    "only the public information that player's opponent sees: the betting history, the community "
    "board, the pot and the stacks -- you do NOT see that player's hole cards. From their betting "
    "behavior, infer how strong their concealed hand most likely is."
)


def ece(y, p, n_bins=10):
    """Expected calibration error: |accuracy - confidence| averaged over confidence bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--real-only", action="store_true", help="probe on real transcript spots only")
    args = ap.parse_args()

    spots = [json.loads(l) for l in SPOTS.read_text().splitlines() if l.strip()]
    if args.real_only:
        spots = [s for s in spots if s["source"] == "real"]
    if args.limit:
        spots = spots[:args.limit]
    print(f"[reader] {len(spots)} spots | model {args.model}", file=sys.stderr, flush=True)

    tok = load_tok(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, args.device)
    n_layers, d = text_dims(model)
    L = n_layers + 1
    dev = next(model.parameters()).device

    X = np.zeros((len(spots), L, d), dtype=np.float16)
    for b0 in range(0, len(spots), args.batch):
        batch = spots[b0:b0 + args.batch]
        seqs = [apply_template(tok, [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": s["obs"]}],
                               add_generation_prompt=True, no_think=True) for s in batch]
        maxlen = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, maxlen - len(s):] = torch.tensor(s)
            att[i, maxlen - len(s):] = 1
        ids, att = ids.to(dev), att.to(dev)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
        for l in range(L):
            X[b0:b0 + len(batch), l, :] = out.hidden_states[l][:, -1, :].float().cpu().numpy().astype(np.float16)
        print(f"[reader] {b0 + len(batch)}/{len(spots)}", file=sys.stderr, flush=True)
        del out

    eq = np.array([s["equity"] for s in spots])
    strong, weak = eq >= 0.60, eq < 0.40
    binmask = strong | weak
    y = strong[binmask].astype(int)
    print(f"[reader] binary spots: {int(binmask.sum())} (strong {int(strong.sum())}, "
          f"weak {int(weak.sum())})", file=sys.stderr, flush=True)

    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    kf = KFold(5, shuffle=True, random_state=0)
    act_auroc, act_r = [], []
    for l in range(L):
        Xl = X[:, l, :].astype(np.float32)
        Xb = Xl[binmask]
        mu, sd = Xb.mean(0), Xb.std(0) + 1e-6
        clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
        try:
            dec = cross_val_predict(clf, (Xb - mu) / sd, y, cv=skf, method="decision_function")
            act_auroc.append(float(roc_auc_score(y, dec)))
        except Exception:
            act_auroc.append(float("nan"))
        muA, sdA = Xl.mean(0), Xl.std(0) + 1e-6
        pred = cross_val_predict(Ridge(alpha=100.0), (Xl - muA) / sdA, eq, cv=kf)
        act_r.append(float(np.corrcoef(pred, eq)[0, 1]))

    best_layer = int(np.nanargmax(act_auroc))
    auroc_best = act_auroc[best_layer]
    r_best_layer = int(np.nanargmax(act_r))

    # ---- calibration at the best layer (CV predicted probabilities) ----
    Xb = X[:, best_layer, :].astype(np.float32)[binmask]
    mu, sd = Xb.mean(0), Xb.std(0) + 1e-6
    clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
    proba = cross_val_predict(clf, (Xb - mu) / sd, y, cv=skf, method="predict_proba")[:, 1]
    brier = float(brier_score_loss(y, proba))
    calib_ece = ece(y, proba)

    # ---- fit final direction on all binary spots + save (performative npz format) ----
    clf.fit((Xb - mu) / sd, y)
    diffmean = Xb[y == 1].mean(0) - Xb[y == 0].mean(0)
    PROBES = D / "probes"; PROBES.mkdir(exist_ok=True)
    npz_path = PROBES / f"probe_leakreader_{args.tag}.npz"
    np.savez(npz_path,
             w=clf.coef_[0].astype(np.float32), b=clf.intercept_.astype(np.float32),
             mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             layer=np.int64(best_layer), C=np.float64(0.5),
             pooling=np.array("last"), diffmean=diffmean.astype(np.float32),
             model=np.array(args.model), strong_thresh=np.float64(0.60), weak_thresh=np.float64(0.40))

    res = {
        "model": args.model, "tag": args.tag, "real_only": args.real_only,
        "n_spots": len(spots), "n_binary": int(binmask.sum()),
        "n_strong": int(strong.sum()), "n_weak": int(weak.sum()), "n_layers_incl_emb": L,
        "reader_leakage": {
            "best_layer": best_layer, "auroc_best": round(auroc_best, 4),
            "auroc_by_layer": [round(a, 4) for a in act_auroc],
            "equity_r_best": round(act_r[r_best_layer], 4), "equity_r_best_layer": r_best_layer,
        },
        "calibration_best_layer": {"brier": round(brier, 4), "ece": round(calib_ece, 4)},
        "probe_npz": str(npz_path),
    }
    (D / f"reader_probe_results_{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # ---- figure: leakage-by-layer + reliability curve ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    xs = list(range(L))
    ax.plot(xs, act_auroc, "-o", ms=3, color="#2a78d6", label="reader probe: AUROC(strong|public residual)")
    ax.plot(xs, act_r, "-o", ms=3, color="#1baf7a", label="reader probe: Pearson r (equity)")
    ax.axhline(0.5, color="#bbb", lw=1)
    ax.scatter([best_layer], [auroc_best], s=140, edgecolors="#0b0b0b", facecolors="#2a78d6",
               zorder=5, label=f"best layer {best_layer}: {auroc_best:.2f}")
    ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel("leakage (AUROC / Pearson r)")
    ax.set_title(f"Leak-reader probe — {args.model}\npublic channel only (hole cards HIDDEN)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(color="#eee")
    # reliability
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(proba, bins) - 1
    bx, by = [], []
    for i in range(10):
        m = idx == i
        if m.sum():
            bx.append(proba[m].mean()); by.append(y[m].mean())
    ax2.plot([0, 1], [0, 1], "--", color="#bbb", label="perfect calibration")
    ax2.plot(bx, by, "-o", color="#e34948", label=f"reader (Brier {brier:.3f}, ECE {calib_ece:.3f})")
    ax2.set_xlabel("predicted P(strong)"); ax2.set_ylabel("empirical fraction strong")
    ax2.set_title("Calibration (best layer)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left"); ax2.grid(color="#eee")
    for a in (ax, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(D / f"reader_probe_{args.tag}.png", dpi=150, bbox_inches="tight")
    print(f"wrote reader_probe_{args.tag}.png / reader_probe_results_{args.tag}.json / {npz_path.name}")


if __name__ == "__main__":
    main()
