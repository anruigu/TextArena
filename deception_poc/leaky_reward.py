#!/usr/bin/env python3
"""Shared DENSE per-bluff reward for the leaky-lies arms race (Phase 1 BoN + Phase 2 RL).

    reward(bluff) = validity_gate * (bluff_gain - lambda * leakage)

  * leakage    = the frozen leak-READER probe's posterior P(strong | public channel) after the
                 action -- a dense, low-variance, per-decision surrogate for "will this get
                 exploited?" (train_reader_probe.py). HIGH leakage = readable bluff = bad.
  * bluff_gain = card-variance-free EV of the aggressive action in chips (fold -> win pot;
                 showdown -> equity * pot, never the realized draw). The audit's de-noiser,
                 identical to poker_bluff.py::action_ev.
  * validity_gate = 0 for an illegal / degenerate action, else 1 (containment: the policy cannot
                 win by emitting garbage, and lie-rate is held fixed elsewhere).

Also exposes realized_chips_reward() -- the SPARSE, high-variance baseline arm (final chips at
showdown). The RLFR thesis is that the dense feature above reaches higher win-rate faster/cheaper
than this sparse signal.

Reader forward pass needs torch; PokerEnv/eval7 need textarena on sys.path. Both live in the
performative venv here (eval7 + nltk/requests/websockets pip-installed; textarena via sys.path):
  /workspace/allie/performative/.venv/bin/python -c "import leaky_reward"
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/allie/performative/scripts")
sys.path.insert(0, "/workspace/allie/TextArena")

D = Path(__file__).resolve().parent
DEFAULT_PROBE = D / "probes" / "probe_leakreader_qwen3_8b.npz"

# MUST match the reader prompt the probe was fit under (train_reader_probe.SYSTEM).
READER_SYSTEM = (
    "You are an expert poker analyst observing ONE player in a heads-up Texas Hold'em hand. You see "
    "only the public information that player's opponent sees: the betting history, the community "
    "board, the pot and the stacks -- you do NOT see that player's hole cards. From their betting "
    "behavior, infer how strong their concealed hand most likely is."
)

WEAK, STRONG = 0.40, 0.60
AGGR_RE = re.compile(r"\[(Bet|Raise)\b", re.I)
ACT_RE = re.compile(r"\[(Check|Call|Fold|Bet|Raise)\b", re.I)
HOLE_RE = re.compile(r"^.*Your hole:.*$\n?", re.MULTILINE)


def is_aggressive(action: str) -> bool:
    return bool(AGGR_RE.search(action or ""))


def is_bluff(equity: float, action: str, weak: float = WEAK) -> bool:
    """A lie = an aggressive act on a weak hand (the game-engine monitor; ground truth exact)."""
    return is_aggressive(action) and equity < weak


def is_valid_action(action: str) -> bool:
    """Legibility gate: exactly one legal bracket token, non-degenerate."""
    return bool(ACT_RE.search(action or ""))


def strip_hole(obs: str) -> str:
    """Actor obs -> opponent's public view (drop the private hole line). Matches build_reader_spots."""
    return HOLE_RE.sub("", obs or "").strip()


def action_ev(pot, bet, equity, opp_folded) -> float | None:
    """Card-variance-free EV (chips) of an aggressive action. fold -> win pot outright (deterministic);
    showdown -> equity*(pot+2*bet) - bet (analytic, no realized-card noise). Identical to
    poker_bluff.py::action_ev. Returns None if unpriced."""
    if pot is None:
        return None
    bet = bet or 0
    if opp_folded:
        return float(pot)
    return equity * (pot + 2 * bet) - bet


def dense_reward(bluff_gain, leakage, lam, valid=True) -> float:
    """validity_gate * (bluff_gain - lambda*leakage). bluff_gain in chips, leakage in [0,1];
    lambda therefore has chip units (sweep it). Non-bluff / unpriced decisions -> 0 contribution."""
    if not valid:
        return 0.0
    if bluff_gain is None:
        return 0.0
    return float(bluff_gain - lam * float(leakage))


def realized_chips_reward(final_chips, pid, starting=1000) -> float:
    """SPARSE baseline arm: net chips at match end (high variance -- dominated by the runout)."""
    try:
        return float(final_chips[str(pid)] - starting)
    except (KeyError, TypeError):
        try:
            return float(final_chips[pid] - starting)
        except Exception:
            return 0.0


class LeakReader:
    """Frozen reader model + fitted linear leak probe. leakage(obs) = P(strong | public residual).
    This is the detector; the outer arms-race loop REFITS the probe (train_reader_probe.py) and
    reloads it here between rounds."""

    def __init__(self, probe_path=DEFAULT_PROBE, device="cuda:0", batch=16, model_name=None):
        from model_utils import load_tok, load_model, text_dims  # noqa: E402
        d = np.load(probe_path, allow_pickle=True)
        self.w = d["w"].astype(np.float32)
        self.b = float(d["b"][0]) if np.ndim(d["b"]) else float(d["b"])
        self.mu = d["mu"].astype(np.float32)
        self.sd = d["sd"].astype(np.float32)
        self.layer = int(d["layer"])
        self.model_name = model_name or (str(d["model"]) if "model" in d else "Qwen/Qwen3-8B")
        self.batch = batch
        self.tok = load_tok(self.model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = load_model(self.model_name, device)
        self.n_layers, self.d = text_dims(self.model)
        self.dev = next(self.model.parameters()).device

    def _residuals(self, obs_texts):
        import torch
        from model_utils import apply_template
        out = np.zeros((len(obs_texts), self.d), dtype=np.float32)
        for b0 in range(0, len(obs_texts), self.batch):
            batch = obs_texts[b0:b0 + self.batch]
            seqs = [apply_template(self.tok, [{"role": "system", "content": READER_SYSTEM},
                                              {"role": "user", "content": t}],
                                   add_generation_prompt=True, no_think=True) for t in batch]
            maxlen = max(len(s) for s in seqs)
            pad = self.tok.pad_token_id
            ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, maxlen - len(s):] = torch.tensor(s)
                att[i, maxlen - len(s):] = 1
            with torch.no_grad():
                o = self.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev),
                               output_hidden_states=True)
            out[b0:b0 + len(batch)] = o.hidden_states[self.layer][:, -1, :].float().cpu().numpy()
            del o
        return out

    def leakage(self, obs_texts, already_public=False):
        """P(strong | public channel) for each actor obs. Pass raw actor obs (hole auto-stripped)
        or already-public text (already_public=True). Returns np.array in [0,1]."""
        if isinstance(obs_texts, str):
            obs_texts = [obs_texts]
        pub = obs_texts if already_public else [strip_hole(t) for t in obs_texts]
        X = self._residuals(pub)
        z = (X - self.mu) / self.sd
        logits = z @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-logits))


if __name__ == "__main__":
    # smoke: load probe, score a few public channels from reader_spots.jsonl
    import json
    spots = [json.loads(l) for l in (D / "reader_spots.jsonl").read_text().splitlines() if l.strip()]
    sample = [s for s in spots if s["source"] == "real"][:8]
    r = LeakReader()
    leak = r.leakage([s["obs"] for s in sample], already_public=True)
    print("layer", r.layer, "model", r.model_name)
    for s, p in zip(sample, leak):
        print(f"  eq={s['equity']:.2f} street={s['street']:<6} action={s['actor_action']:<8} "
              f"leakage(P_strong)={p:.3f}")
