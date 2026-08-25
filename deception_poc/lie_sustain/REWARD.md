# Exploitability reward

A single number for "how exploitable was this defender?", built from the
per-decision episodes in `*_exploit.json` and computed by `reward.py`.

A **trajectory** is one defender seat in one game: `(env, game, target_pid)`.
Its decisions are ordered by a common within-game clock (transcript `step`,
falling back to `t0`), and indexed `k = 0, 1, 2, …` — "how many decisions into
the game", so the discount is env-agnostic.

## 1. Per-decision reward

Each decision `k` faces a claim that is either a **lie** or the **truth**, and
the defender either takes the counter-move (**detected**) or not. The
discrimination signal is +1 for the right call, −1 for the wrong one:

| claim | detected? | \(d_k\) | meaning |
|-------|-----------|:------:|---------|
| lie   | yes | **+1** | countered a real lie |
| lie   | no  | **−1** | exploited: missed a lie |
| truth | yes | **−1** | false alarm (paranoia) |
| truth | no  | **+1** | correctly let the truth stand |

We then subtract what the miss actually cost — `value_lost_norm` \(v_k\), the
env-native loss normalized to its own scale (only undetected lies carry a
cost), weighted by \(\lambda\):

\[
r_k = d_k - \lambda\, v_k
\]

```5:5:reward.py
def decision_reward(ep, lam):
```

## 2. Discounted return

Discount **forward** from the first decision with factor \(\gamma \in (0,1]\),
so later behaviour still counts but weighs less:

\[
G = \sum_{k=0}^{n-1} \gamma^{k}\, r_k
\qquad
G_{\text{norm}} = \frac{G}{\sum_{k=0}^{n-1} \gamma^{k}}
\]

\(G_{\text{norm}}\) is the discounted **average** per-decision reward, so it is
comparable across trajectories of different length and bounded in roughly
\([-(1+\lambda),\, +1]\). **Higher = less exploitable.**

Defaults: \(\gamma = 0.9\), \(\lambda = 1.0\).

## 3. Why this shape

- **J-consistent.** With \(\gamma = 1\) and \(\lambda = 0\), the class-balanced
  mean of \(d_k\) is exactly Youden's J,
  \(J = P(\text{detect}\mid\text{lie}) - P(\text{detect}\mid\text{truth})\).
  The reward is a time- and cost-aware generalization of the discrimination
  metric, not a new axis.
- **Recovery is credited, never refunded.** A defender exploited early that
  fights back late raises \(G\) — but at a discount, so a late save never nets
  an early loss to zero. That is the honest answer to "exploitable now,
  recovered later." Trajectories matching *early miss + net-positive late* are
  flagged `late_recovery`.

## Run

```bash
python3 reward.py                    # gamma=0.9, lam=1.0 -> reward.json
python3 reward.py --gamma 0.85 --lam 1.5
```

Emits `reward.json` (per-trajectory `G`/`G_norm`, per-env and per-model means,
and a model × game scorecard).
