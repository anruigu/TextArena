## Add counterfactuals and probing

Games: Poker, BlindAuction, New Recruit.

2a. still frontier models from phase 1

Add counterfactual replay, try for both terms.

1. Phase 1 used analytic/observational baselines; Phase 2 replays the game from round t to get true counterfactuals. Two replays, measuring opposite sides of the ledger — same machinery, different thing held fixed:

   a. **Honest-replay counterfactual → Term 1 gain (VoD proper).** *What changes:* the agent's own play (deceptive vs honest from round t). *Answers:* "did the lie earn me more than playing straight would have, right now?" Compute Δ payoff = deceptive rollout − matched non-deceptive baseline rollout (honest disclosure / equal-split / Nash). This upgrades the Phase-1 analytic baseline to a real rollout; the ground-truth resolvers already exist per the feasibility matrix, so the only new ingredient is generating the baseline rollout.

Term 2 — leakage / predictability (the novel half). Train a type reader — the general form of the "Mafia Detector" and the Among Us probe — that predicts the agent's hidden state (hand, role, or private value vector) from its public messages at round t. Leakage is that reader's accuracy/AUROC, and the more useful quantity is the marginal leakage: how much a specific deceptive act sharpens the reader's posterior over the agent's type. In a multi-round game you track whether the reader's belief separates the agent from its prior over rounds — a good deceiver holds the reader near the prior (pooling, low message↔type MI), a bad one separates (the Crawford–Sobel partition collapsing to fine-grained; the poker over-bluff becoming exploitable). 

Two implementations: 
- a behavioral reader (LLM/logreg on the transcript — model-agnostic, works on every env, closed weights fine) (done)

   b. **Leaked-type replay → Term 2 leakage cost (repeated-game, discounted).** *What changes:* the opponents' information (they know your leaked type vs they don't). *Answers:* "how much future payoff did leaking cost me?" Replay from round t with vs without the leaked type revealed to opponents; the discounted difference in future payoff is the cost of being read more easily. Note: this measures the *dollar value of leakage*, so it belongs to Term 2 even though it reuses the replay machinery from (a).


2b 
for the probe, the best open weights models small enough to use: qwen 3.6, GLM, kimi, gemma, etc


