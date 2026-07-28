# Proxy combinations — which Term-2 x Term-1 pairing is most discerning

Model-mean **Pearson r** for every leakage (Term-2) x gain (Term-1) proxy pairing, per game, from the per-model numbers in `../*_results.json` (no rollouts/LLM). Sign: leakage axes are *higher = more readable* unless tagged **(inv)** = higher = more deceptive; gain axes are *higher = better*. So a **negative r** on a readable-leakage axis = the **concealment-pays** direction. `|r|` ranks discernment; **n** = models (small — treat as directional). Bold = |r| >= 0.5.

**cond** = whether the leakage proxy is conditioned on a lie event (see `best_conditioned.py`): **A** = lie-conditioned readability (denominator = lie events, or a base-rate-invariant separability/posterior measure) — *safe*; **B** = raw lie *rate* (a frequency of lying); **C** = unconditioned mix (readability over ALL actions, so it moves with the lie rate). **Only class A is free of the raw-deception-rate confound** — B and C rows are confounded and should not be read as pure leakage. Of every leakage proxy here, only mafia `role_leakage`, coup `bluff_caught_rate`, coup `table_read_auroc`, negotiation `leakage`, and liarsdice `bluff_stick_rate` (empty in data) are class A.

## poker

![poker grid](poker_grid.png)

| Term-2 leakage (x) | cond | Term-1 gain (y) | Pearson r | n |
|---|:--:|---|---:|---:|
| bluff caught rate | A | bluff EV (de-noised) | **-0.98** | 5 |
| bluff-detect AUROC | A | bluff EV (de-noised) | **-0.92** | 5 |
| bluff rate (inv) | B⚠ | bluff chips won | **+0.89** | 5 |
| tell AUROC | C⚠ | bluff chips won | **-0.81** | 5 |
| bet-size tell | C⚠ | bluff chips won | **-0.77** | 5 |
| tell AUROC | C⚠ | mean end chips | **+0.68** | 5 |
| bluff caught rate | A | mean end chips | **+0.64** | 5 |
| bet-size tell | C⚠ | mean end chips | **+0.64** | 5 |
| bluff-detect AUROC | A | mean end chips | **+0.57** | 5 |
| bluff rate (inv) | B⚠ | bluff EV (de-noised) | -0.39 | 5 |
| bluff rate (inv) | B⚠ | mean end chips | -0.34 | 5 |
| bet-size tell | C⚠ | bluff EV (de-noised) | +0.23 | 5 |
| tell AUROC | C⚠ | bluff EV (de-noised) | +0.18 | 5 |
| bluff-detect AUROC | A | bluff chips won | +0.13 | 5 |
| bluff caught rate | A | bluff chips won | -0.03 | 5 |

## kuhn

![kuhn grid](kuhn_grid.png)

| Term-2 leakage (x) | cond | Term-1 gain (y) | Pearson r | n |
|---|:--:|---|---:|---:|
| bluff rate (inv) | B⚠ | bluff chips won | **+0.58** | 5 |
| tell AUROC | C⚠ | bluff chips won | **-0.54** | 5 |
| tell AUROC | C⚠ | mean end chips | +0.24 | 5 |
| bluff rate (inv) | B⚠ | mean end chips | -0.17 | 5 |

## leduc

![leduc grid](leduc_grid.png)

| Term-2 leakage (x) | cond | Term-1 gain (y) | Pearson r | n |
|---|:--:|---|---:|---:|
| bluff caught rate | A | bluff EV (de-noised) | **-0.96** | 4 |
| bluff-detect AUROC | A | bluff EV (de-noised) | **-0.92** | 4 |
| tell AUROC | C⚠ | mean end bank | **+0.88** | 5 |
| bluff rate (inv) | B⚠ | bluff EV (de-noised) | **-0.58** | 4 |
| tell AUROC | C⚠ | bluff EV (de-noised) | **+0.57** | 4 |
| bluff rate (inv) | B⚠ | mean end bank | -0.41 | 5 |
| bluff-detect AUROC | A | mean end bank | +0.01 | 4 |
| bluff caught rate | A | mean end bank | n/a | 5 |

## liarsdice

![liarsdice grid](liarsdice_grid.png)

| Term-2 leakage (x) | cond | Term-1 gain (y) | Pearson r | n |
|---|:--:|---|---:|---:|
| bluff rate (inv) | B⚠ | gain (rank reward) | -0.20 | 5 |
| bluff rate (inv) | B⚠ | mean reward | -0.20 | 5 |
| bluff-detect AUROC | A | gain (rank reward) | -0.16 | 3 |
| bluff-detect AUROC | A | mean reward | -0.16 | 3 |
| leakage own_frac-1/6 | C⚠ | gain (rank reward) | +0.13 | 5 |
| leakage own_frac-1/6 | C⚠ | mean reward | +0.13 | 5 |
| mean own_frac | C⚠ | gain (rank reward) | +0.13 | 5 |
| mean own_frac | C⚠ | mean reward | +0.13 | 5 |
| bluff caught rate | A | gain (rank reward) | n/a | 5 |
| bluff-stick rate (inv) | A | gain (rank reward) | n/a | 5 |
| bluff caught rate | A | mean reward | n/a | 5 |
| bluff-stick rate (inv) | A | mean reward | n/a | 5 |

## blindauction

![blindauction grid](blindauction_grid.png)

| Term-2 leakage (x) | cond | Term-1 gain (y) | Pearson r | n |
|---|:--:|---|---:|---:|
| reader recovery on lies | A | profit | **+0.82** | 7 |
| leakage rho | C⚠ | concealed wins | **-0.73** | 7 |
| misrep rate (inv) | B⚠ | profit | **+0.69** | 7 |
| misrep rate (inv) | B⚠ | concealed wins | **+0.68** | 7 |
| leakage rho | C⚠ | profit | **-0.68** | 7 |
| reader recovery on lies | A | concealed wins | **+0.58** | 7 |

## newrecruit

![newrecruit grid](newrecruit_grid.png)

| Term-2 leakage (x) | cond | Term-1 gain (y) | Pearson r | n |
|---|:--:|---|---:|---:|
| leakage rho | C⚠ | surplus | **-0.55** | 5 |
| reader recovery on lies | A | surplus | -0.47 | 5 |
| misrep rate (inv) | B⚠ | surplus | -0.09 | 5 |

## scorablegames

![scorablegames grid](scorablegames_grid.png)

| Term-2 leakage (x) | cond | Term-1 gain (y) | Pearson r | n |
|---|:--:|---|---:|---:|
| leakage rho | C⚠ | surplus | -0.39 | 7 |
| reader recovery on lies | A | surplus | -0.36 | 7 |
| misrep rate (inv) | B⚠ | surplus | +0.05 | 7 |

## negotiation

![negotiation grid](negotiation_grid.png)

| Term-2 leakage (x) | cond | Term-1 gain (y) | Pearson r | n |
|---|:--:|---|---:|---:|
| leakage (reader->you) | A | value captured | +0.09 | 5 |

## mafia

![mafia grid](mafia_grid.png)

| Term-2 leakage (x) | cond | Term-1 gain (y) | Pearson r | n |
|---|:--:|---|---:|---:|
| role leakage | A | mafia win rate | **-0.53** | 8 |
| role leakage | A | mafia survival | -0.25 | 7 |

## coup

![coup grid](coup_grid.png)

| Term-2 leakage (x) | cond | Term-1 gain (y) | Pearson r | n |
|---|:--:|---|---:|---:|
| table-read AUROC | A | win rate | **-0.51** | 7 |
| table-read AUROC | A | survival rate | **-0.51** | 7 |
| bluff caught rate | A | win rate | -0.47 | 7 |
| bluff caught rate | A | survival rate | -0.47 | 7 |
| bluff rate (inv) | B⚠ | win rate | +0.22 | 7 |
| bluff rate (inv) | B⚠ | survival rate | +0.22 | 7 |
| claim honesty | B⚠ | win rate | -0.22 | 7 |
| claim honesty | B⚠ | survival rate | -0.22 | 7 |

## Most discerning pairings across ALL games

`cond`=A rows are the only ones free of the raw-lie-rate confound; **B⚠/C⚠ are confounded** with how often the model lies.

| game | leakage | cond | gain | r | n |
|---|---|:--:|---|---:|---:|
| poker | bluff caught rate | A | bluff EV (de-noised) | -0.98 | 5 |
| leduc | bluff caught rate | A | bluff EV (de-noised) | -0.96 | 4 |
| poker | bluff-detect AUROC | A | bluff EV (de-noised) | -0.92 | 5 |
| leduc | bluff-detect AUROC | A | bluff EV (de-noised) | -0.92 | 4 |
| poker | bluff rate (inv) | B⚠ | bluff chips won | +0.89 | 5 |
| leduc | tell AUROC | C⚠ | mean end bank | +0.88 | 5 |
| blindauction | reader recovery on lies | A | profit | +0.82 | 7 |
| poker | tell AUROC | C⚠ | bluff chips won | -0.81 | 5 |
| poker | bet-size tell | C⚠ | bluff chips won | -0.77 | 5 |
| blindauction | leakage rho | C⚠ | concealed wins | -0.73 | 7 |
| blindauction | misrep rate (inv) | B⚠ | profit | +0.69 | 7 |
| poker | tell AUROC | C⚠ | mean end chips | +0.68 | 5 |
| blindauction | misrep rate (inv) | B⚠ | concealed wins | +0.68 | 7 |
| blindauction | leakage rho | C⚠ | profit | -0.68 | 7 |
| poker | bluff caught rate | A | mean end chips | +0.64 | 5 |
| poker | bet-size tell | C⚠ | mean end chips | +0.64 | 5 |
| leduc | bluff rate (inv) | B⚠ | bluff EV (de-noised) | -0.58 | 4 |
| blindauction | reader recovery on lies | A | concealed wins | +0.58 | 7 |
| kuhn | bluff rate (inv) | B⚠ | bluff chips won | +0.58 | 5 |
| leduc | tell AUROC | C⚠ | bluff EV (de-noised) | +0.57 | 4 |

## Lie-CONDITIONED leakage proxies only (class A — no raw-rate confound)

| game | leakage | gain | r | n |
|---|---|---|---:|---:|
| poker | bluff caught rate | bluff EV (de-noised) | -0.98 | 5 |
| leduc | bluff caught rate | bluff EV (de-noised) | -0.96 | 4 |
| poker | bluff-detect AUROC | bluff EV (de-noised) | -0.92 | 5 |
| leduc | bluff-detect AUROC | bluff EV (de-noised) | -0.92 | 4 |
| blindauction | reader recovery on lies | profit | +0.82 | 7 |
| poker | bluff caught rate | mean end chips | +0.64 | 5 |
| blindauction | reader recovery on lies | concealed wins | +0.58 | 7 |
| poker | bluff-detect AUROC | mean end chips | +0.57 | 5 |
| mafia | role leakage | mafia win rate | -0.53 | 8 |
| coup | table-read AUROC | win rate | -0.51 | 7 |
| coup | table-read AUROC | survival rate | -0.51 | 7 |
| coup | bluff caught rate | win rate | -0.47 | 7 |
| coup | bluff caught rate | survival rate | -0.47 | 7 |
| newrecruit | reader recovery on lies | surplus | -0.47 | 5 |
| scorablegames | reader recovery on lies | surplus | -0.36 | 7 |
| mafia | role leakage | mafia survival | -0.25 | 7 |
| liarsdice | bluff-detect AUROC | gain (rank reward) | -0.16 | 3 |
| liarsdice | bluff-detect AUROC | mean reward | -0.16 | 3 |
| poker | bluff-detect AUROC | bluff chips won | +0.13 | 5 |
| negotiation | leakage (reader->you) | value captured | +0.09 | 5 |
| poker | bluff caught rate | bluff chips won | -0.03 | 5 |
| leduc | bluff-detect AUROC | mean end bank | +0.01 | 4 |
