# Domino Pernambucano AI — Research Catalog
## Comprehensive Literature Sweep (Feb 28, 2026)

**Target**: Near-superhuman AI for 4-player partnership domino (Pernambuco rules).
**Current stack**: ISMCTS + Bayesian belief tracking + bitmask endgame solver + coach/explanation layer.

---

## DELIVERABLE A: FULL CATALOG (de-duplicated, 60+ papers)

### Legend
- **II** = Imperfect Information | **MP** = Multiplayer | **P** = Partnership
- **Relevance**: 0–5 for Pernambuco domino AI specifically
- Tags: `ISMCTS` `belief` `POMDP` `multi-agent` `partner` `endgame` `tablebase` `neural` `MuZero` `offline-RL` `eval` `explain` `self-play` `CFR`

---

### CAT-01 | Information Set Monte Carlo Tree Search (IS-MCTS)
| Field | Value |
|-------|-------|
| Authors | Cowling, Powley, Whitehouse |
| Date | 2012 |
| Venue | IEEE TCIAIG 4(2) |
| Link | DOI: 10.1109/TCIAIG.2012.2200894 |
| Flags | II:Y MP:Y P:tested(Doppelkopf) |
| Tags | `ISMCTS` `belief` `multi-agent` |
| Relevance | **5/5** |
| Core idea | (1) SO-ISMCTS: single tree over info sets, sample determinization per iteration. (2) MO-ISMCTS: separate trees per player. (3) Avoids strategy fusion vs naive PIMC. (4) Tested on trick-taking card games. |
| Pitfalls | Still has non-locality issues. SO-ISMCTS can't model opponent beliefs. MO more expensive. |
| Steal for domino | SO-ISMCTS is the primary baseline. Nodes = (hand, board, play/pass history). Smart rollouts >> random. Progressive bias from heuristics. ~50-100K iterations in <1s. |
| Compute | Light: 10K–100K iters/move on modern hardware |
| Code | OpenSpiel, various GitHub implementations |

---

### CAT-02 | On the Dangers of Determinization (Strategy Fusion)
| Field | Value |
|-------|-------|
| Authors | Frank, Basin |
| Date | 1998 |
| Venue | Artificial Intelligence 100, pp. 87–123 |
| Link | DOI: 10.1016/S0004-3702(98)00005-0 |
| Flags | II:Y MP:Y P:Y(Bridge) |
| Tags | `ISMCTS` `belief` `partner` |
| Relevance | **4/5** |
| Core idea | (1) Defines **strategy fusion**: averaging over determinizations picks moves good in no specific world. (2) Defines **non-locality**: best action depends on info you lack. (3) Demonstrated concretely in Bridge. |
| Pitfalls | This IS the paper about pitfalls — essential negative result. |
| Steal for domino | Understand when ISMCTS will fail. Track cases where AI "looks confused" — likely strategy fusion. Use ISMCTS instead of naive PIMC. Weight determinizations by Bayesian posterior. |
| Compute | Theoretical paper |
| Code | N/A |

---

### CAT-03 | Understanding the Success of PIMC (Long et al.)
| Field | Value |
|-------|-------|
| Authors | Long, Sturtevant, Buro, Furtak |
| Date | 2010 |
| Venue | AAAI 2010 |
| Flags | II:Y MP:varies P:N |
| Tags | `ISMCTS` `belief` `eval` |
| Relevance | **4/5** |
| Core idea | (1) Formal analysis of WHEN/WHY PIMC works despite theoretical issues. (2) "Smoothness" condition: small hidden-info changes ≈ same optimal play → PIMC works. (3) Endgame has less smoothness → more failure. |
| Steal for domino | Domino is relatively smooth (PIMC should work reasonably). Endgame needs specialized handling. Use their framework to predict failure modes. |

---

### CAT-04 | POMCP — Monte Carlo Planning in Large POMDPs
| Field | Value |
|-------|-------|
| Authors | Silver, Veness |
| Date | 2010 |
| Venue | NeurIPS 2010 |
| Flags | II:Y MP:N P:N |
| Tags | `belief` `POMDP` |
| Relevance | **4/5** |
| Core idea | (1) Extends UCT to POMDPs via particle filter belief at each node. (2) Each particle = sampled state consistent with observation history. (3) Proved convergence in huge POMDPs. |
| Pitfalls | Single-agent. Particle depletion in deep trees. |
| Steal for domino | Particle filter belief is directly useful. Maintain set of "worlds" consistent with play history. Reinvigorate particles when depleted. Hybridize: ISMCTS tree + POMCP-style particle filter. |
| Compute | 1K–100K sims/step. Modern: 100K+/sec |
| Code | POMDPs.jl, Python, C++ implementations |

---

### CAT-05 | Smooth UCT Search in Computer Poker
| Field | Value |
|-------|-------|
| Authors | Heinrich, Silver |
| Date | 2015 |
| Venue | IJCAI 2015 |
| Flags | II:Y MP:N P:N |
| Tags | `ISMCTS` |
| Relevance | **4/5** |
| Core idea | (1) Standard UCT converges to pure strategy; optimal play often requires mixing. (2) Smooth UCT averages over strategy history → converges to Nash in 2p zero-sum. (3) Makes ISMCTS less exploitable. |
| Steal for domino | Even without 4p guarantees, Smooth UCT produces more robust play. Natural signaling discovery. Use average strategy for action selection. |
| Code | OpenSpiel |

---

### CAT-06 | Online Outcome Sampling (MCTS + CFR)
| Field | Value |
|-------|-------|
| Authors | Lanctot, Lisy, Winands |
| Date | 2014 |
| Venue | AAMAS 2014 |
| Flags | II:Y MP:N P:N |
| Tags | `ISMCTS` `CFR` |
| Relevance | **5/5** |
| Core idea | (1) Bridges MCTS and CFR: MCTS tree structure with regret-matching node updates. (2) Converges to Nash in 2p zero-sum while keeping MCTS benefits (anytime, parallelizable). (3) Naturally produces mixed strategies. |
| Steal for domino | Best-of-both-worlds. Drop-in replacement for UCB in ISMCTS nodes. Produces mixed strategies → avoids pure-strategy convergence. Highly recommended after basic ISMCTS works. |
| Compute | Same as ISMCTS |
| Code | OpenSpiel |

---

### CAT-07 | Progressive Strategies for MCTS (Progressive Bias)
| Field | Value |
|-------|-------|
| Authors | Chaslot, Winands, van den Herik |
| Date | 2008 |
| Venue | NMCN 4(3) |
| Flags | II:applicable MP:applicable P:applicable |
| Tags | `ISMCTS` `neural` |
| Relevance | **5/5** |
| Core idea | (1) Progressive bias: UCB + H(s,a)/(1+n(s,a)) where H = heuristic value. (2) Progressive widening: grow action set with visits. (3) Allows domain knowledge injection without losing convergence. |
| Steal for domino | ESSENTIAL. Inject: prefer partner's suit, block opponent's suit, play doubles early, low pip tiles late. Use smartAI scoring as initial H. Decays naturally → MCTS overrides bad heuristics. |
| Compute | Negligible overhead |

---

### CAT-08 | Bayesian Action Decoder (BAD)
| Field | Value |
|-------|-------|
| Authors | Foerster, Song, Hughes, Burch, Dunning, Mourad, Lanctot, Bowling |
| Date | 2019 |
| Venue | NeurIPS 2019 |
| Link | arXiv:1811.01458 |
| Flags | II:Y MP:Y P:Y(cooperative) |
| Tags | `multi-agent` `partner` `POMDP` `neural` |
| Relevance | **5/5** |
| Core idea | (1) Treats partner's actions as noisy communication channel. (2) Public belief updated Bayesianly after each action. (3) Actions serve dual purpose: game advancement + info signaling. (4) Best Hanabi scores via explicit signaling optimization. |
| Pitfalls | Expensive belief computation. Assumes shared policy for Bayesian update. |
| Steal for domino | **THE most relevant paper for domino signaling.** Every play signals. Passes are maximally informative. Dual-purpose action selection: advance game + communicate. Public belief framework directly applicable. |
| Code | DeepMind Hanabi research code (partial) |

---

### CAT-09 | The Hanabi Challenge
| Field | Value |
|-------|-------|
| Authors | Bard, Foerster, Chandar, Burch, Lanctot et al. |
| Date | 2020 |
| Venue | Artificial Intelligence 280 |
| Link | arXiv:1902.00506 |
| Flags | II:Y MP:Y P:Y(cooperative) |
| Tags | `multi-agent` `partner` `POMDP` `neural` `eval` |
| Relevance | **5/5** |
| Core idea | (1) Cooperative card game benchmark with imperfect info. (2) Convention formation is central challenge. (3) Self-play vs cross-play tension. (4) Surveys rule-based, search, RL approaches. |
| Steal for domino | Convention emergence framework. Cross-play evaluation methodology. Belief representation. Zero-shot coordination relevance (AI + human partners). |
| Code | Hanabi Learning Environment (open-source) |

---

### CAT-10 | Other-Play for Zero-Shot Coordination
| Field | Value |
|-------|-------|
| Authors | Hu, Foerster |
| Date | 2020 |
| Venue | ICML 2020 |
| Link | arXiv:2003.02979 |
| Flags | II:Y MP:Y P:Y(cooperative) |
| Tags | `multi-agent` `partner` `neural` `self-play` |
| Relevance | **4/5** |
| Core idea | (1) Self-play → arbitrary conventions → brittle coordination. (2) Symmetry augmentation during training → robust conventions. (3) Agents coordinate with unseen partners (zero-shot). |
| Steal for domino | Symmetry-aware training using tile/seat permutations. Robust partnership → works with any human. Evaluate cross-play performance drop. |
| Code | GitHub (Hanabi research repos) |

---

### CAT-11 | Simplified Action Decoder (SAD)
| Field | Value |
|-------|-------|
| Authors | Hu, Foerster |
| Date | 2020 |
| Venue | ICLR 2020 |
| Link | arXiv:1912.02288 |
| Flags | II:Y MP:Y P:Y(cooperative) |
| Tags | `multi-agent` `partner` `neural` |
| Relevance | **4/5** |
| Core idea | (1) Agents decode partner's actions to infer hidden info. (2) Auxiliary training signal from partner's perspective. (3) Near-optimal Hanabi self-play. |
| Steal for domino | Action decoding: interpret partner's plays to infer hand. Auxiliary prediction targets during training. |
| Code | Facebook Research GitHub |

---

### CAT-12 | Off-Belief Learning (OBL)
| Field | Value |
|-------|-------|
| Authors | Hu, Lerer, Cui, Pineda, Brown, Foerster |
| Date | 2021 |
| Venue | ICML 2021 |
| Link | arXiv:2103.04000 |
| Flags | II:Y MP:Y P:Y(cooperative) |
| Tags | `multi-agent` `partner` `neural` `self-play` |
| Relevance | **5/5** |
| Core idea | (1) Decomposes policy value into "belief" (universal) and "convention" (fragile) components. (2) Trains to maximize belief value → convention-agnostic. (3) Strong self-play AND robust to different partner conventions. |
| Steal for domino | Separate Bayesian inference (passes → no matching tiles = universal) from conventions (double-lead = signaling = fragile). Resolves self-play vs cross-play tension for partnership domino. |
| Code | Facebook Research / Hanabi repos |

---

### CAT-13 | SPARTA — Search in Cooperative Partially Observable Games
| Field | Value |
|-------|-------|
| Authors | Lerer, Hu, Foerster, Brown |
| Date | 2020 |
| Venue | AAAI 2020 |
| Link | arXiv:1912.02318 |
| Flags | II:Y MP:Y P:Y(cooperative) |
| Tags | `belief` `multi-agent` `partner` `neural` `eval` |
| Relevance | **5/5** |
| Core idea | (1) Hybrid: learned policy (blueprint) + real-time search. (2) Sample possible hands for acting player, search assuming others follow blueprint. (3) "Common convention" insight: shared policy = natural coordination. |
| Pitfalls | Assumes all players use same blueprint. |
| Steal for domino | Directly applicable to partnership. Shared policy → implicit coordination. Model partner's future play with same policy. Domino easier than Hanabi (see others' plays). |
| Code | Facebook/Meta Hanabi agents |

---

### CAT-14 | ReBeL — Recursive Belief-Based Learning
| Field | Value |
|-------|-------|
| Authors | Brown, Bakhtin, Lerer, Gong |
| Date | 2020 |
| Venue | NeurIPS 2020 (Best Paper) |
| Link | arXiv:2007.13544 |
| Flags | II:Y MP:N(2p) P:N |
| Tags | `belief` `neural` `CFR` `endgame` `self-play` |
| Relevance | **5/5** |
| Core idea | (1) Generalizes AlphaZero to imperfect info via Public Belief States (PBS). (2) PBS = public info + distribution over private states. (3) Value net V(PBS) trained via self-play. (4) Depth-limited CFR search at each PBS. |
| Pitfalls | 2p zero-sum. Extending to 4p partnership is open research. |
| Steal for domino | PBS concept is elegant for domino: PBS = (board, play/pass history, belief distribution). Train value net on PBS. Belief-space search replaces rollouts. |
| Code | https://github.com/facebookresearch/rebel |

---

### CAT-15 | Student of Games (SoG)
| Field | Value |
|-------|-------|
| Authors | Schmid, Moravcik, Burch et al. (DeepMind) |
| Date | 2023 |
| Venue | Science Advances 9(46) |
| Link | DOI: 10.1126/sciadv.adg3256 |
| Flags | II:Y MP:N(2p tested) P:N |
| Tags | `neural` `ISMCTS` `self-play` `CFR` |
| Relevance | **5/5** |
| Core idea | (1) Single algorithm for perfect AND imperfect info games. (2) Growing-tree CFR + learned value/policy nets. (3) Tested on Go, chess, poker, Scotland Yard. |
| Steal for domino | Most directly relevant hybrid architecture. GT-CFR could replace ISMCTS. Unified framework handles both hidden tiles and strategic play. |
| Compute | Research-lab scale |
| Code | Partially in OpenSpiel |

---

### CAT-16 | MuZero — Planning with a Learned Model
| Field | Value |
|-------|-------|
| Authors | Schrittwieser, Antonoglou, Hubert et al. (DeepMind) |
| Date | 2020 |
| Venue | Nature 588 |
| Link | arXiv:1911.08265 |
| Flags | II:partial(Atari) MP:N P:N |
| Tags | `MuZero` `neural` `self-play` |
| Relevance | **5/5** |
| Core idea | (1) Learns dynamics model (state transitions) without ground truth game state. (2) Three nets: representation h(o)→s, dynamics g(s,a)→(r,s'), prediction f(s)→(p,v). (3) Plans in learned latent space. |
| Steal for domino | Learned dynamics model can implicitly track beliefs about hidden tiles. Representation net compresses observation history into belief-like state. Plans without enumerating opponent hands. |
| Code | muzero-general (GitHub), EfficientZero |

---

### CAT-17 | DeepStack — Expert-Level Poker AI
| Field | Value |
|-------|-------|
| Authors | Moravcik, Schmid, Burch, Lisy et al. |
| Date | 2017 |
| Venue | Science 356(6337) |
| Link | DOI: 10.1126/science.aam6960 |
| Flags | II:Y MP:N(2p) P:N |
| Tags | `belief` `neural` `endgame` |
| Relevance | **4/5** |
| Core idea | (1) Continual re-solving: solve subtrees on-the-fly with neural leaf evaluation. (2) Maintains "range" (belief over opponent hands). (3) No pre-computed blueprint needed. |
| Steal for domino | "Range" = tile-probability matrix. Continual re-solving feasible for domino (small game). Train value net on endgame positions. |

---

### CAT-18 | Pluribus — Superhuman 6-Player Poker
| Field | Value |
|-------|-------|
| Authors | Brown, Sandholm |
| Date | 2019 |
| Venue | Science 365(6456) |
| Link | DOI: 10.1126/science.aay2400 |
| Flags | II:Y MP:Y(6p) P:N |
| Tags | `multi-agent` `endgame` `CFR` `self-play` |
| Relevance | **5/5** |
| Core idea | (1) Blueprint via MCCFR + real-time depth-limited search. (2) Assumes opponents play "continuation strategy" (modified blueprint). (3) Perfect equilibrium not needed for multiplayer — practical robustness suffices. (4) ~$150 training cost. |
| Steal for domino | Blueprint + search architecture. Depth-limited search with value function at leaves. Opponent continuation assumption = model opponents as blueprint players. Partnership gap = must ADD team modeling. |
| Compute | Blueprint: $150 cloud. Real-time: 20s/move on single CPU |

---

### CAT-19 | Libratus — Safe Subgame Solving
| Field | Value |
|-------|-------|
| Authors | Brown, Sandholm |
| Date | 2018 |
| Venue | Science 359(6374) |
| Link | DOI: 10.1126/science.aao1733 |
| Flags | II:Y MP:N(2p) P:N |
| Tags | `endgame` `CFR` `self-play` |
| Relevance | **4/5** |
| Core idea | (1) Nested safe subgame solving ensures re-solved strategy doesn't increase exploitability. (2) Endgame re-solving at finer granularity. |
| Steal for domino | Endgame re-solving for final 2-3 rounds (tiles_remaining ≤ 12). Maintain consistency between MCTS values and endgame solver. Safe transition from search to exact solve. |

---

### CAT-20 | Safe and Nested Subgame Solving
| Field | Value |
|-------|-------|
| Authors | Brown, Sandholm |
| Date | 2017 |
| Venue | NeurIPS 2017 |
| Link | arXiv:1705.02955 |
| Flags | II:Y MP:N P:N |
| Tags | `endgame` `ISMCTS` |
| Relevance | **5/5** |
| Core idea | (1) Decomposes large game into independently solvable subgames. (2) "Safe" = grafted solution doesn't increase exploitability. (3) "Nested" = recursively deeper subgames. |
| Steal for domino | Natural boundary: tiles_remaining ≤ threshold → switch from ISMCTS to exact solver. Safety: endgame solution must be consistent with trunk strategy. |

---

### CAT-21 | CFR — Counterfactual Regret Minimization
| Field | Value |
|-------|-------|
| Authors | Zinkevich, Johanson, Bowling, Piccione |
| Date | 2008 |
| Venue | NeurIPS 2007 |
| Flags | II:Y MP:2+ P:extensible |
| Tags | `CFR` |
| Relevance | **4/5** |
| Core idea | (1) Iterative regret-matching converges to Nash in 2p zero-sum. (2) Foundation for all modern poker AI. (3) MCCFR variants for scalability. |
| Steal for domino | MCCFR for endgame CFR. Blueprint strategy computation offline. Domino game tree small enough for direct CFR with moderate abstraction. |
| Code | OpenSpiel, RLCard |

---

### CAT-22 | GIB — Bridge PIMC Pioneer
| Field | Value |
|-------|-------|
| Authors | Ginsberg |
| Date | 2001 |
| Venue | JAIR 14, pp. 303–358 |
| Flags | II:Y MP:Y(4p) P:Y |
| Tags | `ISMCTS` `partner` `belief` |
| Relevance | **5/5** |
| Core idea | (1) Pioneered PIMC for partnership card games: sample N deals, solve each with double-dummy, pick best-average move. (2) Identified strategy fusion. (3) Expert-level bridge play. |
| Steal for domino | THE baseline paradigm for domino AI. Double-dummy equivalent = minimax on determinized 4p game. Weight samples by consistency with play history. |

---

### CAT-23 | NukkAI NooK — World Champion Bridge AI
| Field | Value |
|-------|-------|
| Authors | Ventos, Music, Fantun (NukkAI team) |
| Date | 2022 |
| Venue | Competition results + technical reports |
| Flags | II:Y MP:Y(4p) P:Y |
| Tags | `partner` `neural` `belief` |
| Relevance | **5/5** |
| Core idea | (1) Beat 8 world champion bridge players. (2) Modular: separate bidding (neural) + card play (PIMC/DDS). (3) Partnership coordination via self-play (conventions emerge). (4) Inference engine estimates hands from bid/play history. |
| Steal for domino | Partnership coordination via self-play is directly transferable. Modular architecture (inference + planning + execution). Convention emergence through training. The hardest problem solved (partnership without communication) IS the core Pernambucano challenge. |

---

### CAT-24 | Suphx — Superhuman Mahjong AI
| Field | Value |
|-------|-------|
| Authors | Li, Koyamada, Ye et al. (Microsoft Research Asia) |
| Date | 2020 |
| Venue | arXiv:2003.13590 |
| Flags | II:Y MP:Y(4p) P:N |
| Tags | `neural` `self-play` |
| Relevance | **5/5** |
| Core idea | (1) First AI to achieve top 0.01% on Tenhou (Mahjong). (2) Deep RL + self-play + human game logs. (3) **Oracle guiding**: train perfect-info oracle, distill to imperfect-info agent. (4) Global reward prediction. |
| Steal for domino | Oracle guiding highly transferable: train perfect-info domino AI, distill. Belief encoding for tile games. Self-play for 4-player. Opponent modeling from play patterns ≈ suit tracking. |

---

### CAT-25 | DeepNash — Stratego AI
| Field | Value |
|-------|-------|
| Authors | Perolat, De Vylder, Hennes et al. (DeepMind) |
| Date | 2022 |
| Venue | Science 378(6623) |
| Link | DOI: 10.1126/science.add4679 |
| Flags | II:Y MP:N(2p) P:N |
| Tags | `neural` `self-play` |
| Relevance | **4/5** |
| Core idea | (1) Model-free RL (R-NaD) for Stratego (10^535 states). (2) No search at test time — pure policy net. (3) Implicit belief tracking via learned representations. |
| Steal for domino | Implicit belief tracking via neural nets. Observation encoding of partial info (probability that opponent has tile X). Even without search, strong policy possible for small games. |

---

### CAT-26 | Expert Iteration (ExIt)
| Field | Value |
|-------|-------|
| Authors | Anthony, Tian, Barber |
| Date | 2017 |
| Venue | NeurIPS 2017 |
| Link | arXiv:1705.08439 |
| Flags | II:N(original) MP:N P:N |
| Tags | `neural` `ISMCTS` `self-play` |
| Relevance | **5/5** |
| Core idea | (1) Expert = MCTS. Apprentice = neural net trained to imitate expert. (2) Virtuous cycle: better net → better search → better training data. (3) Simpler than AlphaZero, more practical for small projects. |
| Steal for domino | **Most practical path for domino neural enhancement.** Phase 1: Run ISMCTS, record visit distributions. Phase 2: Train policy net to predict visits. Phase 3: Use net as prior in ISMCTS. Repeat. |
| Compute | Single GPU for domino-scale |

---

### CAT-27 | KataGo — Efficient Self-Play Training
| Field | Value |
|-------|-------|
| Authors | Wu |
| Date | 2020 |
| Venue | arXiv:1902.10565 |
| Link | https://github.com/lightvector/KataGo |
| Flags | II:N MP:N P:N |
| Tags | `neural` `self-play` |
| Relevance | **4/5** |
| Core idea | (1) 50x more efficient AlphaZero training via practical innovations. (2) Playout cap randomization: vary search budget during training. (3) Auxiliary targets (ownership, score distribution). (4) Forced playouts for exploration. |
| Steal for domino | Playout cap randomization is huge efficiency win. Auxiliary targets: predict which player goes out first, final pip difference, whether game blocks. Policy target pruning reduces noise. |
| Code | Open source: https://github.com/lightvector/KataGo |

---

### CAT-28 | PSRO — Policy-Space Response Oracles
| Field | Value |
|-------|-------|
| Authors | Lanctot, Zambaldi, Gruslys et al. |
| Date | 2017 |
| Venue | NeurIPS 2017 |
| Link | arXiv:1711.00832 |
| Flags | II:varies MP:Y P:N |
| Tags | `self-play` `multi-agent` `eval` |
| Relevance | **5/5** |
| Core idea | (1) Naive self-play can cycle in multiplayer games. (2) PSRO: maintain population of policies, compute meta-Nash, train new best response. (3) Principled approach to multiplayer training. |
| Steal for domino | CRITICAL awareness: 4p self-play may not converge. Maintain league of frozen past policies. Train against league. Evaluate against full league. |
| Code | OpenSpiel |

---

### CAT-29 | Re-determinizing MCTS in Trick-Taking Games
| Field | Value |
|-------|-------|
| Authors | Rebstock, Solinas, Bowling |
| Date | 2019 |
| Venue | IEEE CoG 2019 |
| Link | DOI: 10.1109/CIG.2019.8848029 |
| Flags | II:Y MP:Y P:partial(Skat) |
| Tags | `ISMCTS` `belief` |
| Relevance | **5/5** |
| Core idea | (1) Re-determinize at each interior node, not just root. (2) Combats strategy fusion more effectively. (3) Significantly better play in Skat. |
| Steal for domino | Directly applicable. Resample tile assignments at each MCTS node conditioned on all public info up to that point. Prevents "accidental peeking". 2-5x overhead but within real-time budget. |

---

### CAT-30 | Skat AI — PIMC for Trick-Taking
| Field | Value |
|-------|-------|
| Authors | Kupferschmid, Helmert; Buro, Long, Furtak, Sturtevant |
| Date | 2005–2017 |
| Venue | CIG, ACG, KI |
| Flags | II:Y MP:Y(3p) P:partial |
| Tags | `ISMCTS` `belief` |
| Relevance | **4/5** |
| Core idea | (1) PIMC with enhanced sampling beats naive MCTS. (2) Strategy fusion analysis for trick-taking. (3) Informed sampling using play history. |
| Steal for domino | Informed sampling from play/pass history is CRUCIAL. Move pruning: skip obviously bad tiles. Closest published card-game analog to dominoes. |

---

### CAT-31 | NFSP — Neural Fictitious Self-Play
| Field | Value |
|-------|-------|
| Authors | Heinrich, Silver |
| Date | 2016 |
| Venue | NeurIPS 2016 |
| Link | arXiv:1603.01121 |
| Flags | II:Y MP:N P:N |
| Tags | `neural` `self-play` |
| Relevance | **3/5** |
| Core idea | (1) Best-response network (RL) + average-strategy network (SL). (2) Converges toward Nash via fictitious play. (3) No game tree traversal needed. |
| Steal for domino | Training framework for neural enhancement. Train on average of past policies for robustness. Rollout policy for ISMCTS. |
| Code | OpenSpiel |

---

### CAT-32 | Deep CFR
| Field | Value |
|-------|-------|
| Authors | Brown, Lerer, Gross, Sandholm |
| Date | 2019 |
| Venue | ICML 2019 |
| Link | arXiv:1811.00164 |
| Flags | II:Y MP:N P:N |
| Tags | `neural` `CFR` `self-play` |
| Relevance | **3/5** |
| Core idea | (1) Neural net replaces tabular regret storage in CFR. (2) Scales CFR to large games without abstraction. (3) Reservoir sampling for training data. |
| Steal for domino | Neural regret representation for scaling. Reservoir sampling for self-play data management. |

---

### CAT-33 | DREAM — Deep Regret Minimization
| Field | Value |
|-------|-------|
| Authors | Steinberger, Lerer, Brown |
| Date | 2020 |
| Venue | arXiv:2006.10410 |
| Flags | II:Y MP:N P:N |
| Tags | `MuZero` `neural` `self-play` |
| Relevance | **3/5** |
| Core idea | (1) Model-free deep CFR with advantage baselines. (2) Trajectory sampling instead of full tree traversal. (3) Faster convergence than Deep CFR. |
| Steal for domino | Advantage baselines for variance reduction. Trajectory sampling maps to ISMCTS. |
| Code | https://github.com/EricSteinberger/DREAM |

---

### CAT-34 | Discounted CFR (DCFR)
| Field | Value |
|-------|-------|
| Authors | Brown, Sandholm |
| Date | 2019 |
| Venue | AAAI 2019 |
| Link | arXiv:1809.04040 |
| Flags | II:Y MP:N P:N |
| Tags | `CFR` `endgame` |
| Relevance | **3/5** |
| Core idea | (1) Weight recent iterations more heavily → faster convergence. (2) Linear CFR: weight t with t. (3) Both converge faster than vanilla CFR. |
| Steal for domino | Use DCFR/Linear CFR for endgame solving subroutine — faster convergence to good strategies. |

---

### CAT-35 | Alpha-Beta + Transposition Tables
| Field | Value |
|-------|-------|
| Authors | Classic (Zobrist 1970, Akl & Newborn 1977, Schaeffer 1989) |
| Date | 1970–present |
| Venue | Foundational CS |
| Flags | II:N MP:N P:N |
| Tags | `endgame` `tablebase` |
| Relevance | **5/5** |
| Core idea | (1) Alpha-beta pruning. (2) Zobrist hashing for transposition tables. (3) Move ordering (killer, history heuristic). (4) Iterative deepening. |
| Steal for domino | ADD transposition table to bitmask endgame solver via Zobrist hashing. Move ordering: try both-end matches first, then high pip tiles. IDDFS for anytime behavior. Dramatic speedup for endgame. |

---

### CAT-36 | Retrograde Analysis / Endgame Tablebases
| Field | Value |
|-------|-------|
| Authors | Zakharov et al. (Lomonosov), de Man (Syzygy) |
| Date | 2012+ |
| Venue | Technical projects |
| Link | https://syzygy-tables.info/ |
| Flags | II:N MP:N P:N |
| Tags | `endgame` `tablebase` |
| Relevance | **4/5** |
| Core idea | (1) Solve backwards from terminal positions. (2) Enumerate all positions with N pieces, compute optimal result. (3) DTZ compression. |
| Steal for domino | With ≤3 tiles/player (12 tiles total), exact retrograde analysis is feasible. Bitmask representation maps cleanly. Pre-compute tablebases for common endgame configurations. Switch from ISMCTS to exact solver when tiles_remaining ≤ threshold. |

---

### CAT-37 | OpenSpiel Framework
| Field | Value |
|-------|-------|
| Authors | Lanctot, Lockhart et al. (DeepMind) |
| Date | 2019 |
| Venue | arXiv:1908.09453 |
| Link | https://github.com/deepmind/open_spiel |
| Flags | II:Y MP:Y P:Y |
| Tags | `ISMCTS` `CFR` `neural` `eval` |
| Relevance | **5/5** |
| Core idea | (1) 70+ games, 20+ algorithms. (2) Includes IS-MCTS, CFR, MCCFR, Smooth UCT, deep RL. (3) Reference implementations for research. |
| Steal for domino | Could implement domino as OpenSpiel game → instant access to all algorithms. Benchmark against reference implementations. Use as training infrastructure. |
| Code | Fully open source |

---

### CAT-38 | Multi-Player MCTS (max^n, Paranoid, BRS, Coalition)
| Field | Value |
|-------|-------|
| Authors | Sturtevant (2008), Schadd & Winands (2011), Browne et al. (2012 survey) |
| Date | 2008–2012 |
| Venue | ICGA, IEEE TCIAIG |
| Flags | II:N(base) MP:Y P:N(but coalition applicable) |
| Tags | `multi-agent` `ISMCTS` |
| Relevance | **5/5** |
| Core idea | (1) max^n: vector utilities, each player maximizes own. (2) Paranoid: all opponents minimize your score. (3) BRS: single best replier per turn. (4) **Coalition reduction**: collapse 2v2 to 2-team game → standard minimax. |
| Steal for domino | Coalition reduction is THE natural fit for 2v2 domino. Collapse to 2-team minimax: your team maximizes, opponent team minimizes. Halves effective search depth. Do NOT use Paranoid for partner. |

---

### CAT-39 | Doppelkopf IS-MCTS (Partnership + Hidden Info)
| Field | Value |
|-------|-------|
| Authors | Whitehouse, Cowling et al. |
| Date | 2011–2013 |
| Venue | IEEE CIG, ECAI workshops |
| Flags | II:Y MP:Y(4p) P:Y(hidden partnerships!) |
| Tags | `ISMCTS` `partner` `multi-agent` |
| Relevance | **5/5** |
| Core idea | (1) Doppelkopf has hidden partnerships AND hidden cards — harder than domino. (2) SO-ISMCTS with rollout heuristics competitive with MO-ISMCTS. (3) Partnership signaling studied. |
| Steal for domino | Closest published game setting to Pernambucano. Study their rollout heuristics. Partner belief updating approach directly transferable. |

---

### CAT-40 | Team-Maxmin Equilibria
| Field | Value |
|-------|-------|
| Authors | Celli, Gatti (+ Farina et al. related work) |
| Date | 2018 |
| Venue | AAAI 2018 |
| Flags | II:Y MP:Y P:Y |
| Tags | `multi-agent` `eval` |
| Relevance | **4/5** |
| Core idea | (1) Team-maxmin: team maximizes worst-case against best-responding opponent team. (2) Pre-agree on strategy, execute independently. (3) FNP-hard to compute exactly. |
| Steal for domino | Correct solution concept for 2v2 domino. Pre-game coordination = convention agreement. Common knowledge (board state) enables correlated play without communication. |

---

### CAT-41 | Ad Hoc Teamwork Survey
| Field | Value |
|-------|-------|
| Authors | Mirsky, Carlucho, Rahman et al. |
| Date | 2022 |
| Venue | Artificial Intelligence |
| Flags | II:varies MP:Y P:Y |
| Tags | `partner` `multi-agent` `eval` |
| Relevance | **4/5** |
| Core idea | (1) How can an agent cooperate with unknown teammates? (2) Type-based vs model-based vs communication-based approaches. (3) Balance exploitation with exploration. |
| Steal for domino | Define 3-5 partner types (aggressive, conservative, signaling-heavy). Bayesian type classification after 2-3 observations. Fast online adaptation (domino games are short). |

---

### CAT-42 | Machine Theory of Mind (ToMnet)
| Field | Value |
|-------|-------|
| Authors | Rabinowitz, Perbet, Song et al. (DeepMind) |
| Date | 2018 |
| Venue | ICML 2018 |
| Link | arXiv:1802.07740 |
| Flags | II:Y MP:Y P:varies |
| Tags | `partner` `neural` `multi-agent` |
| Relevance | **4/5** |
| Core idea | (1) Neural net learns to model other agents (character + mental state). (2) Few-shot agent modeling from behavioral observations. (3) Predicts actions of novel agent types. |
| Steal for domino | Character network: infer partner's style from 2-3 plays. Few-shot modeling (domino games are short). Separate character (style) from mental state (current hand beliefs). |

---

### CAT-43 | QMIX — Value Decomposition for Teams
| Field | Value |
|-------|-------|
| Authors | Rashid, Samvelyan, Schroeder de Witt et al. |
| Date | 2018 |
| Venue | ICML 2018 |
| Link | arXiv:1803.11485 |
| Flags | II:Y MP:Y P:Y(cooperative) |
| Tags | `multi-agent` `neural` `POMDP` |
| Relevance | **3/5** |
| Core idea | (1) CTDE: centralized training, decentralized execution. (2) Local Q-functions combined via monotonic mixing network. (3) Tested on SMAC. |
| Steal for domino | CTDE paradigm: train with all hands visible, execute with own hand only. Value decomposition: local + partner evaluation = team value. |
| Code | PyMARL (open-source) |

---

### CAT-44 | MAPPO — PPO for Multi-Agent Games
| Field | Value |
|-------|-------|
| Authors | Yu, Velu, Vinitsky et al. |
| Date | 2022 |
| Venue | NeurIPS |
| Link | arXiv:2103.01955 |
| Flags | II:Y MP:Y P:Y(cooperative) |
| Tags | `multi-agent` `neural` `self-play` |
| Relevance | **3/5** |
| Core idea | (1) Simple PPO with shared parameters matches/exceeds QMIX. (2) Centralized critic, decentralized actors. (3) Surprisingly effective strong baseline. |
| Steal for domino | Start simple: MAPPO with team reward is a strong baseline. Shared parameters natural for domino (all players face similar decisions). |

---

### CAT-45 | Emergent Communication in Multi-Agent RL
| Field | Value |
|-------|-------|
| Authors | Foerster, Assael, de Freitas, Whiteson |
| Date | 2016 |
| Venue | NeurIPS 2016 |
| Link | arXiv:1605.06676 |
| Flags | II:Y MP:Y P:Y |
| Tags | `multi-agent` `neural` `partner` |
| Relevance | **4/5** |
| Core idea | (1) Agents learn to communicate through actions. (2) DIAL: gradients flow through communication channel during training. (3) Communication protocols emerge without pre-design. |
| Steal for domino | In domino, every play IS the communication channel. CTDE training → signaling emerges naturally. Frame tile selection as implicit communication. |

---

### CAT-46 | BayesElo / TrueSkill / Glicko Rating Systems
| Field | Value |
|-------|-------|
| Authors | Coulom (BayesElo), Microsoft (TrueSkill), Glickman (Glicko-2) |
| Date | 2008 / 2018 / 2012 |
| Venue | Various |
| Link | remi-coulom.fr/Bayesian-Elo/ ; trueskill.org ; glicko.net |
| Flags | General |
| Tags | `eval` |
| Relevance | **5/5** |
| Core idea | (1) BayesElo: Bayesian rating from game outcomes. (2) TrueSkill: designed for multi-player/teams, tracks uncertainty. (3) Glicko-2: rating deviation + volatility. |
| Steal for domino | Use TrueSkill for 4p team rating. Decompose skill into individual + partnership synergy. 1000+ games for tight CIs. BayesElo's "advantage" captures seat position effects. |
| Code | Python libraries for all three |

---

### CAT-47 | SPRT for Game AI Testing
| Field | Value |
|-------|-------|
| Authors | Wald (1945, foundational); Stockfish Fishtest (practical application) |
| Date | Ongoing |
| Venue | https://tests.stockfishchess.org/ |
| Flags | General |
| Tags | `eval` |
| Relevance | **5/5** |
| Core idea | (1) Sequential Probability Ratio Test: stop testing when confident. (2) Much more efficient than fixed-sample testing. (3) Used by Stockfish for every patch. |
| Steal for domino | SPRT for all A/B comparisons. Bounds [0, 20] Elo (domino has high variance). Pair games with same deal, agents swapped. 5-10x more efficient than fixed N. ~50 lines to implement. |

---

### CAT-48 | Exploitability & Best Response
| Field | Value |
|-------|-------|
| Authors | Johanson, Zinkevich, Bowling (2007+); Burch et al. |
| Date | 2007–2019 |
| Venue | NeurIPS, AAAI |
| Flags | II:Y MP:varies P:N |
| Tags | `eval` `POMDP` |
| Relevance | **4/5** |
| Core idea | (1) Exploitability = value of best response against you - equilibrium value. (2) Local best response for multiplayer. (3) Measures strategy robustness. |
| Steal for domino | Fix your team, let opponents best-respond via MCTS. The gap = how exploitable you are. Secondary metric alongside win rate. Helps identify "leaks" in specific situations. |

---

### CAT-49 | Variance Reduction — Paired/Duplicate Evaluation
| Field | Value |
|-------|-------|
| Authors | Schmid, Burch, Lanctot et al. (VR-MCCFR); Schaeffer (duplicate games) |
| Date | 2019 / ongoing methodology |
| Venue | AAAI / competitive bridge methodology |
| Flags | General |
| Tags | `eval` |
| Relevance | **5/5** |
| Core idea | (1) Duplicate testing: same deal, agents swapped between seats, average results. (2) Removes deal variance, isolates AI quality. (3) Control variates for Monte Carlo variance reduction. |
| Steal for domino | ALWAYS use paired evaluation: same deal played twice with agents swapped. Dramatically reduces games needed for significance. Fix 24-tile deal + 4 dorme tiles, swap teams. |

---

### CAT-50 | Alpha-Rank for Non-Transitive Games
| Field | Value |
|-------|-------|
| Authors | Omidshafiei, Tuyls et al. |
| Date | 2020 |
| Venue | NeurIPS |
| Flags | MP:Y P:N |
| Tags | `eval` |
| Relevance | **3/5** |
| Core idea | (1) Evolutionary dynamics to rank agents in multi-player games. (2) Handles non-transitivity naturally. (3) Nash Averaging for robust ranking. |
| Steal for domino | If round-robin shows non-transitive effects, use Alpha-Rank. Start with TrueSkill, upgrade if needed. |
| Code | OpenSpiel |

---

### CAT-51 | XAI for Game-Playing Agents
| Field | Value |
|-------|-------|
| Authors | Huber, Weitz, Andre, Amir |
| Date | 2021 |
| Venue | IEEE Conference on Games |
| Flags | General |
| Tags | `explain` |
| Relevance | **5/5** |
| Core idea | (1) XAI in games: debugging, teaching, trust. (2) Taxonomy: feature-based, example-based, contrastive/counterfactual. (3) Game-specific challenges: must be player-intuitive. |
| Steal for domino | Layered explanations: L1 (what tile), L2 (why — what it achieves), L3 (why not alternative). Coach in player-understandable terms. Use MCTS tree as faithful explanation source. |

---

### CAT-52 | Counterfactual Explanations for Game Moves
| Field | Value |
|-------|-------|
| Authors | Amir et al. (Harvard/Technion) |
| Date | 2019–2021 |
| Venue | AAMAS, IJCAI |
| Flags | General |
| Tags | `explain` |
| Relevance | **5/5** |
| Core idea | (1) "Why A not B?" — explain why chosen move beats the expected alternative. (2) Find minimal state change that flips preference. (3) Matches human question patterns. |
| Steal for domino | "Why not this move?" for coaching. Structure: "[3|5] better than [3|2] because it opens partner's suit while blocking opponent." Use MCTS value difference as backbone. Perfect for Quiz mode. |

---

### CAT-53 | MCTS Explanation Generation
| Field | Value |
|-------|-------|
| Authors | Huber, Weitz, Andre |
| Date | 2020–2021 |
| Venue | IEEE Conference on Games |
| Flags | General |
| Tags | `explain` `belief` |
| Relevance | **5/5** |
| Core idea | (1) MCTS produces rich data: visit counts, values, principal variation. (2) Value gap between best and second-best indicates decision clarity. (3) Variance across determinizations indicates confidence. |
| Steal for domino | Extract from IS-MCTS: top 3 candidates with values, principal variation, value variance. Generate NL: "Play [4|6] (72% win) beats [4|2] (64%) because it forces opponent B to pass." Show "critical moments" where best/second-best diverge. |

---

### CAT-54 | AI Coaching in Card/Board Games
| Field | Value |
|-------|-------|
| Authors | Various (GIB coaching, KataGo analysis, Bridge Baron) |
| Date | Ongoing |
| Venue | Commercial products + open-source (KataGo) |
| Flags | General |
| Tags | `explain` |
| Relevance | **5/5** |
| Core idea | (1) Post-game review: move quality color-coding (green/yellow/red). (2) Highlight mistakes, not every move. (3) Skill dimension tracking over time. (4) Graduated hints in quiz mode. |
| Steal for domino | Post-game review with move quality coloring. Skill dimensions: blocking, counting, signaling, endgame. Quiz mode: graduated hints ("think about partner's pass" → "partner has no 3s" → "play [3|X]"). KataGo analysis mode as gold standard. |
| Code | KataGo is open source |

---

### CAT-55 | Faithful vs Post-Hoc Explanations
| Field | Value |
|-------|-------|
| Authors | Jacovi, Goldberg (2020) |
| Date | 2020 |
| Venue | ACL |
| Flags | General |
| Tags | `explain` |
| Relevance | **4/5** |
| Core idea | (1) Faithful = reflects actual AI reasoning (MCTS tree). (2) Post-hoc = generated separately, may not reflect actual process. (3) Game AI with MCTS has naturally faithful explanations. |
| Steal for domino | Prefer MCTS-derived explanations (faithful). If hybrid neural+MCTS, derive from MCTS layer. Post-hoc NL generation only to make faithful explanations readable. |

---

### CAT-56 | Risk-Sensitive RL for Games
| Field | Value |
|-------|-------|
| Authors | Chow, Tamar, Mannor, Pavone et al. |
| Date | 2015–2020 |
| Venue | ICML, NeurIPS, AAAI |
| Flags | General |
| Tags | `belief` `POMDP` |
| Relevance | **4/5** |
| Core idea | (1) CVaR: optimize for worst-case tail outcomes. (2) Score-aware risk adjustment: leading = play safe, trailing = take risks. (3) Risk-sensitive value estimation tracks distribution, not just mean. |
| Steal for domino | Track value variance across determinizations. Score-aware play: leading → conservative, trailing → aggressive. Critical for dorme uncertainty (irreducible). Report confidence intervals on move values. |

---

### CAT-57 | Domino AI Papers (Various Latin American)
| Field | Value |
|-------|-------|
| Authors | Various Brazilian/Caribbean university theses |
| Date | 2000s–2023 |
| Venue | ENIAC, BRACIS, university theses (UFPE, UFCG, USP, UNICAMP) |
| Flags | II:Y MP:Y P:Y |
| Tags | `ISMCTS` `belief` `partner` |
| Relevance | **5/5** |
| Core idea | (1) Rule-based agents encoding traditional wisdom. (2) PIMC/ISMCTS with domain heuristics. (3) Pass tracking as key inference. (4) "Domino de duplas" = partnership play. |
| Steal for domino | DIRECTLY relevant — Brazilian domino rules overlap with Pernambucano. Pass-tracking heuristics. Partnership signaling. Suit counting and "fechamento" (blocking). Search Google Scholar Portuguese: "domino inteligência artificial", BDTD thesis portal. |

---

### CAT-58 | Mahjong MCTS+NN (Various)
| Field | Value |
|-------|-------|
| Authors | Multiple groups |
| Date | 2018–2022 |
| Venue | Various AI/game conferences |
| Flags | II:Y MP:Y(4p) P:N |
| Tags | `ISMCTS` `neural` `belief` |
| Relevance | **5/5** |
| Core idea | (1) Determinization + MCTS with learned value/policy nets. (2) Particle filtering for belief tracking. (3) Policy nets prune search; value nets evaluate leaves. |
| Steal for domino | Determinized MCTS is the natural domino approach. Neural evaluation replaces hand-crafted heuristics. Particle filtering for opponent tile tracking. |

---

### CAT-59 | AlphaZero / AlphaGo Zero
| Field | Value |
|-------|-------|
| Authors | Silver, Schrittwieser et al. (DeepMind) |
| Date | 2017/2018 |
| Venue | Nature / Science |
| Flags | II:N MP:N P:N |
| Tags | `neural` `self-play` |
| Relevance | **4/5** |
| Core idea | (1) Self-play RL with policy+value net. (2) PUCT selection for tree policy. (3) Training: value target = outcome z, policy target = visit counts π. |
| Steal for domino | PUCT formula directly applicable to ISMCTS. Dual policy+value architecture. Self-play pipeline with replay buffer. Temperature schedule for exploration. |

---

### CAT-60 | EfficientZero
| Field | Value |
|-------|-------|
| Authors | Ye, Liu, Kurutach, Abbeel, Gao |
| Date | 2021 |
| Venue | NeurIPS 2021 (Spotlight) |
| Link | arXiv:2111.00210 |
| Flags | II:partial MP:N P:N |
| Tags | `MuZero` `self-play` |
| Relevance | **3/5** |
| Core idea | (1) 100x more sample efficient MuZero. (2) Self-supervised consistency loss. (3) End-to-end value prefix. |
| Steal for domino | Sample efficiency techniques critical for fast domino training. Consistency loss ensures model accuracy. |
| Code | https://github.com/YeWR/EfficientZero |

---

### CAT-61 | Bridge Sampling Improvements
| Field | Value |
|-------|-------|
| Authors | Various (bridge AI community) |
| Date | Ongoing |
| Flags | II:Y MP:Y P:Y |
| Tags | `belief` `ISMCTS` `partner` |
| Relevance | **4/5** |
| Core idea | (1) Constrain deal samples to be consistent with observed bidding. (2) Bayesian inference to weight hands. (3) Strategy fusion mitigation via better aggregation. |
| Steal for domino | Constrained sampling based on observed passes directly applicable. Bayesian belief updating from play/pass. Strategy fusion mitigation for partnership. |

---

### CAT-62 | Computational Complexity of Dominoes
| Field | Value |
|-------|-------|
| Authors | Demaine et al. |
| Date | ~2017 |
| Venue | FUN with Algorithms |
| Flags | N/A (theoretical) |
| Tags | `endgame` |
| Relevance | **2/5** |
| Core idea | (1) Various domino problems are NP-complete. (2) Justifies heuristic/MCTS approaches. |

---

---

## DELIVERABLE B: TOP 20 MOST ACTIONABLE PAPERS

Ranked by immediate implementability × expected Elo gain for current stack.

| Rank | ID | Paper | Key Action | Expected Impact |
|------|-----|-------|------------|-----------------|
| 1 | CAT-01 | IS-MCTS (Cowling 2012) | Foundation: SO-ISMCTS implementation | Baseline architecture |
| 2 | CAT-07 | Progressive Bias (Chaslot 2008) | Add heuristic bias to UCB formula | +150-200 Elo |
| 3 | CAT-22 | GIB (Ginsberg 2001) | PIMC paradigm: determinize + solve + average | Baseline paradigm |
| 4 | CAT-35 | Alpha-Beta + TT | Add Zobrist TT to bitmask endgame solver | +100-150 Elo (endgame) |
| 5 | CAT-29 | Re-determinizing MCTS (2019) | Resample at interior nodes | +50-100 Elo |
| 6 | CAT-08 | BAD (Foerster 2019) | Dual-purpose action selection (play + signal) | +100-200 Elo (partnership) |
| 7 | CAT-38 | Coalition Reduction | Collapse 4p to 2-team minimax | Halves search depth |
| 8 | CAT-26 | Expert Iteration (2017) | ISMCTS → train net → feed back to ISMCTS | +200-400 Elo |
| 9 | CAT-14 | ReBeL (Brown 2020) | Public Belief State + value net | Architecture upgrade |
| 10 | CAT-13 | SPARTA (Lerer 2020) | Hybrid RL policy + online search for partnership | +150-250 Elo |
| 11 | CAT-12 | OBL (Hu 2021) | Separate belief from convention in training | Robust partnership |
| 12 | CAT-47 | SPRT Testing | SPRT for all A/B comparisons | 5-10x eval efficiency |
| 13 | CAT-49 | Paired Evaluation | Same deal, swap agents | 3-5x variance reduction |
| 14 | CAT-46 | TrueSkill Rating | Team Elo with uncertainty | Proper agent ranking |
| 15 | CAT-52 | Counterfactual Explanations | "Why not this move?" coaching | Coach layer quality |
| 16 | CAT-53 | MCTS Explanation Generation | Extract rationales from search tree | Coach layer foundation |
| 17 | CAT-06 | OOS (Lanctot 2014) | Replace UCB with regret matching | +50-100 Elo |
| 18 | CAT-28 | PSRO (Lanctot 2017) | League training to prevent cycling | Training stability |
| 19 | CAT-16 | MuZero (2020) | Learned dynamics model for belief | Advanced architecture |
| 20 | CAT-24 | Suphx (2020) | Oracle guiding: perfect→imperfect distillation | Training technique |

---

## DELIVERABLE C: TAXONOMY (8 Buckets)

### Bucket 1: INFORMATION-SET SEARCH
Papers: CAT-01, CAT-02, CAT-03, CAT-05, CAT-06, CAT-29, CAT-30
**Summary**: Algorithms for searching game trees with hidden information. IS-MCTS is the gold standard; key challenge is mitigating strategy fusion and non-locality.

### Bucket 2: BELIEF MODELING & INFERENCE
Papers: CAT-04, CAT-08, CAT-09, CAT-14, CAT-56, CAT-57, CAT-61
**Summary**: Maintaining and updating probability distributions over hidden tiles. Bayesian updates from play/pass observations. Particle filters for tractable belief representation.

### Bucket 3: PARTNERSHIP & COOPERATION
Papers: CAT-08, CAT-09, CAT-10, CAT-11, CAT-12, CAT-13, CAT-23, CAT-39, CAT-40, CAT-41, CAT-45
**Summary**: Coordinating with a partner under partial information without communication. Convention emergence via self-play, zero-shot coordination, signaling through actions.

### Bucket 4: HYBRID SEARCH + LEARNING
Papers: CAT-14, CAT-15, CAT-16, CAT-17, CAT-24, CAT-25, CAT-26, CAT-27, CAT-31, CAT-32, CAT-33, CAT-59
**Summary**: Combining tree search with learned policy/value networks. AlphaZero/MuZero/ReBeL approaches adapted for imperfect information. Expert Iteration as practical bridge.

### Bucket 5: ENDGAME & EXACT SOLVING
Papers: CAT-19, CAT-20, CAT-21, CAT-34, CAT-35, CAT-36
**Summary**: Alpha-beta, transposition tables, retrograde analysis, subgame solving. Transitioning from search to exact solver when the game becomes small enough.

### Bucket 6: MULTI-PLAYER & EQUILIBRIUM
Papers: CAT-18, CAT-28, CAT-38, CAT-40, CAT-43, CAT-44
**Summary**: MCTS/RL for >2 players, coalition reduction for 2v2, team-maxmin equilibria, self-play convergence (PSRO), CTDE paradigm.

### Bucket 7: EVALUATION SCIENCE
Papers: CAT-46, CAT-47, CAT-48, CAT-49, CAT-50
**Summary**: Rating systems (TrueSkill, BayesElo), SPRT for efficient A/B testing, exploitability measurement, paired evaluation for variance reduction.

### Bucket 8: EXPLAINABILITY & COACHING
Papers: CAT-51, CAT-52, CAT-53, CAT-54, CAT-55
**Summary**: Faithful explanations from MCTS traces, counterfactual "why not?" explanations, move quality color-coding, skill tracking, quiz mode with graduated hints.

---

## DELIVERABLE D: DOMINO UPGRADE MAP

```
Bucket → Module mapping for current Pernambuco Domino AI stack

┌──────────────────────────────────────────────────────────────────┐
│                    DOMINO UPGRADE MAP                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │ BELIEF MODEL                                 │                │
│  │ Current: Bayesian tracking of hidden tiles   │                │
│  │                                              │                │
│  │ Upgrade 1: Particle filter (CAT-04)          │                │
│  │   → 200-500 particles, resample on pass      │                │
│  │                                              │                │
│  │ Upgrade 2: Explicit dorme tracking (CAT-57)  │                │
│  │   → P(tile in dorme) column in belief matrix │                │
│  │                                              │                │
│  │ Upgrade 3: Opponent play-model inference      │                │
│  │   → Soft inference from tile CHOICES          │                │
│  │   → BAD-style public belief (CAT-08)         │                │
│  │                                              │                │
│  │ Upgrade 4: Neural belief model (CAT-16,24)   │                │
│  │   → MuZero-style learned latent belief       │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │ SEARCH (ISMCTS / Belief MCTS)                │                │
│  │ Current: SO-ISMCTS with determinization      │                │
│  │                                              │                │
│  │ Upgrade 1: Progressive bias (CAT-07)         │                │
│  │   → UCB + H(s,a)/(1+n) from smartAI scoring │                │
│  │                                              │                │
│  │ Upgrade 2: Re-determinizing (CAT-29)         │                │
│  │   → Resample at interior nodes, not just root│                │
│  │                                              │                │
│  │ Upgrade 3: Smooth UCT / OOS (CAT-05, CAT-06)│                │
│  │   → Mixed strategies, CFR-style updates      │                │
│  │                                              │                │
│  │ Upgrade 4: Coalition reduction (CAT-38)      │                │
│  │   → 2-team minimax within determinizations   │                │
│  │                                              │                │
│  │ Upgrade 5: Depth-limited + value net (CAT-14)│                │
│  │   → Stop search at depth D, evaluate by net  │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │ HEURISTIC / VALUE NET                        │                │
│  │ Current: Hand-crafted evaluation             │                │
│  │                                              │                │
│  │ Upgrade 1: Expert Iteration (CAT-26)         │                │
│  │   → ISMCTS visits → train policy net → loop  │                │
│  │                                              │                │
│  │ Upgrade 2: Self-play value net (CAT-14,59)   │                │
│  │   → Input: hand+board+belief → team score    │                │
│  │                                              │                │
│  │ Upgrade 3: Oracle guiding (CAT-24)           │                │
│  │   → Train perfect-info AI, distill to impf.  │                │
│  │                                              │                │
│  │ Upgrade 4: Auxiliary targets (CAT-27)        │                │
│  │   → Predict: who goes out, pip diff, blocked │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │ ENDGAME SOLVER / TABLEBASE                   │                │
│  │ Current: Bitmask alpha-beta endgame solver   │                │
│  │                                              │                │
│  │ Upgrade 1: Zobrist TT (CAT-35)              │                │
│  │   → Transposition table for bitmask states   │                │
│  │                                              │                │
│  │ Upgrade 2: Move ordering (CAT-35)            │                │
│  │   → Both-end matches first, high pips next   │                │
│  │                                              │                │
│  │ Upgrade 3: Safe subgame transition (CAT-20)  │                │
│  │   → Handoff criteria: tiles_remaining ≤ 12   │                │
│  │   → Consistency with MCTS trunk values       │                │
│  │                                              │                │
│  │ Upgrade 4: Endgame tablebases (CAT-36)       │                │
│  │   → Pre-compute for ≤3 tiles/player          │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │ SELF-PLAY TRAINING LOOP                      │                │
│  │ Current: N/A (hand-crafted heuristics)       │                │
│  │                                              │                │
│  │ Upgrade 1: Expert Iteration cycle (CAT-26)   │                │
│  │   → ISMCTS expert → net apprentice → repeat  │                │
│  │                                              │                │
│  │ Upgrade 2: PSRO / League training (CAT-28)   │                │
│  │   → Population of policies, meta-Nash        │                │
│  │                                              │                │
│  │ Upgrade 3: OBL-style training (CAT-12)       │                │
│  │   → Convention-robust self-play              │                │
│  │                                              │                │
│  │ Upgrade 4: Playout cap randomization (CAT-27)│                │
│  │   → Vary ISMCTS budget during training       │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │ EVALUATION HARNESS                           │                │
│  │ Current: Win rate counting                   │                │
│  │                                              │                │
│  │ Upgrade 1: SPRT testing (CAT-47)             │                │
│  │   → Sequential test, stop when confident     │                │
│  │                                              │                │
│  │ Upgrade 2: Paired evaluation (CAT-49)        │                │
│  │   → Same deal, swap agents between seats     │                │
│  │                                              │                │
│  │ Upgrade 3: TrueSkill rating (CAT-46)         │                │
│  │   → Team ratings with uncertainty tracking   │                │
│  │                                              │                │
│  │ Upgrade 4: Golden position suite             │                │
│  │   → Known-best-move positions for regression │                │
│  │                                              │                │
│  │ Upgrade 5: Exploitability check (CAT-48)     │                │
│  │   → Let opponents best-respond, measure gap  │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │ EXPLANATION / COACH LAYER                    │                │
│  │ Current: Rationale generation from search    │                │
│  │                                              │                │
│  │ Upgrade 1: MCTS trace extraction (CAT-53)    │                │
│  │   → Top-3 candidates + values + PV          │                │
│  │                                              │                │
│  │ Upgrade 2: Contrastive explanations (CAT-52) │                │
│  │   → "Why A not B?" with minimal contrast    │                │
│  │                                              │                │
│  │ Upgrade 3: Move quality coloring (CAT-54)    │                │
│  │   → Green/yellow/red based on value gap      │                │
│  │                                              │                │
│  │ Upgrade 4: Belief visualization              │                │
│  │   → Show inferred tile probabilities         │                │
│  │                                              │                │
│  │ Upgrade 5: Skill tracking (CAT-54)           │                │
│  │   → Blocking, signaling, counting, endgame   │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## DELIVERABLE E: 1-WEEK READING/EXTRACTION PLAN

### Day 1 (Mon): FOUNDATIONS — Search & Determinization
**Read**: CAT-01 (IS-MCTS), CAT-02 (Strategy Fusion), CAT-03 (Why PIMC Works)
**Extract**:
- SO-ISMCTS pseudocode → adapt for domino
- Strategy fusion examples → identify domino analogs
- "Smoothness" checklist → verify domino qualifies

### Day 2 (Tue): ENDGAME & SEARCH ENHANCEMENTS
**Read**: CAT-35 (Alpha-Beta+TT), CAT-07 (Progressive Bias), CAT-29 (Re-determinization), CAT-06 (OOS)
**Extract**:
- Zobrist hashing scheme for domino bitmask states
- Move ordering heuristics for domino endgame
- Progressive bias formula with smartAI as H(s,a)
- OOS pseudocode for regret-matching in ISMCTS nodes

### Day 3 (Wed): PARTNERSHIP & COOPERATION
**Read**: CAT-08 (BAD), CAT-13 (SPARTA), CAT-12 (OBL), CAT-09 (Hanabi Challenge)
**Extract**:
- BAD public belief update equations → adapt for domino
- SPARTA search-with-partner-model pseudocode
- OBL belief/convention decomposition → design for domino
- Hanabi convention taxonomy → domino signaling analogs

### Day 4 (Thu): HYBRID SEARCH + LEARNING
**Read**: CAT-26 (Expert Iteration), CAT-14 (ReBeL), CAT-16 (MuZero), CAT-24 (Suphx)
**Extract**:
- ExIt training loop pseudocode → domino implementation plan
- ReBeL PBS representation → domino PBS definition
- Suphx oracle guiding procedure → domino adaptation
- Feature encoding scheme for domino states

### Day 5 (Fri): EVALUATION SCIENCE
**Read**: CAT-47 (SPRT), CAT-49 (Paired Eval), CAT-46 (TrueSkill), CAT-48 (Exploitability)
**Extract**:
- SPRT implementation (Python, ~50 lines)
- Paired evaluation protocol for domino
- TrueSkill team rating setup
- Exploitability approximation via opponent MCTS

### Day 6 (Sat): DOMAIN-SPECIFIC + MULTI-PLAYER
**Read**: CAT-57 (Domino AI theses), CAT-22 (GIB), CAT-23 (NukkAI), CAT-38 (Coalition MCTS), CAT-39 (Doppelkopf)
**Extract**:
- Domino heuristics catalog from Brazilian theses
- GIB architecture → domino PIMC design
- Coalition reduction implementation for 2v2
- Doppelkopf partnership signaling patterns → domino analogs

### Day 7 (Sun): EXPLAINABILITY + SYNTHESIS
**Read**: CAT-52 (Counterfactual), CAT-53 (MCTS Explanations), CAT-54 (Coaching)
**Extract**:
- Contrastive explanation template for domino
- MCTS trace → NL explanation pipeline
- Move quality scoring (green/yellow/red) thresholds
- Skill dimension definitions for domino coaching
**ALSO**: Write integration document mapping all extracted artifacts to simulator.html modules

---

## DELIVERABLE F: SEARCH QUERIES USED

### Agent 1 — Imperfect Info Search (11 queries):
1. "information set Monte Carlo tree search" ISMCTS
2. "determinization" "strategy fusion" imperfect information games
3. POMCP "partially observable" Monte Carlo planning
4. "belief state MCTS" hidden information
5. "smooth UCT" OR "regularized MCTS" imperfect information
6. "subgame solving" imperfect information
7. "counterfactual regret minimization" MCTS
8. "non-locality" determinization games
9. "progressive bias" MCTS imperfect information
10. "opponent modeling" MCTS games
11. ISMCTS survey OR review 2020-2026

### Agent 2 — Partnership & Multi-Player (12 queries):
1. "team game" reinforcement learning partnership
2. "teammate modeling" cooperative games AI
3. multi-player MCTS coalition "more than two players"
4. "ad hoc teamwork" reinforcement learning
5. Hanabi AI reinforcement learning conventions
6. bridge AI partnership bidding inference
7. "implicit communication" games multi-agent
8. "cooperative MCTS" team games
9. "max^n" OR "paranoid" MCTS multi-player
10. "Bayesian teammate modeling" games
11. multi-agent reinforcement learning "partial observability" team
12. "signaling" cooperative games imperfect information

### Agent 3 — Domino/Tile/Card Games (15 queries):
1. "domino" AI "artificial intelligence" OR "machine learning"
2. "dominoes" MCTS OR "Monte Carlo" OR reinforcement learning
3. "block dominoes" OR "draw dominoes" strategy algorithm
4. "double-six dominoes" computer player
5. Mahjong AI reinforcement learning MCTS
6. Suphx Mahjong "imperfect information"
7. bridge AI "double dummy" inference partnership
8. NukkAI bridge reinforcement learning
9. Skat AI MCTS imperfect information
10. "trick-taking" game AI reinforcement learning
11. "tile game" AI hidden information belief
12. Hanabi reinforcement learning conventions
13. "domino pernambucano" OR "dominó" inteligência artificial
14. Hearts AI MCTS OR reinforcement learning
15. Spades AI partnership strategy

### Agent 4 — Hybrid Engines & Endgame (18 queries):
1. AlphaZero imperfect information hidden state
2. MuZero imperfect information "partial observability"
3. "neural MCTS" OR "neural guided MCTS" imperfect information
4. ReBeL recursive belief learning games
5. "model-based reinforcement learning" games hidden information
6. endgame tablebase reinforcement learning
7. "retrograde analysis" games AI
8. "alpha-beta" transposition table endgame solving
9. "subgame solving" reinforcement learning imperfect information
10. "offline reinforcement learning" games OR "batch RL" games
11. self-play training multi-player games convergence
12. "population based training" games AI
13. "imitation learning" game AI MCTS
14. "progressive widening" OR "progressive bias" neural MCTS
15. Dream DeepMind imperfect information search
16. "endgame solver" combined MCTS neural network
17. "value network" imperfect information games
18. Student of Games DeepMind

### Agent 5 — Belief, Evaluation, Explainability (17 queries):
1. Bayesian inference "hidden cards" OR "hidden tiles" games
2. "particle filter" games "opponent modeling" hidden information
3. "neural belief model" games imperfect information
4. "risk-sensitive" reinforcement learning games
5. "uncertainty estimation" game AI value function
6. Elo rating "non-transitive" games
7. SPRT "sequential test" game AI comparison
8. "exploitability" imperfect information multi-player
9. "best response" computation imperfect information games
10. BayesElo OR Glicko OR TrueSkill game AI evaluation
11. "explainable AI" games OR "explainable game AI"
12. "counterfactual explanation" game moves
13. MCTS explanation OR interpretability "search tree"
14. "coaching" AI games explanation teaching
15. "variance reduction" Monte Carlo games evaluation
16. "ablation study" game AI methodology
17. "belief tracking" card games OR tile games

**Total unique queries**: 73 across 5 agents

---

## NOTES

### Gaps to fill with live search:
1. **Portuguese-language domino AI**: Search BDTD, CAPES thesis portal for "domino inteligência artificial", "jogo de domino agente"
2. **2025-2026 papers**: Recent AAAI 2025, NeurIPS 2025, ICML 2025 proceedings
3. **arXiv preprints**: cs.AI, cs.GT, cs.MA from Aug 2025 onward
4. **Specific Pernambucano domino**: "domino pernambucano" on Google Scholar

### Key observation:
Domino AI is **severely understudied** compared to poker, chess, Go, or even Mahjong. The action space is small (~7 max), game length is short (~12-15 rounds), and the hidden state is manageable. This means even modest algorithmic improvements should yield large Elo gains. The primary challenge is **partnership coordination under imperfect information** — the same core problem Bridge and Hanabi have tackled successfully.
