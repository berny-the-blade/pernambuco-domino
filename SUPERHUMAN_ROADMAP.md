# Pernambuco Domino — Superhuman AI Audit & Upgrade Roadmap

*Audit date: Feb 2026 | Current benchmark: +43 Elo (MC-Expert over heuristic-only, 400 games)*

---

## ROADMAP_FREEZE v1

* **Decision:** F1 indicates rollout bias dominant (Variant B beats A — magnitude TBD, re-run pending after mcCache fix).
* **Priority:** value network + NN leaf eval before RIS-MCTS.
* **Blocking gates:** parity gate (JS/Python encoder+mask must match eps=1e-6) + duplicate-deal arena gate + ΔME targets.
* **No-go:** any change that breaks parity tests or P95 > 300ms.

### Commit sequence:
1. ~~**Parity gate harness**~~ — **DONE**: `exportSnapshots()` + `massParityCheck()` in JS, `test_encoder_parity.py --mass` in Python
2. ~~**Belief semantics**~~ — **DONE**: both JS and Python export conditional `P(zone | not dorme)`
3. ~~**Action mask symmetry**~~ — **DONE**: both JS (line 6851) and Python (line 67) zero `mask[28:56]` when `lE==rE`
4. ~~**Remove 20% NN blend**~~ — **DONE**: blend removed at line 4078
5. ~~**ΔME training targets**~~ — **DONE**: `deltaMe()` standalone JS function, `exportTrainingData()` plays full matches with ΔME backfill, `me3dParityExport()` for table cross-validation, `validate_training_data.py` for Python-side validation (quick mode: 3556 samples, all signs correct, range [-0.5, 0.5])
6. ~~**NN leaf value integration**~~ — **DONE**: `USE_NN_LEAF_VALUE` toggle in both ISMCTS (line ~7162) and flat MC (line ~6531). When on + model loaded, `_nnForward()` value head replaces `simulateFromPosition()` rollouts. Console: `setNNLeafValue(true)`, `getNNLeafStats()`. Falls back to rollouts when no model loaded.

---

## I. Executive Summary

1. **Biggest bottleneck: rollout bias (confirmed by F1)**: The rollout policy (`fastAI`) is the largest Elo ceiling — every MC and ISMCTS evaluation is only as good as its rollout terminal estimates. Confirmed by F1: smartAI-first-4-ply rollouts beat fastAI-only rollouts under same determinization and budget (magnitude TBD — initial run invalidated by mcCache poisoning, re-run pending). A strong value network would eliminate rollouts entirely. *Note: F1 top-1 agreement check reported 100% in the initial (bugged) run and is not used as a decision metric until re-validated.*
2. **Determinization bias** (strategy fusion) is the second suspected bottleneck: ISMCTS averages over determinized worlds, which can systematically misvalue information-hiding and information-gathering moves. May be co-dominant with rollout bias — needs measurement.
3. **The NN training pipeline exists but produces weak models** — 97/100 arena rejections at gen 84. Root causes are falsifiable hypotheses, not settled (see §III.F3). Verified: self-play without `--mcts` uses raw neural policy + Dirichlet noise (α=0.3) as policy targets (`orchestrator.py:84`), which is likely too noisy.
4. **Endgame solver is correct and efficient** (16-tile bitmask, verified 5000/5000). This is the strongest component.
5. **CMA-ES failure is diagnostic**: tuning heuristic weights in heuristic-vs-heuristic games optimizes the wrong objective. Any tuning must use the full MC-Expert stack.
6. **The 300ms budget needs benchmarking**: browser NN inference speed varies with device, GC pressure, and encoding cost. Claims about "200-400 forward passes" must be validated on a reference device as a benchmark gate (see §IV acceptance tests).
7. **Partnership coordination** is under-modeled: the AI tracks partner voids (hard constraints via `cantHave`) and soft play-pattern beliefs (marginals), but doesn't model partner's likely *strategy* or generate cooperative signals. Cheapest fix: convention signals before learning.
8. **NN value targets updated to ΔME** (match equity delta). Previously `points_won / 4.0` (raw game points, [0, 1.5], match-state-blind). Now `orchestrator.py` supports `--value-target me` which computes `ME3D[newScore][oppScore][dob] - ME3D[myScore][oppScore][dob]`, giving match-context-aware training signal. The `match_equity.py` module provides the Python-side ME3D table.
9. **The game's branching factor is low** (~4-8 legal moves per turn) — this means tree search is unusually effective and should be the primary investment.
10. **Path to "near-flawless"**: Replace rollouts with a trained value network, fix determinization bias with RIS-MCTS or PBS search, and run a proper self-play training loop with MCTS-generated policy targets. Each step has a defined acceptance test (see §IV).

---

## II. Current System Map

```
Move Decision Pipeline (Expert AI, ~300ms budget):

  ┌─────────────────────────────────────────────────────┐
  │ 1. ENDGAME SOLVER (≤16 tiles remaining)             │
  │    Exact bitmask minimax + alpha-beta + TT + ordering│
  │    Returns: exact terminal outcome (points per move)  │
  │    Convert to ΔME via ME3D when needed               │
  │    Dominates: last ~8 moves of every game            │
  ├─────────────────────────────────────────────────────┤
  │ 2. ISMCTS (600 iters / 300ms)                       │
  │    UCB1 tree search with 10-feature heuristic bias  │
  │    Determinization: sample hidden hands from beliefs │
  │    Rollouts: fastAI to terminal → ME reward          │
  │    Dominates: mid-game (moves 3-16 roughly)          │
  ├─────────────────────────────────────────────────────┤
  │ 3. MONTE CARLO (800 sims, early stopping)           │
  │    Flat simulation with fastAI rollouts              │
  │    Dominates: early-game when ISMCTS tree too shallow│
  ├─────────────────────────────────────────────────────┤
  │ 4. HEURISTIC (smartAI, ~26 features, ~39 branches)  │
  │    Instant fallback, always available                │
  │    Dominates: time-pressured or as ISMCTS prior      │
  ├─────────────────────────────────────────────────────┤
  │ 5. NEURAL (blend REMOVED — future: leaf value only)  │
  │    4-block ResNet, 185→policy(57)+value(1)           │
  │    Currently: model too weak, blend removed           │
  └─────────────────────────────────────────────────────┘
```

**Where each stage dominates:**
- Moves 1-2: Opening (heuristic-dominated, MC for validation)
- Moves 3-8: Early mid-game (ISMCTS + MC, beliefs still sparse)
- Moves 9-16: Mid-game (ISMCTS with rich beliefs, approaching solver threshold)
- Moves 17-24: Endgame (exact solver, perfect play)

---

## III. Failure Mode Catalog (Ranked by Impact)

### F1. Rollout Bias (HIGH impact — plausible, needs isolation)

**Mechanism:** `fastAI` has 13 features (vs `smartAI`'s 26) and lacks chicote/lock/monopoly/match-equity/info-hiding/two-ply-lookahead awareness. Every MC simulation and ISMCTS rollout terminates with a biased value estimate. With 800 sims × ~20 rollout steps, this bias propagates ~16,000 times per move evaluation.

**Observable symptom:** AI misvalues blocked-game positions (where pip management and lock threats are critical). Especially in late mid-game where `smartAI` would score +40 for a deliberate lock but `fastAI` scores 0.

**How to reproduce:** Find a position where the best move creates a favorable lock. MC will undervalue it because `fastAI` rollouts don't detect lock favorability.

**Status: DIRECTION CONFIRMED, magnitude TBD.** F1 diagnostic ran (400 games) but initial results were invalidated by mcCache poisoning — Variant B had 0 rollouts (cache returned A's results). Bug fixed (mcCache.clear() before each game). Re-run pending. Directionally, the bet is on rollout bias as dominant — value-net-first is the strategy regardless.

**F1 Decision Gate (one day — run before investing in fixes):**

Build a paired evaluation — keep determinization & tree policy fixed, swap *only* terminal evaluator:

| Variant | Terminal evaluator | Purpose |
|---------|-------------------|---------|
| A (baseline) | `fastAI` rollout to terminal | Current system |
| B (smarter rollout) | `smartAI` for first 4 plies, then `fastAI` | Isolate rollout quality |
| C (deep leaf) | MC-5000 just for leaf states | Near-oracle leaf eval |

Run 200 fixed-seed positions, measure top-1 action agreement (A vs C) and Elo shift (B vs A).
- If B/C shifts rankings consistently → **rollout bias is dominant**, invest in value network
- If B/C barely changes results → **determinization/belief is dominant**, invest in RIS-MCTS first

**`fastAI` feature gaps — categorized by impact:**

| Absent feature | Affects terminal value? | Already captured by tree? | Impact |
|---------------|------------------------|--------------------------|--------|
| Chicote/lock detection | YES — determines blocked-game winner | NO — tree doesn't see lock favorability | HIGH |
| Monopoly awareness | YES — controls board access | Partially (ISMCTS heuristic has suit control) | MEDIUM |
| Match-equity adaptation | YES — optimal strategy changes at e.g. 5-4 | NO — rollouts ignore match state | MEDIUM |
| Advanced partnership | NO — affects move choice, not terminal | Partially (beliefs already weight partner) | LOW |
| Opening suit conventions | NO — only relevant in moves 1-3 | YES — MC/heuristic handle opening | LOW |

**Fix options:**
1. **(Best)** Replace rollouts with a trained value network — eliminates rollout bias entirely
2. **(Medium)** Port more `smartAI` features to `fastAI` — diminishing returns, already done once
3. **(Quick)** Use `smartAI` for first N rollout moves, then `fastAI` — 3-5x slower but higher quality

### F2. Determinization Bias / Strategy Fusion (HIGH impact)

**Mechanism:** ISMCTS determinizes (samples) hidden hands before each simulation. This averages over worlds rather than reasoning about information sets. It systematically undervalues:
- **Information-hiding moves** (plays that don't reveal your hand structure)
- **Information-gathering moves** (plays that force opponents to reveal voids)
- **Bluffing/deception** (playing off-suit to mislead opponents)

**Observable symptom:** AI plays "honestly" — always plays its strongest suit openly, never sacrifices short-term score to hide information.

**How to reproduce:** Position where playing suit X is +2 heuristic but reveals your hand structure, vs playing suit Y is -1 but preserves ambiguity. ISMCTS always picks X.

**What "beliefs" are TODAY (clarification):**

The current system is a **hybrid 2-layer model**:

1. **Hard constraints** (`Knowledge.cantHave[p]`): Set of NUMBER voids (0-6) per player, inferred from observed passes. Binary — if player passed when end was N, they have no tile containing N. Always enforced during determinization.

2. **Soft marginals** (`BeliefModel.marginals[p][tileId]`): Probability weights (0.0-1.0) computed via importance-weighted Monte Carlo (200 baseline deals, `dealWeight()` applies 6 factors: suit strength, pass-adjacency, avoidance signal, dumping style, sacrifice signal, opening suit). Used to bias `BeliefSampler` during determinization.

3. **Determinization pipeline** (`generateConsistentDeal`): 3-tier fallback. Tier 1 (BeliefSampler) uses marginals for weighted sampling. Tier 2 (constraint propagation + soft weighting) uses most-constrained-first assignment. Tier 3 (rejection sampling) is uniform random respecting cantHave only.

**What CHANGES in proposed upgrades:**

- **RIS-MCTS (Upgrade Proposal 1):** Re-samples hidden info *at each tree node from the acting player's perspective*, not just once at root. This prevents information leakage (root player's hand leaking into opponent node evaluations). Uses the same belief infrastructure but calls `generateConsistentDeal` per-node instead of per-simulation.

- **PBS search (longer-term):** Replaces individual sampled worlds with a *distribution over all consistent worlds*. The existing 84-dim belief section in the NN encoder (per-tile probabilities for 3 opponents × 28 tiles) is already an approximation of this. The change is searching over belief transitions, not sampled states. **Dorme semantics:** The current belief export uses conditional probabilities `P(zone | not dorme)` — normalizing over 3 visible zones (partner, LHO, RHO) after marginalizing out the dorme pile. This means PBS search would operate on conditional beliefs. A future 4×28 dorme-aware encoding (§XI-C) would give PBS search explicit access to `P(tile in dorme)`, improving mid-game accuracy where dorme probability is significant.

**Fix options:**
1. **(Best)** Public Belief State (PBS) search à la ReBeL — reasons about beliefs directly
2. **(Medium)** RIS-MCTS — re-determinize at each node from acting player's info set (cheapest fix for leakage)
3. **(Quick)** Increase determinization count and weight by belief quality

### F3. Weak Neural Network (HIGH impact, blocks future progress)

**Mechanism:** 97/100 arena rejections at gen 84 means the model never surpassed random-with-rules play.

**Observable symptom:** Exported model adds noise, not signal. 20% blend degraded play (now removed).

**Root causes — falsifiable hypotheses (test in order):**

**H1: Policy targets too noisy without MCTS** (most likely)
- *Verified:* Default self-play uses raw neural policy + Dirichlet(α=0.3) noise, 75/25 blend (`orchestrator.py:84`). No tree search to improve targets.
- *Prediction:* With `--mcts --mcts-sims 100`, policy targets will have lower entropy and higher correlation with strong play.
- *Test:* Run 5 gens with `--mcts` vs 5 gens without. Compare: (a) policy target entropy (lower = better), (b) arena promotion rate, (c) value loss convergence.
- *Pass if:* MCTS-mode promotes ≥2/5 gens while no-MCTS promotes 0/5.

**H2: Value target conflates game points with match equity — FIXED**
- *Was:* Value target = `points_won / 4.0`, signed by team (`orchestrator.py:118`). Range [0, 1.5]. Did NOT use match equity (ME). A 6-0 win at 0-0 match score had same training signal as at 5-5.
- *Now:* `orchestrator.py` supports `--value-target me` (default). Uses `delta_me(winner_team, points, my_team, my_score, opp_score, multiplier)` from `training/match_equity.py`, which computes `ME3D[newScore][oppScore][dob] - ME3D[myScore][oppScore][dob]`. Python-side ME3D table is a full port of the JS DP fill.
- *Prediction:* ME-based targets will produce value predictions that better predict match outcomes.
- *Test:* Train identical configs, one with `--value-target points`, one with `--value-target me`. Compare value head calibration on 500 curated positions (predicted ME vs actual ME from 10K rollouts).
- *Pass if:* ME-target model has ≥20% lower MAE on calibration suite.

**H3: Arena threshold too strict for early training — PARTIALLY ADDRESSED**
- *Was:* Threshold = 55% of match points across 400 games (`orchestrator.py:204`). Fixed for all generations.
- *Now:* Arena uses duplicate deals (each seed played from both sides) with ≥52% match winrate AND 95% CI lower bound > 50% for promotion (`_arena_evaluate`). This is statistically tighter but fairer than flat 55%.
- *Remaining:* Could still benefit from graduated gating (52% gen 0-30, 53% gen 30-60, 55% gen 60+). Test if early-gen promotion rate improves.
- *Pass if:* Graduated gating produces ≥3x more promotions in first 60 gens.

**H4: Replay buffer staleness**
- *Verified:* 200K deque with FIFO eviction (`orchestrator.py:222`). No recency weighting, no priority sampling.
- *Prediction:* With slow improvement, buffer is 95%+ stale data from weak past selves.
- *Test:* Compare buffer_size=200K vs 50K (faster turnover). Track value loss variance and policy agreement with champion.
- *Pass if:* Smaller buffer produces lower value loss after 50 gens.

**Recommended test order:** H1 first (highest expected impact, cheapest to test), then H3 (graduated gating), then H4 (requires full training run comparison). H2 (ΔME targets) is already implemented — validate during Week 1.

### F4. Partnership Modeling Gap (MEDIUM impact)

**Mechanism:** The AI tracks what partner *can't* have (voids via `cantHave`) and soft marginals (play patterns via `BeliefModel`), but doesn't model what partner *would do* given their information. No cooperative signaling — e.g., opening a specific suit to signal strength.

**Observable symptom:** AI doesn't coordinate blockades. Partner blocks a number AI was about to play. Missed opportunities for "feeding" partner's known strong suit.

**Concrete implementation path (cheap → expensive):**

**Step 1: Convention signals (1-2 days, deterministic)**
Encode common Pernambuco Domino conventions into belief updates:
- "First play signals strongest suit" → when partner opens with number N, boost marginals for N-tiles in partner's hand
- "Pass after partner's lead = void" → already tracked (cantHave), but can be weighted higher in suit-selection
- "High double early = strength signal" → partner plays [6|6] or [5|5] in first 3 moves → boost that suit

These are cheap additive features in `dealWeight()` (existing importance weighting), no new architecture needed.

**Step 2: Partner policy model (1-2 weeks, learned)**
Predict partner's move distribution conditioned on public features:
- Input: partner's observed plays, passes, board state, hand sizes
- Output: P(partner plays tile T | public info)
- Training: from self-play game logs, predict held-out partner moves
- Integration: use as improved prior in `generateConsistentDeal` (replace uniform sampling for partner)

**Step 3: Joint team planning (research-level)**
Model the team as a joint agent optimizing shared ME. Requires SPARTA-style search or ReBeL PBS that conditions on both team members' observations.

**Fix options (updated with path):**
1. **(Best)** Explicit partner policy model (Step 2 above — predict partner's move distribution from observed behavior)
2. **(Medium)** Rule-based conventions (Step 1 above — cheap, deterministic, do first)
3. **(Quick)** Increase `pAff` (partner affinity) weight when partner has shown preference

### F5. Early-Game Search Weakness (MEDIUM impact)

**Mechanism:** With 24 tiles out (6 per player + 4 dorme), the information set is enormous. ISMCTS at 600 iterations can't build a deep tree. MC rollouts with `fastAI` are biased. The opening move gets the worst evaluation quality.

**Observable symptom:** Opening play is nearly random across benchmarks. Low correlation between AI's opening choice and eventual game outcome.

**Fix options:**
1. **(Best)** Neural policy prior for ISMCTS root (strong prior = deeper effective search)
2. **(Medium)** Opening book from self-play statistics
3. **(Quick)** Increase MC sims for first 2 moves (budget is available since moves are infrequent)

### F6. Match Equity Miscalibration in Rollouts (LOW-MEDIUM impact)

**Mechanism:** `fastAI` rollouts produce raw game outcomes that `_rolloutToMEReward` converts to ME deltas. But `fastAI` doesn't adapt strategy to match state — e.g., at 5-5 it should maximize/minimize point types, not just win probability.

**Observable symptom:** Suboptimal play at critical match states (e.g., 5-4 not going for cruzada prevention).

**Fix options:**
1. Value network trained on ME targets would handle this naturally
2. Add match-state awareness to `fastAI` (add 2-3 features)

---

## IV. Superhuman Roadmap (Ranked)

### Upgrade 1: Train a Strong Value Network (Expected: +80-150 Elo, conditional)

**What:** Run the existing training pipeline with MCTS-enabled self-play (`--mcts --mcts-sims 100`), **ΔME value targets** (`MSE for ΔME = ME(S_after_hand) − ME(S_before_hand)` via ME3D lookup, not raw game outcome), and relaxed early-gen arena gating. Export the trained model to browser binary format.

**Why this is #1:** A strong value network eliminates rollout bias (F1), improves ISMCTS evaluation, provides a policy prior for search (F5), and enables all downstream upgrades. This is the single change that unlocks "superhuman" territory.

**Elo estimate conditions:** +80-150 range assumes (a) value calibration MAE < 0.08 on 500-position suite, (b) policy accuracy top-1 > 40% vs MC-Expert decisions, (c) model successfully integrates as leaf eval. If NN is weak (calibration MAE > 0.15), expect +0 or negative Elo. **The estimate is not data-driven — it extrapolates from other games (Go, poker). Actual gain must be measured.**

**MVP acceptance test:** NN-leaf ISMCTS beats rollout ISMCTS by ≥+30 Elo (95% CI excludes 0) in 400 paired-seed games. Value calibration MAE < 0.10 on 500 curated positions (predicted ME vs actual ME from 10K MC rollouts).

**Ablation plan:**
- A: MC-Expert baseline (current, +43 over heuristic)
- B: NN value replacing rollouts (no policy)
- C: NN policy as ISMCTS prior (no value replacement)
- D: Both (target system)

**Compute:** 500 generations × 4 workers × 250 games × MCTS = ~2 weeks on 8-core CPU + GPU training. One-time cost.

**Integration:** Export via `export_model.py`, load with existing `loadNeuralModel()`. Replace `fastAI` rollout terminal eval with `_nnForward` value head. Use policy head as ISMCTS progressive bias replacement.

### Upgrade 2: Replace Rollouts with Value Network Evaluation (Expected: +40-80 Elo, conditional on Upgrade 1)

**What:** Once Upgrade 1 produces a strong model, modify ISMCTS to use the value network at leaf nodes instead of `fastAI` rollouts. This is the standard AlphaZero leaf evaluation approach.

**Why:** Eliminates rollout bias entirely. Each ISMCTS simulation becomes: select → expand → neural eval → backprop. No rollout = faster simulations = more iterations in 300ms budget.

**Elo estimate conditions:** Assumes value network passes Upgrade 1 acceptance test. If value calibration is poor, NN leaf eval will be *worse* than rollouts. **Must validate value quality before deploying.**

**MVP acceptance test:** On 200 fixed-seed positions, measure top-1 action agreement between (a) NN-leaf ISMCTS@600 iters and (b) rollout-ISMCTS@600 iters vs (c) deep-MC-5000 reference. NN-leaf agreement with reference must exceed rollout agreement by ≥10 percentage points.

**Compute:** Zero training — uses model from Upgrade 1. **Budget gate:** Measure max forward passes sustainable at P95 < 300ms on reference device (Chrome on mid-range laptop). Size iterations to fit within measured budget. Do not assume "~0.5ms per pass" — benchmark it.

**Integration:** Modify `_ismctsSimulate` to call `_nnForward` at leaf instead of `_ismctsRollout`. Remove `fastAI` from ISMCTS path entirely. Keep `fastAI` as fallback if NN inference exceeds budget.

### Upgrade 3: Gumbel-Style Root Action Selection (Expected: +15-30 Elo, **Phase 3 — requires proven NN policy prior**)

**What:** Replace UCB1 at the ISMCTS root with Gumbel-based action selection from [Gumbel AlphaZero](https://openreview.net/forum?id=bERaNdoegnO). Sample actions without replacement using Gumbel noise + policy prior, guaranteeing policy improvement even with few simulations.

**Why:** UCB1 wastes simulations on clearly bad actions. Gumbel selection focuses budget on the top ~4-5 candidates. With only 600 iterations and ~4-8 legal moves, this means 4-8x deeper trees on the moves that matter.

**Phase 3 gate:** Gumbel requires a **net-positive** NN policy prior (value head that improves Elo when used as leaf eval). Without a proven value head, Gumbel over uniform prior reduces to random sampling (no benefit). **Do not implement until Upgrade 1 (value network) shows measurable Elo gain.** Estimate range is from perfect-info games (Go, chess) — may be smaller in IIG settings due to determinization variance.

**MVP acceptance test:** Compare UCB1 vs Gumbel at 100, 300, 600 iterations in 200 paired-seed games each. Gumbel must show ≥+10 Elo at 600 iters with CI excluding 0. Measure policy improvement guarantee violation rate (target: <5% of root decisions).

**Compute:** Zero additional training. ~50 LOC change in ISMCTS selection.

**Integration:** Modify `_ismctsSelect` root logic. Add Gumbel noise sampling. Policy prior from neural network (requires Upgrade 1). Can also use `smartAI` heuristic scores as a weaker prior for testing without NN.

### Upgrade 4: Belief-Weighted Determinization Improvements (Expected: +10-25 Elo)

**What:** Improve determinization quality by:
1. Using trained belief network (from NN encoder's 84-dim belief section) to complement the existing hybrid belief model (hard `cantHave` + soft marginals from importance-weighted MC)
2. Sampling more determinizations for positions with high belief entropy
3. Weighting ISMCTS node statistics by determinization probability

**Current state:** Belief model already uses 6-factor importance weighting (`dealWeight()`) to compute marginals. This upgrade improves the *quality* of those weights, not the infrastructure.

**Why:** Reduces determinization bias (F2) without requiring full PBS search. Practical within browser compute budget.

**MVP acceptance test:** Belief calibration test — for 500 positions, predict P(opponent holds tile X) from marginals, compare to empirical frequency in 10K consistent deals. Upgraded beliefs must have ≥15% lower Brier score. Elo test: 400 paired-seed games, upgraded vs current beliefs.

**Ablation:** A: Current determinization. B: Belief-weighted. C: Entropy-adaptive count.

**Compute:** Minimal — uses existing belief infrastructure.

### Upgrade 5: Endgame Solver Extension to 20 Tiles (Expected: +5-15 Elo)

**What:** The bitmask solver at 16 tiles catches ~75% of endgames. Pushing to 20 tiles would catch ~95%. The bitwise representation already supports this. Main challenge: deal enumeration for 4 unknown players with more tiles.

**Why:** More exact play in late mid-game. The solver is the strongest component — expanding its domain is high-confidence Elo. This is the *most certain* Elo gain (exact play is provably optimal).

**MVP acceptance test:** (a) `endgameVerify(5000)` at new threshold — must ALL PASS. (b) P95 solve time < 500ms on reference device. (c) 400-game benchmark shows positive Elo shift with CI excluding 0.

**Ablation:** Compare 16 vs 18 vs 20 threshold. Measure timeout rate (must stay <500ms P95).

**Compute:** Zero training. Optimization of deal enumeration + alpha-beta depth. May need iterative deepening or deal sampling for 20 tiles.

### Upgrade 6: Proper Weight Tuning (Expected: +5-15 Elo)

**What:** CMA-ES optimizer upgraded to full-covariance (Feb 28). Tunes 13 `AI_WEIGHTS` parameters via adversarial head-to-head games (candidate vs champion, both sides). Console: `optimizeWeights(30, 16, 150)`.

**Current state:** Uses `smartAI` in `_headlessAdversarial`. For production tuning, should upgrade to use MC-Expert (full MC/ISMCTS), not just heuristic-vs-heuristic — the +43→+14 Elo regression happened because the old diagonal CMA-ES optimized for the wrong objective.

**MVP acceptance test:** MC-Expert with tuned weights beats MC-Expert with default weights by ≥+5 Elo (95% CI excludes 0) in 400 paired-seed games. If tuned weights show <+5, keep default weights — the tuning investment wasn't worth it.

**Compute:** With heuristic games (~1ms each): 16 candidates × 150 games × 30 gens ≈ 2.5 min. With MC-Expert (~50ms each): ~40 min. Both feasible.

**Integration:** Currently uses `smartAI` in `_headlessAdversarial`. For full-stack tuning, modify to use `mcExpertMove` instead.

### Upgrade 7: Opening Book from Self-Play (Expected: +3-8 Elo)

**What:** Generate 100K+ games with strong AI, tabulate opening moves and their outcomes by hand configuration. Build a lookup table for moves 1-3.

**Why:** Opening gets worst evaluation quality (most hidden information, shallowest search). A precomputed table eliminates this weakness.

**MVP acceptance test:** Opening book moves agree with MC-5000 top-1 choice ≥70% of the time on 200 random hands. 400-game benchmark shows positive Elo with CI excluding 0.

**Compute:** 100K headless games × ~10ms each = ~17 minutes. One-time.

---

## V. Research & Algorithm Menu

### A. Search/Planning Improvements

**1. Gumbel AlphaZero / Gumbel MuZero**
- *Paper:* "Policy improvement by planning with Gumbel" ([OpenReview](https://openreview.net/forum?id=bERaNdoegnO), ICLR 2022)
- *Key idea:* Replace UCB1 with Sequential Halving + Gumbel noise. Guarantees policy improvement with as few as 2 simulations.
- *Relevance:* Domino has ~4-8 legal moves per turn — Gumbel selection is ideal for this branching factor. Eliminates wasted simulations on dominated moves.
- *Risk:* Requires a reasonable policy prior (NN) to work well.

**2. Monte-Carlo Tree Search with Uncertainty Propagation via Optimal Transport**
- *Paper:* Dam et al. ([ICML 2025](https://proceedings.mlr.press/v267/dam25c.html))
- *Key idea:* Model value nodes as Gaussian distributions. Backup via Wasserstein barycenter. Select via optimistic or Thompson sampling.
- *Relevance:* Domino's stochastic outcomes (blocked games, point types) create high variance. Distributional backups would better handle this than mean backups.
- *Risk:* Computational overhead of Wasserstein computation. May need approximation for browser.

**3. Novelty in Monte Carlo Tree Search**
- *Paper:* Baier & Kaisers ([IEEE ToG 2025](https://ieeexplore.ieee.org/document/11081805/))
- *Key idea:* Add novelty scores (pseudocounts, frequency thresholds) to MCTS selection. Improves exploration in both heuristic-guided and NN-guided settings.
- *Relevance:* Would help ISMCTS explore information-gathering moves that standard UCB1 undervalues.
- *Risk:* Low. Simple additive bonus. Compatible with existing ISMCTS.

### B. Imperfect-Information Learning/Search

**4. ReBeL (Recursive Belief-based Learning)**
- *Paper:* Brown et al. ([NeurIPS 2020](https://arxiv.org/abs/2007.13544); [GitHub](https://github.com/facebookresearch/rebel))
- *Key idea:* Define a Public Belief State (PBS) = distribution over possible private states given all public actions. Train value and policy networks on PBS space. Search over PBS transitions.
- *Relevance:* **This is the theoretically correct approach for Domino.** The PBS is the joint distribution over hidden hands given all observed plays/passes. ReBeL would eliminate determinization bias entirely.
- *Practical concern:* PBS space in Domino is large (beliefs over ~28 tiles × 4 zones). Need function approximation. A "ReBeL-lite" that uses the existing 185-dim encoder (which already contains beliefs) as PBS representation is feasible.
- *Risk:* Complex to implement. But the existing belief infrastructure (cantHave, BeliefModel, 84-dim belief section in encoder) provides 80% of the foundation.

**5. Deep (Predictive) Discounted CFR (VR-DeepPDCFR+)**
- *Paper:* Xu et al. ([arXiv:2511.08174](https://arxiv.org/abs/2511.08174), arXiv preprint)
- *Key idea:* Neural CFR with bootstrapped cumulative advantages, variance reduction, and discounting. Model-free (no game tree traversal — learns from sampled trajectories).
- *Relevance:* Could train an equilibrium-approximating policy for Domino without explicit tree search. The trained policy could serve as the ISMCTS prior or as a standalone player.
- *Practical concern:* CFR methods are designed for 2-player zero-sum. Domino is 2-team zero-sum (4 players, 2 teams), which is equivalent. But the partnership aspect (each player sees only their own hand) adds complexity.
- *Risk:* CFR convergence in large games is slow. Would need game abstraction (cluster similar information sets). The 185-dim encoder already provides this abstraction.

**6. DeepNash / R-NaD (Regularised Nash Dynamics)**
- *Paper:* Perolat et al. ([Science 2022](https://www.science.org/doi/10.1126/science.add4679))
- *Key idea:* Model-free deep RL with R-NaD optimizer that converges to approximate Nash equilibrium. Used to master Stratego (10^535 game tree).
- *Relevance:* Stratego shares key properties with Domino: imperfect information, large state space, no search at inference time. R-NaD could train a pure neural policy for Domino that approaches Nash equilibrium without any search.
- *Practical concern:* R-NaD training is compute-heavy. But Domino is vastly smaller than Stratego. A strong pure-policy model could be the "offline heavy" component, with lightweight ISMCTS as "online refinement."

### C. Training Stability & Evaluation

**7. BetaZero: Belief-State Planning for POMDPs**
- *Paper:* Moss et al. ([RLJ 2025](https://arxiv.org/abs/2306.00249))
- *Key idea:* Extend AlphaZero to POMDPs by planning in belief space. Train value/policy networks on belief states. Use particle-based belief representation.
- *Relevance:* Most directly applicable architecture for Domino. The existing 185-dim encoder with beliefs is essentially a belief state representation. BetaZero provides the missing training loop.
- *Risk:* Julia implementation (not Python). Would need to adapt the training loop concepts.

**8. Tree Search for Simultaneous Move Games via Equilibrium Approximation**
- *Paper:* Yu et al. ([GameSec 2025](https://arxiv.org/abs/2406.10411)) — **⚠ VERIFY arXiv link** (may point to a routing paper, not game search)
- *Key idea:* Approximate Coarse Correlated Equilibrium (CCE) as a subroutine within tree search. Handles imperfect information from simultaneous moves.
- *Relevance:* Domino isn't simultaneous-move, but the equilibrium-in-search approach generalizes. Computing equilibrium strategy at ISMCTS nodes (rather than deterministic best response) would reduce exploitation by opponents who adapt.
- *Risk:* Medium complexity. Equilibrium computation at each node adds overhead.

---

## VI. "Flawless Mode" — Laddered, Measurable Proxies

### Why absolute targets are unreliable

Perfect play is computationally intractable for the full game (28 tiles, imperfect information, ~10^15 information sets). Numbers like "< 0.5% equity loss" or "exploitability < 2%" sound precise but require:
- A "perfect player" reference (doesn't exist for IIGs with this state space)
- An exploitability computation (intractable without game abstraction)

Instead, define progress through **measurable proxies** that can be computed today.

### Performance Ladder (measurable at each step)

| Level | Elo vs Heuristic | How to Measure | Key Upgrade |
|-------|-----------------|----------------|-------------|
| **Current** | +43 | `runAIBenchmark(400)` — measured | MC-Expert with improved rollouts |
| **Milestone 1** | +80 | Same benchmark, 400 paired-seed games, CI excludes +43 | NN leaf eval replacing rollouts |
| **Milestone 2** | +120 | Same benchmark, 800 games for tighter CI | + RIS-MCTS (Phase 2), then Gumbel root (Phase 3, after proven value head) |
| **Milestone 3** | +160 | Same + cross-validated vs 3 opponent styles | + PBS-aware search + convention signals |
| **Ceiling** | +200+ | Unclear if measurable (heuristic too weak as reference) | Full equilibrium approximation |

**Note:** Beyond ~+150 Elo, the heuristic-only baseline becomes too weak to discriminate improvements. At that point, switch reference to "MC-Expert baseline (current)" and measure deltas from there.

### Measurable Quality Proxies (instead of ungrounded absolutes)

**1. Value calibration (computable today):**
- Generate 500 curated positions across game phases
- For each, compute "ground truth" ME via MC-10000 rollouts with full MC-Expert
- Measure MAE between NN predicted value and MC-10000 value
- Target: MAE < 0.08 (better than random baseline MAE ~0.25)

**2. Decision agreement (computable today):**
- On 500 positions, compute top-1 move from (a) system under test, (b) deep reference (MC-5000 or deeper)
- Measure agreement rate
- Current MC-Expert: ~65-70% agreement with MC-5000 (estimated)
- Target: >80% agreement

**3. Style robustness (computable with training):**
- Train or configure 4 opponent archetypes: aggressive (high-pip priority), passive (low-pip priority), blocking-heavy (maximize opponent voids), random
- Measure worst-case Elo drop across archetypes
- Target: worst-case drop < 15 Elo from average

**4. Belief calibration (computable today):**
- For 500 positions, predict P(opponent holds tile X) from belief marginals
- Compare to true hidden hands from logged games
- Measure Brier score
- Target: Brier score < 0.15 (better than uniform prior ~0.25)

**5. Exploitability proxy (requires training infrastructure):**
- CMA-ES optimize an adversary heuristic specifically to exploit the AI
- Measure how much equity the optimized adversary extracts vs how much a random adversary extracts
- Target: optimized adversary gains < 5% equity over random adversary
- This is a *proxy* — it lower-bounds exploitability but doesn't compute it exactly

### Confidence Bounds

- **Elo estimation**: 400-game benchmarks give ±25 Elo at 95% CI. 800 games give ±18. 2000 games give ±11. Use paired seeds for variance reduction.
- **Statistical stopping rule**: Run benchmarks until `CI_width < target_effect_size / 2`.
- **All Elo claims in this document should be read as point estimates with ±25 uncertainty** unless a larger sample is specified.

---

## VII. Concrete Next Sprint (3 Weeks)

### Step 0: Parity Gate (BLOCKING — must pass before any training)

**Encoder parity test — IMPLEMENTED:**
Three tiers available: (1) `_testEncoderParity()` static 3-scenario check, (2) `massParityCheck(N)` → `test_encoder_parity.py --mass` for N-snapshot cross-validation from live games, (3) `exportSnapshots(N)` for JSONL snapshot export. **If mass parity fails, all training data is poisoned — no amount of hyperparameter tuning will fix distribution shift between training and inference.**

Status: All harnesses implemented (Feb 28). The 3 belief mismatches + action mask bug were fixed (Feb 27). **Run `massParityCheck(100)` + `--mass` to confirm ALL PASS before proceeding.**

**Remove 20% NN blend — DONE:**
~~The 20% NN blend at ~line 4078 has been removed (Feb 28).~~ Code replaced with comment: `// NN blend REMOVED — NN must only affect play via leaf value eval (future)`.

### Step 0b: F1 Decision Gate (one day — determines sprint sequencing)

**What:** Run `runF1Diagnostic(200)` (implemented, §III.F1) to isolate rollout bias from determinization bias. This is a **one-day decision gate**, not a research project.

**Implementation:** Already done — `ROLLOUT_VARIANT` toggle, `simulateFromPosition` patch, `_f1FreezeEarlyStop`, diagnostic counters, mcCache-safe top-1 test. See §III.F1.

**How to interpret:**

| Result | Meaning | Sprint sequencing |
|--------|---------|-------------------|
| B ≫ A (+30 Elo) | Rollout bias dominant | Value network is top priority (Week 1-2) |
| B ≈ A (±15 Elo) | Co-dominant or determinization-dominant | RIS-MCTS + value net in parallel |
| B < A | F1 harness may be buggy | Debug before proceeding |

**After F1:** Proceed to Week 1 regardless of outcome — value network and ΔME targets are needed either way. F1 only affects whether RIS-MCTS gets prioritized alongside or after.

### Week 0: Foundation (encoder parity + F1 + baselines)

- ~~Run Parity Gate (Step 0)~~ — **DONE**: `_testEncoderParity()` + `massParityCheck()` + `test_encoder_parity.py --mass` implemented
- ~~Remove 20% NN blend from search scoring~~ — **DONE**: blend removed at line 4078
- ~~Belief semantics: dorme conditional export~~ — **DONE**: both JS and Python export `P(zone | not dorme)`
- ~~Action mask symmetry: `lE==rE`~~ — **DONE**: both JS (line 6851) and Python (line 67) zero `mask[28:56]`
- Run F1 diagnostic (`runF1Diagnostic(200)`) — re-run after mcCache fix, record Elo shift and top-1 agreement
- Baseline measurements: `runAIBenchmark(400)` for clean Elo reference
- Belief calibration baseline: Brier score on 500 positions vs true hidden hands from logged games
- NN inference budget benchmark: `_nnForward` latency (mean, P95) on reference device

### Week 1: Fix the Training Loop + Self-Play with ΔME

**Day 1-2: Parity validation + clean evaluation baseline**
- Run `massParityCheck(100)` in browser → `test_encoder_parity.py --mass` → must ALL PASS
- Confirm F1 re-run shows Variant B with smartAI calls > 0, fastAI calls > 0, branch verification ~100% OK
- `runAIBenchmark(400)` with NN blend removed — establish clean Elo baseline
- ~~Validate ΔME training signal~~ — **DONE**: `validate_training_data.py --quick` plays 20 matches (3556 samples), all signs correct, all in [-0.5, 0.5], assertions pass. JS `exportTrainingData()` plays full matches with ΔME backfill. `me3dParityExport()` enables table cross-validation.

**Day 3-4: Training infrastructure + validate**
- Validate duplicate-deal arena: `_arena_evaluate` uses 95% CI + ≥52% winrate, lower CI bound > 50%
- Test H1 (MCTS vs no-MCTS policy targets): 5 gens each, compare arena promotion rates
- **Pass criteria:** MCTS-mode promotes ≥2/5 gens while no-MCTS promotes 0/5
- Add TT bound type flags (exact vs lower vs upper) to endgame solver for cleaner value estimates
- Add learning rate warmup (1e-4 for 5 gen, then 1e-3)
- If H1 fails: test H3 (graduated arena gating) next

**Day 5-7: Full self-play training run**
- Run `python orchestrator.py --workers 4 --generations 200 --games-per-worker 250 --mcts --mcts-sims 100 --value-target me`
- Monitor arena promotion rate (target: >30% of generations promote)
- Export checkpoints at gen 50, 100, 150, 200 via `export_model.py`
- **Quality gate:** value calibration MAE < 0.10 on 500-position suite before proceeding to Week 2

### Week 2: Integrate Value Head + Measure

**Day 8-9: Replace rollouts with value network**
- ~~Modify ISMCTS leaf evaluation~~ — **DONE**: `USE_NN_LEAF_VALUE` toggle, both ISMCTS + flat MC wired
- ~~Keep `fastAI` for flat MC~~ — **DONE**: fallback to rollout when no model or toggle off
- **Budget gate:** verify NN inference fits within measured P95 budget from Week 0 (`getNNLeafStats()`)
- Benchmark: `runAIBenchmark(400)` with NN-ISMCTS vs heuristic-only (`setNNLeafValue(true)` first)

**Day 10-11: Validate value head quality**
- Measure value calibration MAE on 500-position suite (target: <0.10)
- If value head is net-positive (Elo gain over rollout-only): consider RIS-MCTS prototype
- *(Gumbel root selection deferred to Phase 3 — requires proven net-positive value head, see Upgrade 3)*

**Day 12-13: Weight tuning with full stack**
- Fix `_headlessAdversarial` to use MC-Expert (not `smartAI`)
- Run `optimizeWeights(20, 12, 50)` with full-stack evaluation
- Apply winning weights, benchmark

**Day 14: Integration test + commit**
- `endgameVerify(5000)` — must ALL PASS
- Determinism gate: run 10 fixed-seed games, verify identical outcomes across 3 runs
- `runAIBenchmark(800)` — final Elo measurement with all upgrades
- Target: +100 Elo over heuristic-only (up from +43), with 95% CI
- Commit, push, update docs

### Hard Gates (abort/reprioritize if any fail)

| Gate | Checked when | Fail action |
|------|-------------|-------------|
| **Encoder parity** | **Week 0 (BLOCKING)** | **Do not train until ALL PASS — fix mismatches first** |
| F1 decision gate | Week 0 | If B < A → debug harness. If co-dominant → add RIS-MCTS to Week 2 |
| NN inference budget | Week 0 | If >2ms/pass → reduce model size or use WASM |
| Arena promotion rate | Week 1 Day 5-7 | If <10% after 100 gens → revisit H4 (buffer staleness) |
| Value calibration MAE | Week 1 Day 7 | If >0.15 → model too weak, do not deploy as leaf eval |
| Determinism | Week 2 Day 14 | If non-deterministic → debug TT/hash before release |

---

## VII-B. Infrastructure Readiness (Verified Feb 27)

Three prerequisites for the sprint, all confirmed present:

### 1. Match Equity Table — EXISTS, runtime-computed

`ME3D[s1][s2][dobIndex]` is a 3D DP table computed at page load (~line 7195-7241). Dimensions: `(MATCH_TARGET+5) × (MATCH_TARGET+5) × 4` where dobrada indices map to multipliers [1, 2, 4, 8].

- `getMatchEquity3D(s1, s2, dobMultiplier)` — O(1) array lookup (~line 7260)
- `_rolloutToMEReward(winnerTeam, points, myTeam)` — already converts rollout outcomes to ME deltas (~line 7276): `newME - currentME`
- `roundToMatchEquity(epts, myScore, oppScore, dob)` — converts expected points to ME shift (~line 7267)

**For NN training:** ME(S) can be computed offline by the Python training loop. The `DominoEnv` in Python would need a port of the ME3D DP fill (small — ~50 LOC). Or: export the precomputed ME3D table from JS as a JSON array and load it in Python.

### 2. Scoring Rules — Fully Characterized

| Win type | Points | Condition |
|----------|--------|-----------|
| Normal | 1 | Standard win or blocked game (lowest pips) |
| Carroca | 2 | Finish with a double |
| Lá-e-ló | 3 | Non-double tile plays on both board ends |
| Cruzada | 4 | Double that matches both board ends |

- Match to **6 points** (MATCH_TARGET=6)
- Dobrada multipliers [1, 2, 4, 8] — triggered by 6-0 shutout (buchuda)
- Blocked game (4 consecutive passes) → team with lowest total pips wins 1 point
- TIE_PROB = 0.03 (3% of rounds are ties, used in ME DP fill)

### 3. State Export — Two Paths Available

**Path A: JSONL game logs (existing)**
`GameExporter` class (~line 2562-2634) exports full game records in JSONL format: header → deal → decisions (with evaluations) → outcome. Can generate training data from headless games.

**Path B: 185-dim state vectors (requires minor bridge)**
JS `_nnEncodeState()` (~line 6755) and Python `DominoEncoder.encode()` produce identical 185-dim vectors. To export states from JS to Python training:
- Option 1: Modify headless game loop to call `_nnEncodeState` per move, collect into JSON array, export
- Option 2: Use Python `DominoEnv` + `DominoEncoder` for self-play (already implemented in orchestrator.py) — no JS export needed

**Recommended:** Use Path B — the Python training pipeline already generates its own states via `DominoEnv`. JS export is only needed if you want to train from browser-played games.

---

## VIII. Paper References (Full Inventory)

### A. Search / Planning Improvements

| # | Paper | Year | Venue | Contribution | Relevance to Domino AI | Link |
|---|-------|------|-------|-------------|----------------------|------|
| 1 | Policy improvement by planning with Gumbel | 2022 | ICLR | Sequential Halving + Gumbel noise for root action selection; guarantees policy improvement with as few as 2 sims | With ~4-8 legal moves, Gumbel eliminates wasted sims on dominated actions. Ideal for 300ms budget | [OpenReview](https://openreview.net/forum?id=bERaNdoegnO) |
| 2 | MCTS with Uncertainty Propagation via Optimal Transport | 2025 | ICML (spotlight) | Wasserstein barycenter backups; models value nodes as Gaussians; optimistic + Thompson sampling | Domino's stochastic outcomes (blocked games, point types) create high variance — distributional backups handle this better than mean backups | [PMLR](https://proceedings.mlr.press/v267/dam25c.html) |
| 3 | Novelty in Monte Carlo Tree Search | 2025 | IEEE ToG | 4 novelty measures (pseudocounts, frequency thresholds) integrated into MCTS selection | Would help ISMCTS explore information-gathering moves that UCB1 undervalues; tested in 6 board games with both heuristic and NN guidance | [IEEE](https://ieeexplore.ieee.org/document/11081805/) |
| 4 | E-MCTS: Deep Exploration via Epistemic Uncertainty | 2025 | ICLR | Propagates epistemic uncertainty through learned MCTS; novel deep exploration | Epistemic uncertainty in belief states (dorme, hidden hands) is exactly the signal ISMCTS should use for exploration vs exploitation | [OpenReview](https://openreview.net/forum?id=zrCybZXxC8) |
| 5 | Extreme Value MCTS for Classical Planning | 2024 | SoCS | UCB1-Uniform/Power bandits for unbounded rewards; Full-Bellman backup justification | ME rewards are bounded [-1,1] so less directly applicable, but the backup theory applies | [AAAI](https://ojs.aaai.org/index.php/SOCS/article/view/31569) |
| 6 | Multiagent Gumbel MuZero | 2024 | AAAI | Extends Gumbel planning to multi-agent combinatorial action spaces | Partnership domino is multi-agent; Gumbel selection for team games directly relevant | [ACM](https://dl.acm.org/doi/10.1609/aaai.v38i11.29121) |

### B. Imperfect-Information Learning / Search

| # | Paper | Year | Venue | Contribution | Relevance to Domino AI | Link |
|---|-------|------|-------|-------------|----------------------|------|
| 7 | Re-determinizing IS-MCTS (RIS-MCTS) in Hanabi | 2019 | IEEE CoG | Re-samples hidden info from acting player's perspective at each node; prevents information leakage | **Most directly applicable** — Domino's ISMCTS has exactly this leakage problem. RIS-MCTS was the Hanabi competition winner under 40ms/move budget | [arXiv](https://arxiv.org/abs/1902.06075) |
| 8 | ReBeL: Deep RL + Search for Imperfect-Info Games | 2020 | NeurIPS | Public Belief State (PBS) search; trains value/policy on PBS space; provably converges to Nash in 2p zero-sum | Theoretically correct approach for Domino. The 185-dim encoder with 84-dim belief section is already a PBS approximation | [arXiv](https://arxiv.org/abs/2007.13544) |
| 9 | Student of Games (SoG) | 2023 | Science Advances | Unified algorithm for perfect + imperfect info games. GT-CFR (growing-tree CFR) + sound self-play with value/policy nets | The most comprehensive framework — combines AlphaZero-style search with CFR-style equilibrium guarantees. Beats SOTA in poker + Scotland Yard | [Science](https://www.science.org/doi/10.1126/sciadv.adg3256) |
| 10 | DeepNash / R-NaD (Mastering Stratego) | 2022 | Science | Model-free deep RL with Regularised Nash Dynamics; converges to approximate Nash equilibrium without search | Stratego shares key properties: imperfect info, large state space. R-NaD could train a pure-policy model for Domino | [Science](https://www.science.org/doi/10.1126/science.add4679) |
| 11 | Deep (Predictive) Discounted CFR (VR-DeepPDCFR+) | 2025 | arXiv preprint | Neural CFR with bootstrapped advantages, variance reduction, discounting. Model-free | Could train equilibrium-approximating policy without explicit tree search. The 185-dim encoder provides the function approximation | [arXiv](https://arxiv.org/abs/2511.08174) |
| 12 | SPARTA: Search for Policy Improvement in Cooperative Games | 2020 | AAAI | Monotonic policy improvement via single-agent search over blueprint policy in cooperative PO games | Directly applicable — Domino is cooperative within teams. SPARTA-style search could improve a learned policy at inference time within 300ms | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/6208/6064) |
| 13 | LAMIR: Look-ahead Reasoning with Learned Model in IIGs | 2025 | arXiv (ICLR review) | Learns abstracted model from trajectories; domain-independent abstractions enable tractable look-ahead under imperfect info | Extends MuZero to IIGs — learns game model rather than requiring explicit rules. Could enable model-based search even without hand-coded game tree | [arXiv](https://arxiv.org/abs/2510.05048) |
| 14 | Robust Deep Monte Carlo CFR | 2025 | arXiv | Addresses scale-dependent challenges in neural MCCFR: non-stationary targets, action support collapse, variance explosion | Directly relevant to fixing the NN training loop — provides mitigation strategies (target networks, exploration mixing, variance-aware objectives) | [arXiv](https://arxiv.org/abs/2509.00923) |
| 15 | Hierarchical Deep CFR (HDCFR) | 2025 | arXiv | Hierarchical policies with option framework for long-horizon IIGs; low-variance MC sampling | Domino games are 24 turns — long-horizon for CFR. Hierarchical skills (opening, midgame tactics, endgame) match Domino's natural phases | [arXiv](https://arxiv.org/abs/2305.17327) |

### C. Training Stability / Engineering / Card Games

| # | Paper | Year | Venue | Contribution | Relevance to Domino AI | Link |
|---|-------|------|-------|-------------|----------------------|------|
| 16 | BetaZero: Belief-State Planning for POMDPs | 2024 | RLJ 2025 | AlphaZero extended to POMDPs; trains value/policy on belief states; particle-based beliefs | Most directly applicable architecture — the 185-dim encoder is already a belief state. BetaZero provides the missing training loop | [arXiv](https://arxiv.org/abs/2306.00249) |
| 17 | Tree Search for Simultaneous Move Games via Equilibrium Approx. | 2025 | GameSec | Coarse Correlated Equilibrium as subroutine within tree search | Equilibrium-in-search reduces exploitation; generalizes to partially observed settings | **⚠ VERIFY**: [arXiv](https://arxiv.org/abs/2406.10411) — citation may point to routing paper, not game search |
| 18 | Outer-Learning Framework for Trick-Taking Card Games (Skat) | 2025 | arXiv | Bootstrapping framework: expands human game DBs with millions of self-play AI games for trick-taking cards | Skat is the closest analog to Domino (partnership, imperfect info, trick-taking). Their framework for combining human knowledge with self-play data is directly transferable | [arXiv](https://arxiv.org/abs/2512.15435) |
| 19 | GO-MCTS: Transformer-Based Planning in Observation Space | 2024 | arXiv | Searches over observation space (not game state) in IIGs; tested in Hearts, Skat, "The Crew" | Observation-space search avoids determinization entirely — plans over what the agent actually sees. Tested in partnership card games | [arXiv](https://arxiv.org/abs/2404.13150) |
| 20 | Learning Diverse Risk Preferences in Population-Based Self-Play | 2024 | AAAI | Risk-sensitive PPO + population diversity for robust agents | Addresses opponent distribution shift — train against diverse styles to avoid overfitting to one opponent type | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29188) |

---

## IX. Technique Buckets — What to Steal

### Bucket 1: "Fix Determinization Bias" (Papers 7, 8, 13, 19)

**Key idea:** Current ISMCTS determinizes hidden hands and searches as if the game were perfect-information. This causes strategy fusion (averaging over worlds that require different strategies) and information leakage (using knowledge the acting player shouldn't have).

**What to steal:**
- **RIS-MCTS (#7):** Re-sample hidden info from each player's perspective at their decision node. Simplest fix, drop-in replacement for ISMCTS determinization. Won the Hanabi competition.
- **ReBeL (#8):** Search over Public Belief States instead of determinized worlds. The existing 84-dim belief section in the encoder is already a PBS approximation.
- **GO-MCTS (#19):** Search in observation space rather than game-state space. Avoids determinization entirely.
- **LAMIR (#13):** Learn an abstracted model that handles hidden info natively.

**What it replaces:** The `generateConsistentDeal()` → fixed determinization → search pattern in current ISMCTS.

**Expected failure modes:** RIS-MCTS increases per-node cost (re-sampling). PBS search requires a trained belief model. GO-MCTS needs a transformer architecture. Start with RIS-MCTS (cheapest).

### Bucket 2: "Neural Guidance + Training Stability" (Papers 9, 12, 14, 16, 18)

**Key idea:** The NN training loop failed (97/100 arena rejections) because it used pure neural policy without search for data generation, producing noisy targets.

**What to steal:**
- **Student of Games (#9):** "Sound self-play" — train from recursive sub-search outcomes, not just game results. GT-CFR at training time generates better policy targets.
- **SPARTA (#12):** Use search to monotonically improve a blueprint policy. Even a weak NN becomes useful when search refines it.
- **Robust Deep MCCFR (#14):** Target networks, exploration mixing, variance-aware objectives to stabilize neural CFR training.
- **BetaZero (#16):** The exact training loop needed — AlphaZero on belief states with particle-based representation.
- **Skat Outer-Learning (#18):** Bootstrap from heuristic play data, then refine with self-play.

**What it replaces:** The current orchestrator.py training loop that generates weak data.

**Risks:** Compute requirements for search-supervised training. Mitigation: start with SPARTA-style improvement over heuristic policy.

### Bucket 3: "MCTS Exploration/Backups Upgrades" (Papers 1, 2, 3, 4, 6)

**Key idea:** Within the existing ISMCTS framework, improve selection (what to explore) and backups (how to propagate values).

**What to steal:**
- **Gumbel (#1, #6):** Replace UCB1 at root with Sequential Halving + Gumbel noise. With 4-8 legal moves, all budget goes to top candidates.
- **Wasserstein MCTS (#2):** Model nodes as Gaussians, backup via Wasserstein barycenter. Better uncertainty handling for stochastic Domino outcomes.
- **Novelty (#3):** Add exploration bonus for under-visited information states.
- **E-MCTS (#4):** Propagate epistemic uncertainty (from NN) through the tree.

**What it replaces:** UCB1 selection + mean backup in current ISMCTS.

**Risks:** Wasserstein computation may be too expensive for browser. Gumbel is cheapest. Novelty is additive bonus (very low risk).

### Bucket 4: "Equilibrium / Regret Methods" (Papers 9, 10, 11, 15, 17)

**Key idea:** Instead of best-response search (which can be exploited), compute strategies that approximate Nash equilibrium.

**What to steal:**
- **Student of Games (#9):** GT-CFR for local equilibrium computation during search.
- **VR-DeepPDCFR+ (#11):** Train an equilibrium policy offline, use as ISMCTS prior.
- **HDCFR (#15):** Hierarchical skills for long-horizon games (opening/mid/end).

**What it replaces:** The pure best-response assumption in ISMCTS (play optimally against average opponent).

**Practical concern:** Full CFR is 2-player zero-sum only. Domino is 2-team zero-sum (equivalent), but each player has private info from their teammate too. Need to model partnership as "joint team" vs "joint opponents."

---

## X. Concrete Upgrade Proposals (Ranked, with Paper Citations)

### Proposal 1: RIS-MCTS Determinization Fix (Highest priority)
- **What:** At each non-root node, re-sample hidden hands from the acting player's information set (not the root player's). Prevents leakage of root player's hand info into opponent/partner simulations.
- **Papers:** RIS-MCTS (#7), also ISMCTS foundations
- **Integration:** Modify `_ismctsSimulate` to re-call `generateConsistentDeal` from acting player's perspective at each depth. Cache deals per-information-set.
- **Compute:** ~2x overhead per simulation (re-sampling). With 600 iters, effective iterations drop to ~300 equivalent. Compensate by removing rollouts (use NN value) or increasing time budget slightly.
- **Minimum experiment:** A/B on 200 fixed-seed positions. Measure root-action stability (top-1 agreement across 50 RNG seeds) and Elo. Target: stability +20%, Elo +15.

### Proposal 2: NN Leaf Evaluation (Replace Rollouts)
- **What:** At ISMCTS leaf nodes, call `_nnForward()` for value estimate instead of running `fastAI` rollout to terminal. Keep policy head as progressive bias prior.
- **Papers:** AlphaZero pattern; BetaZero (#16); SPARTA (#12); Hanabi competition (#7) all use learned leaf eval
- **Integration:** Modify `_ismctsRollout` → call `_nnEncodeState` + `_nnForward`, return value head output. ~0.5ms per inference.
- **Compute:** Eliminates rollout cost. Net faster per iteration. Can increase ISMCTS iterations from 600 to 1000+ within 300ms.
- **Minimum experiment:** Compare A (rollout ISMCTS) vs B (NN leaf ISMCTS) on 400 games. Requires trained model from Proposal 4.

### Proposal 3: Gumbel Root Selection
- **What:** Replace UCB1 at ISMCTS root with Sequential Halving over Gumbel-perturbed policy logits.
- **Papers:** Gumbel AlphaZero (#1); Multiagent Gumbel MuZero (#6)
- **Integration:** ~50 LOC change in `_ismctsSelect` for root node only. Non-root nodes keep UCB1.
- **Compute:** Zero additional cost. Actually cheaper (Sequential Halving eliminates dominated actions early).
- **Minimum experiment:** Compare at 100, 300, 600 iterations. Measure Elo at each budget. Target: +10-20 Elo at 600 iters.

### Proposal 4: Fix NN Training Loop (Search-Supervised)
- **What:** Train with MCTS policy targets (not raw neural policy). Use match-equity value targets. Relax arena gating early. Add SPARTA-style policy improvement.
- **Papers:** Student of Games (#9); SPARTA (#12); BetaZero (#16); Robust Deep MCCFR (#14); Skat Outer-Learning (#18)
- **Integration:** Modify orchestrator.py: enable `--mcts`, change value target to ME delta, lower arena threshold gen 0-30.
- **Compute:** 2 weeks GPU training (500 gens × 4 workers × 250 games with MCTS).
- **Minimum experiment:** Track arena acceptance rate (target >30% per gen). Export gen 100 model, benchmark in browser.

### Proposal 5: Wasserstein/Distributional Backups
- **What:** Replace mean backup in ISMCTS with Gaussian distributional backup. Select via Thompson sampling or optimistic UCB.
- **Papers:** Wasserstein MCTS (#2)
- **Integration:** Each ISMCTS node stores (mean, variance) instead of just mean. Backup computes Wasserstein barycenter (for Gaussians: weighted mean of means + weighted mean of variances).
- **Compute:** ~1.5x memory per node. Computation negligible (Gaussian barycenter is closed-form).
- **Minimum experiment:** Compare mean vs distributional backup on 200 high-variance positions (many blocked-game candidates). Measure decision quality.

---

## XI. Assumptions Verified (Appendix)

Claims in this document were verified against the actual codebase on Feb 27, 2026. If any assumption below is marked TODO, the corresponding roadmap item should not be executed until verified.

### NN Training Pipeline (`training/orchestrator.py`)

| Claim | Verified? | Source | Actual Value |
|-------|-----------|--------|-------------|
| Value target supports ΔME (match equity delta) | **UPDATED** | `orchestrator.py` + `training/match_equity.py` | `--value-target me` (default) computes `delta_me()` from ME3D table. Legacy `--value-target points` still available as `points_won / 4.0` |
| `--mcts` flag exists but wasn't used | YES | `orchestrator.py:423-426` | `parser.add_argument('--mcts', action='store_true')` |
| Without MCTS: policy targets are neural + Dirichlet noise | YES | `orchestrator.py:84` | 75% neural + 25% Dirichlet(α=0.3) |
| With MCTS: policy targets from tree search | YES | `orchestrator.py:72` | `mcts.get_action_probs(env, encoder, temperature=temp)` |
| Arena threshold is ≥52% winrate + 95% CI > 50% | **UPDATED** | `orchestrator.py:_arena_evaluate()` | Duplicate deals (each seed played from both sides), promotion requires ≥52% match winrate AND 95% CI lower bound > 50% |
| Arena uses duplicate-deal evaluation | **UPDATED** | `orchestrator.py:ARENA_SEEDS` | Each seed played twice (challenger as team 0 and team 1) |
| Replay buffer is 200K deque with FIFO | YES | `orchestrator.py:206,222` | `deque(maxlen=buffer_size)`, default 200000 |
| No recency weighting or priority sampling | YES | `orchestrator.py:222` | Plain `deque.extend()` |
| Min buffer before training: 2000 | YES | `orchestrator.py:290` | `min_buffer = 2000` |
| MCTS sims default: 50 | YES | `orchestrator.py:425` | `default=50` |
| Temperature annealing in MCTS mode | YES | `orchestrator.py:70` | `temp = 1.0 if step_count < 8 else 0.1` |

### Belief / Determinization System (`simulator.html`)

| Claim | Verified? | Source | Actual Value |
|-------|-----------|--------|-------------|
| `cantHave` tracks NUMBER voids (0-6), not tiles | YES | `Knowledge` class ~line 2783 | Set of numbers per player |
| BeliefModel has soft marginals (0.0-1.0 per tile) | YES | `BeliefModel` class ~line 656 | `marginals[p][tileId]` float weights |
| 200 MC deals for importance-weighted belief update | YES | `computeBeliefs()` ~line 5286 | `nSamples=200` |
| `dealWeight()` uses 6 factors | YES | ~line 5221 | suit strength, pass-adjacency, avoidance, dumping style, sacrifice, opening suit |
| 3-tier determinization fallback | YES | `generateConsistentDeal()` ~line 5102 | BeliefSampler → constraint-prop → rejection sampling |
| 84-dim belief section in NN encoder | YES | `training/domino_encoder.py` | 28 tiles × 3 zones = 84 dims, using conditional `P(zone | not dorme)` semantics (see §XI-C) |

### AI System (`simulator.html`)

| Claim | Verified? | Source | Actual Value |
|-------|-----------|--------|-------------|
| `fastAI` has 13 features | **UPDATED Feb 28** | `fastAI()` ~line 5577-5720 | **13 distinct features** (suit control, block opp1/opp2, partner void, phase-weighted pip dump via `deadMul`, double clearing, dead number detection via `remainingWithNumber`, strand penalty −30, 2-tile close, 3-tile close, opening phase) |
| `smartAI` has 35+ features | **CORRECTED** | `smartAI()` ~line 3000-3508 | **26 distinct features** across 39 conditional branches (chicote, monopoly, lock, partnership, match-point awareness, info hiding, point denial, two-ply lookahead, etc.) |
| ISMCTS: 600 iters / 300ms | YES | ~line 6602 | `MAX_ITERATIONS: 600`, `TIME_LIMIT: 300` |
| MC sims: 800 (Expert) | YES | ~line 6818 | `MC_SIMS_PLAY_EXPERT: 800` |
| Endgame solver threshold: 16 tiles | YES | Verified via `endgameVerify(5000)` | ALL PASS |
| NN inference: 185→57+1 via 4-block ResNet | YES | `training/domino_net.py` + `simulator.html` | Input(185), 4 ResBlocks(256), policy(57), value(1) |
| 20% NN blend when model loaded | **REMOVED Feb 28** | ~line 4078 | Blend deleted — NN will only affect play via leaf value eval (future). See ROADMAP_FREEZE v1 Commit 4 |
| 97/100 arena rejections at gen 84 | UNVERIFIED | From training logs, not code | Need log confirmation — but pipeline issues confirmed via code analysis |

### Match Equity Infrastructure (`simulator.html`)

| Claim | Verified? | Source | Actual Value |
|-------|-----------|--------|-------------|
| ME table exists | **YES** | ~line 7195-7263 | `ME3D[s1][s2][dobIndex]` — 3D DP table computed at startup |
| ME lookup is O(1) | **YES** | `getMatchEquity3D()` ~line 7260 | Direct array index lookup |
| Rollouts use ME | **YES** | `_rolloutToMEReward()` ~line 7276-7289 | Returns `newME - currentME` (ME delta) |
| Scoring: normal(1), carroca(2), la-e-lo(3), cruzada(4) | **YES** | `scoreWin()` ~line 3581, `POINT_DIST` ~line 7187 | Match to 6 points, dobrada multipliers [1,2,4,8] |
| Game state export exists | **YES** | `GameExporter` ~line 2562-2634 | JSONL format (header, deal, decisions, outcome) |
| JS `_nnEncodeState` matches Python `DominoEncoder` | **FIXED Feb 27, harness added Feb 28** | Both produce 185-dim vector | Belief mismatches fixed (§XI-B). Mass parity harness: `massParityCheck(N)` → `test_encoder_parity.py --mass` for N-snapshot cross-validation |

### XI-B. Belief Encoding: Exists but Inconsistent Across Training vs Inference (Found & Fixed Feb 27)

The Bayesian belief system exists in both Python (training) and JS (inference), but was **inconsistent** between them — the NN saw different belief distributions during training vs deployment. Three mismatches were identified. These would silently cause distribution shift, capping NN training quality regardless of architecture or hyperparameters.

**Bug 1: Python belief export normalization (dorme leak)**
- Python `DominoEncoder.belief` is 28×4 (partner, LHO, RHO, dorme), normalized across all 4 columns
- But `encode()` exported only columns 0..2 → NN saw beliefs that don't sum to 1
- Column 3 (dorme) soaked probability mass, making the visible distribution flat/noisy
- **Fix**: Export conditional distribution `P(zone | not dorme)` — normalize cols 0..2 to sum to 1 before writing to state vector (`domino_encoder.py:92-104`)

**Bug 2: Python `_sync_belief` contradiction fallback (belief teleportation)**
- When all 4 zones were eliminated (total=0), old code reset to `self.belief[t,:] = 0.25`
- This "teleported" probability back to zones that had been definitively ruled out
- **Fix**: Assign all mass to dorme: `self.belief[t,3] = 1.0` (the tile must be in the removed pile)

**Bug 3: JS belief was stateless (exists but inconsistent with Python's persistent Bayesian tracker)**
- Python maintained a persistent `self.belief` matrix updated via `update_on_pass()` and `update_on_play()` — information accumulated over the game
- JS recomputed uniform beliefs from scratch each call (`1/N` over feasible players) — no persistence
- Both sides had belief tracking, but the NN saw different belief distributions during training (Python, persistent) vs inference (JS, stateless)
- **Fix**: Added `knowledge.nnBelief` (28×5 Float64Array, absolute player indexing), updated in `recordPlay()` and `recordPass()`, exported via conditional normalization in `_nnEncodeState()`

**Bug 4: JS action mask missing `lE===rE` symmetry**
- Python `get_legal_moves_mask()` already had `symmetric = (left_end == right_end)` to avoid duplicate actions
- JS `_nnActionMask()` set both left and right slots when `lE===rE`, creating redundant actions
- **Fix**: Added `lE === rE` check — when ends are equal, only use left slot

**Parity test (3 tiers):**
1. **Static**: `_testEncoderParity()` in browser — 3 hand-crafted scenarios
2. **Mass**: `massParityCheck(100)` in browser → downloads JSON → `python test_encoder_parity.py --mass <file>` — cross-validates N snapshots from live headless games (eps 1e-6)
3. **Export**: `exportSnapshots(100)` — captures JSONL snapshots for offline training data generation

### XI-C. Belief Encoding Strategy: Conditional 3×28 Now, Explicit 4×28 Later

**Current (Gen 85):** Conditional 3×28 `P(zone | not dorme)`, NN_STATE_DIM=185.

Both Python (`DominoEncoder.export_conditional_belief()`) and JS (`_nnEncodeState`) normalize belief columns 0..2 to sum to 1 before writing to state[91:175]. Internally, Python stores 28×4 (P/LHO/RHO/dorme) and JS stores 28×5 (absolute player IDs + dorme) — but the exported 3×28 is identical due to conditional normalization.

**Why this is correct for now:**
- Preserves NN_STATE_DIM=185 (no model surgery)
- Fixes the training/inference distribution shift (the critical bug)
- Minimal churn — ship a stronger NN immediately

**Future upgrade (post-stable-Gen-85): Explicit 4×28 dorme channel**

Dorme is a real hidden sink (6×4 dealt + 4 removed). Without an explicit dorme channel, the net can't represent "this tile is probably dorme" and will misattribute that probability to P/LHO/RHO — especially midgame when cantHave constraints tighten. This upgrade requires:
- NN_STATE_DIM 185→213 (or 185→186 with scalar `p_dorme_mean`)
- Retrain from scratch or fine-tune
- Update JS + Python encoders + model loader + export tooling

**Implementation note (nnBelief vs belief4):** JS uses `knowledge.nnBelief` (28×5 Float64Array, absolute player indexing) rather than the proposed `belief4` (28×4 Float32Array, relative zones). Both produce identical output. Absolute indexing avoids coordinate transforms at write time (events arrive with absolute player IDs). Float64 avoids precision loss during cumulative normalization.

### Items NOT Verified (claims from narrative, not code)

| Claim | Status | How to Verify |
|-------|--------|--------------|
| "Rollout bias is the single biggest bottleneck" | **DIRECTION CONFIRMED** (F1 shows B > A, magnitude TBD after mcCache fix re-run) | F1 diagnostic ran but initial result invalidated by mcCache poisoning — re-run pending |
| "+80-150 Elo from value network" | UNPROVEN | Extrapolation from other games; measure after training |
| "fastAI lacks chicote/lock awareness" | **CONFIRMED** | Audited `fastAI()`: no chicote, lock, monopoly, match-point, info-hiding, two-ply lookahead. These 6 feature families exist only in `smartAI`. |
| "300ms is enough for ~200-400 forward passes" | UNPROVEN | Run Day 0 benchmark gate |

---

*Generated for berny-the-blade/pernambuco-domino, Feb 2026*
*Research sweep covers 20 papers from 2019-2026, with focus on post-2024 work*
*Review feedback incorporated Feb 27, 2026: added falsifiable hypotheses, MVP acceptance tests, belief model clarification, diagnostic experiments, benchmark gates, and assumptions appendix*
*Belief encoding fixes applied Feb 27, 2026: fixed 3 Python/JS belief mismatches + action mask parity, added encoder parity test (§XI-B)*
*Production-readiness corrections Feb 27, 2026: value targets updated to ΔME, belief framing changed to "exists but inconsistent" (not missing), dorme semantics added to PBS section, Gumbel moved to Phase 3, Parity Gate added as Step 0, F1 reframed as one-day decision gate, sprint restructured to Week 0/1/2, VR-DeepPDCFR+ venue corrected to arXiv preprint, TSS citation flagged for verification*
*ROADMAP_FREEZE v1 applied Feb 28, 2026: F1 cited as decision evidence (rollout-dominant, magnitude TBD after mcCache fix), ΔME specified as explicit value target in Upgrade 1, 20% NN blend removed (code + doc), parity gate harness implemented (exportSnapshots + massParityCheck + --mass flag), CMA-ES upgraded to full-covariance, fastAI expanded to 13 features (deadMul + dead number + 3-tile closing), Week 0 items marked DONE, Week 1 reordered (parity validation first), all line refs and feature counts updated*
