# MCTS Search vs Network Bottleneck — Diagnostic Framework

> From analysis session 2026-03-07. Apply after gen 50.

## The Core Asymmetry

If value loss improves but policy loss stalls and strength barely moves:
→ the problem is almost certainly search targets, not the network.

MCTS visit distributions are the policy labels. If MCTS is weak or noisy,
the policy head has nothing useful to learn from.

---

## Quick Decision Tree

### Search is probably the bottleneck if:
- Value improves, policy flat ← **already showing this**
- Deeper sims materially improve strength
- Root targets change a lot with more sims
- Entropy drops meaningfully with more sims
- Reanalysis produces sharper targets
- Anchor strength lags internal loss improvement ← **already showing this**

### Network is probably the bottleneck if:
- Deeper sims barely help
- Root targets already stable at 200 sims
- Entropy already low
- Raw priors poor even in easy states
- Larger network improves strength immediately
- Training loss still meaningfully reducible

---

## The 5 Post-Gen50 Tests (priority order)

### Test 1: Search Scaling Eval
Same checkpoint (gen 50) vs gen 46 at multiple sim budgets.

| Sims | Win% vs Gen46 | ELO d | Notes |
|------|--------------|-------|-------|
| 50   |              |       |       |
| 100  |              |       |       |
| 200  |              |       | (current training budget) |
| 400  |              |       |       |
| 800  |              |       | if feasible |

**Interpretation:**
- Big gain 50→400: search bottleneck. Training targets at 200 sims are too weak.
- Flat 50→400: network/representation bottleneck.

**Pass threshold:** < 3% spread across sim budgets = search not the bottleneck.
**Fail threshold:** > 8% spread = search is clearly bottlenecking.

---

### Test 2: Root Target Stability
Take 200 fixed public states. Run MCTS at 100, 200, 400 sims.
Measure KL divergence of visit distributions and top-1 agreement.

| Comparison     | Mean KL  | Top-1 Agreement | Top-2 Gap |
|----------------|----------|-----------------|-----------|
| 100 vs 200 sim |          |                 |           |
| 200 vs 400 sim |          |                 |           |

**Pass threshold:** 200 vs 400 top-1 agreement > 85% = targets mature enough.
**Fail threshold:** < 70% agreement = labels still moving, policy supervision is noisy.

---

### Test 3: Anchor Eval (already running via anchor_eval.py)
Gen 50 vs gen 46 and gen 1 with duplicate deals, 400 games.

See: logs/anchor_eval.jsonl

**Pass threshold:** Win% vs gen46 > 55% by gen 50.
**Fail threshold:** Still ~50% = no measured strength improvement after 50 gens.

---

### Test 4: Particle Disagreement (Belief Sensitivity)
Same 200 public states. For each state, sample N particles, run search on each.
Measure top-action agreement across particles.

| States         | Mean Top-1 Agreement | Std Dev | Notes |
|----------------|---------------------|---------|-------|
| Easy (few legal moves) |         |         |       |
| Medium         |                     |         |       |
| Complex        |                     |         |       |

**Pass threshold:** > 80% top-1 agreement in medium states.
**Fail threshold:** < 60% = belief uncertainty is smearing policy targets.
If this fails, the bottleneck is belief modeling, not sim count.

---

### Test 5: Replay Diagnostics
From self-play data of the current run, measure:

| Metric              | Value | Target Range | Status |
|---------------------|-------|--------------|--------|
| Forced move %       |       | 10–30%       |        |
| Mean legal move count |     | > 3.0        |        |
| Root entropy (mean) |       | 1.0–2.0      |        |
| Avg game length     |       | —            |        |
| Sample diversity    |       | —            |        |

**Warning:** Forced move % > 35% = policy diluted by trivial states.
**Warning:** Root entropy > 2.5 consistently = search is confused (too many equivalent moves or belief noise).

---

## The One Heuristic That Matters Most

> If doubling sims gives more strength gain than doubling training,
> the bottleneck is search.
> If doubling training gives more gain than doubling sims,
> the bottleneck is network/data.

---

## Current Priors (as of gen 15)

Based on current evidence, bottleneck is likely **search/data quality** because:
- Value loss improving, policy loss flat ✓ (search bottleneck indicator)
- Anchor win rates all ~50% at gen 15 ✓ (strength not tracking loss)
- 200 sims in 4-player hidden-info partnership game is modest ✓
- Belief modeling is early-stage ✓
- Imperfect-information noise likely high ✓

**Next action after gen 50:** Run Test 1 (search scaling) first.
Result determines everything else.

---

## Decision Matrix After Tests

| Test 1 result | Test 2 result | Test 4 result | Recommended next step |
|--------------|---------------|---------------|-----------------------|
| Big sim gain | Targets unstable | — | Increase sims to 400–800, possibly add reanalysis |
| Big sim gain | Targets stable | — | Belief modeling is bottleneck, not raw sims |
| Flat sim gain | — | Low agreement | Fix belief constraints / particle diversity |
| Flat sim gain | — | High agreement | Network capacity or encoder quality |
| Flat sim gain | Targets stable | High agreement | Data diversity / replay quality |

---

## Reanalysis Option (if search is bottleneck)

Take stored replay states, re-run with 2–4× sims, replace stored policy targets.
If reanalyzed targets are significantly sharper → original labels were limiting policy learning.
Reanalysis is one of the highest-leverage interventions available.
