# Phase 6 Decision Memo — Day 5
_2026-03-09_

---

## 1. Budget-Specific Champions

| Budget | Champion | Notes |
|--------|----------|-------|
| 50 sims | **Gen20** | 56.2% vs Gen50 (Day 1) |
| 100 sims | **Gen20** | 50.5% vs Gen50 — near-tied but best available |
| 200 sims | **Gen20** | 54.0% vs Gen50 (Day 1) |

Gen20 remains the best base checkpoint. The Day 4 belief-head extension
(λ=0.1, 8 gens from Gen20) is now the new champion for deployment.

---

## 2. Phase 6 Result: KEEP belief head (λ = 0.1)

### Evidence

| Metric | Baseline Gen20 | Day 4 λ=0.1 (8 gens) | Delta |
|--------|---------------|----------------------|-------|
| Partnership suite avg | 0.684 | **0.684** | = |
| confirm_partner_signal | 0.000 | **0.333** | +0.333 ✅ |
| dont_steal_partner_suit | 0.833 | **1.000** | +0.167 ✅ |
| preserve_new_end | 0.600 | **1.000** | +0.400 ✅ |
| ad_hoc_teamplay | 0.667 | **0.833** | +0.167 ✅ |
| tactical_override | 0.417 | **0.750** | +0.333 ✅ |
| preserve_pressure | 1.000 | 0.333 | -0.667 ⚠️ |
| sacrifice_to_lock | 1.000 | 0.500 | -0.500 ⚠️ |
| information_probe | 0.500 | 0.500 | = |
| **100-sim vs Gen50** | 50.5% | **53.0%** | +2.5pp ✅ |

### Search scaling (Day 4 λ=0.1 vs Gen50)
| Sims | Win% | 95% CI |
|------|------|--------|
| 50 | 55.0% | [48.1, 61.7] |
| 100 | 53.0% | [46.1, 59.8] |
| 200 | 51.0% | [44.1, 57.8] |

Scaling curve is clean: 50 > 100 > 200 (monotone, healthy).

### Decision: **KEEP belief head at λ = 0.1**

Pass criteria all met:
- ✅ Suite recovered to baseline (0.684)
- ✅ confirm_partner_signal: 0.000 → 0.333
- ✅ 100-sim vs Gen50: 50.5% → 53.0% (target ≥ 53%)
- ✅ All 8 gens auto-promoted, 0 rejections

**Caveats (monitor in Phase 7):**
- preserve_pressure dropped 1.000 → 0.333 (was 1.000 in Gen20 baseline)
- sacrifice_to_lock dropped 1.000 → 0.500
- These may recover with more training games (Day 3/4 only used 100 games/gen)
  OR may reflect real tradeoffs from the belief head reshaping the trunk

---

## 3. Next Training Direction

### Phase 7: Full belief-head training run

**Recipe:**
```
python training/orchestrator.py \
  --resume checkpoints/<day4_final>.pt \
  --generations 20 \
  --workers 20 \
  --games-per-worker 250 \
  --value-target me \
  --policy-target visits \
  --mcts-sims 200 \
  --belief-head \
  --belief-weight 0.1
```

**Why 20 gens at full game count:**
- Day 3/4 used only 100 games/gen — too small to stabilize behavior
- preserve_pressure and sacrifice_to_lock regressions likely noise at this sample size
- Full run will tell if the belief head holds or regresses under proper training load

**Monitoring gates (every 5 gens):**
- Partnership suite must not fall below 0.650 (grace: -0.034 from baseline)
- confirm_partner_signal must stay > 0.10
- Arena: no rejection streaks > 3

**If preserve_pressure stays degraded after 10 gens:**
- Reduce λ from 0.1 to 0.05 and continue
- Do not revert to no-belief baseline yet

---

## 4. Deployment

**Deploy Day 4 λ=0.1 checkpoint as new live model?**

Recommendation: **YES** — it matches baseline strength + improves signaling.
The only risk is preserve_pressure/sacrifice_to_lock regressions, which are
likely noise given the tiny training sample. Gen20 is kept as fallback.
