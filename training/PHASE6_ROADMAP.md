# Phase 6 + Checkpoint Selection Mini-Roadmap (5 Days)

_Received 2026-03-09 from AI reviewer_

## Success Criteria

By end of Day 5:
- One budget-specific champion selected for 50 / 100 / 200 sims
- One clear decision on Phase 6 MVP (keep belief head / tune λ / drop)
- Short written decision memo: "deploy checkpoint X at budget Y"

---

## Day 1 — Budget-specific champion selection

**Goal:** Stop assuming "latest checkpoint" is best.

**Candidates:** Gen10, Gen20, Gen50 (minimum set)  
**Budgets:** 50 sims, 100 sims (+ optional 200 sims)  
**Protocol:** duplicate deals, NN leaf on, PUCT on, C=0.7, progBias=false

**Pass/fail gate:** champion = best duplicate-deal result per budget; if within noise → "near-tied"

**Output:** `training/BUDGET_CHAMPIONS.md`

| Budget | Best ckpt | Runner-up | Notes |
|--------|----------:|----------:|-------|
| 50 sims | ? | ? | |
| 100 sims | ? | ? | |
| 200 sims | ? | ? | |

---

## Day 2 — Baseline suite before belief head

**Goal:** Freeze a baseline before Phase 6.

Run on budget-specific champion from Day 1:
- Partnership suite (base + per-theme scores)
- Search scaling: 50 / 100 / 200 / 400 sims
- Anchor match: champion vs current deployed at deployment budget

**Output:** `training/PHASE6_BASELINE.md`

---

## Day 3 — Phase 6 belief-head probe (3-way λ sweep)

**Goal:** Test whether joint belief training helps without using it in search.

**Variants:** λ=0.1 / λ=0.2 / λ=0.3, 5 gens each from same checkpoint

```bash
python training/orchestrator.py \
  --resume checkpoints/<best_checkpoint>.pt \
  --generations 5 \
  --workers 10 \
  --games-per-worker 10 \
  --value-target me \
  --policy-target visits \
  --mcts \
  --mcts-sims 50 \
  --belief-head \
  --belief-weight 0.1   # repeat for 0.2 and 0.3
```

**Pass/fail gate:** λ survives if belief loss decreases, no arena regression, training stable.

---

## Day 4 — Evaluate best belief-head variant

Compare best λ from Day 3 vs Day 2 baseline on:
- Partnership suite (especially preserve_new_end, suit_stealing, ad_hoc_teamplay)
- Search scaling at 50/100/200/400 sims
- Anchor at deployment budget

**Hard pass:** suite +0.05 OR anchor winrate +2.5pp OR cleaner scaling  
**Hard fail:** suite worse >0.03 OR benchmark worse >2.5pp OR training unstable

---

## Day 5 — Decision memo and deployment roadmap

**Output:** `training/PHASE6_PROBE_RESULTS.md` + written decision:

1. Budget-specific champions table
2. Phase 6 result: KEEP (λ=X) / KEEP (tune) / DROP
3. Next training direction

---

## Experiment Matrix

### Champion selection
| Compare | 50 sims | 100 sims | 200 sims |
|---------|--------:|---------:|---------:|
| Gen10 vs Gen20 | ✓ | ✓ | optional |
| Gen20 vs Gen50 | ✓ | ✓ | ✓ |
| Gen10 vs Gen50 | optional | ✓ | optional |

### Phase 6 λ sweep
| Variant | λ_belief | Gens |
|---------|--------:|-----:|
| A | 0.1 | 5 |
| B | 0.2 | 5 |
| C | 0.3 | 5 |

---

## Key thresholds

- Champion: lower CI > 50% = winner; heavy overlap = tie → prefer cheaper ckpt
- Phase 6 pass: partnership suite +0.05 OR anchor +2.5pp OR cleaner scaling
- Phase 6 fail: suite worse >0.03 OR benchmark worse >2.5pp OR unstable training
