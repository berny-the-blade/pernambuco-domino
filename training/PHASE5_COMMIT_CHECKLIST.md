# PHASE5_COMMIT_CHECKLIST.md
# Execution Checklist — Next 8 Commits
# Source: Bernie analysis 2026-03-07

---

## Commit 7 — rootVisitPolicy export
`commit 7: rootVisitPolicy export`

**Goal:** Replace heuristic-softmax labels with true 57-dim visit-count policy targets.

**Files:** simulator.html, training data export helper / JSONL exporter

**Changes:**
- Add `rootVisitPolicy(root, legalMask, tau=1.0)`:
  - builds pi[57] from child.visits
  - no dedup by tile
  - respects symmetry masking
  - normalizes to sum 1
- Update `exportTrainingData()` to write: x, pi, v=deltaMe, mask, metadata (gen, sims, team_pov, scores, dobrada)

**Pass gate:**
- sum(pi)=1 ± 1e-6
- sum(pi[mask==0]) < 1e-6
- no right-side mass when ends are equal
- validate 100 rows with validate_training_data.py

---

## Commit 8 — visits-vs-heuristic policy target flag
`commit 8: visits-vs-heuristic policy target flag`

**Goal:** Train from visits or heuristic behind a flag.

**Files:** training/orchestrator.py, trainer config parser, dataset loader

**Changes:**
- CLI flag: `--policy-target visits|heuristic`
- Default new experiments to `--policy-target visits`
- Log config at startup

**Pass gate:** training log prints `policy_target=visits`, `value_target=me`

---

## Commit 9 — root-only PUCT prior
`commit 9: root-only puct prior`

**Goal:** Make higher sims useful by guiding root exploration with NN policy.

**Files:** simulator.html

**Changes:**
- Add `USE_NN_POLICY_PRIOR`
- P' = (1-alpha)*Uniform + alpha*Pnn
- Start: alpha=0.2, c_puct=1.0
- Keep C=0.7, progBias=false in NN mode

**Pass gate:**
- 50 sims doesn't regress vs current best
- 100 sims >= 50 sims

**Benchmark:** 200 games at 50 and 100 sims — PUCT off vs PUCT on

---

## Commit 10 — visits-max root choice
`commit 10: visits-max root choice`

**Goal:** Improve high-sim stability, align play with visit-target training.

**Files:** simulator.html, root decision block

**Changes:**
- Final move = max child.visits
- Tie-break by average reward

**Pass gate:** 100 and 200 sims no worse than before, lower variance across runs

---

## Commit 11 — partnership regression suite
`commit 11: partnership regression suite`

**Goal:** Capture human partnership concepts the AI still misses.

**Files:**
- `training/tests/partnership_suite.json` (new)
- `training/tests/test_partnership_suite.py` (new)

**Positions to include:**
- don't steal partner's suit
- don't collapse newly opened end
- preserve pressure on weak end
- probe for voids
- sacrifice to lock
- KEY EXAMPLE: open 55, opponent plays 51 → partner continues 1-side NOT reconnect 5

**Pass gate:** suite runs end-to-end, baseline score recorded

---

## Commit 12 — new-end preservation heuristic
`commit 12: new-end preservation heuristic`

**Goal:** Fast Elo from explicit teamplay concept.

**Files:** simulator.html, smartAI scoring section, heuristic scoring helpers

**Rule:** If opponent opens new end x, and partner can legally continue x,
penalize moves that remove x unless immediate tactical gain > threshold.

**Pass gate:** partnership suite improves, no broad benchmark regression

---

## Commit 13 — duplicate-deals arena
`commit 13: duplicate-deals arena`

**Goal:** Reduce evaluation variance, stop wasting compute on noisy promotions.

**Files:** training/orchestrator.py, arena evaluation code, match runner

**Changes:**
- Each seed played twice with swapped sides
- Aggregate result over the pair

**Pass gate:** CI width shrinks noticeably, promotion decisions stabilize

---

## Commit 14 — league promotion gate
`commit 14: league promotion gate`

**Goal:** Stop overfitting to latest champion, make promotions durable.

**Files:** arena/promotion logic, checkpoint manager

**Changes:**
- Compare challenger vs: current champion, previous 3-5 champs, heuristic-only
- Quick gate at 200 games, extend to 400/600 if close
- Promote if lower CI > 50% and no major league regression

**Pass gate:** fewer lucky promotions, fewer false rejections

---

# Benchmark Protocol After Commit 10

```
setNNLeafValue(true)
setUsePUCT(false); MC_SIMS=50; runAIBenchmark(200)
setUsePUCT(true);  MC_SIMS=50; runAIBenchmark(200)
                   MC_SIMS=100; runAIBenchmark(200)
                   MC_SIMS=200; runAIBenchmark(200)
                   MC_SIMS=400; runAIBenchmark(100)
```

**Success:** PUCT-on > PUCT-off; 100>=50; 200>=100 or close; 400 doesn't collapse

---

# Training Run After Commit 8

```bash
python -u orchestrator.py \
  --resume checkpoints/domino_gen_0050.pt \
  --generations 20 \
  --workers 20 \
  --games-per-worker 25 \
  --value-target me \
  --policy-target visits \
  > full_run_visits_me.log 2>&1

# Quick probe first:
python -u orchestrator.py \
  --resume checkpoints/domino_gen_0050.pt \
  --generations 5 \
  --workers 10 \
  --games-per-worker 10 \
  --value-target me \
  --policy-target visits \
  > probe_visits_me.log 2>&1
```

---

# Stop/Continue Rules

**Continue if:**
- policy loss trends down
- promotions happen at least every few gens
- 100+ sims becomes useful

**Stop and inspect if:**
- visit targets produce illegal pi mass
- PUCT regresses badly even at low alpha
- higher sims still collapse after Commit 10
- partnership feature helps suite but hurts broad arena badly

---

# Recommended Order
1. Commit 7 → 2. Commit 8 → 3. Commit 9 → 4. Commit 10 → **benchmark** → 5. Commit 11 → 6. Commit 12 → 7. Commit 13 → 8. Commit 14

**Fastest near-term Elo: Commits 7-10 first, benchmark, then 11-14.**
