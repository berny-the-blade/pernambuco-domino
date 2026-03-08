# PHASE5_14DAY_PLAN.md
# 14-Day Implementation Plan — Post-Gen50
# Source: Bernie analysis 2026-03-07

## Goal
3 highest-value upgrades:
1. Visit-count policy training + stronger PUCT prior
2. Partnership/signaling awareness
3. League evaluation with duplicate deals + CI/SPRT

---

# WEEK 1 — Make policy actually drive search

## Day 1 — Export true visit-count policy targets
**Goal:** stop training policy from heuristic softmax; use root visit counts.

- Add `rootVisitPolicy(root, mask, tau=1.0) -> pi[57]` to simulator.html root extraction
- Keep dedup for human display only; no dedup for training rows
- Training row format: x (213-dim state), pi (57-dim visit counts), v (delta ME), mask, metadata

**Acceptance:** sum(pi)=1, sum(pi[mask==0]) < 1e-6

---

## Day 2 — Wire policy-source switch into training
**Goal:** train from visits or heuristic behind a flag.

- `training/orchestrator.py`: add `--policy-target visits|heuristic`
- Default: heuristic for old runs, visits for new

**Acceptance:** training logs print `policy_target=visits`

---

## Day 3 — Make PUCT prior real but safe
**Goal:** use NN policy prior in PUCT without overtrusting it.

- Root-only first. Blended prior: P'(a) = (1-alpha)*U(a) + alpha*P_nn(a)
- Start: alpha=0.2, c_puct=1.0
- Final move selection = most visited (tie-break by Q)

**Acceptance:** 50/100 sims don't regress vs current NN-leaf baseline

---

## Day 4 — Tune root PUCT quickly
- Sweep: alpha in {0.1, 0.2, 0.3} x c_puct in {0.5, 1.0, 2.0}
- 200 games each at 50 and 100 sims
- Pick config where 100 sims >= 50 sims, no regression

---

## Day 5 — Re-run search-scaling curve
**Goal:** verify PUCT prior fixed the "more sims hurts" pathology.

- 50/100/200/400 sims, same model, NN leaf on, PUCT on, 200 games
- **Success:** monotone or near-monotone Elo curve

---

## Day 6 — Short ExIt run with visit targets
- 20 workers, 25 games/worker, ME value target, visits policy target
- 5-10 generations first

**Acceptance:** policy loss trends down, at least 1 promotion in 5 gens

---

## Day 7 — Export best checkpoint and browser benchmark
- A/B: old best vs new visit-target model
- NN leaf on, PUCT on, 50/100/200 sims
- **Decision:** if new model wins at 100+ sims, Week 1 succeeded

---

# WEEK 2 — Teach partnership and make evaluation trustworthy

## Day 8 — Build partnership/signaling tactical suite
**Goal:** capture human partnership concepts the AI misses.

30-50 curated positions covering:
- "don't steal partner's suit"
- "don't collapse newly opened end"
- "preserve pressure on weak end"
- "probe for voids"
- "sacrifice to lock"
- e.g. 55 opened → opponent plays 51 → partner continues 1-side, not reconnect 5

Store in JSON/fixture form with expected preferred move set.

---

## Day 9 — Add heuristic features for clearest partnership concept
- Penalize moves that remove newly opened end when partner can keep it alive
- Convention penalty, not hard rule
- Keep it small

**Acceptance:** tactical suite score improves, no regression in arena

---

## Day 10 — Add auxiliary partnership targets (lightweight)
- Auxiliary prediction: partner void probabilities (pips 0-6), or "does move preserve opened end?"
- Does not need to affect inference yet — shapes representation

**Acceptance:** auxiliary loss decreases, no policy/value regression

---

## Day 11 — Duplicate deals in arena
- Each deal played both sides; aggregate result across pair
- **Acceptance:** CIs tighten significantly, promotion decisions more stable

---

## Day 12 — CI/SPRT league gate
- League: current champion + 3-5 previous champs + heuristic-only + MC-expert rollout
- Quick gate at 200 games, extend to 400/600 if close
- Promote if lower CI > 50% vs champion and no major league regression

---

## Day 13 — 20-generation main run
- value_target=me, policy_target=visits, graduated/CI gating, duplicate deals, best PUCT
- Run 20 gens (not 50 — cheaper, enough to confirm compounding)

**Success:** >2 promotions in 20 gens, browser benchmark beats current deployed model

---

## Day 14 — Freeze and deploy
- Export best checkpoint to domino_model.bin
- Backup previous binary
- Verify load in simulator + mobile
- Final benchmark: vs heuristic-only, vs previous deployed model, 50/100/200 sims, tactical suite score
- Update roadmap: what worked, what didn't, next bottleneck

---

# Highest ROI subset (if time tight)
1. Visit-count policy targets
2. Root-only PUCT with blended prior
3. Most-visited final move
4. Duplicate deals arena
5. Partnership tactical suite

---

# NOT prioritized yet
- Full RIS-MCTS
- Determinization pooling
- Larger NN
- CMA-ES on legacy heuristic weights

---

# Deliverables checklist
- [ ] policy_target=visits end-to-end
- [ ] Root-only PUCT live and tuned
- [ ] Most-visited move selection live
- [ ] Partnership tactical suite in repo
- [ ] Duplicate deals + CI/SPRT arena
- [ ] Best checkpoint deployed to simulator and mobile
- [ ] New benchmark table showing gains at 100+ sims
