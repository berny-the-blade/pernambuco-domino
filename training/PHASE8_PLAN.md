# Phase 8 Plan — Probe B Production Challenge
_2026-03-11_

## Objective

Determine whether the Phase 6.5 Probe B continuation checkpoint beats the current production champion at the live deployment sim budget.

Promote only if it wins. Keep as research branch if it doesn't.

---

## Context

| Item | Status |
|------|--------|
| Architecture winner | Probe B (belief+support, λ=0.1/0.1) — confirmed by Phase 6.5 probe |
| Continuation run | 6 gens completed; `checkpoints/best_100sims.pt` = gen6 |
| Gen6 suite | **NOT YET MEASURED** — must run before Step B |
| Head-to-head vs gen50 | **NOT YET RUN** — this is the main Phase 8 gate |
| Production champion | `training/checkpoints/domino_gen_0050.pt` (gen50) |

---

## Step A — Identify Best Continuation Checkpoint

Find the single best Probe B checkpoint across the 6-gen continuation run.

### Criteria (in priority order)

1. Partnership suite (mcts100) ≥ 0.700
2. `confirm_partner_signal` ≥ 0.600
3. Arena win rate vs gen50 at 100 sims (use for tiebreak)

### What to run

```bash
# Measure suite on each of gen1–gen6 from checkpoints/
python training/run_phase65_probe.py \
  --checkpoint checkpoints/domino_gen_000X.pt \
  --suite-only --sims 100
```

Check each gen. The best suite score wins. If gen6 is already best (likely), skip earlier gens.

**Starting point: just measure gen6 first.** Only go back to earlier gens if gen6 fails suite.

### Expected outcome

`best_100sims.pt` (gen6) is probably the winner. Confirm it before Step B.

---

## Step B — Head-to-Head: Best Probe B vs Production Champion

This is the decision test.

### Setup

| Role | Checkpoint |
|------|-----------|
| Model A (challenger) | Best checkpoint from Step A (expected: `checkpoints/best_100sims.pt`) |
| Model B (champion) | `training/checkpoints/domino_gen_0050.pt` |

### Test budget

| Sims | Pairs | Total games |
|------|-------|-------------|
| 50 | 400 | 800 |
| 100 | 400 | 800 |
| 200 | 400 | 800 |

Use duplicate deals throughout.

### Command

```bash
python training/search_scaling_eval.py \
  --model-a checkpoints/best_100sims.pt \
  --model-b training/checkpoints/domino_gen_0050.pt \
  --sims 50 100 200 \
  --pairs 400 \
  --duplicate \
  --tag phase8_challenge \
  --seed 9000
```

---

## Step C — Partnership Suite on Both

Run the full partnership suite on both checkpoints at 100 sims.

```bash
python training/run_phase65_probe.py \
  --checkpoint checkpoints/best_100sims.pt \
  --suite-only --sims 100

python training/run_phase65_probe.py \
  --checkpoint training/checkpoints/domino_gen_0050.pt \
  --suite-only --sims 100
```

Confirm Probe B does not regress suite vs gen50.

---

## Promotion Rule

**Promote Probe B if ALL of:**

| Gate | Threshold |
|------|-----------|
| Arena win rate @ live budget | ≥ 50% (ties count as pass) |
| Arena win rate @ live budget | Preferably > 52% for confidence |
| Partnership suite | Not worse than gen50 baseline |
| `confirm_partner_signal` | Materially above 0 (≥ 0.400) |

**Reject if ANY of:**

- Loses clearly at live budget (< 48%, CI upper bound below 50%)
- Partnership suite drops below gen50 baseline by > 0.05
- `confirm_partner_signal` regresses below 0.333

---

## Outcome Paths

### Path A — Probe B passes → Deploy

1. Export best checkpoint to `domino_model.bin`
2. Back up old champion to `domino_model_prev.bin`
3. Tag commit as `phase8-promotion`
4. Update `FROZEN_STATE.md`

### Path B — Probe B fails → Continue as research branch

Do not abandon the architecture. Options:

- Reduce aux weights: 0.05 / 0.05 (less interference with core training)
- Continue training longer (10–15 more gens from gen6)
- Try no-aux-detach variant (allow auxiliary gradients to flow into trunk)

Keep gen6 as the research branch starting point.

---

## The gen3 > gen100 Signal

Probe B gen3 beat gen100 at 53.7%. This is a positive architecture signal.

**Do not treat this as the main decision variable.** Gen100 may have been a weaker checkpoint from an earlier era.

The only result that matters for deployment: **does the best Probe B checkpoint beat gen50 at the live budget?**

Treat the gen100 result as: "architecture is not dead, worth continuing." Nothing more.

---

## Files

| File | Description |
|------|-------------|
| `FROZEN_STATE.md` | Checkpoint inventory and frozen state |
| `PHASE8_PLAN.md` | This file |
| `PHASE65_PLAN.md` | Architecture design for Phase 6.5 |
| `PHASE6_VERDICT.md` | Why Phase 7/Phase 6 MVP was vetoed |
| `checkpoints/best_100sims.pt` | Phase 6.5 Probe B continuation best (gen6) |
| `training/checkpoints/domino_gen_0050.pt` | Production champion reference |
