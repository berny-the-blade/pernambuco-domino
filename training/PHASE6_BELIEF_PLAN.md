# PHASE6_BELIEF_PLAN.md
# Belief-Aware Representation Learning (7-Day MVP)
# Source: Bernie analysis 2026-03-08

---

## Objective

Improve partnership and hidden-information play by adding a lightweight belief
auxiliary head that teaches the network to model what other players likely hold.

This phase targets:
- preserving newly opened ends
- not stealing partner's line
- better pressure on opponent weak ends
- better ad-hoc teamplay

**Without yet changing the search algorithm.**

Core insight: The belief head does NOT need to be consumed by search or
inference immediately. Joint training alone can improve the shared trunk's
internal representation, and that may already improve policy/value behavior
and the partnership suite.

---

## Why This Phase

Remaining mistakes are NOT "search too shallow" mistakes. They are:
- good local move, bad partner move
- collapsed useful end
- stole partner's line
- relieved opponent pressure
- failed to exploit likely voids

This phase gives the trunk a reason to represent those latent structures.

---

## Scope

### In scope
- 21-output auxiliary belief head
- Exact belief labels from self-play logs
- Joint training with current policy/value objectives
- Measure: partnership suite, search scaling, ad-hoc partner, belief calibration

### Explicitly NOT in scope this week
- No search-time use of belief outputs
- No belief-conditioned PUCT
- No tile-location marginal head (28×4)
- No ReBeL-style public belief search
- No browser inference API changes

---

## Architecture

### Current
```
Shared trunk → policy head + value head
```

### New
```
Shared trunk → policy head
             → value head
             → belief head (21 logits)
```

### Belief Head Target Layout
```
Index 0..6   = partner has pip 0..6
Index 7..13  = LHO has pip 0..6
Index 14..20 = RHO has pip 0..6
```

Each output is binary:
- 1 if that player holds any tile containing that pip
- 0 otherwise

Activation/loss: raw logits → BCEWithLogitsLoss

---

## Training Losses

```
L = L_policy + 1.0 * L_value + λ_belief * L_belief

Starting: λ_belief = 0.2
Too slow: try 0.3
Policy/value regression: reduce to 0.1
```

---

## Label Generation

Labels are EXACT (not estimated) — full hidden hands are known at generation time.

```
For each training state:
  For each other player (partner, LHO, RHO):
    For each pip n in 0..6:
      label[player][n] = 1 if any tile in player's hand contains pip n, else 0

Example:
  partner hand = [1|4], [1|6], [0|3]
  partner pip labels = [1, 1, 0, 1, 1, 0, 1]
```

### Data Record Schema (Option A — preferred)
```json
{
  "x": [...],          // encoded state
  "pi": [...],         // policy target (visit counts)
  "v": float,          // ΔME target
  "mask": [...],       // legal move mask
  "belief_target": [21 ints],  // NEW
  "metadata": {...}
}
```

---

## Success Criteria

### Primary (one of these must improve without regressions)
1. Partnership suite score improves ≥ +0.05 absolute
2. Search scaling becomes more monotone or stronger at 100–200 sims
3. Ad-hoc partner eval improves meaningfully

### Secondary
- Belief head trains cleanly (loss decreases, AUROC improves)
- No meaningful regression in policy/value losses

### Failure condition
- Policy/value benchmark regresses
- Partnership suite unchanged
- Belief head fails to learn

If failure: reduce auxiliary weight or revisit labels.

---

## Day-by-Day Plan

### Day 1 — Belief target generation
**Goal:** Generate exact 21-dim labels from self-play data.

Tasks:
- Add `build_belief_target(hidden_hands, player_to_move)`
- Extend training record export to include `belief_target`

Files: self-play/export script, training data schema, validator

Pass gate: Sanity check 20 random samples — labels match actual hidden hands.

---

### Day 2 — Add belief head to model
**Goal:** Extend PyTorch model with a third head.

Tasks:
- Add 21-logit belief head
- Add BCE loss
- Add config flags: `--belief-head`, `--belief-weight 0.2`

Files: model definition, trainer, config parser

Pass gate: Forward pass works, all three losses print cleanly.

---

### Day 3 — Small pilot run
**Goal:** Check auxiliary task learns without breaking main tasks.

Run:
- 3–5 generations, current stable settings
- One run with belief head, one without

Metrics: policy loss, value loss, belief loss, quick arena, partnership suite

Pass gate: Belief loss decreases, no obvious regression, suite doesn't worsen.

---

### Day 4 — Belief evaluation metrics
**Goal:** Measure whether head is learning meaningful inference.

Add:
- BCE per player
- AUROC or accuracy per pip/player
- Optional: Brier score

Target report:
```
partner pip AUROC = 0.81
LHO pip AUROC    = 0.74
RHO pip AUROC    = 0.76
```

Pass gate: Belief performance clearly above random.

---

### Day 5 — Partnership suite + ad-hoc partner eval
**Goal:** See whether trunk improved WITHOUT using belief at inference.

Tests:
1. Baseline suite score
2. New belief-head model suite score
3. Optional: ad-hoc partner benchmark (model + weaker heuristic partner vs strong opponents)

Pass gate: Suite improves ≥ +0.05, or ad-hoc partner winrate improves.

---

### Day 6 — Search scaling benchmark
**Goal:** See if better internal belief modeling helps search.

Benchmark: 50 / 100 / 200 / 400 sims (same as Phase 4 eval)

Success pattern (don't need full monotone):
- 100 > 50
- 200 >= previous baseline
- 400 degradation flatter or higher peak

Pass gate: At least one of the above.

---

### Day 7 — Decide Phase 6 outcome and branch

**If successful:**
Freeze Phase 6 MVP → plan Phase 6.5:
- Consume belief outputs explicitly in policy/value heads
- Small belief summary vector fed into heads
- Later: belief outputs in search

**If mixed:**
Keep auxiliary head, lower weight, continue training.

**If no gain:**
Abort integration, keep logs, try:
- Stronger labels
- Different auxiliary target
- Partner-only head instead of all 3 players

---

## Experiment Matrix

| Variant    | Config              |
|------------|---------------------|
| Baseline   | no belief head      |
| Variant A  | λ_belief = 0.2      |
| Variant B  | λ_belief = 0.1      |
| Variant C  | λ_belief = 0.3      |

**Note:** 5–10 generation run is enough. You do NOT need 50 gens for this MVP.

---

## Risks and Mitigations

| Risk | Symptom | Fix |
|------|---------|-----|
| Auxiliary loss dominates | Policy/value regression | Reduce λ_belief to 0.1 |
| Labels too easy / not useful | Head learns but gameplay unchanged | Switch to behaviorally relevant labels (preserve-new-end, void heads) |
| No gameplay lift despite good belief metrics | Trunk learns beliefs, policy/value ignore them | Phase 6.5: explicitly concatenate belief logits back into policy/value heads |

---

## Estimated Elo Impact

~+40–60 Elo (after Phase 5 is working)

---

## The Domino Example This Fixes

```
55 opened → opponent plays 51 → partner should continue 1-side, NOT reconnect to 5
```

A belief-aware model learns:
- Opponent opening 1 changes public belief
- Partner responding on 1 preserves pressure AND signals support
- Reconnecting to 5 destroys that signal and relieves opponent pressure

Plain value head cannot consistently infer this from reward alone.

---

## Status

- [ ] 7-day plan written ✅ (2026-03-08)
- [ ] Label generation pseudocode
- [ ] Model head pseudocode
- [ ] Day 1: belief target generation
- [ ] Day 2: belief head in model
- [ ] Day 3: pilot run
- [ ] Day 4: belief eval metrics
- [ ] Day 5: partnership suite + ad-hoc eval
- [ ] Day 6: search scaling benchmark
- [ ] Day 7: branch decision

---

## References

- ReBeL: NeurIPS 2020 — Combining Deep RL and Search for Imperfect-Information Games
- BAD: Bayesian Action Decoder for Deep Multi-Agent RL (ICML 2019)
- Other-Play: Zero-Shot Coordination (ResearchGate)
