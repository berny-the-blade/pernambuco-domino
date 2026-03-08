# PHASE5_ROADMAP.md
# Next 3 Algorithmic Improvements (priority order)
# Source: Bernie analysis 2026-03-07 post-Gen50

---

## Priority 1: Visit-count policy training + stronger PUCT prior
**Expected upside: +10 to +30 Elo**

Current weak link: policy head. Value head is useful but policy is still lagging.
Fix makes 100→200→400 sims monotone (instead of plateauing).

### What to do
- Train policy on root visit counts from MCTS, not heuristic softmax
- Use policy prior in PUCT: Q + c_puct * P(a) * sqrt(N) / (1 + n(a))
- Final move selection: most-visited node, not best-Q

### Acceptance test
- 50/100/200/400 sims benchmark curve becomes monotone
- PUCT-on beats PUCT-off with same value model

---

## Priority 2: Partnership-awareness / signaling features
**Expected upside: +10 to +25 Elo + much better human-like teamplay**

Biggest human concept gap. Example: partner opens 55, opponent plays 51 —
engine should continue the 1-side, not reconnect to 5-side.

### Option A: Tactical heuristic (fast)
- Penalize "collapsing newly opened end" when partner can keep it alive

### Option B: Auxiliary training targets (better long-term)
- Auxiliary heads: predict partner void probabilities, opponent voids,
  whether a move preserves/destroys information pressure

### Themes to cover
- Not stealing partner's suit
- Preserving pressure on newly opened end
- Probing for information
- Avoiding "locally OK, globally dumb" moves

### Acceptance test
- Curated suite of 50-200 partnership positions
- Track % of positions where engine picks human-correct coordination move

---

## Priority 3: League evaluation + duplicate deals + CI/SPRT promotion
**Expected upside: +10 to +20 Elo in real robustness; prevents regressions**

### What to do
- Duplicate deals in evaluation (already done ✓)
- SPRT / CI-based promotion, not fixed thresholds
- Maintain league/pool of past champions (not just latest vs current)
- Track winrate vs: heuristic-only, MC-expert, previous champion, 3-5 older champs

### Acceptance test
- Promotion decisions stabilize
- No "improved in training, regressed in browser" incidents
- Wider opponent-pool winrate improves

---

## Summary

| Priority | Change | Est. Elo |
|----------|--------|----------|
| 1 | Visit-count policy + PUCT prior | +10 to +30 |
| 2 | Partnership signaling features | +10 to +25 |
| 3 | League eval + SPRT | +10 to +20 |

Total potential: +30 to +75 Elo on top of current ~+20-30 baseline.
Realistic ceiling after all three: ~+80-120 Elo over heuristic.

---

## Current state (post-Gen50 baseline)
- Value head: useful and improving
- Policy head: still weak link — causes search to plateau or degrade at high sims
- Phase transition crossed: gen50 prefers deeper search (unlike gen15 which preferred shallow)
- Forced move %: ~47-58% — data bottleneck co-present (many trivial positions)
