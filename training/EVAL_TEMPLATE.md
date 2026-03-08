# Domino AI Training Run — Evaluation Template

Use this table after gens 10, 20, 30, 40, 50.

| Gen | Train Loss | Value Loss | Policy Loss | Replay Size | Games/Gen | Avg Game Length | Root Entropy | Forced Move % | Winrate vs Gen46 | Winrate vs Gen1 | Duplicate Margin |
|-----|------------|------------|-------------|-------------|-----------|-----------------|--------------|---------------|------------------|-----------------|------------------|
| 10  |            |            |             |             |           |                 |              |               |                  |                 |                  |
| 20  |            |            |             |             |           |                 |              |               |                  |                 |                  |
| 30  |            |            |             |             |           |                 |              |               |                  |                 |                  |
| 40  |            |            |             |             |           |                 |              |               |                  |                 |                  |
| 50  |            |            |             |             |           |                 |              |               |                  |                 |                  |

---

## Known data points (Phase 4)

### Gen 15 anchor eval (2026-03-07, greedy play, 200 games per matchup)

| Anchor | Win% | 95% CI     | ELO d | Verdict |
|--------|------|------------|-------|---------|
| gen 46 | 50.0% | [43.1–56.9] | ±0   | flat    |
| gen 1  | 49.5% | [42.6–56.4] | −3.5 | flat    |
| gen 5  | 50.5% | [43.6–57.4] | +3.5 | flat    |
| gen 10 | 49.5% | [42.6–56.4] | −3.5 | flat    |

**Interpretation:** Gen 15 is statistically indistinguishable from all anchors.
Training is stable but no measured strength improvement yet.

---

## Column Definitions

### Train / Value / Policy Loss
From training logs (epoch 5 value each gen).

**Watch for:**
- Value loss: steady decline = good calibration
- Policy loss: flat = search targets not improving (key warning sign)

### Replay Size
Buffer cap = 200k samples. Verify stable.

### Games / Gen
Expected: 31 workers × 100 games = 3100. Verify stable.

### Avg Game Length
- Increasing = deeper play
- Decreasing = quicker finishes

### Root Entropy
| Value  | Meaning          |
|--------|-----------------|
| < 1.0  | very decisive   |
| 1.0–2.0 | normal         |
| > 2.5  | confused search |

### Forced Move %
Fraction of positions with only 1 legal move.
Typical domino range: 10–30%. Too high = training dominated by trivial states.

### Winrate vs Gen46 (primary strength signal)
400–1000 games, duplicate deals, fixed seeds.
- Gen 10: expect 51–53% if healthy
- Gen 20: expect 54–57%
- Gen 30: expect 57–60%
- Stays ~50% = training not improving

### Winrate vs Gen1 (sanity anchor)
Expected > 60% by gen 30. If not, training is stuck.

### Duplicate Deal Margin
Points scored difference per deal (e.g. +0.18 pts/deal).
Lower variance than win rate — better signal.

---

## Evaluation Procedure (every 10 gens)

1. Freeze checkpoint: `domino_gen_00XX.pt`
2. Run: `python anchor_eval.py --current checkpoints/domino_gen_00XX.pt --games 400`
3. Log results into table above
4. Plot: Value Loss, Policy Loss, Winrate vs Gen46 (3 curves)

---

## Warning Signs

| Signal | Likely Cause | Fix |
|--------|-------------|-----|
| Value improves, strength flat | Value head overfitting self-play distribution | Stronger search, more sims, replay diversity |
| Policy loss flat throughout | Search too shallow / targets too noisy | Increase MCTS sims, improve data quality |
| Duplicate margin near zero | Model strength unchanged | Reassess pipeline |

---

## Healthy Example

| Gen | Train Loss | V Loss | P Loss | Win vs 46 |
|-----|------------|--------|--------|-----------|
| 10  | 0.445      | 0.027  | 0.420  | 52%       |
| 20  | 0.443      | 0.026  | 0.418  | 55%       |
| 30  | 0.441      | 0.025  | 0.415  | 58%       |
| 40  | 0.439      | 0.024  | 0.412  | 60%       |
| 50  | 0.437      | 0.023  | 0.408  | 63%       |

---

## Pipeline Stats (Phase 4 baseline)

- Self-play throughput: ~3100 games / 27 min ≈ 6900 games/day
- 50-gen run total: ~155k games generated
- Buffer: 200k samples, MCTS 200 sims (10% at 800)

---

## Next Decision Points (after gen 50)

Based on results, decide whether to change:
- MCTS sims (current: 200)
- Replay buffer size (current: 200k)
- Evaluation gating (currently: auto-promote)
- Belief constraints / hidden-info handling
