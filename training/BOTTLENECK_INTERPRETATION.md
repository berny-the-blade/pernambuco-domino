# BOTTLENECK_INTERPRETATION.md
# How to interpret post-Gen50 eval results

Written 2026-03-07. Decision framework for reading search_scaling_eval + particle_disagreement_eval outputs.

---

## Decision Tree (quick)

```
Does strength improve with more sims?
  YES  --> SEARCH BOTTLENECK
  NO   --> Particle disagreement high?
             YES --> BELIEF BOTTLENECK
             NO  --> NETWORK BOTTLENECK

Policy loss flat AND forced move % high --> DATA BOTTLENECK
```

---

## 1. Search Bottleneck

Strength clearly rises with sims. Targets still moving.

| Indicator | Pattern |
|-----------|---------|
| Margin (50→400 sims) | clear upward trend (+0.03 to +0.16) |
| Top1 agreement (100v200, 200v400) | 82–86% |
| Particle disagreement | moderate |
| Policy loss | flat |

**Fix**: increase sims (200→400) or add reanalysis pass.

---

## 2. Belief Bottleneck (hidden-info noise)

Very likely in 4-player partnership domino. Search depth barely matters.

| Indicator | Pattern |
|-----------|---------|
| Margin (50→400 sims) | ~flat (53% → 55%) |
| Non-forced top1 agreement | 60–70% |
| Pairwise JSD | high |
| Policy loss | flat |

**Fix**: stronger belief model — pass-derived constraints, tile scarcity inference, particle weighting, partner inference.

---

## 3. Network Bottleneck

Search produces good targets but network can't represent them.

| Indicator | Pattern |
|-----------|---------|
| Margin (50→400 sims) | barely changes (55% → 56.2%) |
| Non-forced top1 agreement | 85–95% |
| Root targets | stable by 200 sims |
| Policy loss | still moderately high |

**Fix**: better encoder, larger network, richer features, longer training.

---

## 4. Data Bottleneck

Low-information states dominate replay. Common in domino.

| Indicator | Pattern |
|-----------|---------|
| Forced move % | >40% |
| Avg legal moves | ~2 |
| Policy loss | flat (trivial supervision) |

**Fix**: bias training toward non-forced, high-entropy states; downweight trivial positions.

---

## Prediction for this run (Bernie, 2026-03-07)

> 60% Search Bottleneck / 30% Belief Bottleneck / 10% other

Rationale:
- value loss improving (network is learning something)
- policy loss flat (search quality likely the ceiling)
- 200 sims is modest for 4-player hidden-info partnership game
- forced move % already tracking high (47-58% in smoke tests) — data bottleneck co-present

---

## Priority metrics to check after Gen50

1. **Gen50 vs Gen46 duplicate margin at 200 sims** — is the run doing anything at all?
2. **200 vs 400 sims margin difference** — is search limiting?
3. **Non-forced top1 agreement** — is belief the issue?

---

## Rule: don't change anything until after Gen50 analysis

No architecture, training params, belief model, or sim count changes until clean eval data is in hand.

---

## Gen 50 Post-Run Analysis (2026-03-07)

### Anchor Eval Results
| Opponent | Win% | ELO | Verdict |
|----------|------|-----|---------|
| vs gen 46 | 50.7% | +5 | flat |
| vs gen 1 | 49.8% | -2 | flat |
| vs gen 5 | 52.5% | +17 | improved |
| vs gen 10 | 51.5% | +10 | improved |

### Search Scaling (200 pairs each, gen50 vs gen46)
| Sims | Margin | Win% |
|------|--------|------|
| 50 | -0.048 | ~47% |
| 100 | +0.048 | ~53% |
| 200 | ~+0.030-0.130 | ~52-56% (in progress) |
| 400 | (pending) | - |

### Key Interpretation (Bernie)
- Training did NOT stall � shifted bottleneck from model -> search budget
- Phase transition crossed: old models preferred shallow search, gen50 prefers deeper search
- This is exactly what AlphaZero-style systems should do once value head is meaningful
- Estimated strength: current ~+20-30 Elo; after search scaling ~+40-70; after partner-awareness ~+80-120

### Next Bottleneck: Partner Signaling
- cantHave/belief probabilities are tracked, but NO convention inference
- Network not trained to reason about partner signals (e.g. suit strength, protecting dorme)
- Fix: auxiliary head predicting partner tiles, or symmetry/Other-Play style training
- Not urgent until search scaling is fully exploited
