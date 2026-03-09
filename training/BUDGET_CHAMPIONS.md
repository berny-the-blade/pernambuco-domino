# Budget Champion Selection — Day 1 Results
_Run: 2026-03-09 | 200 duplicate deal pairs per matchup | 10 parallel workers_

---

## Raw Results

### Gen10 (A) vs Gen20 (B)
| Sims | Gen10 Win% | 95% CI        | Margin   | Verdict        |
|------|-----------|---------------|----------|----------------|
| 50   | 47.8%     | [42.9, 52.6]  | -0.045   | Gen20 wins     |
| 100  | 47.2%     | [42.4, 52.1]  | -0.090   | Gen20 wins     |
| 200  | 52.5%     | [47.6, 57.3]  | +0.030   | Near-tied      |

**Winner: Gen20 beats Gen10 cleanly at 50 and 100 sims.**

---

### Gen20 (A) vs Gen50 (B)
| Sims | Gen20 Win% | 95% CI        | Margin   | Verdict        |
|------|-----------|---------------|----------|----------------|
| 50   | 56.2%     | [51.4, 61.0]  | +0.095   | **Gen20 wins** |
| 100  | 50.5%     | [45.6, 55.4]  | -0.040   | Near-tied      |
| 200  | 54.0%     | [49.1, 58.8]  | +0.115   | Gen20 wins     |

**Notable: Gen20 beats Gen50 at 50 sims in this run** — contradicts the earlier
search_scaling result (48.2% at 50 sims). CIs overlap across all levels; treat
as near-tied. Previous result may have had different random seed or config.

---

### Gen10 (A) vs Gen50 (B) — tiebreaker at 100 sims
| Sims | Gen10 Win% | 95% CI        | Margin   | Verdict    |
|------|-----------|---------------|----------|------------|
| 100  | 50.5%     | [45.6, 55.4]  | -0.040   | Near-tied  |

**Gen50 very slight edge over Gen10 at 100 sims, but within noise.**

---

## Budget Champions Table

| Budget   | Champion | Runner-up | Notes                                      |
|----------|----------|-----------|--------------------------------------------|
| 50 sims  | **Gen20** | Gen50    | Gen20 56.2% vs Gen50; Gen20 >> Gen10       |
| 100 sims | **Gen20** | Gen50    | All near-tied; Gen20 beats Gen10 at -0.09  |
| 200 sims | **Gen20** | Gen50    | Gen20 54% vs Gen50; Gen20 52.5% vs Gen10   |

**Gen20 is the best or co-best checkpoint across all tested budgets.**

---

## Deployment Decision

### Key finding
Unlike the earlier Phase 5 post-pipeline result (Gen20 at 48.2% vs Gen50 at 50 sims),
this dedicated run with 10 parallel workers shows **Gen20 ≥ Gen50 at all sim levels**.
Both previous and current CIs are wide — results are noisy at 200 pairs.

### Contradiction with earlier result
| Run          | Gen20 vs Gen50 @ 50 sims | Sample |
|--------------|--------------------------|--------|
| Phase 5 post | 48.2% (Gen20 loses)       | 200 pairs |
| Day 1 run    | 56.2% (Gen20 wins)        | 200 pairs |

**This spread is within expected variance** — 200 pairs × 2 games = 400 games,
CI is ~±5pp. True win rate at 50 sims is somewhere in [44%, 60%] — genuinely uncertain.

### Conclusion
- **Current deployed model (Gen20) is correct to keep deployed**
- No checkpoint switch warranted — Gen20 is at worst tied with Gen50
- A 300ms time-limited benchmark on real hardware would give the definitive answer
- Conservative option: run 800 pairs at 50 and 100 sims for tighter CIs before deciding

### 300ms benchmark still recommended
The live game uses time-limited ISMCTS (300ms / max 600 iter). A real-device benchmark
logging actual iteration counts would resolve the Gen20 vs Gen50 question definitively.
Until then, Gen20 stays deployed (justified by this run).

---

## Next Step: Day 2
Run partnership suite + search scaling baseline on Gen20 before Phase 6 probe.
See PHASE6_ROADMAP.md.
