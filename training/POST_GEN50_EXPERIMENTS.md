# Post-Gen50 Experiment Plan

> From analysis session 2026-03-07. Run these after gen 50 completes.

---

## Experiment 1 — Search Scaling

Run Gen50 vs Gen46 at 50 / 100 / 200 / 400 sims.
Same seeds, duplicate deals, ≥ 400 games per setting (800 preferred).

| Sims | Winrate vs 46 | Duplicate Margin | Root Entropy | Top1 Mass |
|------|--------------|-----------------|--------------|-----------|
| 50   |              |                 |              |           |
| 100  |              |                 |              |           |
| 200  |              |                 |              | (current) |
| 400  |              |                 |              |           |

**Search bottleneck likely if:**
- 50→200 sims improves winrate ≥ 3pp, OR
- 100→400 sims improves duplicate margin > 0.05 pts/deal

**Search probably NOT bottleneck if:**
- 100→400 sims changes winrate < 1.5pp
- Entropy/top1 barely change

Script: `search_scaling_eval.py` (to build)

---

## Experiment 2 — Root Target Stability

200 fixed states from replay. Run search at 100 / 200 / 400 sims each.
Compare visit distributions.

| Comparison | Top1 Agreement | Mean JSD | Mean Entropy Change |
|------------|---------------|----------|-------------------|
| 100 vs 200 |               |          |                   |
| 200 vs 400 |               |          |                   |

**Targets unstable / search-limited if:**
- 200 vs 400 top-1 agreement < 85%, OR
- JSD still materially high, OR
- Entropy still drops a lot from 200→400

**Targets mature if:**
- 200 vs 400 top-1 agreement ≥ 90–95%
- Entropy/top-2 gap barely move

Script: `target_stability_eval.py` (to build)

---

## Experiment 3 — Anchor Strength

Already running via `anchor_eval.py`. Run at gen 50 with 400+ games.

| Matchup      | Winrate | Duplicate Margin | 95% CI |
|--------------|---------|-----------------|--------|
| Gen50 vs Gen1  |         |                 |        |
| Gen50 vs Gen10 |         |                 |        |
| Gen50 vs Gen46 |         |                 |        |

**Healthy if:**
- Gen50 vs Gen1 > 60%
- Gen50 vs Gen46 > 52–55%

**Concerning if:**
- Gen50 vs Gen1 near coinflip
- Gen50 vs Gen46 flat despite better losses

---

## Experiment 4 — Particle Disagreement

200 fixed public states. For each: sample 8 particles, run search on each,
compare top action and visit distribution across particles.

| Metric                          | Value |
|---------------------------------|-------|
| Top1 agreement across particles |       |
| Mean JSD across particles       |       |
| States with strong disagreement |       |

**Belief instability is a major issue if:**
- Top-1 agreement across particles < 70–75%
- Many states show very different preferred actions across particles
→ Policy targets being smeared by hidden-world ambiguity

Script: `particle_disagreement_eval.py` (to build)

---

## Experiment 5 — Replay Quality Diagnostics

For gens 1, 10, 20, 30, 40, 50:

| Gen | Forced % | Avg Legal Count | Root Entropy | Top1 Mass | Avg Game Length |
|-----|----------|----------------|--------------|-----------|----------------|
| 1   |          |                |              |           |                |
| 10  |          |                |              |           |                |
| 20  |          |                |              |           |                |
| 30  |          |                |              |           |                |
| 40  |          |                |              |           |                |
| 50  |          |                |              |           |                |

**Data quality concern if:**
- Forced-move % very high (> 35%)
- Root entropy never sharpens
- Legal count low, most states trivial

Script: needs instrumentation in `orchestrator.py` / `vectorized_mcts.py`

---

## Decision Matrix

| Exp 1 | Exp 2 | Exp 4 | Diagnosis | Next move |
|-------|-------|-------|-----------|-----------|
| Big sim gain | Targets unstable | — | Search bottleneck | Increase sims, add reanalysis |
| Big sim gain | Targets stable | High disagreement | Belief bottleneck | Fix hidden-info inference |
| Flat sim gain | — | High disagreement | Belief bottleneck | Improve particle diversity |
| Flat sim gain | Targets stable | Low disagreement | Network/repr bottleneck | Better encoder, more capacity |
| All clean | — | Low disagreement | Healthy | Continue, add eval league |

---

## Case Summaries

### Case A — Search Bottleneck
- Strength rises with more sims
- Target stability poor at 200 sims
- Policy loss flat
- Particle disagreement moderate/high
→ Increase training sims, improve belief constraints, consider reanalysis

### Case B — Belief Bottleneck
- Large particle disagreement
- Public-state collisions common
- Deeper sims help only somewhat
- Targets unstable mainly across particles
→ Improve hidden-info inference, harder pass-history incorporation,
  split public-tree vs particle-tree modes

### Case C — Network/Representation Bottleneck
- Targets stable by 200 sims
- More sims barely help
- Anchor strength still weak
- Policy/value both underwhelming
→ Improve encoder, increase model capacity, re-balance training

### Case D — Mostly Healthy
- Gen50 clearly beats anchors
- Targets stable
- More sims help a bit but not massively
- Replay diagnostics normal
→ Continue training, add eval league, optimize throughput

---

## Minimal Deliverables After Gen50

These 5 outputs make the next step obvious:

1. **Search scaling table** (Exp 1)
2. **Target stability table** (Exp 2)
3. **Anchor eval table** (Exp 3) — already automated
4. **Particle disagreement summary** (Exp 4)
5. **Replay diagnostics table** (Exp 5)

---

## Scripts Status

| Script                          | Status | Command |
|---------------------------------|--------|---------|
| `anchor_eval.py`                | ✅ Done | `python anchor_eval.py --games 400` |
| `search_scaling_eval.py`        | ✅ Done | `python search_scaling_eval.py --model-a checkpoints/domino_gen_0050.pt --model-b checkpoints/domino_gen_0046.pt --sim-list 50,100,200,400 --deal-pairs 800 --seed-base 5000 --output results/search_scaling_gen50_vs_gen46.json` |
| `target_stability_eval.py`      | ✅ Done | `python target_stability_eval.py --states 200 --sims 100 200 400` |
| `particle_disagreement_eval.py` | ✅ Done | `python particle_disagreement_eval.py --model checkpoints/domino_gen_0050.pt --states 200 --particles 8 --sims 200 --phase-buckets early,mid,late --seed-base 9000 --output results/particle_disagreement_gen50.json` |
| Replay diagnostics              | TODO   | needs instrumentation in orchestrator.py |

### Results location
All outputs go to `results/` (JSON + CSV per run) and `logs/` (JSONL append logs).
