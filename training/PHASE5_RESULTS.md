# Phase 5 Post-Training Results

**Generated:** 2026-03-09 06:49

## Checkpoints
- Gen 20: `C:\Users\bernd\pernambuco-domino\training\checkpoints\domino_gen_0020.pt`
- Gen 50 (Phase 4 best): `C:\Users\bernd\pernambuco-domino\training\checkpoints\domino_gen_0050.pt`

## Search Scaling

Gen 20 vs Gen 50 | 200 duplicate deal pairs per sim level

| Sims | Gen20 Win% | CI 95%       | Margin | Verdict          |
|------|-----------|--------------|--------|------------------|
| 50   | 48.2%     | [43.4, 53.1] | -0.028 | Gen20 loses      |
| 100  | 55.8%     | [50.9, 60.5] | +0.120 | Gen20 wins       |
| 200  | 51.5%     | [46.6, 56.4] | +0.010 | Gen20 wins (weak)|
| 400  | 53.2%     | [48.4, 58.1] | +0.095 | Gen20 wins       |

**Verdict: SEARCH_BOTTLENECK_LIKELY**
Non-monotone curve. Gen20 needs minimum search budget (~100 sims) to outperform Gen50.

**Important deployment note:** The live game uses **time-limited ISMCTS (300ms / max 600 iter)** - not a fixed sim count.
On mobile hardware this translates to roughly 80-150 effective iterations per move.
At that budget, Gen20 is likely close to the 50-sim regime where it underperforms Gen50.
Consider deploying Gen50 for mobile until Gen20 is further trained or sim budget is increased.

## Expert Analysis Summary (received 2026-03-09)

1. "0 rejections" is NOT a success metric - promotion policy too permissive
2. Budget-specific champion selection needed: best_50sims.pt, best_100sims.pt, best_200sims.pt
3. Phase 6 belief probe (lambda=0.1/0.2/0.3) is higher ROI than bumping training sims
4. Mixed-sim schedule (80% at 200, 20% at 400) preferred over global bump

## Revised Next Steps

- [ ] Determine actual effective sim count on target mobile hardware
- [ ] Budget-specific champion selection (run Gen20 vs Gen50 at actual deploy budget)
- [ ] Run Phase 6 belief-head probe (lambda=0.1/0.2/0.3), 5-gen probes each
- [ ] Re-run anchor eval with 800+ games or duplicate deals
- [ ] Only then: mixed 200/400 sim training schedule

## Files
- Search scaling: `training/logs/search_scaling_phase5.json`
- Browser model: `domino_model_gen20.bin`

## Next Steps
- [ ] Check search-scaling monotonicity (goal: 50%→55%→57%→59%)
- [ ] If monotone → start Phase 6 probe (belief head λ=0.1/0.2/0.3)
- [ ] If non-monotone → diagnose (PUCT priors, more gens)
