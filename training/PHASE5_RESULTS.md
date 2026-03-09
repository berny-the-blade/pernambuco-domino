# Phase 5 Post-Training Results

**Generated:** 2026-03-09 06:49  
**Updated with expert analysis:** 2026-03-09

## Checkpoints
- Gen 20: `training/checkpoints/domino_gen_0020.pt`
- Gen 50 (Phase 4 best): `training/checkpoints/domino_gen_0050.pt`
- Gen 100 (Phase 5 latest): `training/checkpoints/domino_gen_0100.pt`

---

## Post-Training Analysis Summary

### Search Scaling Results (Gen20 vs Gen50, 200 duplicate deal pairs each)

| Sims | Gen20 Win% | Verdict |
|------|-----------|---------|
| 50   | 48.2% | Gen20 LOSES |
| 100  | 55.8% | Gen20 wins |
| 200  | 51.5% | Gen20 wins |
| 400  | 53.2% | Gen20 wins |

**Verdict: SEARCH_BOTTLENECK_LIKELY**  
Non-monotone win curve: Gen20 is budget-dependent. Loses at 50 sims but wins at 100+.  
This means the deployed checkpoint MUST match the live sim budget.

### Anchor Eval (Gen20 vs historical, 400 games each)

| Anchor | Win% | ELO delta | Verdict |
|--------|------|-----------|---------|
| Gen 1  | 50.7% | +5.2  | flat (no progress vs Gen1) |
| Gen 5  | 47.8% | -15.6 | worse (likely noise) |
| Gen 10 | 49.5% | -3.5  | flat |
| Gen 46 | 54.2% | +29.6 | better ✓ |
| Gen 50 | 50.2% | +1.7  | flat |

**Key finding:** Gen20 is only clearly better than Gen46. Flat/noisy vs other anchors = Gen20 did not
reliably surpass Phase 4 quality across the board. Training signal is weak at 20 gens.

### Promotion Policy Analysis: 0 Rejections

The arena gate was explicitly removed in the Phase 5 orchestrator ("Arena gate REMOVED per Gemini
recommendation"). This means:
- **Every gen was auto-promoted** — "champion = latest", not "champion = best"
- The 0-rejection count is an artifact of the disabled gate, not a sign of consistent improvement
- Real arena evaluations (SPRT with 100–600 seeds) were never run during Phase 5

---

## Live Browser Sim Budget

**Finding from `index.html` review:**
```javascript
const MAX_ITERATIONS = 600;
const TIME_LIMIT = 300;   // ms
```
The browser uses **wall-clock limited** ISMCTS (300ms cutoff, max 600 iterations).  
This is NOT a fixed simulation count — it depends on device hardware.

**Python equivalent:** ~100–200 sims on a modern laptop/desktop.  
**Deployment recommendation:** Use `best_100sims.pt` as primary, `best_200sims.pt` as secondary.

---

## Expert Recommendations (Ordered Priority)

### 1. ✅ Budget-Specific Champion Selection (IMPLEMENTED)

**Problem:** `domino_model.bin` is always overwritten by the latest gen, which may not be best at
the deployment sim budget (100–200 sims).

**Fix implemented in `orchestrator.py`:**
- Added `_track_budget_checkpoints()` method
- After each auto-promoted gen, runs a quick 50-pair eval at each of `[50, 100, 200]` sims
- Saves `checkpoints/best_50sims.pt`, `best_100sims.pt`, `best_200sims.pt` if new gen wins
- Deploy `best_100sims.pt` for the browser (closest to TIME_LIMIT=300ms budget)

**Immediate action required:**
```bash
# Run Gen20 vs Gen50 at 100 sims with 400+ pairs (larger sample, more confidence):
python training/search_scaling_eval.py \
  --model-a training/checkpoints/domino_gen_0020.pt \
  --model-b training/checkpoints/domino_gen_0050.pt \
  --sim-list 100,200 \
  --deal-pairs 400
# Then deploy the winner as domino_model.bin
```

### 2. ✅ Phase 6 Belief-Head Probe (IMPLEMENTED)

**Problem:** Non-monotone search scaling and flat anchor evals suggest the network's internal
representation is the bottleneck — not search depth alone.

**Fix implemented in `phase6_probe.py`:**
- Runs 3 independent 5-gen training probes (λ=0.1, 0.2, 0.3)
- All start from the same base checkpoint (prefers `best_100sims.pt`)
- Evaluates each probe on: partnership suite + search scaling + anchor at 100 sims
- Writes results to `training/logs/phase6_probe_results.json`
- Recommends best λ based on composite score

**To run:**
```bash
python training/phase6_probe.py
# Optional: dry run to test eval pipeline
python training/phase6_probe.py --dry-run
```

**Success criteria:**
- Partnership suite improves ≥ +0.05 vs baseline (0.462)
- Search scaling at 100 sims improves or becomes more monotone
- Anchor eval (vs Gen50 at 100 sims) shows ELO Δ > +20

### 3. Mixed-Budget Training (LATER — only if belief head helps)

**Scope:** NOT a global jump to 400/800 sims.  
**Schedule:** 80–90% at 200 sims, 10–20% at 400 sims.  
**When:** Only after Phase 6 belief probe shows measurable improvement.

The `orchestrator.py` already supports `--high-sim-fraction` and `--high-sim-multiplier` for this.

---

## Revised Next Steps Checklist

- [ ] **Deploy best gen for live game:** Run Gen20 vs Gen50 at 100 sims (400 pairs), export winner as `domino_model.bin`
- [ ] **Run Phase 6 probe:** `python training/phase6_probe.py` — 3×5-gen probes to find best λ_belief
- [ ] **Review probe results** at `training/logs/phase6_probe_results.json`
- [ ] **If belief helps:** Run Phase 6 full training (20–30 gens) with best λ from probe
- [ ] **After Phase 6:** Re-run search scaling + anchor eval to confirm improvement
- [ ] **Mixed-budget training:** Only if belief head improves AND scaling is still non-monotone

---

## Files
- Search scaling results: `training/logs/search_scaling_phase5.json`
- Phase 6 probe results: `training/logs/phase6_probe_results.json` (written after probe run)
- Phase 6 plan: `training/PHASE6_BELIEF_PLAN.md`
- Phase 6 patch guide: `training/PHASE6_PATCH_GUIDE.md`
- Budget tracking script: `training/phase6_probe.py`
