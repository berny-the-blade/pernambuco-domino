# Phase 6 Baseline — Gen20
_Frozen: 2026-03-09 | This is the pre-Phase-6 reference point. All Phase 6 results compare against this._

---

## 1. Deployed Champion

| Field | Value |
|-------|-------|
| Checkpoint | `checkpoints/domino_gen_0020.pt` |
| Params | 647,877 |
| Phase | Phase 5 Gen 20 |
| Deployed | `domino_model.bin` (live on GitHub Pages) |

**Deployment rationale (Day 1 results vs Gen50, 200 duplicate pairs each):**
| Budget | Gen20 Win% | CI 95% | Verdict |
|--------|-----------|--------|---------|
| 50 sims | 56.2% | [51.4, 61.0] | Gen20 wins |
| 100 sims | 50.5% | [45.6, 55.4] | Near-tied |
| 200 sims | 54.0% | [49.1, 58.8] | Gen20 wins |

---

## 2. Partnership Suite Baseline

_Run: 100 sims, 34 cases, Gen20_

| Metric | Value |
|--------|-------|
| **Overall avg score** | **0.684** |
| Total score | 23.25 / 34 |
| Preferred choices | 21 / 34 |
| Acceptable choices | 4 / 34 |
| Discouraged choices | 8 / 34 |
| Fallback choices | 1 / 34 |
| Pass threshold (0.700) | ❌ FAIL |

**Per-theme scores:**
| Theme | Score | Notes |
|-------|------:|-------|
| preserve_pressure | 1.000 | ✅ Perfect |
| preserve_new_end_forced_exception | 1.000 | ✅ Perfect |
| dont_steal_partner_suit_forced_exception | 1.000 | ✅ Perfect |
| double-side-control | 1.000 | ✅ Perfect |
| sacrifice_to_lock | 1.000 | ✅ Perfect |
| dont_rescue_opponent | 1.000 | ✅ Perfect |
| dont_steal_partner_suit | 0.833 | ✅ Strong |
| ad_hoc_teamplay | 0.667 | 🟡 OK |
| maintain_partner_options | 0.667 | 🟡 OK |
| preserve_new_end | 0.600 | 🟡 Weak |
| information_probe | 0.500 | 🟡 Weak |
| **tactical_override** | **0.417** | ❌ Weakest |
| **confirm_partner_signal** | **0.000** | ❌ Fails all |

**Key weaknesses for Phase 6 to target:**
- `confirm_partner_signal`: 0/3 — completely blind to partner signals
- `tactical_override`: 5/12 — struggles to override habits when tactics demand it
- `information_probe`: 1/2 — misses void-probing opportunities

---

## 3. Search Scaling Baseline

_Gen20 (A) vs Gen50 (B) | 200 duplicate deal pairs per level_

| Sims | Gen20 Win% | 95% CI | Margin | Source |
|------|-----------|--------|--------|--------|
| 50 | 56.2% | [51.4, 61.0] | +0.095 | Day 1 |
| 100 | 50.5% | [45.6, 55.4] | -0.040 | Day 1 |
| 200 | 54.0% | [49.1, 58.8] | +0.115 | Day 1 |
| **400** | **53.2%** | **[48.4, 58.1]** | **+0.110** | Day 2 |

**Verdict: SEARCH_BOTTLENECK_UNLIKELY** — flat sim response across all levels.
Network/data/belief quality is the bottleneck, not search depth.
This makes Phase 6 (belief head) the highest-ROI next step.

---

## 4. Anchor Snapshot

_Gen20 vs Gen10 | 200 duplicate pairs | 50/100/200 sims_

| Sims | Gen20 Win% | CI 95% | Margin |
|------|-----------|--------|--------|
| 50 | 52.2% | [47.4, 57.1] | +0.045 |
| 100 | 52.8% | [47.9, 57.7] | +0.090 |
| 200 | 47.5% | [42.6, 52.4] | -0.030 |

Gen20 slightly ahead of Gen10 at 50/100 sims, near-tied at 200.

---

## Phase 6 Targets (to beat this baseline)

For Phase 6 to be considered a pass, the belief-head variant must improve at least one of:

| Metric | Current | Target |
|--------|---------|--------|
| Partnership suite avg | 0.684 | ≥ 0.734 (+0.05) |
| confirm_partner_signal | 0.000 | > 0.000 (any improvement) |
| tactical_override | 0.417 | ≥ 0.500 |
| 100-sim win% vs Gen50 | 50.5% | ≥ 53.0% (+2.5pp) |

Without regression on metrics already at 1.000.
