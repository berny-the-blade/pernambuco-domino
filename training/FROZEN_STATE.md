# FROZEN STATE — 2026-03-10
_Do not rename, move, or overwrite these files._

---

## 1. Production Champion

**Currently live in game. Do NOT replace until Phase 8 promotion gate passes.**

| Label | Path | Note |
|-------|------|------|
| `best_prod_current` | `training/checkpoints/domino_gen_0050.pt` | Phase 4/5 MCTS champion; used as gen50 anchor in all scaling tests |
| Deployed binary | `domino_model.bin` | Browser export of Phase 5 champion |

**This is the bar.** Every new checkpoint must beat this at the live deployment budget to be promoted.

---

## 2. Architecture Winner — Phase 6.5 Probe B gen3

**Architecture probe winner. Starting point for continuation. NOT yet production champion.**

| Label | Path | Note |
|-------|------|------|
| `best_phase65_probeB_gen3` | `checkpoints/phase65_probe_B/final_gen3.pt` | Probe B: belief+support balanced (λ=0.1/0.1), gen 3 |

Suite at gen3: avg **0.772**, confirm_partner_signal **0.833**, all gates ✅  
Arena vs gen50 at gen3: **46.8%** (loses — expected at this early stage)  
Arena vs gen100 at gen3: **53.7%** (beats — positive signal)

---

## 3. Phase 6.5 Probe B Continuation — Best Checkpoints

**Continuation run of Probe B, 6 gens from gen3 starting point. Best-per-sim-budget saved.**

| Label | Path | Gen | Note |
|-------|------|-----|------|
| `best_phase65_cont_50sims` | `checkpoints/best_50sims.pt` | 6 | Best checkpoint at 50-sim budget |
| `best_phase65_cont_100sims` | `checkpoints/best_100sims.pt` | 6 | Best checkpoint at 100-sim budget |
| `best_phase65_cont_200sims` | `checkpoints/best_200sims.pt` | 6 | Best checkpoint at 200-sim budget |

Suite at gen5: avg **0.676** (mcts100), **0.654** (greedy)  
⚠️ Gen6 suite scores NOT yet measured. Run before Phase 8 Step B.  
Head-to-head vs gen50: **NOT YET RUN.** This is the open gate for Phase 8.

---

## 4. Phase 7 Final (Phase 6 MVP — Vetoed)

**Continuation of old belief-head architecture. Verdict: do not deploy.**

| Label | Path | Note |
|-------|------|------|
| `best_phase7_final_gen20` | `training/checkpoints/domino_gen_0020.pt` | Phase 7, 20 gens, belief-head λ=0.1 only |

Arena vs gen50: **48.0%** at 200 sims — does not beat production champion.  
Partnership suite: **0.559** (below 0.700 threshold). Vetoed by PHASE6_VERDICT.md.

---

## Summary

| Role | Checkpoint | Status |
|------|-----------|--------|
| Production champion | `training/checkpoints/domino_gen_0050.pt` | ✅ Live |
| Architecture winner | `checkpoints/phase65_probe_B/final_gen3.pt` | Research branch |
| Best continuation | `checkpoints/best_100sims.pt` (gen6) | Awaiting Phase 8 test |
| Phase 6 MVP | `training/checkpoints/domino_gen_0020.pt` | ❌ Vetoed |

**Probe B is the architecture to continue. It is not yet the production champion.**
