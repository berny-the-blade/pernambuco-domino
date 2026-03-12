"""
run_phase91.py -- Phase 9.1 Checkpoint Playoff (full automation)

Follows the checklist:
  Step 1: Quick elimination at 100 sims (400 pairs)
  Step 2: Drop candidates with WR < 48%
  Step 3: Full playoff at 50 / 100 / 200 sims for survivors
  Step 4: Promotion rule (>=52% @ 100 sim)
  Step 5: Tiebreaker (100-sim WR, then partnership)
  Step 6: Optional confidence run if winner is close

Usage:
    python run_phase91.py           # full run
    python run_phase91.py --fast    # smoke test (20 pairs, skip 200sim)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_arena import run_arena
import torch

# ---- Checkpoint definitions -------------------------------------------------
TRAINING_CKPTS = os.path.join(os.path.dirname(__file__), "checkpoints")

PRODUCTION = os.path.join(TRAINING_CKPTS, "domino_gen_0050.pt")

CANDIDATES = [
    {"label": "gen07", "path": os.path.join(TRAINING_CKPTS, "domino_gen_0007.pt"), "partnership": 0.669},
    {"label": "gen15", "path": os.path.join(TRAINING_CKPTS, "domino_gen_0015.pt"), "partnership": 0.632},
    {"label": "gen19", "path": os.path.join(TRAINING_CKPTS, "domino_gen_0019.pt"), "partnership": 0.625},
]

# ---- Gates ------------------------------------------------------------------
ELIMINATE_WR      = 0.48
PROMOTE_WR        = 0.52
SECONDARY_50_WR   = 0.49
SECONDARY_200_WR  = 0.50
CLOSE_CALL_MARGIN = 0.02
CONFIDENCE_PAIRS  = 800


def divider(title=""):
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "-" * pad + " " + title + " " + "-" * (w - pad - len(title) - 2))
    else:
        print("\n" + "=" * w)


def run_phase91(pairs_elim=400, pairs_full=400, fast=False, seed=42):
    device = torch.device("cpu")  # DominoMCTS hardcodes CPU

    if fast:
        pairs_elim = 20
        pairs_full = 20

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    log = {"production": PRODUCTION, "candidates": [], "survivors": [], "winner": None}

    # ---- Step 1: Elimination round @ 100 sims -------------------------------
    divider("STEP 1 -- ELIMINATION @ 100 SIMS")
    print("  %d duplicate pairs per candidate" % pairs_elim)
    print("  Eliminate: WR < %.0f%%" % (ELIMINATE_WR * 100))

    for c in CANDIDATES:
        if not os.path.exists(c["path"]):
            print("\n  [SKIP] %s: not found" % c["label"])
            continue

        print("\n  Challenging %s (partnership %.3f)..." % (c["label"], c["partnership"]))
        r = run_arena(c["path"], PRODUCTION,
                      num_pairs=pairs_elim, num_sims=100,
                      duplicate_deals=True, seed_base=seed,
                      device=device, verbose=True)
        c["elim_100"] = r["wr"]

    # ---- Step 2: Elimination ------------------------------------------------
    divider("STEP 2 -- ELIMINATION RESULTS")
    survivors = []
    for c in CANDIDATES:
        if "elim_100" not in c:
            continue
        wr = c["elim_100"]
        if wr < ELIMINATE_WR:
            status = "[NO] ELIMINATED  (%.1f%% < %.0f%%)" % (wr * 100, ELIMINATE_WR * 100)
        else:
            status = "[OK] SURVIVES    (%.1f%% >= %.0f%%)" % (wr * 100, ELIMINATE_WR * 100)
            survivors.append(c)
        print("  %-10s  WR=%.1f%%  partnership=%.3f  %s" % (
            c["label"], wr * 100, c["partnership"], status))

    log["candidates"] = [
        {"label": c["label"], "partnership": c["partnership"], "elim_100": c.get("elim_100")}
        for c in CANDIDATES
    ]
    log["survivors"] = [c["label"] for c in survivors]

    if not survivors:
        divider("RESULT")
        print("  [NO] No survivors. All candidates eliminated.")
        print("  --> Keep production champion (gen50).")
        print("  --> Proceed with support-summary tweak on Phase 10.")
        _save(log, results_dir)
        return log

    print("\n  %d survivor(s): %s" % (len(survivors), [c["label"] for c in survivors]))

    # ---- Step 3: Full playoff for survivors ---------------------------------
    divider("STEP 3 -- FULL PLAYOFF")
    sim_budgets = [50, 100, 200] if not fast else [50, 100]

    for c in survivors:
        c["playoff"] = {}
        for sims in sim_budgets:
            print("\n  %s vs production @ %d sims (%d pairs)..." % (
                c["label"], sims, pairs_full))
            r = run_arena(c["path"], PRODUCTION,
                          num_pairs=pairs_full, num_sims=sims,
                          duplicate_deals=True, seed_base=seed + sims * 1000,
                          device=device, verbose=True)
            c["playoff"][sims] = r

    # ---- Step 4+5: Promotion rule + tiebreaker ------------------------------
    divider("STEP 4 -- PROMOTION VERDICT")

    promotable = []
    for c in survivors:
        wr_100 = c["playoff"].get(100, {}).get("wr", 0)
        wr_50  = c["playoff"].get(50,  {}).get("wr", 0)
        wr_200 = c["playoff"].get(200, {}).get("wr") if 200 in c["playoff"] else None

        passes_main    = wr_100 >= PROMOTE_WR
        passes_50      = wr_50  >= SECONDARY_50_WR
        passes_200     = wr_200 >= SECONDARY_200_WR if wr_200 is not None else True
        passes_partner = c["partnership"] >= 0.60

        checks = []
        checks.append("100sim %.1f%%  %s" % (wr_100 * 100, "[OK]" if passes_main else "[NO]"))
        checks.append(" 50sim %.1f%%  %s" % (wr_50  * 100, "[OK]" if passes_50   else "[!!]"))
        if wr_200 is not None:
            checks.append("200sim %.1f%%  %s" % (wr_200 * 100, "[OK]" if passes_200 else "[!!]"))
        checks.append("partn  %.3f  %s" % (c["partnership"], "[OK]" if passes_partner else "[!!]"))

        verdict = "[WIN] PROMOTABLE" if passes_main else "[NO] REJECTED"
        if passes_main:
            promotable.append(c)

        print("\n  %-10s  %s" % (c["label"], verdict))
        for ch in checks:
            print("    %s" % ch)

    # ---- Step 5: Tiebreaker -------------------------------------------------
    winner = None
    if len(promotable) > 1:
        divider("STEP 5 -- TIEBREAKER")
        promotable.sort(key=lambda c: (c["playoff"][100]["wr"], c["partnership"]), reverse=True)
        winner = promotable[0]
        runner = promotable[1]
        diff = winner["playoff"][100]["wr"] - runner["playoff"][100]["wr"]
        print("  Winner:    %s  %.1f%% @ 100sim" % (winner["label"], winner["playoff"][100]["wr"] * 100))
        print("  Runner-up: %s  %.1f%% @ 100sim" % (runner["label"], runner["playoff"][100]["wr"] * 100))
        print("  Gap:       %.1f%%" % (diff * 100))
    elif len(promotable) == 1:
        winner = promotable[0]

    # ---- Step 6: Confidence run if close ------------------------------------
    if winner and not fast:
        wr_100 = winner["playoff"][100]["wr"]
        margin_above_gate = wr_100 - PROMOTE_WR
        if margin_above_gate < CLOSE_CALL_MARGIN:
            divider("STEP 6 -- CONFIDENCE RUN (close call)")
            print("  %s is only %.1f%% above gate. Running %d pairs..." % (
                winner["label"], margin_above_gate * 100, CONFIDENCE_PAIRS))
            r = run_arena(winner["path"], PRODUCTION,
                          num_pairs=CONFIDENCE_PAIRS, num_sims=100,
                          duplicate_deals=True, seed_base=seed + 99999,
                          device=device, verbose=True)
            winner["confidence_run"] = r
            if r["wr"] >= PROMOTE_WR:
                print("  [OK] Confidence confirmed: %.1f%%" % (r["wr"] * 100))
            else:
                print("  [NO] Confidence FAILED: %.1f%% -- reverting to no promotion" % (r["wr"] * 100))
                winner = None

    # ---- Final result -------------------------------------------------------
    divider("FINAL RESULT")
    if winner:
        log["winner"] = winner["label"]
        print("\n  [WIN] PROMOTE: %s" % winner["label"])
        print("     100-sim WR:   %.1f%%" % (winner["playoff"][100]["wr"] * 100))
        print("     Partnership:  %.3f" % winner["partnership"])
        print("     Path:         %s" % winner["path"])
        print("\n  Next steps:")
        print("    1. python export_model.py --checkpoint %s --output domino_model.bin" % winner["path"])
        print("    2. Deploy domino_model.bin as new production model")
        print("    3. Archive old gen50 as fallback")
        print("    4. Then proceed with support-summary architecture tweak (Phase 10)")
    else:
        log["winner"] = None
        print("\n  [NO] No checkpoint beats production at the required margin.")
        print("     --> Keep production champion (gen50)")
        print("     --> Proceed with support-summary tweak on Phase 10 continuation")

    _save(log, results_dir)
    return log


def _save(log, results_dir):
    out = os.path.join(results_dir, "phase91_results.json")
    with open(out, "w") as f:
        json.dump(log, f, indent=2)
    print("\n  Results saved: %s" % out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",       action="store_true", help="Smoke test: 20 pairs, skip 200sim")
    parser.add_argument("--pairs-elim", type=int, default=400)
    parser.add_argument("--pairs-full", type=int, default=400)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_phase91(
        pairs_elim=args.pairs_elim,
        pairs_full=args.pairs_full,
        fast=args.fast,
        seed=args.seed,
    )
