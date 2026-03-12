"""
run_phase91_gen15.py -- Phase 9.1 final playoff for gen15 only.

gen15 was the sole candidate to pass the 100-sim elimination gate (52.4%).
This runs the full 50/100/200 sim playoff with paired margin scoring
on a fixed seed set.

Usage:
    python run_phase91_gen15.py
    python run_phase91_gen15.py --pairs 200   # faster
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_arena import run_arena
import torch

TRAINING_CKPTS = os.path.join(os.path.dirname(__file__), "checkpoints")
CHALLENGER = os.path.join(TRAINING_CKPTS, "domino_gen_0015.pt")
PRODUCTION = os.path.join(TRAINING_CKPTS, "domino_gen_0050.pt")

FIXED_SEED   = 42       # same 400 deals for every budget
PROMOTE_WR   = 0.52
PROMOTE_MARG = 0.0      # pair margin must be > 0


def divider(title=""):
    w = 62
    pad = (w - len(title) - 2) // 2
    print("\n" + "-" * pad + " " + title + " " + "-" * (w - pad - len(title) - 2))


def run(num_pairs=400):
    device = torch.device("cpu")
    results = {}

    print("\nPhase 9.1 Final Playoff: gen15 vs gen50 (production)")
    print("Pairs: %d  |  Fixed seed: %d  |  Budgets: 50/100/200 sims" % (
        num_pairs, FIXED_SEED))

    for sims in [50, 100, 200]:
        divider("@ %d SIMS" % sims)
        r = run_arena(
            CHALLENGER, PRODUCTION,
            num_pairs=num_pairs,
            num_sims=sims,
            seed_base=FIXED_SEED,   # same deals at every budget
            device=device,
            verbose=True,
        )
        results[sims] = r

    # Final verdict
    divider("FINAL VERDICT")
    r100 = results.get(100, {})
    r50  = results.get(50,  {})
    r200 = results.get(200, {})

    wr100   = r100.get("wr", 0)
    marg100 = r100.get("margin", 0)
    mlo100  = r100.get("margin_lo", 0)

    passes_wr     = wr100   >= PROMOTE_WR
    passes_margin = marg100 > PROMOTE_MARG
    margin_confident = mlo100 > 0   # lower CI bound positive

    print("\n  Budget    WR        Margin     CI_margin")
    print("  " + "-" * 50)
    for sims in [50, 100, 200]:
        r = results[sims]
        print("  %3dsims   %.1f%%  %s   %.4f     [%.4f, %.4f]" % (
            sims,
            r["wr"] * 100,
            "[OK]" if r["wr"] >= PROMOTE_WR else "[  ]",
            r["margin"],
            r["margin_lo"],
            r["margin_hi"],
        ))

    print()
    if passes_wr and passes_margin:
        if margin_confident:
            verdict = "[WIN] PROMOTE -- WR>=52%% and margin positive with CI above 0"
        else:
            verdict = "[WIN] PROMOTE -- WR>=52%% and margin positive (CI borderline)"
    elif passes_wr:
        verdict = "[!!] MARGINAL -- WR passes but margin not conclusive"
    else:
        verdict = "[NO] REJECT -- does not beat production at 100-sim budget"

    print("  " + verdict)

    if "[WIN]" in verdict or "[!!]" in verdict:
        print("\n  Next: python export_model.py --checkpoint %s --output domino_model.bin" % CHALLENGER)
        print("  Then deploy and archive gen50 as fallback.")
    else:
        print("\n  Recommendation: keep gen50 as production.")
        print("  Next experiment: support-summary architecture tweak on Phase 10.")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "phase91_gen15_final.json")
    with open(out, "w") as f:
        json.dump({
            "challenger": CHALLENGER,
            "production": PRODUCTION,
            "pairs": num_pairs,
            "seed": FIXED_SEED,
            "results": {str(k): {
                "wr": v["wr"], "wr_lo": v["wr_lo"], "wr_hi": v["wr_hi"],
                "margin": v["margin"], "margin_lo": v["margin_lo"], "margin_hi": v["margin_hi"],
            } for k, v in results.items()},
            "verdict": verdict,
        }, f, indent=2)
    print("\n  Saved: %s" % out)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=400)
    args = parser.parse_args()
    run(num_pairs=args.pairs)
