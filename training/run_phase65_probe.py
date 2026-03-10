"""
Phase 6.5 Day 2 Probe - 3-variant sweep

Runs three 3-gen probes from the same starting checkpoint, all with
aux_detach=True and the Phase 6.5 conditioned architecture.

Probe A - support-only conditioning
  belief_weight=0.0, support_weight=0.1

Probe B - belief + support, balanced (main candidate)
  belief_weight=0.1, support_weight=0.1

Probe C - belief + support, stronger support signal
  belief_weight=0.1, support_weight=0.2

After each probe, runs:
  - Partnership suite
  - Quick arena vs Gen50 (50 pairs at 100 sims)

Results written to: training/logs/phase65_probe_results.json

Usage:
    python training/run_phase65_probe.py --checkpoint checkpoints/<phase5_champ>.pt
    python training/run_phase65_probe.py --checkpoint checkpoints/<phase5_champ>.pt --generations 5
"""

import argparse
import json
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
import sys
import shutil
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from domino_net import DominoNet
from domino_trainer import Trainer, ReplayDataset
from orchestrator import (
    Orchestrator, build_belief_target, build_support_target,
    arena_match, TILES,
)
from domino_env import DominoEnv, DominoMatch
from domino_encoder import DominoEncoder
from domino_mcts import DominoMCTS

SUITE_PATH = Path(__file__).parent / "tests" / "partnership_suite.json"
LOG_PATH   = Path(__file__).parent / "logs" / "phase65_probe_results.json"
BASELINE_SUITE_SCORE = 0.684   # Phase 5 Gen20 baseline


# -----------------------------------------------------------------------------
# Probe configuration
# -----------------------------------------------------------------------------

PROBES = [
    {
        "name": "A",
        "label": "support-only",
        "belief_weight": 0.0,
        "support_weight": 0.1,
        "belief_head": False,
        "support_head": True,
        "aux_detach": True,
    },
    {
        "name": "B",
        "label": "belief+support balanced",
        "belief_weight": 0.1,
        "support_weight": 0.1,
        "belief_head": True,
        "support_head": True,
        "aux_detach": True,
    },
    {
        "name": "C",
        "label": "belief+support strong-support",
        "belief_weight": 0.1,
        "support_weight": 0.2,
        "belief_head": True,
        "support_head": True,
        "aux_detach": True,
    },
]


# -----------------------------------------------------------------------------
# Partnership suite evaluation
# -----------------------------------------------------------------------------

def run_suite(model, device, sims=0):
    """Run the partnership suite against model. Returns full report dict."""
    try:
        sys.path.insert(0, str(SUITE_PATH.parent))
        from test_partnership_suite import evaluate_suite, make_engine_fn
        engine_fn = make_engine_fn(model, sims=sims, device=str(device))
        return evaluate_suite(engine_fn, SUITE_PATH)
    except Exception as e:
        print(f"  [Suite error: {e}]")
        return None


def suite_summary(report):
    if report is None:
        return {"avg": None, "confirm_partner_signal": None, "preserve_pressure": None}
    return {
        "avg": round(report["avg_score"], 4),
        "confirm_partner_signal": round(
            report["theme_avg"].get("confirm_partner_signal", 0.0), 4),
        "preserve_pressure": round(
            report["theme_avg"].get("preserve_pressure", 0.0), 4),
        "theme_avg": {k: round(v, 4) for k, v in report["theme_avg"].items()},
    }


# -----------------------------------------------------------------------------
# Quick arena: probe checkpoint vs Gen50 reference at fixed sims
# -----------------------------------------------------------------------------

def quick_arena(probe_weights, ref_weights, num_pairs=50, sims=100, seed_base=70000):
    """Duplicate-deal arena. Returns challenger win rate."""
    device = torch.device("cpu")

    probe = DominoNet().to(device)
    probe.load_state_dict(probe_weights, strict=False)
    probe.eval()

    ref = DominoNet().to(device)
    ref.load_state_dict(ref_weights, strict=False)
    ref.eval()

    mcts_p = DominoMCTS(probe, num_simulations=sims)
    mcts_r = DominoMCTS(ref,   num_simulations=sims)

    wins_p = 0
    total  = 0
    enc_p  = DominoEncoder()
    enc_r  = DominoEncoder()

    with torch.no_grad():
        for i in range(num_pairs):
            seed = seed_base + i
            for p_side in [0, 1]:
                env = DominoEnv()
                enc_p.reset(); enc_r.reset()
                obs = env.reset(seed=seed)
                step = 0
                while not env.is_over() and step < 200:
                    mask = env.get_legal_moves_mask()
                    if mask.sum() == 0:
                        break
                    team = env.current_player % 2
                    if team == p_side:
                        pi = mcts_p.get_action_probs(env, enc_p, temperature=0.1)
                    else:
                        pi = mcts_r.get_action_probs(env, enc_r, temperature=0.1)
                    action = int(np.argmax(pi * mask))
                    obs, _, done, _ = env.step(action)
                    step += 1
                if env.game_over and env.winner_team == p_side:
                    wins_p += 1
                total += 1

    return round(wins_p / total, 4) if total > 0 else 0.5


# -----------------------------------------------------------------------------
# Run a single probe variant
# -----------------------------------------------------------------------------

def run_probe(probe_cfg, start_checkpoint, generations, workers, games_per_worker,
              mcts_sims, ref_weights, device):
    name  = probe_cfg["name"]
    label = probe_cfg["label"]
    print(f"\n{'='*60}")
    print(f"  PROBE {name}: {label}")
    print(f"  belief_weight={probe_cfg['belief_weight']}  "
          f"support_weight={probe_cfg['support_weight']}  "
          f"aux_detach={probe_cfg['aux_detach']}")
    print(f"{'='*60}")

    probe_dir = Path("checkpoints") / f"phase65_probe_{name}"
    probe_dir.mkdir(parents=True, exist_ok=True)

    orch = Orchestrator(
        num_workers=workers,
        buffer_size=50000,
        use_mcts=True,
        mcts_sims=mcts_sims,
        value_target="me",
        policy_target="visits",
        high_sim_fraction=0.0,   # no high-sim games in probes - keep it fast
        use_belief_head=probe_cfg["belief_head"],
        belief_weight=probe_cfg["belief_weight"],
        use_support_head=probe_cfg["support_head"],
        support_weight=probe_cfg["support_weight"],
        aux_detach=probe_cfg["aux_detach"],
    )
    orch.load_checkpoint(start_checkpoint)

    # Override checkpoint save dir for this probe
    original_ckpt_dir = None  # orchestrator saves to ./checkpoints/
    # We'll rename after run

    orch.run(total_generations=generations, games_per_worker=games_per_worker)

    # Grab final checkpoint
    final_ckpt = Path("checkpoints") / f"domino_gen_{generations:04d}.pt"
    probe_ckpt = probe_dir / f"final_gen{generations}.pt"
    if final_ckpt.exists():
        shutil.copy2(final_ckpt, probe_ckpt)
        print(f"  Saved probe checkpoint: {probe_ckpt}")
    else:
        # Find latest checkpoint
        ckpts = sorted(Path("checkpoints").glob("domino_gen_*.pt"))
        if ckpts:
            shutil.copy2(ckpts[-1], probe_ckpt)
            print(f"  Saved probe checkpoint (latest): {probe_ckpt}")

    # -- Evaluate ----------------------------------------------------------
    orch.model.eval()
    probe_weights = {k: v.cpu().clone() for k, v in orch.model.state_dict().items()}

    print(f"\n  Running partnership suite...")
    suite_report = run_suite(orch.model, device, sims=0)  # greedy NN for speed
    suite = suite_summary(suite_report)
    print(f"  Suite avg: {suite['avg']}  "
          f"confirm_signal: {suite['confirm_partner_signal']}  "
          f"preserve_pressure: {suite['preserve_pressure']}")

    print(f"\n  Running quick arena vs Gen50 (100 sims, 50 pairs)...")
    arena_wr = quick_arena(probe_weights, ref_weights, num_pairs=50, sims=100)
    print(f"  Arena win rate vs Gen50: {arena_wr:.1%}")

    # -- Pass gate ---------------------------------------------------------
    suite_ok     = suite["avg"] is not None and suite["avg"] >= BASELINE_SUITE_SCORE - 0.05
    signal_ok    = (suite["confirm_partner_signal"] or 0.0) > 0.0
    pressure_ok  = (suite["preserve_pressure"] or 0.0) > 0.0
    arena_ok     = arena_wr >= 0.48
    advances     = suite_ok and arena_ok and (signal_ok or pressure_ok)

    print(f"\n  Gates: suite_ok={suite_ok} signal_ok={signal_ok} "
          f"pressure_ok={pressure_ok} arena_ok={arena_ok}")
    print(f"  => {'ADVANCES to Day 3' if advances else 'DOES NOT advance'}")

    return {
        "probe": name,
        "label": label,
        "config": probe_cfg,
        "generations": generations,
        "suite": suite,
        "arena_win_rate_vs_gen50_100sims": arena_wr,
        "gate": {
            "suite_ok": suite_ok,
            "signal_ok": signal_ok,
            "pressure_ok": pressure_ok,
            "arena_ok": arena_ok,
            "advances": advances,
        },
        "checkpoint": str(probe_ckpt),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 6.5 Day 2 Probe Sweep")
    parser.add_argument("--checkpoint", required=True,
                        help="Starting checkpoint (Phase 5 champion)")
    parser.add_argument("--ref-checkpoint", default=None,
                        help="Reference checkpoint for arena (default: best_100sims.pt or Gen50)")
    parser.add_argument("--generations", type=int, default=3,
                        help="Generations per probe (default: 3)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Self-play workers (default: 10)")
    parser.add_argument("--games-per-worker", type=int, default=20,
                        help="Games per worker per gen (default: 20)")
    parser.add_argument("--mcts-sims", type=int, default=50,
                        help="MCTS sims per move (default: 50)")
    parser.add_argument("--probes", nargs="+", default=["A", "B", "C"],
                        help="Which probes to run (default: A B C)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load reference for arena
    ref_path = args.ref_checkpoint
    if ref_path is None:
        candidates = [
            "checkpoints/best_100sims.pt",
            "checkpoints/domino_gen_0050.pt",
        ]
        for c in candidates:
            if Path(c).exists():
                ref_path = c
                break
    if ref_path is None:
        print("ERROR: no reference checkpoint found. Pass --ref-checkpoint.")
        sys.exit(1)

    print(f"Reference checkpoint: {ref_path}")
    ref_ckpt    = torch.load(ref_path, map_location="cpu", weights_only=True)
    ref_weights = ref_ckpt.get("model_state_dict", ref_ckpt)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    t_start = time.time()

    for probe_cfg in PROBES:
        if probe_cfg["name"] not in args.probes:
            continue
        result = run_probe(
            probe_cfg=probe_cfg,
            start_checkpoint=args.checkpoint,
            generations=args.generations,
            workers=args.workers,
            games_per_worker=args.games_per_worker,
            mcts_sims=args.mcts_sims,
            ref_weights=ref_weights,
            device=device,
        )
        results.append(result)

    # -- Final summary ------------------------------------------------------
    print(f"\n{'='*60}")
    print("  PHASE 6.5 DAY 2 PROBE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Probe':<8} {'Suite':>8} {'Signal':>8} {'Pressure':>10} {'Arena':>8} {'Advances':>10}")
    print("-" * 60)
    for r in results:
        s = r["suite"]
        print(f"  {r['probe']:<6} {str(s['avg']):>8} "
              f"{str(s['confirm_partner_signal']):>8} "
              f"{str(s['preserve_pressure']):>10} "
              f"{r['arena_win_rate_vs_gen50_100sims']:>8.1%} "
              f"{'YES' if r['gate']['advances'] else 'no':>10}")

    advancing = [r for r in results if r["gate"]["advances"]]
    if advancing:
        best = max(advancing, key=lambda r: (r["suite"]["avg"] or 0) + r["arena_win_rate_vs_gen50_100sims"])
        print(f"\n  Best advancing probe: {best['probe']} ({best['label']})")
        print(f"  => Run Day 3 full eval on: {best['checkpoint']}")
    else:
        print("\n  No probes advanced. Review results and consider abort criteria.")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/3600:.1f}h")

    # Write results
    with open(LOG_PATH, "w") as f:
        json.dump({"probes": results, "baseline_suite": BASELINE_SUITE_SCORE}, f, indent=2)
    print(f"Results saved: {LOG_PATH}")


if __name__ == "__main__":
    main()


