"""
Phase 9.1 — Checkpoint Playoff

Challenge Phase 7 candidates against production champion (gen50) at multiple sim budgets.
Phase 7 gen20 was already tested (48% @ 200sims — failed).
This script tests gen 7, 15, 19 which were never individually evaluated.

Usage:
    python phase91_playoff.py
    python phase91_playoff.py --pairs 50 --sims 50,100,200

Pass gate: win rate > 52% at 100-sim budget (deployment budget).
Bonus gate: also wins at 200 sims (higher confidence).
"""

import argparse
import json
import os
import sys
import time
import random

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domino_net import DominoNet
from domino_env import DominoMatch
from domino_mcts import DominoMCTS
from orchestrator import safe_load_state_dict

TRAINING_CKPTS = os.path.join(os.path.dirname(__file__), "checkpoints")
PRODUCTION_CKPT = os.path.join(TRAINING_CKPTS, "domino_gen_0050.pt")

# Challengers: (label, path, known_partnership_score)
CHALLENGERS = [
    ("phase7_gen07", os.path.join(TRAINING_CKPTS, "domino_gen_0007.pt"), 0.669),
    ("phase7_gen15", os.path.join(TRAINING_CKPTS, "domino_gen_0015.pt"), 0.632),
    ("phase7_gen19", os.path.join(TRAINING_CKPTS, "domino_gen_0019.pt"), 0.625),
    ("phase7_gen20", os.path.join(TRAINING_CKPTS, "domino_gen_0020.pt"), 0.574),  # known: 48% @ 200sim
]

# Pass/fail gates
GATE_MAIN_SIM  = 100   # deployment budget
GATE_MAIN_WR   = 0.52  # must beat this to promote
GATE_BONUS_SIM = 200   # bonus confirmation
GATE_BONUS_WR  = 0.50  # just needs to not lose badly at 200


def load_model(path, device):
    net = DominoNet().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    incompat = safe_load_state_dict(net, ckpt["model_state_dict"], strict=False)
    if incompat.missing_keys:
        missing = [k for k in incompat.missing_keys if "aux_proj" not in k]
        if missing:
            print(f"  [warn] unexpected missing keys: {missing}")
    gen = ckpt.get("generation", "?")
    net.eval()
    return net, gen


def play_duplicate_pair(chall_mcts, ref_mcts, seed):
    """
    One duplicate pair: play the same deal twice with sides swapped.
    Returns (challenger_wins, total_matches).
    challenger_wins counts full matches won (first to 6 points).
    """
    results = []
    for swap in [False, True]:
        match = DominoMatch(target_points=6)
        obs = match.reset(seed=seed)
        done = False
        while not done:
            player = match.current_player()
            # Teams: challenger = team 0 (players 0,2), ref = team 1 (players 1,3)
            # When swapped: challenger = team 1
            if (not swap and player % 2 == 0) or (swap and player % 2 == 1):
                mcts = chall_mcts
            else:
                mcts = ref_mcts
            policy = mcts.get_policy(obs, temperature=0.0)
            action = int(np.argmax(policy))
            obs, reward, done, info = match.step(action)

        winner = info.get("winner", -1)
        # challenger_team = 0 if not swap else 1
        chall_team = 0 if not swap else 1
        results.append(1 if winner == chall_team else 0)

    return sum(results), len(results)


def eval_at_budget(chall_path, ref_path, num_sims, num_pairs, seed_base, device):
    print(f"    Loading models for {num_sims}-sim eval...", flush=True)
    chall_model, chall_gen = load_model(chall_path, device)
    ref_model,   ref_gen   = load_model(ref_path,   device)

    chall_mcts = DominoMCTS(chall_model, num_simulations=num_sims)
    ref_mcts   = DominoMCTS(ref_model,   num_simulations=num_sims)

    wins = 0
    total = 0
    for i in range(num_pairs):
        seed = seed_base + i * 7919
        w, t = play_duplicate_pair(chall_mcts, ref_mcts, seed)
        wins += w
        total += t
        if (i + 1) % 10 == 0:
            wr = wins / total if total > 0 else 0
            print(f"      pair {i+1}/{num_pairs}: {wins}/{total} ({wr:.1%})", flush=True)

    win_rate = wins / total if total > 0 else 0.0
    return win_rate, wins, total


def run_playoff(num_pairs=50, sim_budgets=(50, 100, 200), seed_base=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Production champion: {PRODUCTION_CKPT}")
    print(f"Pairs per budget: {num_pairs}")
    print(f"Sim budgets: {sim_budgets}")
    print("=" * 60)

    results = {}

    for label, chall_path, known_partnership in CHALLENGERS:
        if not os.path.exists(chall_path):
            print(f"\n[SKIP] {label}: file not found at {chall_path}")
            continue

        print(f"\n{'='*60}")
        print(f"CHALLENGER: {label}  (known partnership: {known_partnership:.3f})")
        print(f"  Path: {chall_path}")
        print(f"  Ref:  {PRODUCTION_CKPT}")

        row = {"label": label, "partnership": known_partnership, "budgets": {}}

        for num_sims in sim_budgets:
            print(f"\n  [{num_sims} sims] running {num_pairs} duplicate pairs...")
            t0 = time.time()
            wr, wins, total = eval_at_budget(
                chall_path, PRODUCTION_CKPT, num_sims, num_pairs,
                seed_base=seed_base + num_sims * 1000, device=device
            )
            elapsed = time.time() - t0
            row["budgets"][num_sims] = {"wr": wr, "wins": wins, "total": total}

            gate = ""
            if num_sims == GATE_MAIN_SIM:
                gate = "✅ PASS" if wr >= GATE_MAIN_WR else "❌ FAIL"
            elif num_sims == GATE_BONUS_SIM:
                gate = "✅ BONUS" if wr >= GATE_BONUS_WR else "⚠️  weak"

            print(f"  [{num_sims} sims] WR={wr:.1%} ({wins}/{total})  {gate}  ({elapsed:.0f}s)")

        # Verdict
        main_wr = row["budgets"].get(GATE_MAIN_SIM, {}).get("wr", 0)
        bonus_wr = row["budgets"].get(GATE_BONUS_SIM, {}).get("wr", 0)
        passes_main = main_wr >= GATE_MAIN_WR
        passes_bonus = bonus_wr >= GATE_BONUS_WR

        if passes_main and passes_bonus:
            verdict = "🏆 PROMOTE — beats production at both 100 and 200 sims"
        elif passes_main:
            verdict = "⚠️  MARGINAL — passes 100-sim gate, weak at 200"
        else:
            verdict = "❌ REJECT — does not beat production at deployment budget"

        row["verdict"] = verdict
        results[label] = row
        print(f"\n  VERDICT: {verdict}")

    # Summary table
    print(f"\n{'='*60}")
    print("PLAYOFF SUMMARY")
    print(f"{'='*60}")
    header = f"{'Checkpoint':<20} {'Partn':>6}"
    for s in sim_budgets:
        header += f"  {s}sim"
    header += "  Verdict"
    print(header)
    print("-" * 70)

    best_label = None
    best_100_wr = 0

    for label, row in results.items():
        line = f"{label:<20} {row['partnership']:>6.3f}"
        for s in sim_budgets:
            wr = row["budgets"].get(s, {}).get("wr", float("nan"))
            line += f"  {wr:>4.1%}"
        line += f"  {row['verdict'][:12]}"
        print(line)

        main_wr = row["budgets"].get(GATE_MAIN_SIM, {}).get("wr", 0)
        if main_wr > best_100_wr:
            best_100_wr = main_wr
            best_label = label

    if best_label and results[best_label]["budgets"].get(GATE_MAIN_SIM, {}).get("wr", 0) >= GATE_MAIN_WR:
        print(f"\n🏆 Best candidate: {best_label} ({best_100_wr:.1%} @ {GATE_MAIN_SIM} sims)")
        print(f"   → Recommend promoting this as new production champion")
        print(f"   → Then proceed with support-summary architecture tweak (Phase 10+)")
    else:
        print(f"\n❌ No checkpoint beats production at {GATE_MAIN_SIM} sims")
        print(f"   → Keep current production champion (gen50)")
        print(f"   → Proceed with support-summary tweak on Phase 10 continuation")

    # Save results JSON
    out_path = os.path.join(os.path.dirname(__file__), "results", "phase91_playoff.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs",  type=int, default=50,
                        help="Duplicate pairs per sim budget (default 50 = 100 games)")
    parser.add_argument("--sims",   type=str, default="50,100,200",
                        help="Comma-separated sim budgets (default 50,100,200)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--fast",   action="store_true",
                        help="Quick smoke test: 10 pairs, 50+100 sims only")
    args = parser.parse_args()

    if args.fast:
        num_pairs  = 10
        sim_budgets = (50, 100)
    else:
        num_pairs  = args.pairs
        sim_budgets = tuple(int(s) for s in args.sims.split(","))

    run_playoff(num_pairs=num_pairs, sim_budgets=sim_budgets, seed_base=args.seed)
