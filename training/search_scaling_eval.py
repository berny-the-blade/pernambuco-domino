"""
search_scaling_eval.py -- Search scaling evaluation for Pernambuco Domino AI.

Test 1 from POST_GEN50_EXPERIMENTS.md.

Evaluates model-a vs model-b at multiple MCTS sim budgets using duplicate deals
and fixed seeds. Both models use MCTS. Same deal schedule across all sim levels.

Measures: win rate, duplicate margin (pts/deal), root entropy/top1/top2,
          forced_move_pct, avg_game_length.

Usage:
    python search_scaling_eval.py \\
        --model-a checkpoints/domino_gen_0050.pt \\
        --model-b checkpoints/domino_gen_0046.pt \\
        --sim-list 50,100,200,400 \\
        --duplicate-deals \\
        --deal-pairs 800 \\
        --seed-base 5000 \\
        --output-json results/search_scaling_gen50_vs_gen46.json \\
        --output-csv  results/search_scaling_gen50_vs_gen46.csv \\
        --tag gen50_vs_gen46
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import glob

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domino_env import DominoEnv
from domino_encoder import DominoEncoder
from domino_net import DominoNet
from domino_mcts import DominoMCTS

CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

DEFAULT_SIM_LIST  = [50, 100, 200, 400]
DEFAULT_DEAL_PAIRS = 200   # bump to 400-800 for real eval
DEFAULT_SEED_BASE  = 5000
DEFAULT_ANCHOR_GEN = 46


# ============================================================
# Model helpers
# ============================================================

def load_model(path: str) -> DominoNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    input_dim = sd["input_fc.weight"].shape[1]
    m = DominoNet(input_dim=input_dim, hidden_dim=256, num_actions=57, num_blocks=4)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def latest_checkpoint():
    pattern = os.path.join(CHECKPOINTS_DIR, "domino_gen_????.pt")
    files = [f for f in glob.glob(pattern) if "BACKUP" not in f]
    if not files:
        raise FileNotFoundError("No checkpoints found")
    p = max(files, key=os.path.getmtime)
    gen = int(os.path.basename(p).replace("domino_gen_", "").replace(".pt", ""))
    return gen, p


def ckpt_for_gen(gen: int) -> str:
    return os.path.join(CHECKPOINTS_DIR, f"domino_gen_{gen:04d}.pt")


def gen_from_path(path: str) -> int:
    try:
        return int(os.path.basename(path).replace("domino_gen_", "").replace(".pt", ""))
    except ValueError:
        return -1


# ============================================================
# Statistics
# ============================================================

def bootstrap_ci(values, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval on the mean."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values)
    means = [np.mean(arr[np.random.randint(0, len(arr), len(arr))]) for _ in range(n_boot)]
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return float(np.mean(arr)), lo, hi


def wilson_ci(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ============================================================
# Game play
# ============================================================

def play_game_mcts(mcts_a: DominoMCTS, mcts_b: DominoMCTS,
                   seed: int, collect_root_stats: bool = False):
    """
    Play one full game. A=team0, B=team1.
    Returns (winner_team, points_won, root_stats, game_length, forced_count, total_moves).

    DominoEncoder.encode(obs) calls _sync_belief(obs) internally, so beliefs
    are always current without manual update_on_pass/update_on_play in the outer loop.
    """
    env = DominoEnv()
    enc_a = DominoEncoder(); enc_a.reset()
    enc_b = DominoEncoder(); enc_b.reset()
    env.reset(seed=seed)

    root_stats = []   # (entropy, top1_mass, top2_gap) for model-A turns
    total_moves = 0
    forced_count = 0

    while not env.is_over():
        mask = env.get_legal_moves_mask()
        legal = np.where(mask > 0)[0]
        total_moves += 1
        if len(legal) == 1:
            forced_count += 1

        team = env.current_player % 2
        if team == 0:
            pi = mcts_a.get_action_probs(env, enc_a, temperature=0.1)
            if collect_root_stats:
                vis = pi[pi > 1e-9]
                if len(vis) >= 1:
                    ent = float(-np.sum(vis * np.log(vis + 1e-10)))
                    sv = np.sort(vis)[::-1]
                    t1 = float(sv[0])
                    t2g = float(sv[0] - sv[1]) if len(sv) > 1 else 1.0
                    root_stats.append((ent, t1, t2g))
        else:
            pi = mcts_b.get_action_probs(env, enc_b, temperature=0.1)

        action = int(np.argmax(pi * mask))
        env.step(action)

    return env.winner_team, env.points_won, root_stats, total_moves, forced_count


def play_duplicate_pair(mcts_a: DominoMCTS, mcts_b: DominoMCTS,
                         seed: int, collect_root_stats: bool = True):
    """
    Two games with same seed, A and B swap seats.
    Returns result dict with margin and stats for model A.

    Margin: positive = A scored more than B.
      game1 (A=team0): g1_margin = +pts if team0 wins, -pts if team1 wins
      game2 (A=team1): g2_margin = +pts if team1 wins, -pts if team0 wins
      pair_margin = (g1_margin + g2_margin) / 2
    """
    w1, p1, rs1, moves1, forced1 = play_game_mcts(mcts_a, mcts_b, seed, collect_root_stats)
    g1_margin_a = p1 if w1 == 0 else -p1

    w2, p2, rs2, moves2, forced2 = play_game_mcts(mcts_b, mcts_a, seed, False)
    g2_margin_a = p2 if w2 == 1 else -p2

    total_moves = moves1 + moves2
    total_forced = forced1 + forced2

    return {
        "a_wins": (1 if w1 == 0 else 0) + (1 if w2 == 1 else 0),
        "pair_margin": (g1_margin_a + g2_margin_a) / 2.0,
        "root_stats": rs1,
        "game_length_mean": total_moves / 2.0,
        "forced_pct": total_forced / total_moves if total_moves > 0 else 0.0,
    }


# ============================================================
# Eval at one sim level
# ============================================================

def eval_at_sims(model_a: DominoNet, model_b: DominoNet,
                 num_sims: int, num_pairs: int, seed_base: int,
                 verbose: bool = True) -> dict:
    mcts_a = DominoMCTS(model_a, num_simulations=num_sims)
    mcts_b = DominoMCTS(model_b, num_simulations=num_sims)

    total_games = num_pairs * 2
    a_wins = 0
    pair_margins, game_lengths, forced_pcts = [], [], []
    entropies, top1s, top2_gaps = [], [], []

    for i in range(num_pairs):
        seed = seed_base + i
        result = play_duplicate_pair(mcts_a, mcts_b, seed,
                                      collect_root_stats=(i < 100))
        a_wins += result["a_wins"]
        pair_margins.append(result["pair_margin"])
        game_lengths.append(result["game_length_mean"])
        forced_pcts.append(result["forced_pct"])
        for ent, t1, t2g in result["root_stats"]:
            entropies.append(ent)
            top1s.append(t1)
            top2_gaps.append(t2g)

        if verbose and (i + 1) % 50 == 0:
            wr = a_wins / ((i + 1) * 2)
            mg = float(np.mean(pair_margins))
            print(f"    [{num_sims} sims] Pair {i+1}/{num_pairs}  "
                  f"win%={wr*100:.1f}  margin={mg:+.3f}", flush=True)

    win_rate = a_wins / total_games
    ci_lo, ci_hi = wilson_ci(a_wins, total_games)
    margin_mean, margin_ci_lo, margin_ci_hi = bootstrap_ci(pair_margins)
    margin_std = float(np.std(pair_margins)) if pair_margins else 0.0

    return {
        "sims": num_sims,
        "games_total": total_games,
        "pairs_total": num_pairs,
        "wins_a": a_wins,
        "wins_b": total_games - a_wins,
        "draws": 0,
        "winrate_a": round(win_rate, 5),
        "win_pct_a": round(win_rate * 100, 1),
        "win_ci95_lo": round(ci_lo * 100, 1),
        "win_ci95_hi": round(ci_hi * 100, 1),
        "mean_duplicate_margin_a": round(margin_mean, 4),
        "std_duplicate_margin_a": round(margin_std, 4),
        "ci95_low_margin_a": round(margin_ci_lo, 4),
        "ci95_high_margin_a": round(margin_ci_hi, 4),
        "root_entropy_mean": round(float(np.mean(entropies)), 4) if entropies else 0.0,
        "root_top1_mass_mean": round(float(np.mean(top1s)), 4) if top1s else 0.0,
        "root_top2_gap_mean": round(float(np.mean(top2_gaps)), 4) if top2_gaps else 0.0,
        "forced_move_pct": round(float(np.mean(forced_pcts)), 4) if forced_pcts else 0.0,
        "avg_game_length": round(float(np.mean(game_lengths)), 2) if game_lengths else 0.0,
    }


# ============================================================
# Verdict
# ============================================================

def compute_verdict(results: list[dict]) -> dict:
    """
    Domino-tuned verdict logic.
    Computes delta_50_200 and delta_100_400 on duplicate margin.
    """
    by_sims = {r["sims"]: r for r in results}

    delta_50_200 = delta_100_400 = None
    if 50 in by_sims and 200 in by_sims:
        delta_50_200 = by_sims[200]["mean_duplicate_margin_a"] - by_sims[50]["mean_duplicate_margin_a"]
    if 100 in by_sims and 400 in by_sims:
        delta_100_400 = by_sims[400]["mean_duplicate_margin_a"] - by_sims[100]["mean_duplicate_margin_a"]

    # Fall back to widest available spread
    if len(results) >= 2:
        lo = results[0]["mean_duplicate_margin_a"]
        hi = results[-1]["mean_duplicate_margin_a"]
        spread_margin = hi - lo
        spread_win = results[-1]["win_pct_a"] - results[0]["win_pct_a"]
    else:
        spread_margin = spread_win = 0.0

    if (delta_50_200 is not None and delta_50_200 >= 0.08) or \
       (delta_100_400 is not None and delta_100_400 >= 0.05) or \
       spread_margin >= 0.08 or spread_win >= 3.0:
        label = "SEARCH_BOTTLENECK_LIKELY"
        detail = "Strength rises materially with more sims -- increase training sims or add reanalysis"
    elif (delta_50_200 is not None and delta_50_200 >= 0.04) or \
         (delta_100_400 is not None and delta_100_400 >= 0.03) or \
         spread_margin >= 0.04 or spread_win >= 1.5:
        label = "SEARCH_BOTTLENECK_MODERATE"
        detail = "Moderate sim sensitivity -- search still matters but not the only bottleneck"
    else:
        label = "SEARCH_BOTTLENECK_UNLIKELY"
        detail = "Flat sim response -- network/data/belief quality more likely the bottleneck"

    return {
        "label": label,
        "detail": detail,
        "delta_50_200_margin": round(delta_50_200, 4) if delta_50_200 is not None else None,
        "delta_100_400_margin": round(delta_100_400, 4) if delta_100_400 is not None else None,
        "spread_margin": round(spread_margin, 4),
        "spread_win_pct": round(spread_win, 2),
    }


# ============================================================
# Output
# ============================================================

def save_outputs(results: list[dict], verdict: dict,
                  gen_a: int, gen_b: int,
                  path_a: str, path_b: str,
                  args) -> None:
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    payload = {
        "meta": {
            "script": "search_scaling_eval.py",
            "model_a": path_a,
            "model_b": path_b,
            "gen_a": gen_a,
            "gen_b": gen_b,
            "sim_list": [r["sims"] for r in results],
            "duplicate_deals": True,
            "deal_pairs": args.deal_pairs,
            "seed_base": args.seed_base,
            "tag": args.tag or f"gen{gen_a}_vs_gen{gen_b}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": results,
        "verdict": verdict,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_fields = [
        "sims", "pairs_total", "wins_a", "wins_b", "winrate_a", "win_pct_a",
        "win_ci95_lo", "win_ci95_hi",
        "mean_duplicate_margin_a", "std_duplicate_margin_a",
        "ci95_low_margin_a", "ci95_high_margin_a",
        "root_entropy_mean", "root_top1_mass_mean", "root_top2_gap_mean",
        "forced_move_pct", "avg_game_length",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in csv_fields})

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV:  {args.output_csv}")


def print_summary(results: list[dict], verdict: dict, gen_a: int, gen_b: int):
    print(f"\n{'='*85}")
    print(f"  Search Scaling: Gen {gen_a} (A) vs Gen {gen_b} (B)")
    print(f"{'='*85}")
    hdr = (f"  {'Sims':<6} {'Pairs':>6} {'Win%':>7} {'Win CI':>14} "
           f"{'Margin':>9} {'95% CI Margin':>16} "
           f"{'Entropy':>9} {'Top1':>7} {'Top2Gap':>8} "
           f"{'Forced%':>8} {'GmLen':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        win_ci = f"[{r['win_ci95_lo']:.1f}-{r['win_ci95_hi']:.1f}]"
        mg_ci  = f"[{r['ci95_low_margin_a']:+.3f},{r['ci95_high_margin_a']:+.3f}]"
        print(f"  {r['sims']:<6} {r['pairs_total']:>6} "
              f"{r['win_pct_a']:>6.1f}%  {win_ci:<14} "
              f"{r['mean_duplicate_margin_a']:>+8.4f}  {mg_ci:<16} "
              f"{r['root_entropy_mean']:>9.4f} {r['root_top1_mass_mean']:>7.4f} "
              f"{r['root_top2_gap_mean']:>8.4f} "
              f"{r['forced_move_pct']*100:>7.1f}% {r['avg_game_length']:>7.1f}")

    print(f"\n  INTERPRETATION")
    print(f"  Gen {gen_a} vs Gen {gen_b}")
    if verdict['delta_50_200_margin'] is not None:
        print(f"    50->200 sims margin delta:  {verdict['delta_50_200_margin']:+.4f}")
    if verdict['delta_100_400_margin'] is not None:
        print(f"    100->400 sims margin delta: {verdict['delta_100_400_margin']:+.4f}")
    print(f"  Verdict: {verdict['label']}")
    print(f"  {verdict['detail']}")
    print()


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Search scaling evaluation")
    parser.add_argument("--model-a", type=str, default=None)
    parser.add_argument("--model-b", type=str, default=None)
    parser.add_argument("--gen-a", type=int, default=None)
    parser.add_argument("--gen-b", type=int, default=None)
    parser.add_argument("--sim-list", type=str,
                        default=",".join(map(str, DEFAULT_SIM_LIST)))
    parser.add_argument("--duplicate-deals", action="store_true", default=True)
    parser.add_argument("--deal-pairs", type=int, default=DEFAULT_DEAL_PAIRS)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv",  type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    sim_levels = [int(x) for x in args.sim_list.split(",")]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")

    # Resolve model A
    if args.model_a:
        path_a = args.model_a
        gen_a = args.gen_a or gen_from_path(path_a)
    else:
        gen_a, path_a = latest_checkpoint()
        print(f"Model A: auto-detected gen {gen_a}")

    # Resolve model B
    if args.model_b:
        path_b = args.model_b
        gen_b = args.gen_b or gen_from_path(path_b)
    else:
        gen_b = DEFAULT_ANCHOR_GEN
        path_b = ckpt_for_gen(gen_b)
        print(f"Model B: default anchor gen {gen_b}")

    for label, path in [("model-a", path_a), ("model-b", path_b)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    base_name = f"search_scaling_gen{gen_a:04d}_vs_gen{gen_b:04d}_{ts}"
    if not args.output_json:
        args.output_json = os.path.join(RESULTS_DIR, base_name + ".json")
    if not args.output_csv:
        args.output_csv = os.path.join(RESULTS_DIR, base_name + ".csv")

    print(f"\nGen {gen_a} vs Gen {gen_b}")
    print(f"Sim levels: {sim_levels}")
    print(f"Deal pairs: {args.deal_pairs} per level ({args.deal_pairs * 2} games)")
    print(f"Seed base:  {args.seed_base}")
    print()

    model_a = load_model(path_a)
    model_b = load_model(path_b)

    all_results = []
    for sims in sim_levels:
        print(f"--- {sims} sims ({args.deal_pairs} pairs = {args.deal_pairs*2} games) ---")
        r = eval_at_sims(model_a, model_b, sims, args.deal_pairs, args.seed_base)
        r["gen_a"] = gen_a
        r["gen_b"] = gen_b
        all_results.append(r)

    verdict = compute_verdict(all_results)
    print_summary(all_results, verdict, gen_a, gen_b)
    save_outputs(all_results, verdict, gen_a, gen_b, path_a, path_b, args)


if __name__ == "__main__":
    main()
