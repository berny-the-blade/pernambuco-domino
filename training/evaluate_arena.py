"""
evaluate_arena.py -- Head-to-head arena evaluation with paired margin scoring.

Noise-reduction design:
  - Duplicate deals: same seed played twice with sides swapped (cancels tile luck)
  - Continuous margin: delta_ME per game, pair_margin = dME_A(game0) + dME_A(game1)
  - Fixed seed set: pass --seed to ensure all candidates see identical deals
  - Reports pair win rate + mean pair margin + CI on both

Usage:
    python evaluate_arena.py \
        --modelA checkpoints/domino_gen_0015.pt \
        --modelB checkpoints/domino_gen_0050.pt \
        --pairs 400 --sims 100 --seed 42

Promotion gates:
    pair_win_rate >= 52% @ 100 sims  (primary)
    pair_margin   >  0               (secondary, confirms direction)
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domino_net import DominoNet
from domino_env import DominoEnv
from domino_encoder import DominoEncoder
from domino_mcts import DominoMCTS
from match_equity import delta_me
from orchestrator import safe_load_state_dict

# Promotion gates
GATE_PROMOTE_WR   = 0.52
GATE_ELIMINATE_WR = 0.48


def load_model(path, device=torch.device("cpu")):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError("Checkpoint not found: %s" % path)
    net = DominoNet().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    safe_load_state_dict(net, ckpt["model_state_dict"], strict=False)
    net.eval()
    return net


def play_game(mcts_A, mcts_B, seed, a_team):
    """
    Play one game to completion.
    a_team: which team model A controls (0 or 1).
    Returns delta_ME from Team A's perspective.
    """
    enc_A = DominoEncoder()
    enc_B = DominoEncoder()
    env = DominoEnv()
    obs = env.reset(seed=seed)
    enc_A.reset()
    enc_B.reset()
    step = 0

    with torch.no_grad():
        while not env.is_over() and step < 200:
            mask = env.get_legal_moves_mask()
            if mask.sum() == 0:
                break
            team = env.current_player % 2
            if team == a_team:
                pi = mcts_A.get_action_probs(env, enc_A, temperature=0.0)
            else:
                pi = mcts_B.get_action_probs(env, enc_B, temperature=0.0)
            action = int(np.argmax(pi * mask))
            obs, _, done, _ = env.step(action)
            step += 1

    if not env.game_over:
        return 0.0  # timed out, call it a draw

    # delta_ME from A's perspective (scores=0,0 since standalone game)
    dme = delta_me(env.winner_team, env.points_won,
                   my_team=a_team, my_score=0, opp_score=0)
    return dme


def play_duplicate_pair(mcts_A, mcts_B, seed):
    """
    Duplicate pair: same deal, A plays team 0 then team 1.
    pair_margin = dME_A(game0) + dME_A(game1)
    pair_win    = 1 if pair_margin > 0, 0.5 if == 0, 0 if < 0
    """
    dme0 = play_game(mcts_A, mcts_B, seed, a_team=0)
    dme1 = play_game(mcts_A, mcts_B, seed, a_team=1)
    pair_margin = dme0 + dme1
    if pair_margin > 1e-9:
        pair_win = 1.0
    elif pair_margin < -1e-9:
        pair_win = 0.0
    else:
        pair_win = 0.5
    return pair_margin, pair_win


def wilson_ci(wins, total, z=1.96):
    """Wilson score CI for win rate."""
    if total == 0:
        return 0.0, 0.0, 1.0
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def mean_ci(values, z=1.96):
    """Mean +/- CI for a list of values (t-style using sample std)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mu = sum(values) / n
    if n == 1:
        return mu, mu, mu
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return mu, mu - z * se, mu + z * se


def run_arena(model_a_path, model_b_path, num_pairs, num_sims,
              seed_base=42, device=None, verbose=True):

    # DominoMCTS hardcodes CPU inference
    device = torch.device("cpu")

    label_A = os.path.basename(model_a_path)
    label_B = os.path.basename(model_b_path)

    if verbose:
        print("\n" + "=" * 60)
        print("  modelA: %s" % label_A)
        print("  modelB: %s" % label_B)
        print("  sims: %d  |  pairs: %d  |  seeds: %d..%d" % (
            num_sims, num_pairs, seed_base, seed_base + num_pairs - 1))
        print("=" * 60)

    model_A = load_model(model_a_path, device)
    model_B = load_model(model_b_path, device)
    mcts_A = DominoMCTS(model_A, num_simulations=num_sims)
    mcts_B = DominoMCTS(model_B, num_simulations=num_sims)

    pair_wins    = []   # 0/0.5/1 per pair
    pair_margins = []   # continuous dME per pair
    t0 = time.time()

    for i in range(num_pairs):
        seed = seed_base + i  # fixed seed set — same deals across all candidates
        margin, win = play_duplicate_pair(mcts_A, mcts_B, seed)
        pair_wins.append(win)
        pair_margins.append(margin)

        if verbose and (i + 1) % 50 == 0:
            wr   = sum(pair_wins) / len(pair_wins)
            mu_m = sum(pair_margins) / len(pair_margins)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (num_pairs - i - 1)
            print("  pair %4d/%d  WR=%.1f%%  margin=%.4f  ETA %.0fs" % (
                i + 1, num_pairs, wr * 100, mu_m, eta), flush=True)

    elapsed = time.time() - t0

    # Stats
    wr, wr_lo, wr_hi = wilson_ci(sum(pair_wins), len(pair_wins))
    mu_m, m_lo, m_hi = mean_ci(pair_margins)

    if verbose:
        print("\n  PAIR WIN RATE: %.1f%%  95%%CI=[%.1f%%, %.1f%%]" % (
            wr * 100, wr_lo * 100, wr_hi * 100))
        print("  PAIR MARGIN:   %.4f  95%%CI=[%.4f, %.4f]" % (mu_m, m_lo, m_hi))
        print("  Time: %.0fs (%.2fs/pair)" % (elapsed, elapsed / num_pairs))

        # Verdict
        margin_positive = m_lo > 0
        if num_sims == 100:
            if wr >= GATE_PROMOTE_WR and mu_m > 0:
                verdict = "PROMOTE  (WR>=52%% and margin>0)"
            elif wr >= GATE_PROMOTE_WR:
                verdict = "MARGINAL  (WR passes but margin not confident)"
            elif wr >= GATE_ELIMINATE_WR:
                verdict = "BORDERLINE  (WR 48-52%%)"
            else:
                verdict = "ELIMINATE  (WR<48%%)"
        else:
            if wr >= GATE_PROMOTE_WR and mu_m > 0:
                verdict = "STRONG  (WR>=52%% + margin>0)"
            elif mu_m > 0:
                verdict = "OK  (margin positive)"
            else:
                verdict = "WEAK  (margin <= 0)"
        print("  Verdict @ %dsims: %s" % (num_sims, verdict))

    return {
        "wr": wr, "wr_lo": wr_lo, "wr_hi": wr_hi,
        "margin": mu_m, "margin_lo": m_lo, "margin_hi": m_hi,
        "sims": num_sims, "pairs": num_pairs,
        "raw_wins": sum(pair_wins), "raw_margins": pair_margins,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelA",  required=True)
    parser.add_argument("--modelB",  required=True)
    parser.add_argument("--pairs",   type=int, default=400)
    parser.add_argument("--sims",    type=int, default=100)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    result = run_arena(
        model_a_path=args.modelA,
        model_b_path=args.modelB,
        num_pairs=args.pairs,
        num_sims=args.sims,
        seed_base=args.seed,
        verbose=True,
    )

    print("\nSUMMARY wr=%.4f margin=%.4f wr_ci=[%.4f,%.4f] margin_ci=[%.4f,%.4f] sims=%d" % (
        result["wr"], result["margin"],
        result["wr_lo"], result["wr_hi"],
        result["margin_lo"], result["margin_hi"],
        result["sims"]))


if __name__ == "__main__":
    main()
