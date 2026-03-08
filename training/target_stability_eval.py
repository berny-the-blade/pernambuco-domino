"""
target_stability_eval.py -- Measure how stable MCTS policy targets are
as sim count increases.

This is Test 2 from POST_GEN50_EXPERIMENTS.md.

Takes a set of fixed game states (sampled from self-play) and computes
MCTS visit distributions at different sim counts. Measures:
  - Top-1 agreement between sim levels
  - Jensen-Shannon Divergence between visit distributions
  - Mean entropy change

If 200 vs 400 sims still produce very different targets, the current
training labels are not mature enough -- search bottleneck likely.

Usage:
    python target_stability_eval.py
    python target_stability_eval.py --checkpoint checkpoints/domino_gen_0050.pt
    python target_stability_eval.py --states 200 --sims 100 200 400
"""

import argparse
import json
import os
import sys
import time
import glob

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)), )
from domino_env import DominoEnv
from domino_encoder import DominoEncoder
from domino_net import DominoNet
from domino_mcts import DominoMCTS

CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

DEFAULT_SIM_LEVELS = [100, 200, 400]
DEFAULT_NUM_STATES = 200


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(path: str) -> DominoNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    input_dim = state_dict["input_fc.weight"].shape[1]
    model = DominoNet(input_dim=input_dim, hidden_dim=256, num_actions=57, num_blocks=4)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def latest_checkpoint() -> tuple[int, str]:
    pattern = os.path.join(CHECKPOINTS_DIR, "domino_gen_????.pt")
    files = [f for f in glob.glob(pattern) if "BACKUP" not in f]
    if not files:
        raise FileNotFoundError("No checkpoints found")
    latest = max(files, key=os.path.getmtime)
    gen = int(os.path.basename(latest).replace("domino_gen_", "").replace(".pt", ""))
    return gen, latest


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two distributions."""
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def entropy(p: np.ndarray) -> float:
    p = p[p > 1e-10]
    return float(-np.sum(p * np.log(p)))


# ── state sampling ─────────────────────────────────────────────────────────────

def sample_states(num_states: int, seed_offset: int = 3000) -> list[tuple]:
    """
    Play quick games (greedy, no MCTS) and sample random mid-game states.
    Returns list of (env_snapshot, encoder_snapshot) tuples.

    We snapshot the game state at a random move in each game.
    """
    states = []
    game_idx = 0
    np.random.seed(42)

    print(f"Sampling {num_states} game states...", flush=True)

    while len(states) < num_states:
        env = DominoEnv()
        enc = DominoEncoder()
        enc.reset()
        env.reset(seed=seed_offset + game_idx)
        game_idx += 1

        # Collect all non-terminal positions in this game
        positions = []
        while not env.is_over():
            mask = env.get_legal_moves_mask()
            legal = np.where(mask > 0)[0]

            # Only keep non-trivial positions (>= 2 legal moves)
            if len(legal) >= 2:
                positions.append((env.clone(), enc.clone()))

            # Advance with greedy network-less random play (for speed)
            action = int(np.random.choice(legal))
            if action == 56:
                obs = env.get_obs()
                player = env.current_player
                me = 0
                diff = (player - me) % 4
                rel = {2: 0, 1: 1, 3: 2}.get(diff)
                if rel is not None:
                    enc.update_on_pass(rel, obs['left_end'], obs['right_end'])
            else:
                tile = action if action < 28 else action - 28
                player = env.current_player
                me = 0
                diff = (player - me) % 4
                rel = {2: 0, 1: 1, 3: 2}.get(diff)
                if rel is not None:
                    enc.update_on_play(rel, tile)

            env.step(action)

        # Sample one random position from this game
        if positions:
            idx = np.random.randint(len(positions))
            states.append(positions[idx])

        if len(states) % 20 == 0 and len(states) > 0:
            print(f"  Collected {len(states)}/{num_states} states "
                  f"({game_idx} games played)", flush=True)

    return states[:num_states]


# ── main eval ─────────────────────────────────────────────────────────────────

def compute_visit_distribution(mcts: DominoMCTS, env, encoder) -> np.ndarray:
    """Run MCTS and return normalized visit distribution over all 57 actions."""
    pi = mcts.get_action_probs(env, encoder, temperature=1.0)
    return pi


def run_stability_eval(model: DominoNet, states: list[tuple],
                        sim_levels: list[int], verbose: bool = True) -> dict:
    """
    For each state, compute visit distributions at each sim level.
    Returns aggregated statistics.
    """
    # Build MCTS at each sim level
    mcts_by_sims = {s: DominoMCTS(model, num_simulations=s) for s in sim_levels}

    # For each state, store distributions at each sim level
    # distributions[sim_level] = list of pi arrays
    distributions = {s: [] for s in sim_levels}
    entropies_by_sims = {s: [] for s in sim_levels}

    for i, (env_snap, enc_snap) in enumerate(states):
        for s in sim_levels:
            pi = compute_visit_distribution(mcts_by_sims[s], env_snap.clone(), enc_snap.clone())
            distributions[s].append(pi)
            entropies_by_sims[s].append(entropy(pi[pi > 0]))

        if verbose and (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(states)} states", flush=True)

    # Compute pairwise stats between consecutive sim levels
    pairs = []
    for idx in range(len(sim_levels) - 1):
        s_lo = sim_levels[idx]
        s_hi = sim_levels[idx + 1]

        jsds = []
        top1_agreements = []
        entropy_deltas = []

        for j in range(len(states)):
            pi_lo = distributions[s_lo][j]
            pi_hi = distributions[s_hi][j]

            jsds.append(jsd(pi_lo, pi_hi))
            top1_lo = int(np.argmax(pi_lo))
            top1_hi = int(np.argmax(pi_hi))
            top1_agreements.append(1 if top1_lo == top1_hi else 0)
            entropy_deltas.append(entropies_by_sims[s_hi][j] - entropies_by_sims[s_lo][j])

        pairs.append({
            "sims_lo": s_lo,
            "sims_hi": s_hi,
            "mean_jsd": round(float(np.mean(jsds)), 5),
            "std_jsd": round(float(np.std(jsds)), 5),
            "top1_agreement": round(float(np.mean(top1_agreements)) * 100, 1),
            "mean_entropy_lo": round(float(np.mean(entropies_by_sims[s_lo])), 4),
            "mean_entropy_hi": round(float(np.mean(entropies_by_sims[s_hi])), 4),
            "mean_entropy_delta": round(float(np.mean(entropy_deltas)), 4),
        })

    # Per-sim-level summary
    sim_summary = {}
    for s in sim_levels:
        sim_summary[s] = {
            "mean_entropy": round(float(np.mean(entropies_by_sims[s])), 4),
            "std_entropy": round(float(np.std(entropies_by_sims[s])), 4),
        }

    return {"pairs": pairs, "sim_summary": sim_summary}


def interpret_results(pairs: list[dict]) -> str:
    # Focus on the highest sim pair
    highest = pairs[-1]
    agreement = highest["top1_agreement"]
    jsd_val = highest["mean_jsd"]
    label = f"{highest['sims_lo']} vs {highest['sims_hi']} sims"

    if agreement >= 90 and jsd_val < 0.05:
        return f"STABLE ({label}: {agreement:.0f}% agreement, JSD={jsd_val:.4f}) -- targets mature, search not bottleneck"
    elif agreement >= 80:
        return f"MOSTLY STABLE ({label}: {agreement:.0f}% agreement, JSD={jsd_val:.4f}) -- mild sim sensitivity"
    elif agreement >= 70:
        return f"MODERATE INSTABILITY ({label}: {agreement:.0f}% agreement, JSD={jsd_val:.4f}) -- possible search bottleneck"
    else:
        return f"UNSTABLE ({label}: {agreement:.0f}% agreement, JSD={jsd_val:.4f}) -- SEARCH BOTTLENECK likely"


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Target stability evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--states", type=int, default=DEFAULT_NUM_STATES,
                        help=f"Number of states to sample (default: {DEFAULT_NUM_STATES})")
    parser.add_argument("--sims", type=int, nargs="+", default=DEFAULT_SIM_LEVELS,
                        help=f"Sim levels (default: {DEFAULT_SIM_LEVELS})")
    args = parser.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)

    if args.checkpoint:
        ckpt_path = args.checkpoint
        current_gen = int(os.path.basename(ckpt_path).replace("domino_gen_", "").replace(".pt", ""))
    else:
        current_gen, ckpt_path = latest_checkpoint()
        print(f"Auto-detected: gen {current_gen}")

    print(f"Checkpoint: gen {current_gen}")
    print(f"Sim levels: {args.sims}")
    print(f"States: {args.states}")
    print()

    model = load_model(ckpt_path)

    # Sample states
    states = sample_states(args.states)
    print(f"\nRunning stability eval across {len(args.sims)} sim levels...")

    results = run_stability_eval(model, states, args.sims)
    results["gen"] = current_gen
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["num_states"] = args.states

    # Print summary
    print(f"\n{'='*65}")
    print(f"  Target Stability: Gen {current_gen}  ({args.states} states)")
    print(f"{'='*65}")
    print(f"\n  Per-sim entropy:")
    for s, stats in results["sim_summary"].items():
        print(f"    {s:>5} sims: entropy={stats['mean_entropy']:.4f} "
              f"(+/-{stats['std_entropy']:.4f})")

    print(f"\n  Pairwise stability:")
    print(f"  {'Comparison':<20} {'Top1 Agree':>12} {'Mean JSD':>10} {'Entropy Delta':>14}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*14}")
    for p in results["pairs"]:
        cmp_str = f"{p['sims_lo']} vs {p['sims_hi']} sims"
        print(f"  {cmp_str:<20} {p['top1_agreement']:>11.1f}% {p['mean_jsd']:>10.5f} "
              f"{p['mean_entropy_delta']:>+14.4f}")

    verdict = interpret_results(results["pairs"])
    print(f"\n  Verdict: {verdict}")

    # Save
    jsonl_path = os.path.join(LOGS_DIR, "target_stability.jsonl")
    txt_path = os.path.join(LOGS_DIR, "target_stability.txt")

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(results) + "\n")

    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*65}\n")
        f.write(f"Target Stability: Gen {current_gen}  [{results['timestamp']}]\n")
        f.write(f"States: {args.states}, Sims: {args.sims}\n")
        f.write(f"{'='*65}\n")
        for p in results["pairs"]:
            cmp_str = f"{p['sims_lo']} vs {p['sims_hi']} sims"
            f.write(f"  {cmp_str:<20} top1={p['top1_agreement']:.1f}%  "
                    f"JSD={p['mean_jsd']:.5f}  entropy_delta={p['mean_entropy_delta']:+.4f}\n")
        f.write(f"  {verdict}\n")

    print(f"\nSaved to {txt_path}")


if __name__ == "__main__":
    main()
