"""
move_value_scatter.py -- Move Value Advantage Scatter

For a set of midgame positions, computes:
  v_A = value the model assigns after its best move (argmax policy)
  v_B = same for the reference model

Then plots v_A vs v_B with the y=x diagonal.
Points above diagonal = model A picks better moves = A is likely stronger.

Usage:
    # Compare gen07 vs gen50 (production):
    python move_value_scatter.py \
        --modelA checkpoints/domino_gen_0007.pt \
        --modelB checkpoints/domino_gen_0050.pt \
        --positions 400 \
        --label-a "gen07 (p=0.669)" \
        --label-b "gen50 production"

    # Compare all candidates at once:
    python move_value_scatter.py --all --positions 400

    # Save plot without displaying (headless):
    python move_value_scatter.py --all --positions 400 --save scatter.png --no-show
"""

import argparse
import os
import sys
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domino_net import DominoNet
from domino_env import DominoEnv
from domino_encoder import DominoEncoder
from orchestrator import safe_load_state_dict

TRAINING_CKPTS = os.path.join(os.path.dirname(__file__), "checkpoints")
PRODUCTION     = os.path.join(TRAINING_CKPTS, "domino_gen_0050.pt")

CANDIDATES = [
    {"label": "gen07 (p=0.669)", "path": os.path.join(TRAINING_CKPTS, "domino_gen_0007.pt")},
    {"label": "gen15 (p=0.632)", "path": os.path.join(TRAINING_CKPTS, "domino_gen_0015.pt")},
    {"label": "gen19 (p=0.625)", "path": os.path.join(TRAINING_CKPTS, "domino_gen_0019.pt")},
]


def load_model(path, device=torch.device("cpu")):
    net = DominoNet().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    safe_load_state_dict(net, ckpt["model_state_dict"], strict=False)
    net.eval()
    return net


def best_move_value(model, env, encoder, device=torch.device("cpu")):
    """
    Get the value the model assigns to the position after making its best move.
    Returns (value, action) where value is in [-1, 1] from Team 0's perspective.
    If the game is already over or no legal moves, returns None.
    """
    mask = env.get_legal_moves_mask()
    if mask.sum() == 0:
        return None, None

    obs = env.get_obs()
    state = encoder.encode(obs)

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        mask_t  = torch.tensor(mask,  dtype=torch.float32).unsqueeze(0).to(device)
        policy, value = model(state_t, valid_actions_mask=mask_t)

    # Best move = argmax policy
    best_action = int(torch.argmax(policy[0]).item())

    # Return the value estimate at current position (before move)
    # This is the model's prediction of Team 0's advantage from here
    return value[0].item(), best_action


def collect_positions(num_positions, seed_base=12345, min_moves_played=4, max_moves_played=20):
    """
    Play random games and collect midgame positions.
    Filters: non-trivial positions (>=2 legal moves, midgame phase).
    Returns list of (env_snapshot, encoder_state) tuples.
    """
    positions = []
    game_seed = seed_base
    attempts = 0

    print("Collecting %d midgame positions..." % num_positions, flush=True)

    while len(positions) < num_positions:
        attempts += 1
        env = DominoEnv()
        enc = DominoEncoder()
        obs = env.reset(seed=game_seed)
        enc.reset()
        game_seed += 1
        moves_played = 0

        while not env.is_over() and moves_played < 56:
            mask = env.get_legal_moves_mask()
            if mask.sum() == 0:
                break

            # Collect if in midgame window and has real choice (>=2 legal moves)
            if (min_moves_played <= moves_played <= max_moves_played
                    and mask.sum() >= 2):
                # Snapshot: store env state as encoded tensor
                state = enc.encode(obs).copy()
                mask_copy = mask.copy()
                player = env.current_player
                positions.append({
                    "state": state,
                    "mask": mask_copy,
                    "player": player,
                    "move_num": moves_played,
                    "seed": game_seed - 1,
                })
                if len(positions) >= num_positions:
                    break

            # Play random move to advance
            legal_actions = np.where(mask > 0)[0]
            action = int(np.random.choice(legal_actions))
            obs, _, done, _ = env.step(action)
            moves_played += 1

    print("Collected %d positions from %d games." % (len(positions), attempts))
    return positions[:num_positions]


def evaluate_positions(model, positions, device=torch.device("cpu")):
    """
    For each position, run the model and get its value estimate.
    Returns list of values in [-1, 1].
    """
    values = []
    model.eval()
    with torch.no_grad():
        for pos in positions:
            state_t = torch.tensor(pos["state"], dtype=torch.float32).unsqueeze(0).to(device)
            mask_t  = torch.tensor(pos["mask"],  dtype=torch.float32).unsqueeze(0).to(device)
            _, value = model(state_t, valid_actions_mask=mask_t)
            values.append(value[0].item())
    return values


def scatter_two(vals_A, vals_B, label_a, label_b, ax, alpha=0.35):
    """Plot one scatter panel."""
    vals_A = np.array(vals_A)
    vals_B = np.array(vals_B)

    # Color by which model thinks it's better at this position
    above = vals_A > vals_B   # A thinks it has better position
    below = ~above

    ax.scatter(vals_B[above], vals_A[above], alpha=alpha, s=12,
               color="#2196F3", label="%s better (%d)" % (label_a, above.sum()))
    ax.scatter(vals_B[below], vals_A[below], alpha=alpha, s=12,
               color="#F44336", label="%s better (%d)" % (label_b, below.sum()))

    # Diagonal
    lim = [-1, 1]
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5, label="equal")

    # Mean difference annotation
    diff = vals_A.mean() - vals_B.mean()
    sign = "+" if diff >= 0 else ""
    ax.set_title("%s vs %s\nmean diff = %s%.4f" % (label_a, label_b, sign, diff), fontsize=9)
    ax.set_xlabel("%s value" % label_b, fontsize=8)
    ax.set_ylabel("%s value" % label_a, fontsize=8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)

    # Above/below ratio
    frac_above = above.mean()
    return diff, frac_above


def run_scatter(model_a_path, model_b_path, label_a, label_b,
                num_positions=400, save_path=None, show=True, seed=12345):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cpu")

    print("Loading models...")
    model_A = load_model(model_a_path, device)
    model_B = load_model(model_b_path, device)

    positions = collect_positions(num_positions, seed_base=seed)

    print("Evaluating model A (%s)..." % label_a, flush=True)
    vals_A = evaluate_positions(model_A, positions, device)
    print("Evaluating model B (%s)..." % label_b, flush=True)
    vals_B = evaluate_positions(model_B, positions, device)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    diff, frac_above = scatter_two(vals_A, vals_B, label_a, label_b, ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved: %s" % save_path)
    if show:
        plt.show()
    plt.close()

    # Text summary
    direction = label_a if diff > 0 else label_b
    print("\nSCATTER SUMMARY")
    print("  Mean value diff (A - B):  %+.4f" % diff)
    print("  Positions where A > B:    %.1f%%" % (frac_above * 100))
    print("  Likely stronger model:    %s" % direction)
    if abs(diff) < 0.01:
        print("  Signal: models are very similar")
    elif abs(diff) < 0.03:
        print("  Signal: slight advantage, borderline arena result expected")
    else:
        print("  Signal: clear advantage, should show up in arena")

    return {"diff": diff, "frac_above": frac_above, "label_a": label_a, "label_b": label_b}


def run_all(num_positions=400, save_dir=None, show=True, seed=12345):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cpu")

    print("Loading production model...")
    model_ref = load_model(PRODUCTION, device)

    # Collect positions once, reuse for all
    positions = collect_positions(num_positions, seed_base=seed)

    print("Evaluating reference (gen50)...")
    vals_ref = evaluate_positions(model_ref, positions, device)

    candidates_with_vals = []
    for c in CANDIDATES:
        if not os.path.exists(c["path"]):
            print("  [SKIP] %s: not found" % c["label"])
            continue
        print("Evaluating %s..." % c["label"], flush=True)
        model_c = load_model(c["path"], device)
        vals_c = evaluate_positions(model_c, positions, device)
        candidates_with_vals.append((c["label"], vals_c))

    n = len(candidates_with_vals)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    print("\nSCATTER SUMMARY")
    print("  %-25s  %8s  %8s  %s" % ("Challenger", "MeanDiff", "AbovePct", "Signal"))
    print("  " + "-" * 60)

    for i, (label, vals_c) in enumerate(candidates_with_vals):
        diff, frac_above = scatter_two(vals_c, vals_ref, label, "gen50", axes[i])
        direction = label if diff > 0 else "gen50"
        if abs(diff) < 0.01:
            signal = "similar"
        elif abs(diff) < 0.03:
            signal = "slight edge -> " + direction
        else:
            signal = "clear edge -> " + direction
        print("  %-25s  %+.4f    %.1f%%     %s" % (label, diff, frac_above * 100, signal))

    fig.suptitle("Move Value Scatter: Candidates vs Production (gen50)\n%d positions" % num_positions,
                 fontsize=11)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "move_value_scatter_all.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print("\nSaved: %s" % out)
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelA",     default=None)
    parser.add_argument("--modelB",     default=PRODUCTION)
    parser.add_argument("--label-a",    default="challenger")
    parser.add_argument("--label-b",    default="gen50 production")
    parser.add_argument("--positions",  type=int, default=400)
    parser.add_argument("--all",        action="store_true",
                        help="Compare all candidates vs production in one plot")
    parser.add_argument("--save",       default=None, help="Save plot to this path")
    parser.add_argument("--no-show",    action="store_true")
    parser.add_argument("--seed",       type=int, default=12345)
    args = parser.parse_args()

    show = not args.no_show

    if args.all:
        save_dir = os.path.dirname(args.save) if args.save else \
                   os.path.join(os.path.dirname(__file__), "results")
        run_all(num_positions=args.positions, save_dir=save_dir, show=show, seed=args.seed)
    else:
        if not args.modelA:
            parser.error("--modelA required (or use --all)")
        run_scatter(
            model_a_path=args.modelA,
            model_b_path=args.modelB,
            label_a=args.label_a,
            label_b=args.label_b,
            num_positions=args.positions,
            save_path=args.save,
            show=show,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
