"""
collect_positions.py — Sample diagnostic positions from self-play.

Plays self-play games and saves a pickle of position records that can be
loaded by diagnose_policy_value.py.  Each record contains:

  state_np   : (213,) float32  — encoded observable state
  mask_np    : (57,)  float32  — legal moves mask
  v_pred     : float           — value head prediction at this state
  outcome    : float           — final ΔME (current player's POV)
  env        : DominoEnv       — deep-copied env snapshot (for MCTS in Diag B)
  encoder    : DominoEncoder   — deep-copied encoder snapshot
  my_score   : int
  opp_score  : int
  multiplier : int
  player     : int             — absolute player index
  left_end   : int | None
  right_end  : int | None
  board_len  : int
  move_idx   : int             — move number within game

Usage:
    python training/collect_positions.py \\
        --checkpoint checkpoints/best_100sims.pt \\
        --positions 1000 \\
        --out diagnostics/positions.pkl
"""

import os, sys, copy, pickle, argparse, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domino_env     import DominoMatch
from domino_net     import DominoNet
from domino_encoder import DominoEncoder
from match_equity   import delta_me


def load_model(path, device):
    net  = DominoNet()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    return net.to(device).eval()


def collect(model, device, n_positions=1000, sample_every=3, seed=42):
    np.random.seed(seed)
    records = []
    games   = 0

    while len(records) < n_positions:
        match   = DominoMatch(target_points=6)
        encoder = DominoEncoder()
        encoder.reset()
        match.new_game()

        scores_before = list(match.scores)
        mult_before   = match.multiplier
        game_buf      = []
        step          = 0

        while not match.env.is_over() and step < 200:
            env  = match.env
            mask = env.get_legal_moves_mask()
            if mask.sum() == 0:
                break

            team      = env.current_team
            my_sc     = match.scores[team]
            opp_sc    = match.scores[1 - team]
            obs       = env.get_obs()
            state_np  = encoder.encode(obs, my_sc, opp_sc, match.multiplier)

            if step % sample_every == 0:
                with torch.no_grad():
                    _, v_pred = model.predict(state_np, mask, device)

                board = env.board if hasattr(env, "board") else []
                game_buf.append(dict(
                    state_np  = state_np.copy(),
                    mask_np   = mask.copy(),
                    v_pred    = float(v_pred),
                    env       = copy.deepcopy(env),
                    encoder   = copy.deepcopy(encoder),
                    my_score  = my_sc,
                    opp_score = opp_sc,
                    multiplier= match.multiplier,
                    player    = env.current_player,
                    left_end  = int(env.left_end)  if len(board) > 0 else None,
                    right_end = int(env.right_end) if len(board) > 0 else None,
                    board_len = len(board),
                    move_idx  = step,
                ))

            probs, _ = model.predict(state_np, mask, device)
            action   = int(np.random.choice(len(probs), p=probs))
            obs, _, done, _ = env.step(action)
            encoder.update(obs)
            step += 1
            if done:
                break

        # Assign outcomes
        outcome_t0 = delta_me(
            winner_team  = match.env.winner_team,
            points       = match.env.points_won,
            my_team      = 0,
            my_score     = scores_before[0],
            opp_score    = scores_before[1],
            multiplier   = mult_before,
        )
        for rec in game_buf:
            t = rec["env"].current_team
            rec["outcome"] = float(outcome_t0) if t == 0 else float(-outcome_t0)
            records.append(rec)

        games += 1
        if games % 5 == 0:
            print(f"  {len(records)}/{n_positions} positions ({games} games)...", flush=True)

        if not match.match_over:
            try:
                match.new_game(); encoder.reset()
                scores_before = list(match.scores)
                mult_before   = match.multiplier
            except Exception:
                pass

    return records[:n_positions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions",  type=int, default=1000)
    ap.add_argument("--out",        default="diagnostics/positions.pkl")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Checkpoint: {args.checkpoint}")

    model   = load_model(args.checkpoint, device)
    t0      = time.time()
    records = collect(model, device, args.positions, seed=args.seed)
    print(f"Collected {len(records)} positions in {time.time()-t0:.1f}s")

    with open(args.out, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
