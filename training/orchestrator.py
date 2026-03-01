"""
Self-Play Orchestrator for Pernambuco Domino AI training.
Uses torch.multiprocessing to spawn parallel CPU workers for data generation,
then trains the master network on GPU with the collected replay buffer.

Value targets: ΔME (match equity delta) instead of raw points/4.
Arena: duplicate deals with side-swap for fair evaluation.

Usage:
    python orchestrator.py                    # Default: 4 workers, 100 generations
    python orchestrator.py --workers 8 --generations 200 --games-per-worker 500
    python orchestrator.py --mcts --mcts-sims 100 --value-target me
"""

import os
import sys
import time
import argparse
from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Add parent dir for imports when running from training/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domino_env import DominoEnv, DominoMatch
from domino_net import DominoNet
from domino_encoder import DominoEncoder
from domino_mcts import DominoMCTS
from domino_trainer import Trainer, ReplayDataset
from match_equity import get_match_equity, delta_me, DOB_VALUES


def self_play_worker(worker_id, model_state_dict, num_games, use_mcts,
                     mcts_sims, result_queue, value_target='me'):
    """
    Isolated CPU process. Plays games against itself and pipes training data back.

    Each data point: (state_np, mask_np, pi_np, v_target)

    value_target: 'me' for ΔME (match equity), 'points' for points/4 (legacy)
    """
    device = torch.device("cpu")
    model = DominoNet().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    torch.set_grad_enabled(False)

    # Unique seed per worker
    np.random.seed(int(time.time() * 1000) % (2**31) + worker_id * 1000)

    mcts = DominoMCTS(model, num_simulations=mcts_sims) if use_mcts else None
    worker_data = []

    for game_idx in range(num_games):
        # Play a full match (first to 6) for ME context
        match = DominoMatch(target_points=6)
        match_history = []  # list of (game_history, match_state_before)

        while not match.match_over:
            env = match.env
            encoder = DominoEncoder()
            obs = match.new_game()
            encoder.reset()

            # Capture match state BEFORE this game (for ΔME)
            scores_before = list(match.scores)
            multiplier_before = match.multiplier

            game_history = []
            step_count = 0

            while not env.is_over() and step_count < 200:
                current_team = env.current_team
                valid_mask = env.get_legal_moves_mask()

                if valid_mask.sum() == 0:
                    break

                # Encode with match context
                my_team = current_team
                my_score = match.scores[my_team]
                opp_score = match.scores[1 - my_team]
                state_np = encoder.encode(obs, my_score=my_score,
                                          opp_score=opp_score,
                                          multiplier=match.multiplier)

                if use_mcts and valid_mask.sum() > 1:
                    temp = 1.0 if step_count < 8 else 0.1
                    target_pi = mcts.get_action_probs(env, encoder, temperature=temp)
                else:
                    policy_probs, _ = model.predict(state_np, valid_mask, device)

                    valid_indices = np.where(valid_mask > 0)[0]
                    target_pi = policy_probs.copy()

                    if len(valid_indices) > 1:
                        noise = np.random.dirichlet([0.3] * len(valid_indices))
                        target_pi[valid_indices] = (
                            0.75 * target_pi[valid_indices] + 0.25 * noise
                        )
                        total = target_pi.sum()
                        if total > 0:
                            target_pi /= total

                chosen_action = np.random.choice(57, p=target_pi)

                game_history.append({
                    'state': state_np,
                    'mask': valid_mask,
                    'pi': target_pi,
                    'team': current_team,
                })

                # Execute action
                obs, _, done, info = env.step(chosen_action)

                # Update encoder beliefs for played tiles
                if chosen_action < 56:
                    tile = chosen_action if chosen_action < 28 else chosen_action - 28
                    encoder.update_on_play(0, tile)

                # Update encoder beliefs for passes
                if chosen_action == 56:
                    passer = (env.current_player - 1) % 4
                    me_player = game_history[-1].get('_me_player', obs['player'])
                    # Pass updates are handled by _sync_belief via env.cant_have

                step_count += 1

            # Record game result in match
            if env.game_over:
                match.record_game_result(env.winner_team, env.points_won)

                # Compute value targets
                if value_target == 'me':
                    # ΔME: match equity change from this game
                    for step in game_history:
                        team = step['team']
                        my_s = scores_before[team]
                        opp_s = scores_before[1 - team]
                        v = delta_me(
                            winner_team=env.winner_team,
                            points=env.points_won,
                            my_team=team,
                            my_score=my_s,
                            opp_score=opp_s,
                            multiplier=multiplier_before
                        )
                        worker_data.append((
                            step['state'], step['mask'], step['pi'], v
                        ))
                else:
                    # Legacy: points/4
                    winner_team = env.winner_team
                    reward_magnitude = env.points_won / 4.0
                    for step in game_history:
                        if step['team'] == winner_team:
                            v_target = reward_magnitude
                        else:
                            v_target = -reward_magnitude
                        worker_data.append((
                            step['state'], step['mask'], step['pi'], v_target
                        ))

    result_queue.put(worker_data)


def arena_match(champion_weights, challenger_weights, seed, challenger_team=0):
    """
    Play one deterministic match (first to 6): challenger vs champion.
    No noise, greedy policy (argmax). Returns (match_winner, games_played).
    """
    device = torch.device("cpu")
    champion = DominoNet().to(device)
    champion.load_state_dict(champion_weights)
    champion.eval()

    challenger = DominoNet().to(device)
    challenger.load_state_dict(challenger_weights)
    challenger.eval()

    match = DominoMatch(target_points=6)
    game_seed = seed
    games_played = 0

    with torch.no_grad():
        while not match.match_over and games_played < 20:  # safety limit
            env = match.env
            encoder = DominoEncoder()
            obs = match.new_game(seed=game_seed + games_played)
            encoder.reset()
            step = 0

            while not env.is_over() and step < 200:
                mask = env.get_legal_moves_mask()
                if mask.sum() == 0:
                    break

                team = env.current_team
                my_score = match.scores[team]
                opp_score = match.scores[1 - team]
                state = encoder.encode(obs, my_score=my_score,
                                       opp_score=opp_score,
                                       multiplier=match.multiplier)

                # Select model based on team
                model = challenger if team == challenger_team else champion
                policy, _ = model.predict(state, mask, device)

                # Greedy: argmax (no noise, no temperature)
                action = int(np.argmax(policy))

                obs, _, done, info = env.step(action)
                if action < 56:
                    tile = action if action < 28 else action - 28
                    encoder.update_on_play(0, tile)
                step += 1

            if env.game_over:
                match.record_game_result(env.winner_team, env.points_won)
            games_played += 1

    return match.match_winner, games_played


def arena_worker(worker_id, champion_weights, challenger_weights,
                 game_seeds, result_queue):
    """Run duplicate-deal arena matches in a subprocess.
    Each seed is played twice (challenger as team 0 and team 1)."""
    challenger_wins = 0
    champion_wins = 0
    total_matches = 0

    for seed in game_seeds:
        for c_team in [0, 1]:
            winner, _ = arena_match(champion_weights, challenger_weights,
                                    seed, challenger_team=c_team)
            total_matches += 1
            if winner == c_team:
                challenger_wins += 1
            elif winner >= 0:
                champion_wins += 1

    result_queue.put((challenger_wins, champion_wins, total_matches))


class Orchestrator:
    """Manages the generational self-play + training loop with arena gatekeeper.

    Value targets: ΔME (match equity delta) or legacy points/4.
    Arena: duplicate-deal matches with side-swap. Promotion requires >52% winrate
    with lower 95% CI > 50%.
    """

    ARENA_SEEDS = 100       # duplicate deals (×2 side-swap = 200 matches)
    ARENA_MIN_WINRATE = 0.52  # challenger must win ≥52% of matches

    def __init__(self, num_workers=4, buffer_size=200000, use_mcts=False,
                 mcts_sims=50, value_target='me'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Orchestrator device: {self.device}")
        print(f"Workers: {num_workers}, Buffer: {buffer_size}, "
              f"MCTS: {use_mcts} ({mcts_sims} sims), Value: {value_target}")

        self.model = DominoNet().to(self.device)
        self.champion_weights = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        self.trainer = Trainer(self.model, lr=1e-3)

        self.num_workers = num_workers
        self.use_mcts = use_mcts
        self.mcts_sims = mcts_sims
        self.value_target = value_target
        self.replay_buffer = deque(maxlen=buffer_size)
        self.generation = 0
        self.rejections = 0  # count of failed arena challenges

    def run(self, total_generations=100, games_per_worker=250):
        """Run the full generational training loop."""
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        os.makedirs("checkpoints", exist_ok=True)

        for gen in range(1, total_generations + 1):
            self.generation = gen
            print(f"\n{'='*50}")
            print(f"  GENERATION {gen}/{total_generations}")
            print(f"{'='*50}")

            # === PHASE 1: PARALLEL DATA GENERATION ===
            t0 = time.time()
            print(f"Spawning {self.num_workers} workers "
                  f"({games_per_worker} games each)...")

            # Extract weights for CPU workers
            shared_weights = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }

            ctx = mp.get_context('spawn')
            result_queue = ctx.Queue()
            processes = []

            for w_id in range(self.num_workers):
                p = ctx.Process(
                    target=self_play_worker,
                    args=(w_id, shared_weights, games_per_worker,
                          self.use_mcts, self.mcts_sims, result_queue,
                          self.value_target)
                )
                p.start()
                processes.append(p)

            # Collect results
            collected = 0
            total_samples = 0
            while collected < self.num_workers:
                try:
                    worker_data = result_queue.get(timeout=3600)
                    self.replay_buffer.extend(worker_data)
                    total_samples += len(worker_data)
                    collected += 1
                    print(f"  Worker {collected}/{self.num_workers} done: "
                          f"{len(worker_data)} samples")
                except Exception as e:
                    print(f"  Worker collection error: {e}")
                    collected += 1

            for p in processes:
                p.join(timeout=30)

            elapsed = time.time() - t0
            total_games = self.num_workers * games_per_worker
            print(f"Self-play: {total_games} games in {elapsed:.1f}s "
                  f"({total_games/elapsed:.0f} games/s)")
            print(f"Buffer: {len(self.replay_buffer)} samples "
                  f"(+{total_samples} new)")

            # === PHASE 2: TRAINING ===
            min_buffer = 2000
            if len(self.replay_buffer) >= min_buffer:
                print("Training neural network...")
                dataset = ReplayDataset(list(self.replay_buffer))
                dataloader = DataLoader(
                    dataset, batch_size=256, shuffle=True,
                    num_workers=0, pin_memory=(self.device.type == 'cuda')
                )

                for epoch in range(5):
                    loss, v_loss, p_loss = self.trainer.train_epoch(dataloader)
                    print(f"  Epoch {epoch+1}/5 | "
                          f"Loss: {loss:.4f} (V: {v_loss:.4f}, P: {p_loss:.4f})")

                # === PHASE 3: ARENA EVALUATION ===
                challenger_weights = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

                # Skip arena for first generation (no champion to compare against)
                if gen == 1:
                    self.champion_weights = challenger_weights
                    promoted = True
                    print("First generation — auto-promoting to champion.")
                else:
                    promoted = self._arena_evaluate(challenger_weights)

                if promoted:
                    self.champion_weights = challenger_weights
                    ckpt_path = f"checkpoints/domino_gen_{gen:04d}.pt"
                    torch.save({
                        'generation': gen,
                        'model_state_dict': self.model.state_dict(),
                        'buffer_size': len(self.replay_buffer),
                    }, ckpt_path)
                    print(f"Saved champion checkpoint: {ckpt_path}")
                else:
                    # Revert to champion weights
                    self.model.load_state_dict(
                        {k: v.to(self.device) for k, v in self.champion_weights.items()}
                    )
                    self.trainer = Trainer(self.model, lr=1e-3)
                    self.rejections += 1
                    print(f"Reverted to champion. "
                          f"Total rejections: {self.rejections}")
            else:
                print(f"Buffer too small ({len(self.replay_buffer)}/{min_buffer}). "
                      f"Gathering more data...")

        print(f"\nTraining complete. {total_generations} generations. "
              f"Rejections: {self.rejections}")

    def _arena_evaluate(self, challenger_weights):
        """Run arena: challenger vs champion with duplicate deals.
        Returns True if challenger promoted (≥52% winrate, lower 95% CI > 50%)."""
        total_matches = self.ARENA_SEEDS * 2  # each seed played from both sides
        print(f"Arena: {self.ARENA_SEEDS} seeds × 2 sides = "
              f"{total_matches} matches (threshold: {self.ARENA_MIN_WINRATE:.0%})...")
        t0 = time.time()

        # Generate deterministic seeds
        seeds = list(range(self.generation * 10000,
                           self.generation * 10000 + self.ARENA_SEEDS))

        # Split seeds across workers
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        n_workers = min(self.num_workers, len(seeds))
        chunk = max(1, len(seeds) // n_workers)
        processes = []

        for w_id in range(n_workers):
            start = w_id * chunk
            end = start + chunk if w_id < n_workers - 1 else len(seeds)
            if start >= len(seeds):
                break
            worker_seeds = seeds[start:end]
            p = ctx.Process(
                target=arena_worker,
                args=(w_id, self.champion_weights, challenger_weights,
                      worker_seeds, result_queue)
            )
            p.start()
            processes.append(p)

        # Collect results (3-tuple: challenger_wins, champion_wins, total_matches)
        challenger_wins = 0
        champion_wins = 0
        matches_played = 0
        for _ in range(len(processes)):
            try:
                c_w, ch_w, t_m = result_queue.get(timeout=300)
                challenger_wins += c_w
                champion_wins += ch_w
                matches_played += t_m
            except Exception as e:
                print(f"  Arena worker error: {e}")

        for p in processes:
            p.join(timeout=30)

        if matches_played == 0:
            win_rate = 0.5
        else:
            win_rate = challenger_wins / matches_played

        # 95% CI using normal approximation: p ± 1.96 * sqrt(p*(1-p)/n)
        if matches_played > 0:
            ci_half = 1.96 * np.sqrt(win_rate * (1 - win_rate) / matches_played)
            ci_lower = win_rate - ci_half
        else:
            ci_lower = 0.0

        elapsed = time.time() - t0
        print(f"  Arena result: Challenger {challenger_wins}W vs "
              f"Champion {champion_wins}W / {matches_played} matches "
              f"({win_rate:.1%}, 95% CI lower: {ci_lower:.1%}) in {elapsed:.1f}s")

        if win_rate >= self.ARENA_MIN_WINRATE and ci_lower > 0.50:
            print(f"  PROMOTED! ({win_rate:.1%} >= {self.ARENA_MIN_WINRATE:.0%}, "
                  f"CI lower {ci_lower:.1%} > 50%)")
            return True
        else:
            reason = []
            if win_rate < self.ARENA_MIN_WINRATE:
                reason.append(f"winrate {win_rate:.1%} < {self.ARENA_MIN_WINRATE:.0%}")
            if ci_lower <= 0.50:
                reason.append(f"CI lower {ci_lower:.1%} <= 50%")
            print(f"  REJECTED. ({', '.join(reason)})")
            return False

    def load_checkpoint(self, path):
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.champion_weights = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        self.generation = ckpt.get('generation', 0)
        print(f"Loaded checkpoint: {path} (gen {self.generation})")


def main():
    parser = argparse.ArgumentParser(description='Pernambuco Domino AI Trainer')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1),
                        help='Number of parallel self-play workers')
    parser.add_argument('--generations', type=int, default=100,
                        help='Total training generations')
    parser.add_argument('--games-per-worker', type=int, default=250,
                        help='Games each worker plays per generation')
    parser.add_argument('--buffer-size', type=int, default=200000,
                        help='Replay buffer capacity')
    parser.add_argument('--mcts', action='store_true',
                        help='Use IS-MCTS (slower but stronger)')
    parser.add_argument('--mcts-sims', type=int, default=50,
                        help='MCTS simulations per move')
    parser.add_argument('--value-target', type=str, default='me',
                        choices=['me', 'points'],
                        help='Value target: "me" for ΔME (default), '
                             '"points" for legacy points/4')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    args = parser.parse_args()

    orch = Orchestrator(
        num_workers=args.workers,
        buffer_size=args.buffer_size,
        use_mcts=args.mcts,
        mcts_sims=args.mcts_sims,
        value_target=args.value_target,
    )

    if args.resume:
        orch.load_checkpoint(args.resume)

    orch.run(
        total_generations=args.generations,
        games_per_worker=args.games_per_worker,
    )


if __name__ == "__main__":
    main()
