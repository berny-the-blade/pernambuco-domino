"""
Self-Play Orchestrator for Pernambuco Domino AI training.
Uses torch.multiprocessing to spawn parallel CPU workers for data generation,
then trains the master network on GPU with the collected replay buffer.

Usage:
    python orchestrator.py                    # Default: 4 workers, 100 generations
    python orchestrator.py --workers 8 --generations 200 --games-per-worker 500
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

from domino_env import DominoEnv
from domino_net import DominoNet
from domino_encoder import DominoEncoder
from domino_mcts import DominoMCTS
from domino_trainer import Trainer, ReplayDataset


def self_play_worker(worker_id, model_state_dict, num_games, use_mcts,
                     mcts_sims, result_queue):
    """
    Isolated CPU process. Plays games against itself and pipes training data back.

    Each data point: (state_np, mask_np, pi_np, v_target)
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
        env = DominoEnv()
        encoder = DominoEncoder()
        obs = env.reset()
        encoder.reset()

        game_history = []
        step_count = 0

        while not env.is_over() and step_count < 200:
            current_team = env.current_team
            valid_mask = env.get_legal_moves_mask()

            if valid_mask.sum() == 0:
                break  # Shouldn't happen, but safety

            state_np = encoder.encode(obs)

            if use_mcts and valid_mask.sum() > 1:
                # Temperature: explore early, exploit late
                temp = 1.0 if step_count < 8 else 0.1
                target_pi = mcts.get_action_probs(env, encoder, temperature=temp)
            else:
                # Fast neural intuition (no tree search)
                policy_probs, _ = model.predict(state_np, valid_mask, device)

                # Dirichlet noise for exploration
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

            # Choose action
            chosen_action = np.random.choice(57, p=target_pi)

            # Record state before the move
            game_history.append({
                'state': state_np,
                'mask': valid_mask,
                'pi': target_pi,
                'team': current_team,
            })

            # Execute
            obs, _, done, info = env.step(chosen_action)

            # Update encoder beliefs
            if chosen_action == 56:
                # Pass: the player who just acted (previous current_player) passed
                passer = (env.current_player - 1) % 4
                me_perspective = 0  # We encode from each player's perspective in turn
                # For training, we track from the acting player's perspective
            elif chosen_action < 56:
                tile = chosen_action if chosen_action < 28 else chosen_action - 28
                encoder.update_on_play(0, tile)  # Mark tile as played

            step_count += 1

        # === RETROACTIVE CREDIT ASSIGNMENT ===
        if env.game_over:
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
        # If game didn't end (safety), skip this game's data

        if (game_idx + 1) % 50 == 0:
            pass  # Progress tracking happens in main process

    result_queue.put(worker_data)


class Orchestrator:
    """Manages the generational self-play + training loop."""

    def __init__(self, num_workers=4, buffer_size=200000, use_mcts=False,
                 mcts_sims=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Orchestrator device: {self.device}")
        print(f"Workers: {num_workers}, Buffer: {buffer_size}, "
              f"MCTS: {use_mcts} ({mcts_sims} sims)")

        self.model = DominoNet().to(self.device)
        self.trainer = Trainer(self.model, lr=1e-3)

        self.num_workers = num_workers
        self.use_mcts = use_mcts
        self.mcts_sims = mcts_sims
        self.replay_buffer = deque(maxlen=buffer_size)
        self.generation = 0

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
                          self.use_mcts, self.mcts_sims, result_queue)
                )
                p.start()
                processes.append(p)

            # Collect results
            collected = 0
            total_samples = 0
            while collected < self.num_workers:
                try:
                    worker_data = result_queue.get(timeout=600)
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

                # Save checkpoint
                ckpt_path = f"checkpoints/domino_gen_{gen:04d}.pt"
                torch.save({
                    'generation': gen,
                    'model_state_dict': self.model.state_dict(),
                    'buffer_size': len(self.replay_buffer),
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
            else:
                print(f"Buffer too small ({len(self.replay_buffer)}/{min_buffer}). "
                      f"Gathering more data...")

        print(f"\nTraining complete. {total_generations} generations.")

    def load_checkpoint(self, path):
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
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
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    args = parser.parse_args()

    orch = Orchestrator(
        num_workers=args.workers,
        buffer_size=args.buffer_size,
        use_mcts=args.mcts,
        mcts_sims=args.mcts_sims,
    )

    if args.resume:
        orch.load_checkpoint(args.resume)

    orch.run(
        total_generations=args.generations,
        games_per_worker=args.games_per_worker,
    )


if __name__ == "__main__":
    main()
