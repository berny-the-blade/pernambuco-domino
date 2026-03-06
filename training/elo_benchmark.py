import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domino_env import DominoEnv
from domino_encoder import DominoEncoder
from domino_net import DominoNet


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    # Auto-detect input_dim from checkpoint weights
    input_dim = state_dict['input_fc.weight'].shape[1]
    model = DominoNet(input_dim=input_dim, hidden_dim=256, num_actions=57, num_blocks=4)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def play_game(model_a, model_b, seed=None):
    env = DominoEnv()
    enc_a = DominoEncoder()
    enc_b = DominoEncoder()
    enc_a.reset()
    enc_b.reset()
    env.reset(seed=seed)
    while not env.is_over():
        team = env.current_player % 2
        obs = env.get_obs()
        mask = env.get_legal_moves_mask()
        model = model_a if team == 0 else model_b
        enc = enc_a if team == 0 else enc_b
        policy, _ = model.predict(enc.encode(obs), mask)
        env.step(int(np.argmax(policy * mask)))
    return env.winner_team


def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(ra, rb, score_a, k=32):
    ea = expected_score(ra, rb)
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


def run_benchmark(checkpoints_dir, games_per_pair=200):
    files = sorted([
        f for f in os.listdir(checkpoints_dir)
        if f.endswith('.pt') and 'domino_gen' in f and 'BACKUP' not in f
    ])
    print(f"Found {len(files)} checkpoints")

    models = {}
    elos = {}
    for f in files:
        gen = int(f.replace('domino_gen_', '').replace('.pt', ''))
        m = load_model(os.path.join(checkpoints_dir, f))
        # Skip old input_dim=185 models — incompatible encoder
        if m.input_fc.in_features != 213:
            print(f"  Skipping gen {gen:04d} (input_dim={m.input_fc.in_features})")
            continue
        models[gen] = m
        elos[gen] = 1000.0
        print(f"  Loaded gen {gen:04d}")

    gens = sorted(models.keys())
    total = len(gens) * (len(gens) - 1) // 2
    print(f"Running {total} matchups x {games_per_pair} games...")

    for i, ga in enumerate(gens):
        for gb in gens[i + 1:]:
            wins_a = 0
            for g in range(games_per_pair):
                if g % 2 == 0:
                    winner = play_game(models[ga], models[gb], seed=g)
                    wins_a += 1 if winner == 0 else 0
                else:
                    winner = play_game(models[gb], models[ga], seed=g)
                    wins_a += 1 if winner == 1 else 0
            score_a = wins_a / games_per_pair
            elos[ga], elos[gb] = update_elo(elos[ga], elos[gb], score_a)
            print(f"  Gen {ga:04d} vs {gb:04d}: {score_a*100:.1f}%  ELO {elos[ga]:.0f} vs {elos[gb]:.0f}")

    print("\n=== FINAL ELO RANKINGS ===")
    for gen in sorted(elos, key=lambda g: elos[g], reverse=True):
        print(f"  Gen {gen:04d}: {elos[gen]:.0f}")

    # Save results
    out_path = os.path.join(checkpoints_dir, 'elo_results.txt')
    with open(out_path, 'w') as f:
        for gen in sorted(elos, key=lambda g: elos[g], reverse=True):
            f.write(f"Gen {gen:04d}: {elos[gen]:.0f}\n")
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints-dir', default='checkpoints/')
    parser.add_argument('--games', type=int, default=200)
    args = parser.parse_args()
    run_benchmark(args.checkpoints_dir, args.games)
