"""
Automated checkpoint monitor for Phase 6.5 training.

Waits for a target generation checkpoint to appear, then collects:
  - aux_gate value
  - policy / value / belief / support losses (from checkpoint)
  - greedy suite score (sims=0)
  - MCTS suite score (sims=100)

Usage:
    python training/monitor_checkpoint.py --gen 5
    python training/monitor_checkpoint.py --gen 10 --reduce-aux-weights
"""
import os, sys, time, json, argparse, subprocess, torch
import numpy as np

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR   = os.path.join(REPO_ROOT, 'training')
CKPT_DIR    = os.path.join(REPO_ROOT, 'checkpoints')
SUITE_JSON  = os.path.join(TRAIN_DIR, 'tests', 'partnership_suite.json')
SUITE_PY    = os.path.join(TRAIN_DIR, 'tests', 'test_partnership_suite.py')

sys.path.insert(0, TRAIN_DIR)


def wait_for_checkpoint(gen, poll_interval=30):
    path = os.path.join(CKPT_DIR, f'domino_gen_{gen:04d}.pt')
    print(f'Waiting for {path} ...', flush=True)
    while not os.path.exists(path):
        time.sleep(poll_interval)
    # Small extra wait to ensure file is fully written
    time.sleep(5)
    print(f'Checkpoint found: {path}', flush=True)
    return path


def read_checkpoint_stats(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    sd = ckpt.get('model_state_dict', ckpt)
    gate = sd.get('aux_gate', None)
    stats = {
        'gen':      ckpt.get('generation', '?'),
        'aux_gate': round(gate.item(), 6) if gate is not None else None,
    }
    # Losses stored in checkpoint if orchestrator saves them
    for key in ('loss', 'v_loss', 'p_loss', 'b_loss', 's_loss'):
        stats[key] = ckpt.get(key, None)
    return stats


def run_suite(checkpoint_path, sims):
    """Run partnership suite, return avg score or None on failure."""
    result = subprocess.run(
        [sys.executable, SUITE_PY,
         '--checkpoint', checkpoint_path,
         '--suite', SUITE_JSON,
         '--sims', str(sims),
         '--device', 'cpu'],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=300
    )
    output = result.stdout + result.stderr
    # Parse avg score from final PASS/FAIL line
    for line in reversed(output.splitlines()):
        if 'avg=' in line:
            try:
                avg_str = line.split('avg=')[1].split()[0].rstrip('>')
                return float(avg_str)
            except Exception:
                pass
    print(f'  [suite sims={sims}] Could not parse output:\n{output[-500:]}')
    return None


def report(gen, stats, greedy_score, mcts_score):
    sep = '=' * 55
    lines = [
        sep,
        f'  Phase 6.5 Gen {gen} Checkpoint Report',
        sep,
        f'  aux_gate       : {stats["aux_gate"]}',
        f'  value_loss     : {stats.get("v_loss")}',
        f'  policy_loss    : {stats.get("p_loss")}',
        f'  belief_loss    : {stats.get("b_loss")}',
        f'  support_loss   : {stats.get("s_loss")}',
        '',
        f'  Suite (greedy) : {greedy_score}',
        f'  Suite (100sim) : {mcts_score}',
        '',
        f'  Gap            : {round(greedy_score - mcts_score, 3) if greedy_score and mcts_score else "N/A"}',
        sep,
    ]
    print('\n'.join(lines), flush=True)

    out_path = os.path.join(REPO_ROOT, 'training', 'logs',
                            f'phase65_gen{gen:02d}_report.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'gen': gen,
            **stats,
            'suite_greedy': greedy_score,
            'suite_mcts100': mcts_score,
            'gap': round(greedy_score - mcts_score, 3) if greedy_score and mcts_score else None,
        }, f, indent=2)
    print(f'  Saved: {out_path}', flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', type=int, required=True,
                        help='Generation number to monitor')
    parser.add_argument('--poll', type=int, default=30,
                        help='Polling interval in seconds (default 30)')
    args = parser.parse_args()

    ckpt_path = wait_for_checkpoint(args.gen, poll_interval=args.poll)
    stats = read_checkpoint_stats(ckpt_path)

    print(f'  Running greedy suite (sims=0)...', flush=True)
    greedy = run_suite(ckpt_path, sims=0)

    print(f'  Running MCTS suite (sims=100)...', flush=True)
    mcts = run_suite(ckpt_path, sims=100)

    report(args.gen, stats, greedy, mcts)


if __name__ == '__main__':
    main()
