"""
anchor_eval.py — Scheduled anchor evaluation for Pernambuco Domino training.

Evaluates a "current" checkpoint against a set of fixed anchor checkpoints
using duplicate deals (side-swapped pairs) for fair, low-variance comparison.

Results are appended to:
  logs/anchor_eval.jsonl   — machine-readable, one JSON object per run
  logs/anchor_eval.txt     — human-readable summary

Usage:
    # Evaluate latest checkpoint vs default anchors (gen 46, 1, 5, 10)
    python anchor_eval.py

    # Specify current checkpoint explicitly
    python anchor_eval.py --current checkpoints/domino_gen_0020.pt

    # Specify anchors explicitly (gen numbers)
    python anchor_eval.py --anchors 46 1 5 10 20

    # More games for tighter confidence intervals
    python anchor_eval.py --games 400

    # Auto-watch mode: poll checkpoints dir and eval every N new gens
    python anchor_eval.py --watch --eval-every 5
"""

import argparse
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


# ── helpers ──────────────────────────────────────────────────────────────────

CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

DEFAULT_ANCHORS = [46, 1, 5, 10]   # gen numbers; extend as run progresses


def ensure_logs_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)


def ckpt_path(gen: int) -> str:
    return os.path.join(CHECKPOINTS_DIR, f"domino_gen_{gen:04d}.pt")


def latest_phase4_checkpoint() -> tuple[int, str]:
    """Return (gen, path) of the most recently modified checkpoint.
    Uses mtime so that a fresh phase-4 gen 15 beats an older gen 100."""
    pattern = os.path.join(CHECKPOINTS_DIR, "domino_gen_????.pt")
    files = [f for f in glob.glob(pattern) if "BACKUP" not in f]
    if not files:
        raise FileNotFoundError("No checkpoints found in " + CHECKPOINTS_DIR)
    latest = max(files, key=os.path.getmtime)
    def gen_num(p):
        return int(os.path.basename(p).replace("domino_gen_", "").replace(".pt", ""))
    return gen_num(latest), latest


def load_model(path: str) -> DominoNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    input_dim = state_dict["input_fc.weight"].shape[1]
    model = DominoNet(input_dim=input_dim, hidden_dim=256, num_actions=57, num_blocks=4)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ── game play ─────────────────────────────────────────────────────────────────

def play_game(model_a, model_b, seed=None) -> int:
    """Play one game; returns winning team (0 = model_a's team, 1 = model_b's team)."""
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


def play_duplicate_pair(model_a, model_b, seed: int) -> tuple[int, int]:
    """
    Play two games with the same seed, sides swapped.
    Returns (wins_for_a_game1, wins_for_a_game2) where each is 0 or 1.
    """
    # Game 1: model_a = team 0
    w1 = play_game(model_a, model_b, seed=seed)
    wins_a_g1 = 1 if w1 == 0 else 0

    # Game 2: model_a = team 1 (sides swapped)
    w2 = play_game(model_b, model_a, seed=seed)
    wins_a_g2 = 1 if w2 == 1 else 0

    return wins_a_g1, wins_a_g2


# ── stats ─────────────────────────────────────────────────────────────────────

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def elo_delta(win_rate: float) -> float:
    """Estimated ELO difference from win rate (relative to 50%)."""
    win_rate = max(0.001, min(0.999, win_rate))
    return -400 * math.log10(1 / win_rate - 1)


# ── core eval ─────────────────────────────────────────────────────────────────

def evaluate_vs_anchor(
    current_model,
    current_gen: int,
    anchor_model,
    anchor_gen: int,
    num_pairs: int = 100,   # duplicate pairs → 2× games total
    verbose: bool = True,
) -> dict:
    """
    Evaluate current vs anchor using num_pairs duplicate pairs.
    Returns a result dict.
    """
    total_games = num_pairs * 2
    wins_current = 0

    for i in range(num_pairs):
        seed = 1000 + i   # fixed seeds for reproducibility across evals
        w1, w2 = play_duplicate_pair(current_model, anchor_model, seed)
        wins_current += w1 + w2
        if verbose and (i + 1) % 10 == 0:
            wr = wins_current / ((i + 1) * 2)
            print(f"    Pair {i+1}/{num_pairs}  win%={wr*100:.1f}", flush=True)

    win_rate = wins_current / total_games
    ci_lo, ci_hi = wilson_ci(wins_current, total_games)
    elo_diff = elo_delta(win_rate)

    result = {
        "current_gen": current_gen,
        "anchor_gen": anchor_gen,
        "pairs": num_pairs,
        "total_games": total_games,
        "wins_current": wins_current,
        "win_rate": round(win_rate, 4),
        "win_pct": round(win_rate * 100, 1),
        "ci_lo": round(ci_lo * 100, 1),
        "ci_hi": round(ci_hi * 100, 1),
        "elo_delta": round(elo_diff, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return result


# ── output ────────────────────────────────────────────────────────────────────

def log_results(results: list[dict]):
    ensure_logs_dir()
    jsonl_path = os.path.join(LOGS_DIR, "anchor_eval.jsonl")
    txt_path = os.path.join(LOGS_DIR, "anchor_eval.txt")

    # Append JSONL
    with open(jsonl_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Append human-readable
    with open(txt_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Eval at {results[0]['timestamp']}  (current gen {results[0]['current_gen']})\n")
        f.write(f"{'='*60}\n")
        f.write(f"{'Anchor':<10} {'Win%':>7} {'95% CI':>16} {'ELO d':>8}\n")
        f.write(f"{'-'*10} {'-'*7} {'-'*16} {'-'*8}\n")
        for r in results:
            ci_str = f"[{r['ci_lo']:.1f}-{r['ci_hi']:.1f}]"
            elo_str = f"{r['elo_delta']:+.1f}"
            verdict = "> better" if r['elo_delta'] > 10 else ("< worse" if r['elo_delta'] < -10 else "~ flat")
            f.write(f"gen {r['anchor_gen']:<6} {r['win_pct']:>6.1f}% {ci_str:>16} {elo_str:>8}  {verdict}\n")

    print(f"\nResults appended to {txt_path}")
    print(f"JSONL appended to   {jsonl_path}")


def print_summary(results: list[dict]):
    current_gen = results[0]["current_gen"]
    print(f"\n{'='*60}")
    print(f"  Gen {current_gen} anchor evaluation")
    print(f"{'='*60}")
    print(f"  {'Anchor':<12} {'Win%':>7} {'95% CI':>16} {'ELO d':>8}  Verdict")
    print(f"  {'-'*12} {'-'*7} {'-'*16} {'-'*8}  -------")
    for r in results:
        ci_str = f"[{r['ci_lo']:.1f}-{r['ci_hi']:.1f}]"
        elo_str = f"{r['elo_delta']:+.1f}"
        verdict = "> better" if r['elo_delta'] > 10 else ("< worse" if r['elo_delta'] < -10 else "~ flat")
        print(f"  vs gen {r['anchor_gen']:<6} {r['win_pct']:>6.1f}%  {ci_str:<16} {elo_str:>8}  {verdict}")
    print()


# ── watch mode ────────────────────────────────────────────────────────────────

def watch_and_eval(anchors: list[int], num_pairs: int, eval_every: int):
    """
    Poll checkpoints dir and run anchor evals every `eval_every` new generations.
    Tracks which gens have already been evaluated in logs/anchor_eval_state.json.
    """
    state_path = os.path.join(LOGS_DIR, "anchor_eval_state.json")
    ensure_logs_dir()

    if os.path.exists(state_path):
        with open(state_path, encoding='utf-8') as f:
            state = json.load(f)
    else:
        state = {"evaluated_gens": [], "next_eval_at": None}

    print(f"Watch mode: eval every {eval_every} new gens. Ctrl-C to stop.")
    print(f"Anchors: {anchors}")

    already_evaluated = set(state.get("evaluated_gens", []))

    while True:
        try:
            current_gen, current_path = latest_phase4_checkpoint()
        except FileNotFoundError:
            time.sleep(60)
            continue

        # Determine when to next eval
        next_eval_at = state.get("next_eval_at")
        if next_eval_at is None:
            # First run: eval immediately
            next_eval_at = current_gen

        if current_gen >= next_eval_at and current_gen not in already_evaluated:
            print(f"\n[{time.strftime('%H:%M:%S')}] Gen {current_gen} reached — running anchor evals...")
            run_eval(current_gen, current_path, anchors, num_pairs)
            already_evaluated.add(current_gen)
            state["evaluated_gens"] = sorted(already_evaluated)
            state["next_eval_at"] = current_gen + eval_every
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Gen {current_gen} — next eval at gen {next_eval_at}. Waiting...")

        time.sleep(120)   # check every 2 min


# ── main eval run ─────────────────────────────────────────────────────────────

def run_eval(current_gen: int, current_path: str, anchors: list[int], num_pairs: int):
    print(f"Loading current checkpoint: gen {current_gen} ({current_path})")
    current_model = load_model(current_path)

    results = []
    for anchor_gen in anchors:
        anchor_path = ckpt_path(anchor_gen)
        if not os.path.exists(anchor_path):
            print(f"  Anchor gen {anchor_gen}: checkpoint not found, skipping.")
            continue
        if anchor_gen == current_gen:
            print(f"  Anchor gen {anchor_gen}: same as current, skipping.")
            continue

        print(f"\nEvaluating gen {current_gen} vs anchor gen {anchor_gen} ({num_pairs} pairs = {num_pairs*2} games)...")
        anchor_model = load_model(anchor_path)
        r = evaluate_vs_anchor(current_model, current_gen, anchor_model, anchor_gen,
                               num_pairs=num_pairs)
        results.append(r)
        del anchor_model   # free memory

    if results:
        log_results(results)
        print_summary(results)
    else:
        print("No valid anchor matchups to run.")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anchor evaluation for domino training")
    parser.add_argument("--current", type=str, default=None,
                        help="Path to current checkpoint (default: latest in checkpoints/)")
    parser.add_argument("--current-gen", type=int, default=None,
                        help="Gen number of current checkpoint (auto-detected if not set)")
    parser.add_argument("--anchors", type=int, nargs="+", default=DEFAULT_ANCHORS,
                        help=f"Anchor gen numbers (default: {DEFAULT_ANCHORS})")
    parser.add_argument("--games", type=int, default=200,
                        help="Total games per anchor matchup (must be even; default: 200 = 100 pairs)")
    parser.add_argument("--watch", action="store_true",
                        help="Watch mode: keep running and eval every --eval-every new gens")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="In watch mode, eval every N new gens (default: 5)")
    args = parser.parse_args()

    num_pairs = args.games // 2

    if args.watch:
        watch_and_eval(args.anchors, num_pairs, args.eval_every)
        return

    # Single run
    if args.current:
        current_path = args.current
        if args.current_gen:
            current_gen = args.current_gen
        else:
            # Try to parse gen from filename
            bn = os.path.basename(current_path)
            try:
                current_gen = int(bn.replace("domino_gen_", "").replace(".pt", ""))
            except ValueError:
                current_gen = -1
    else:
        current_gen, current_path = latest_phase4_checkpoint()
        print(f"Auto-detected latest checkpoint: gen {current_gen}")

    run_eval(current_gen, current_path, args.anchors, num_pairs)


if __name__ == "__main__":
    main()
