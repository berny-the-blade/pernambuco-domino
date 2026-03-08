"""
test_partnership_suite.py

Evaluate engine move choices against the partnership tactical regression suite.

Usage:
    python tests/test_partnership_suite.py                              # greedy NN, threshold 0.70
    python tests/test_partnership_suite.py 0.85                         # stricter threshold
    python tests/test_partnership_suite.py --checkpoint PATH --sims 200 # MCTS
    python tests/test_partnership_suite.py --stub                       # smoke test (stub returns preferred)

Scoring:
    preferred   = 1.0
    acceptable  = 0.5
    discouraged = 0.0
    fallback    = 0.25  (engine chose something not explicitly categorized)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domino_env import DominoEnv, TILES as ALL_TILES, NUM_TILES
from domino_encoder import DominoEncoder
from domino_net import DominoNet
from domino_mcts import DominoMCTS
import torch

SUITE_PATH = Path("tests/partnership_suite.json")
TILE_BY_LR: dict[tuple, int] = {}
for _i, (_l, _r) in enumerate(ALL_TILES):
    TILE_BY_LR[(_l, _r)] = _i
    TILE_BY_LR[(_r, _l)] = _i


# ============================================================
# Canonical scoring helpers
# ============================================================

def canonical_move(move: dict | None) -> tuple:
    """Normalize move spec for comparison. Handles tile symmetry ([1,4] == [4,1])."""
    if move is None:
        return ("invalid",)
    side = move.get("side")
    if side == "pass":
        return ("pass",)
    tile = move.get("tile")
    if not isinstance(tile, (list, tuple)) or len(tile) != 2 or side is None:
        return ("invalid",)
    a, b = sorted(tile)
    return (a, b, side)


def move_set(moves: list[dict]) -> set:
    return {canonical_move(m) for m in moves}


def score_case(pred_move: dict, case: dict) -> tuple[float, str]:
    pred = canonical_move(pred_move)
    exp  = case["expected"]
    sc   = case.get("scoring", {})

    preferred   = move_set(exp.get("preferred_moves",   []))
    acceptable  = move_set(exp.get("acceptable_moves",  []))
    discouraged = move_set(exp.get("discouraged_moves", []))

    if pred in preferred:   return sc.get("preferred_score",   1.00), "preferred"
    if pred in acceptable:  return sc.get("acceptable_score",  0.50), "acceptable"
    if pred in discouraged: return sc.get("discouraged_score", 0.00), "discouraged"
    return sc.get("fallback_score", 0.25), "fallback"


def pretty_move(move: dict | None) -> str:
    if move is None: return "None"
    if move.get("side") == "pass": return "PASS"
    return f"{move.get('tile')} @ {move.get('side')}"


# ============================================================
# Suite loader
# ============================================================

def load_suite(path: Path = SUITE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Engine adapters
# ============================================================

def build_env_from_case(case: dict) -> tuple[DominoEnv, dict]:
    """Inject board state from suite case into DominoEnv."""
    board = case["board"]
    player = case["player_to_move"]

    env = DominoEnv()
    env.reset(seed=0)
    env.hands = [[] for _ in range(4)]
    for pk, tiles in case["hands"].items():
        p = int(pk.split("_")[1])
        env.hands[p] = [TILE_BY_LR[tuple(lr)] for lr in tiles if lr]

    env.left_end       = board["left_end"]
    env.right_end      = board["right_end"]
    env.board          = list(range(board["board_length"]))
    env.current_player = player
    env.game_over      = False
    env.pass_count     = 0
    env.played         = {TILE_BY_LR[tuple(e["tile"])] for e in board.get("history", [])}
    env.cant_have      = [set() for _ in range(4)]
    env.plays_by       = [[] for _ in range(4)]

    return env, env.get_obs()


def make_engine_fn(model, sims: int, device: str):
    """Returns engine_fn(case) -> {tile, side} using real NN/MCTS."""
    def engine_fn(case: dict) -> dict:
        try:
            env, obs = build_env_from_case(case)
        except Exception as e:
            return {"side": "pass", "_error": str(e)}

        mask = env.get_legal_moves_mask()
        if mask.sum() == 0:
            return {"side": "pass"}

        enc = DominoEncoder(); enc.reset()
        state_np = enc.encode(obs)

        if sims > 0 and model is not None:
            mcts = DominoMCTS(model, num_simulations=sims)
            pi   = mcts.get_action_probs(env, enc, temperature=0.0)
        else:
            pi, _ = model.predict(state_np, mask) if model else (mask / mask.sum(), 0.0)

        action = int(np.argmax(pi * mask))
        if action == 56:
            return {"side": "pass"}
        side     = "right" if action >= 28 else "left"
        tile_idx = action - (28 if action >= 28 else 0)
        l, r     = ALL_TILES[tile_idx]
        return {"tile": [l, r], "side": side}

    return engine_fn


def engine_fn_stub(case: dict) -> dict:
    """Smoke-test stub — returns first preferred move so suite always passes."""
    preferred = case["expected"].get("preferred_moves", [])
    if preferred: return preferred[0]
    acceptable = case["expected"].get("acceptable_moves", [])
    if acceptable: return acceptable[0]
    return {"side": "pass"}


# ============================================================
# Suite evaluator
# ============================================================

def evaluate_suite(engine_fn, suite_path: Path = SUITE_PATH) -> dict:
    suite = load_suite(suite_path)
    cases = suite["cases"]

    total_score    = 0.0
    results        = []
    theme_scores   = defaultdict(float)
    theme_counts   = defaultdict(int)
    bucket_counts  = defaultdict(int)

    for case in cases:
        pred_move       = engine_fn(case)
        score, bucket   = score_case(pred_move, case)
        theme           = case.get("theme", "unknown")

        total_score          += score
        theme_scores[theme]  += score
        theme_counts[theme]  += 1
        bucket_counts[bucket]+= 1

        results.append({
            "id":            case["id"],
            "theme":         theme,
            "description":   case.get("description", ""),
            "pred_move":     pred_move,
            "pred_move_str": pretty_move(pred_move),
            "score":         score,
            "bucket":        bucket,
        })

    avg_score = total_score / max(1, len(cases))
    theme_avg = {t: theme_scores[t] / theme_counts[t] for t in theme_scores}

    return {
        "suite_name":   suite.get("name", "Unnamed Suite"),
        "num_cases":    len(cases),
        "total_score":  total_score,
        "avg_score":    avg_score,
        "bucket_counts": dict(bucket_counts),
        "theme_avg":    theme_avg,
        "results":      results,
    }


# ============================================================
# Report printer
# ============================================================

def print_report(report: dict) -> None:
    print("=" * 72)
    print(report["suite_name"])
    print("=" * 72)
    print(f"Cases      : {report['num_cases']}")
    print(f"Avg Score  : {report['avg_score']:.3f}")
    print(f"Total Score: {report['total_score']:.3f}")
    print()
    print("Bucket counts:")
    for k in ["preferred", "acceptable", "discouraged", "fallback"]:
        print(f"  {k:12s}: {report['bucket_counts'].get(k, 0)}")
    print()
    print("Theme averages:")
    for theme, avg in sorted(report["theme_avg"].items()):
        print(f"  {theme:40s} {avg:.3f}")
    print()
    print("Per-case results:")
    for r in report["results"]:
        print(f"  {r['id']:30s} {r['bucket']:11s} {r['score']:.2f}  {r['pred_move_str']}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Partnership tactical suite evaluator")
    parser.add_argument("threshold", nargs="?", type=float, default=0.70)
    parser.add_argument("--suite",       default=str(SUITE_PATH))
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--sims",        type=int, default=0)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--stub",        action="store_true", help="Use stub engine (smoke test)")
    args = parser.parse_args()

    if args.stub:
        engine_fn = engine_fn_stub
    else:
        model = None
        if args.checkpoint:
            ckpt       = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            state_dict = ckpt.get("model_state_dict", ckpt)
            input_dim  = state_dict["input_fc.weight"].shape[1]
            model      = DominoNet(input_dim=input_dim, hidden_dim=256, num_actions=57, num_blocks=4)
            model.load_state_dict(state_dict)
            model.to(args.device).eval()
            print(f"Loaded checkpoint: {args.checkpoint} (dim={input_dim})")
        engine_fn = make_engine_fn(model, args.sims, args.device)

    report = evaluate_suite(engine_fn, Path(args.suite))
    print_report(report)

    if report["avg_score"] < args.threshold:
        print(f"\nFAIL: avg={report['avg_score']:.3f} < threshold={args.threshold:.3f}")
        sys.exit(1)
    print(f"\nPASS: avg={report['avg_score']:.3f} >= threshold={args.threshold:.3f}")


if __name__ == "__main__":
    main()
