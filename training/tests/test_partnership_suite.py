"""
test_partnership_suite.py

Evaluate engine move choices against the partnership tactical suite.

Usage:
    python tests/test_partnership_suite.py
    python tests/test_partnership_suite.py --suite tests/partnership_suite.json
    python tests/test_partnership_suite.py --checkpoint checkpoints/domino_gen_0050.pt --sims 200

Scoring:
    preferred  = 1.0   (engine chose a preferred move)
    acceptable = 0.5   (engine chose an acceptable move)
    discouraged = 0.0  (engine chose a discouraged move)
    other      = 0.25  (engine chose something not explicitly categorized)

Pass threshold: mean score >= 0.70 across all cases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domino_env import DominoEnv, TILES as ALL_TILES, NUM_TILES
# TILES is list of (left, right) tuples; build lookup dicts
TILE_INDEX = {lr: i for i, lr in enumerate(ALL_TILES)}  # (l,r) -> index
TILE_BY_LR = {}
for i, (l, r) in enumerate(ALL_TILES):
    TILE_BY_LR[(l, r)] = i
    TILE_BY_LR[(r, l)] = i
from domino_encoder import DominoEncoder
from domino_net import DominoNet
from domino_mcts import DominoMCTS
import torch


# ============================================================
# Scoring constants
# ============================================================
SCORE_PREFERRED   = 1.0
SCORE_ACCEPTABLE  = 0.5
SCORE_DISCOURAGED = 0.0
SCORE_OTHER       = 0.25
PASS_THRESHOLD    = 0.70


# ============================================================
# Data helpers
# ============================================================

@dataclass
class CaseResult:
    case_id: str
    theme: str
    engine_tile: tuple[int, int]
    engine_side: str
    outcome: str      # preferred | acceptable | discouraged | other
    score: float
    preferred: list
    discouraged: list


def canonical_move(tile: tuple | None, side: str | None) -> tuple:
    """Canonical (sorted_tile, side) key — handles [1,4] == [4,1] symmetry."""
    if side == "pass" or tile is None:
        return ("pass",)
    if side is None:
        return ("invalid",)
    a, b = sorted(tile)
    return (a, b, side)


def canonical_spec(move_spec: dict) -> tuple:
    if move_spec is None:
        return ("invalid",)
    if move_spec.get("side") == "pass":
        return ("pass",)
    tile = move_spec.get("tile")
    side = move_spec.get("side")
    if not isinstance(tile, (list, tuple)) or len(tile) != 2 or side is None:
        return ("invalid",)
    a, b = sorted(tile)
    return (a, b, side)


def score_move(engine_tile, engine_side, case: dict) -> tuple[str, float]:
    pred = canonical_move(engine_tile, engine_side)
    expected = case["expected"]
    scoring  = case.get("scoring", {})

    preferred   = {canonical_spec(m) for m in expected.get("preferred_moves", [])}
    acceptable  = {canonical_spec(m) for m in expected.get("acceptable_moves", [])}
    discouraged = {canonical_spec(m) for m in expected.get("discouraged_moves", [])}

    if pred in preferred:
        return "preferred",   scoring.get("preferred_score",   SCORE_PREFERRED)
    if pred in acceptable:
        return "acceptable",  scoring.get("acceptable_score",  SCORE_ACCEPTABLE)
    if pred in discouraged:
        return "discouraged", scoring.get("discouraged_score", SCORE_DISCOURAGED)
    return "other", scoring.get("fallback_score", SCORE_OTHER)


# ============================================================
# Engine interface
# ============================================================

def build_env_from_case(case: dict) -> tuple[DominoEnv, dict]:
    """
    Reconstruct a DominoEnv state from a suite case by direct field injection.
    Returns (env, obs) with the board at the described position.
    """
    board = case["board"]
    hands_raw = case["hands"]
    player = case["player_to_move"]

    # Convert [[l,r],...] specs to tile indices
    env = DominoEnv()
    env.reset(seed=0)  # Initialize all fields

    env.hands = [[] for _ in range(4)]
    for p_key, tiles in hands_raw.items():
        p = int(p_key.split("_")[1])
        env.hands[p] = [TILE_BY_LR[tuple(lr)] for lr in tiles if lr]

    env.left_end  = board["left_end"]
    env.right_end = board["right_end"]
    env.board     = list(range(board["board_length"]))  # placeholder length
    env.current_player = player
    env.game_over = False
    env.pass_count = 0

    # Reconstruct played set from all non-hand tiles listed in history
    env.played = set()
    for entry in board.get("history", []):
        lr = tuple(entry["tile"])
        idx = TILE_BY_LR.get(lr)
        if idx is not None:
            env.played.add(idx)

    # cant_have from case spec
    env.cant_have = [set() for _ in range(4)]
    for k, pips in case.get("cant_have", {}).items():
        env.cant_have[int(k)] = set(pips)

    env.plays_by = [[] for _ in range(4)]

    obs = env.get_obs()
    return env, obs


def get_engine_move(case: dict, model: Any, sims: int, device: str) -> tuple[tuple, str] | None:
    """
    Run engine on the case position and return (tile, side).
    Falls back to greedy if sims=0.
    """
    try:
        env, obs = build_env_from_case(case)
    except Exception as e:
        print(f"    [SKIP] Could not build env: {e}")
        return None

    mask = env.get_legal_moves_mask()
    if mask.sum() == 0:
        return None

    enc = DominoEncoder()
    enc.reset()
    state_np = enc.encode(obs)

    if sims > 0 and model is not None:
        mcts = DominoMCTS(model, num_simulations=sims)
        pi = mcts.get_action_probs(env, enc, temperature=0.0)
    else:
        pi, _ = model.predict(state_np, mask) if model else (mask / mask.sum(), 0)

    action = int(np.argmax(pi * mask))

    # Map action index back to (tile, side)
    if action == 56:
        return (None, None), "pass"
    side = "right" if action >= 28 else "left"
    tile_idx = action - (28 if action >= 28 else 0)
    l, r = ALL_TILES[tile_idx]
    return (l, r), side


# ============================================================
# Suite runner
# ============================================================

def run_suite(suite_path: str, checkpoint: str | None, sims: int, device: str) -> list[CaseResult]:
    with open(suite_path, encoding="utf-8") as f:
        suite = json.load(f)

    model = None
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        input_dim = state_dict["input_fc.weight"].shape[1]
        model = DominoNet(input_dim=input_dim, hidden_dim=256, num_actions=57, num_blocks=4)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        print(f"Loaded checkpoint: {checkpoint} (dim={input_dim})")
    else:
        print("No checkpoint — using random policy (baseline)")

    results = []
    for case in suite["cases"]:
        cid = case["id"]
        theme = case["theme"]
        print(f"\n[{cid}] {case['description'][:70]}")

        result = get_engine_move(case, model, sims, device)
        if result is None:
            print(f"    SKIP")
            continue

        engine_tile, engine_side = result
        outcome, score = score_move(engine_tile, engine_side, case)

        mark = "OK" if outcome == "preferred" else ("~" if outcome == "acceptable" else "XX")
        print(f"    Engine: {engine_tile} {engine_side} -> {outcome.upper()} {mark} (score={score:.2f})")

        results.append(CaseResult(
            case_id=cid, theme=theme,
            engine_tile=engine_tile, engine_side=engine_side,
            outcome=outcome, score=score,
            preferred=case["expected"].get("preferred_moves", []),
            discouraged=case["expected"].get("discouraged_moves", []),
        ))

    return results


def print_summary(results: list[CaseResult]) -> float:
    if not results:
        print("\nNo results.")
        return 0.0

    mean_score = sum(r.score for r in results) / len(results)
    n_pref = sum(1 for r in results if r.outcome == "preferred")
    n_disc = sum(1 for r in results if r.outcome == "discouraged")
    n_other = sum(1 for r in results if r.outcome == "other")

    print(f"\n{'='*60}")
    print(f"PARTNERSHIP SUITE RESULTS")
    print(f"  Cases:       {len(results)}")
    print(f"  Preferred:   {n_pref}/{len(results)} ({n_pref/len(results)*100:.0f}%)")
    print(f"  Discouraged: {n_disc}/{len(results)} ({n_disc/len(results)*100:.0f}%)")
    print(f"  Other:       {n_other}/{len(results)}")
    print(f"  Mean score:  {mean_score:.3f}")
    print(f"  Threshold:   {PASS_THRESHOLD}")
    verdict = "PASS" if mean_score >= PASS_THRESHOLD else "FAIL"
    print(f"  Verdict:     {verdict}")
    print(f"{'='*60}")

    # By theme
    themes: dict[str, list] = {}
    for r in results:
        themes.setdefault(r.theme, []).append(r.score)
    print("\nBy theme:")
    for theme, scores in sorted(themes.items()):
        print(f"  {theme:<40} mean={sum(scores)/len(scores):.2f}  n={len(scores)}")

    return mean_score


def main():
    parser = argparse.ArgumentParser(description="Partnership tactical suite evaluator")
    parser.add_argument("--suite", default="tests/partnership_suite.json")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--sims", type=int, default=0, help="MCTS sims per move (0=greedy)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    results = run_suite(args.suite, args.checkpoint, args.sims, args.device)
    score = print_summary(results)
    sys.exit(0 if score >= PASS_THRESHOLD else 1)


if __name__ == "__main__":
    main()
