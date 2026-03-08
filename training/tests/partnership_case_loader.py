"""
partnership_case_loader.py

Converts symbolic suite moves → 57-action indices and validates cases
against the actual engine tile ordering.

Action space:
    0  .. 27  play tile on left
    28 .. 55  play tile on right
    56        pass

Tile ordering: imported directly from domino_env to guarantee alignment.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from domino_env import TILES, NUM_TILES  # single source of truth

SUITE_PATH = Path("tests/partnership_suite.json")
NN_NUM_ACTIONS = 57

# Build lookup: (sorted tuple) → tile index, using engine ordering
TILE_TO_INDEX: dict[tuple, int] = {
    (min(l, r), max(l, r)): i for i, (l, r) in enumerate(TILES)
}


# ── Conversion helpers ──────────────────────────────────────────────────────

def tile_to_index(tile: list) -> int:
    key = (min(tile[0], tile[1]), max(tile[0], tile[1]))
    if key not in TILE_TO_INDEX:
        raise ValueError(f"Unknown tile: {tile}")
    return TILE_TO_INDEX[key]


def symbolic_move_to_action_idx(move: dict) -> int:
    """
    Convert symbolic move → canonical 57-action index.
      0-27  = play on left
      28-55 = play on right
      56    = pass
    Opening moves (side='open') treated as left by convention.
    """
    side = move.get("side")
    if side == "pass":
        return 56
    tile = move.get("tile")
    if tile is None:
        raise ValueError(f"Move missing tile: {move}")
    idx = tile_to_index(tile)
    if side in ("left", "open"):
        return idx
    if side == "right":
        return 28 + idx
    raise ValueError(f"Unknown move side: {side!r}")


def action_idx_to_symbolic(action_idx: int) -> dict:
    """Reverse: action index → symbolic move dict."""
    if action_idx == 56:
        return {"side": "pass"}
    if 0 <= action_idx < 28:
        l, r = TILES[action_idx]
        return {"tile": [l, r], "side": "left"}
    if 28 <= action_idx < 56:
        l, r = TILES[action_idx - 28]
        return {"tile": [l, r], "side": "right"}
    raise ValueError(f"Invalid action index: {action_idx}")


def case_expected_action_sets(case: dict) -> dict[str, set]:
    """Return {preferred, acceptable, discouraged} as sets of action indices."""
    exp = case["expected"]
    return {
        "preferred":   {symbolic_move_to_action_idx(m) for m in exp.get("preferred_moves",   [])},
        "acceptable":  {symbolic_move_to_action_idx(m) for m in exp.get("acceptable_moves",  [])},
        "discouraged": {symbolic_move_to_action_idx(m) for m in exp.get("discouraged_moves", [])},
    }


def score_action(action_idx: int, case: dict) -> tuple[float, str]:
    """Score an action index against a case's expected sets."""
    sets   = case_expected_action_sets(case)
    sc     = case.get("scoring", {})
    if action_idx in sets["preferred"]:
        return sc.get("preferred_score",   1.00), "preferred"
    if action_idx in sets["acceptable"]:
        return sc.get("acceptable_score",  0.50), "acceptable"
    if action_idx in sets["discouraged"]:
        return sc.get("discouraged_score", 0.00), "discouraged"
    return sc.get("fallback_score", 0.25), "fallback"


# ── Case observation builder ─────────────────────────────────────────────────

def build_case_observation(case: dict) -> dict:
    """
    Build a lightweight observation dict for Python-side tests / heuristic adapters.

    Returned keys match what downstream adapters typically need:
      player        - int, player to move
      hand_indices  - list[int], tile indices in hand
      left_end      - int
      right_end     - int
      board_length  - int
      played        - list[int], tile indices played so far
      cant_have     - dict[int, list[int]], player → list of absent suits
    """
    p        = case["player_to_move"]
    board    = case["board"]
    hand_key = f"player_{p}"

    hand_indices = [tile_to_index(t) for t in case["hands"].get(hand_key, [])]
    played       = [tile_to_index(h["tile"]) for h in board.get("history", [])]

    cant_have_raw = case.get("cant_have", {})
    cant_have = {int(k): list(v) for k, v in cant_have_raw.items()}

    hands_raw = case.get("hands", {})
    hand_sizes = {
        i: len(hands_raw.get(f"player_{i}", [])) or 6
        for i in range(4)
    }

    return {
        "player":       p,
        "hand_indices": hand_indices,
        "hand":         hand_indices,            # alias for adapter compatibility
        "left_end":     board["left_end"],
        "right_end":    board["right_end"],
        "board_length": board["board_length"],
        "played":       played,
        "cant_have":    cant_have,
        "hand_sizes":   hand_sizes,
        "history":      board.get("history", []),
        "raw_case":     case,
    }


# ── Suite loader ─────────────────────────────────────────────────────────────

def load_partnership_suite(path: Path | str = SUITE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Suite validator ───────────────────────────────────────────────────────────

def validate_suite(path: Path = SUITE_PATH) -> bool:
    """
    Validate every move in the suite can be converted to a valid action index.
    Prints errors and returns True if clean, False if any case is broken.
    """
    with open(path, encoding="utf-8") as f:
        suite = json.load(f)

    errors = 0
    for case in suite["cases"]:
        cid = case["id"]
        exp = case["expected"]
        for bucket in ("preferred_moves", "acceptable_moves", "discouraged_moves"):
            for m in exp.get(bucket, []):
                try:
                    idx = symbolic_move_to_action_idx(m)
                    assert 0 <= idx < NN_NUM_ACTIONS, f"out of range: {idx}"
                except Exception as e:
                    print(f"  ERROR [{cid}] {bucket} {m}: {e}")
                    errors += 1

    if errors == 0:
        print(f"Suite OK — {len(suite['cases'])} cases, all moves valid.")
        return True
    else:
        print(f"\n{errors} error(s) found.")
        return False


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate and inspect partnership suite actions")
    parser.add_argument("--suite",    default=str(SUITE_PATH))
    parser.add_argument("--validate", action="store_true", help="Validate all suite moves")
    parser.add_argument("--case",     default=None, help="Print action indices for a single case id")
    args = parser.parse_args()

    if args.validate or args.case is None:
        ok = validate_suite(Path(args.suite))
        sys.exit(0 if ok else 1)

    with open(args.suite, encoding="utf-8") as f:
        suite = json.load(f)

    match = next((c for c in suite["cases"] if c["id"] == args.case), None)
    if match is None:
        print(f"Case not found: {args.case}")
        sys.exit(1)

    sets = case_expected_action_sets(match)
    obs  = build_case_observation(match)
    print(f"\n{match['id']}")
    print(f"  Board  : L={obs['left_end']} R={obs['right_end']} len={obs['board_length']}")
    print(f"  Hand   : {[list(TILES[i]) for i in obs['hand_indices']]}")
    for bucket, idxs in sets.items():
        syms = [action_idx_to_symbolic(i) for i in sorted(idxs)]
        print(f"  {bucket:11s}: {idxs}  -> {syms}")
