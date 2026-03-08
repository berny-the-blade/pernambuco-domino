"""
render_partnership_case.py

Print curated partnership suite cases in readable ASCII for debugging.

Usage:
    python tests/render_partnership_case.py                  # all cases
    python tests/render_partnership_case.py LOCK_BUILD_001   # one case by id
    python tests/render_partnership_case.py sacrifice_to_lock # filter by theme
"""

import json
import sys
from pathlib import Path

SUITE_PATH = Path("tests/partnership_suite.json")


def fmt_tile(t: list) -> str:
    return f"[{t[0]}|{t[1]}]"


def move_str(m: dict) -> str:
    if m is None:
        return "None"
    if m.get("side") == "pass":
        return "PASS"
    return f"{fmt_tile(m['tile'])} @ {m['side']}"


def render_board(board: dict) -> None:
    le = board["left_end"]
    re = board["right_end"]
    ln = board["board_length"]
    tiles = " ... ".join([f"[{le}]"] + ["[ ]"] * max(0, ln - 2) + [f"[{re}]"])
    print(f"  {tiles}")


def render_case(case: dict) -> None:
    print("=" * 72)
    print(f"  {case['id']}  |  theme: {case.get('theme', '')}")
    print(f"  {case.get('description', '')}")
    print("-" * 72)

    board = case["board"]
    print(f"  Player to move : P{case['player_to_move']} (team {case['team_to_move']})")
    print(f"  Board          : L={board['left_end']}  ...({board['board_length']} tiles)...  R={board['right_end']}")
    print()

    history = board.get("history", [])
    if history:
        print("  History:")
        for h in history:
            print(f"    P{h['player']}: {fmt_tile(h['tile'])} @ {h.get('side','?')}")
        print()

    print("  Hands:")
    for p in range(4):
        key = f"player_{p}"
        hand = case["hands"].get(key, [])
        if hand:
            hand_str = "  ".join(fmt_tile(t) for t in hand)
        else:
            hand_str = "(hidden/empty)"
        marker = " <-- to move" if p == case["player_to_move"] else ""
        print(f"    P{p}: {hand_str}{marker}")
    print()

    cant = case.get("cant_have", {})
    active_cant = {k: v for k, v in cant.items() if v}
    if active_cant:
        print("  Can't have:")
        for p, suits in active_cant.items():
            print(f"    P{p}: lacks suit(s) {suits}")
        print()

    exp = case["expected"]
    preferred   = exp.get("preferred_moves",   [])
    acceptable  = exp.get("acceptable_moves",  [])
    discouraged = exp.get("discouraged_moves", [])

    print("  Expected:")
    if preferred:
        print(f"    PREFERRED  : {' | '.join(move_str(m) for m in preferred)}")
    if acceptable:
        print(f"    ACCEPTABLE : {' | '.join(move_str(m) for m in acceptable)}")
    if discouraged:
        for m in discouraged:
            reason = m.get("reason", "")
            print(f"    DISCOURAGED: {move_str(m)}  ({reason})")

    tags = case.get("tags", [])
    if tags:
        print(f"\n  Tags: {', '.join(tags)}")

    print("=" * 72)
    print()


def load_suite(path: Path = SUITE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    suite = load_suite()
    cases = suite["cases"]
    total = len(cases)

    filter_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if filter_arg:
        matched = [c for c in cases
                   if filter_arg.upper() in c["id"].upper()
                   or filter_arg.lower() in c.get("theme", "").lower()]
        if not matched:
            print(f"No cases matching '{filter_arg}'")
            sys.exit(1)
        cases = matched

    print(f"\n{suite['name']}")
    print(f"Showing {len(cases)}/{total} cases\n")

    for case in cases:
        render_case(case)
