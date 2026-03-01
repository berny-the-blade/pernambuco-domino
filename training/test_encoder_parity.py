"""
Encoder parity test: Python DominoEncoder vs JS _nnEncodeState.

Uses the canonical Raw Snapshot Schema (v1).
Run this to produce expected vectors, then compare against _testEncoderParity() in browser.

Usage:
  python test_encoder_parity.py              # run built-in scenarios
  python test_encoder_parity.py snap.json    # encode a snapshot file
"""

import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domino_env import TILES, NUM_TILES, TILE_ID
from domino_encoder import DominoEncoder


# ========== Canonical Snapshot Loader ==========

def load_snapshot(line):
    """Load a v1 snapshot from JSON string or dict."""
    snap = json.loads(line) if isinstance(line, str) else line
    assert snap["version"] == 1, f"Unknown snapshot version: {snap['version']}"
    return snap


# ========== Canonical Action Mask (must match JS exactly) ==========

def nn_action_mask(snap):
    """Compute 57-dim action mask from snapshot. Matches JS _nnActionMask."""
    mask = np.zeros(57, dtype=np.float32)

    hand = snap["hand"]
    lE = snap["left_end"]
    rE = snap["right_end"]
    bLen = snap["board_length"]

    def can_play(tile_idx):
        left, right = TILES[tile_idx]
        if bLen == 0:
            return True
        return (left == lE or right == lE or
                left == rE or right == rE)

    playable = [t for t in hand if can_play(t)]

    if not playable:
        mask[56] = 1.0
        return mask

    for t in playable:
        left, right = TILES[t]
        if bLen == 0:
            mask[t] = 1.0
        else:
            if left == lE or right == lE:
                mask[t] = 1.0
            if left == rE or right == rE:
                mask[28 + t] = 1.0

    # Symmetry: when both ends equal, only use left slot
    if bLen > 0 and lE == rE:
        mask[28:56] = 0.0

    return mask


# ========== Snapshot → DominoEncoder obs dict ==========

def snapshot_to_obs(snap):
    """Convert canonical snapshot to DominoEnv-style obs dict."""
    me = snap["player"]

    # Build cant_have as dict of sets
    cant_have = {}
    for p_str, nums in snap["cant_have"].items():
        cant_have[int(p_str)] = set(nums)

    # Derive hand sizes: 6 - tiles_played_by_player
    hand_sizes = [0, 0, 0, 0]
    for p_str, tiles in snap["plays_by"].items():
        p = int(p_str)
        hand_sizes[p] = max(0, 6 - len(tiles))
    # Override my hand size with actual
    hand_sizes[me] = len(snap["hand"])

    return {
        'player': me,
        'hand': list(snap["hand"]),
        'played': list(snap["played"]),
        'left_end': snap["left_end"],
        'right_end': snap["right_end"],
        'cant_have': cant_have,
        'hand_sizes': hand_sizes,
        'board_length': snap["board_length"],
    }


def encode_snapshot(snap):
    """Encode a snapshot using DominoEncoder. Returns (state, mask)."""
    me = snap["player"]
    partner = (me + 2) % 4
    lho = (me + 1) % 4
    rho = (me + 3) % 4

    encoder = DominoEncoder()

    # Replay events: update_on_play for other players' tiles
    for p_str, tiles in snap["plays_by"].items():
        p = int(p_str)
        if p == me:
            continue
        # Map absolute player to relative
        if p == partner:
            rel = 0
        elif p == lho:
            rel = 1
        elif p == rho:
            rel = 2
        else:
            continue
        for tile_idx in tiles:
            encoder.update_on_play(rel, tile_idx)

    # Replay passes: cant_have entries imply passes
    # For each other player, if they can't have numbers that match board ends,
    # that means they passed. We use update_on_pass for the pass events.
    # However, cant_have accumulates numbers from ALL passes throughout the game,
    # and we don't have the exact board-end context for each pass.
    # The _sync_belief will apply cantHave constraints anyway.
    # For proper persistent belief, we'd need the event log.
    # For now, _sync_belief handles it.

    # Build obs dict
    obs = snapshot_to_obs(snap)

    # Derive scores
    my_team = me % 2
    my_score = snap["match_score"][my_team]
    opp_score = snap["match_score"][1 - my_team]
    multiplier = snap["score_multiplier"]

    state = encoder.encode(obs, my_score=my_score, opp_score=opp_score, multiplier=multiplier)
    mask = nn_action_mask(snap)

    return state, mask


# ========== Parity Check ==========

def check_parity(state_py, mask_py, js_state, js_mask, eps=1e-6):
    """Compare Python and JS outputs. Returns (pass, mismatches)."""
    mismatches = []

    for i in range(185):
        if abs(float(state_py[i]) - float(js_state[i])) > eps:
            mismatches.append(f"STATE[{i}]: PY={state_py[i]:.8f} JS={js_state[i]:.8f}")

    for i in range(57):
        if abs(float(mask_py[i]) - float(js_mask[i])) > eps:
            mismatches.append(f"MASK[{i}]: PY={mask_py[i]:.8f} JS={js_mask[i]:.8f}")

    return len(mismatches) == 0, mismatches


# ========== Built-in Test Scenarios ==========

SCENARIO_1 = {
    "version": 1,
    "player": 0,
    "hand": [1, 2, 3, 4, 5],
    "played": [0, 6, 12],
    "left_end": 1,
    "right_end": 0,
    "board_length": 3,
    "cant_have": {"0": [], "1": [], "2": [], "3": [0, 1]},
    "plays_by": {"0": [0], "1": [6], "2": [12], "3": []},
    "match_score": [0, 0],
    "score_multiplier": 1
}

SCENARIO_2 = {
    "version": 1,
    "player": 2,
    "hand": [13, 14, 15, 16],
    "played": [0, 7, 1, 8, 6, 12],
    "left_end": 3,
    "right_end": 3,
    "board_length": 6,
    "cant_have": {"0": [5], "1": [3, 5], "2": [], "3": [3]},
    "plays_by": {"0": [0, 1], "1": [7, 8], "2": [6, 12], "3": []},
    "match_score": [2, 4],
    "score_multiplier": 2
}

SCENARIO_3 = {
    "version": 1,
    "player": 1,
    "hand": [20, 21, 22],
    "played": [0, 7, 13, 18, 1, 8, 14, 19, 2, 9, 15, 27],
    "left_end": 6,
    "right_end": 5,
    "board_length": 12,
    "cant_have": {"0": [2, 4], "1": [], "2": [3, 6], "3": [2, 5, 6]},
    "plays_by": {"0": [0, 1, 2], "1": [7, 8, 9], "2": [13, 14, 15], "3": [18, 19, 27]},
    "match_score": [5, 5],
    "score_multiplier": 4
}


def run_tests():
    """Run all built-in scenarios and dump results."""
    scenarios = [
        ("Scenario 1: Early game, P0 observes", SCENARIO_1),
        ("Scenario 2: Mid game, P2 observes, symmetric ends", SCENARIO_2),
        ("Scenario 3: Late game, P1 observes, match point", SCENARIO_3),
    ]

    all_results = []

    for name, snap in scenarios:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        state, mask = encode_snapshot(snap)

        # Print belief section
        print(f"Belief [91:175] (3 zones × 28 tiles):")
        for t in range(NUM_TILES):
            l, r = TILES[t]
            b = [state[91 + z * 28 + t] for z in range(3)]
            s = sum(b)
            marker = ""
            if t in snap["hand"]:
                marker = " [HAND]"
            elif t in snap["played"]:
                marker = " [PLAYED]"
            if abs(s - 1.0) > 0.001 and marker == "":
                marker = " *** SUM != 1 ***"
            print(f"  tile {t:2d} ({l}-{r}): P={b[0]:.4f} L={b[1]:.4f} R={b[2]:.4f} Σ={s:.4f}{marker}")

        # Print active mask
        print(f"\nAction mask (active):")
        for i in range(57):
            if mask[i] > 0:
                if i < 28:
                    l, r = TILES[i]
                    print(f"  [{i:2d}] tile ({l}-{r}) on LEFT")
                elif i < 56:
                    l, r = TILES[i - 28]
                    print(f"  [{i:2d}] tile ({l}-{r}) on RIGHT")
                else:
                    print(f"  [56] PASS")

        all_results.append({
            'name': name,
            'snapshot': snap,
            'state': [round(float(x), 8) for x in state],
            'mask': [round(float(x), 8) for x in mask],
        })

    # Dump to JSON
    out_path = os.path.join(os.path.dirname(__file__), 'parity_expected.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results written to {out_path}")
    print("Load in browser: _testEncoderParity(await fetch('parity_expected.json').then(r=>r.json()))")

    return all_results


def run_mass_parity(json_path, eps=1e-6):
    """Cross-validate against JS massParityCheck() output.

    Usage:
      1. In browser: massParityCheck(100) → downloads parity_check_NNNN.json
      2. python test_encoder_parity.py --mass parity_check_NNNN.json
    """
    with open(json_path) as f:
        data = json.load(f)

    total = len(data)
    passed = 0
    failed = 0
    state_mismatches = 0
    mask_mismatches = 0

    for i, entry in enumerate(data):
        snap = entry['snapshot']
        js_state = entry['js_state']
        js_mask = entry['js_mask']

        py_state, py_mask = encode_snapshot(snap)

        s_ok = True
        m_ok = True

        for j in range(185):
            if abs(float(py_state[j]) - float(js_state[j])) > eps:
                if s_ok:  # print first mismatch per snapshot
                    print(f"FAIL snap {i} (player={snap['player']}, bLen={snap['board_length']}): "
                          f"STATE[{j}] PY={py_state[j]:.8f} JS={js_state[j]:.8f}")
                s_ok = False
                state_mismatches += 1

        for j in range(57):
            if abs(float(py_mask[j]) - float(js_mask[j])) > eps:
                if m_ok:
                    print(f"FAIL snap {i}: MASK[{j}] PY={py_mask[j]:.8f} JS={js_mask[j]:.8f}")
                m_ok = False
                mask_mismatches += 1

        if s_ok and m_ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Mass parity: {passed}/{total} PASS, {failed}/{total} FAIL")
    print(f"  State mismatches: {state_mismatches}")
    print(f"  Mask mismatches:  {mask_mismatches}")
    if failed == 0:
        print("✓ ALL PARITY TESTS PASSED")
    else:
        print(f"✗ {failed} SNAPSHOTS FAILED")
    print(f"{'='*60}")
    return failed == 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--mass':
        if len(sys.argv) < 3:
            print("Usage: python test_encoder_parity.py --mass <parity_check.json>")
            sys.exit(1)
        ok = run_mass_parity(sys.argv[2])
        sys.exit(0 if ok else 1)
    elif len(sys.argv) > 1:
        # Encode a specific snapshot file
        with open(sys.argv[1]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                snap = load_snapshot(line)
                state, mask = encode_snapshot(snap)
                print(json.dumps({
                    'state': [round(float(x), 8) for x in state],
                    'mask': [round(float(x), 8) for x in mask],
                }))
    else:
        run_tests()
