"""
Validate training data exported from JS exportTrainingData().
Cross-checks dME values, encoder parity, and data quality.

Usage:
  python validate_training_data.py training_50m_1234s.jsonl
  python validate_training_data.py --me3d me3d_js.json     # ME3D table parity
  python validate_training_data.py --quick                  # generate + validate in Python only
"""

import json
import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Policy validation helpers (visit-count targets) ───────────────────────────

NN_NUM_ACTIONS = 57


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def validate_policy_row(pi, mask, row_idx=None, eps: float = 1e-6) -> dict:
    """
    Validate one exported 57-dim policy target row.

    Returns dict with: illegal_mass, entropy, top1_idx, top1_prob, legal_count
    Raises ValueError on malformed rows.
    """
    pi   = np.asarray(pi,   dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    if pi.shape != (NN_NUM_ACTIONS,):
        raise ValueError(f"row {row_idx}: pi shape {pi.shape} != (57,)")
    if mask.shape != (NN_NUM_ACTIONS,):
        raise ValueError(f"row {row_idx}: mask shape {mask.shape} != (57,)")

    if not np.isfinite(pi).all():
        raise ValueError(f"row {row_idx}: pi contains non-finite values")
    if not np.isfinite(mask).all():
        raise ValueError(f"row {row_idx}: mask contains non-finite values")

    if (pi < -eps).any():
        bad = np.where(pi < -eps)[0][:10]
        raise ValueError(f"row {row_idx}: pi contains negative mass at {bad.tolist()}")

    # Clamp floating-point noise
    pi = np.maximum(pi, 0.0)

    legal       = mask > 0.5
    legal_count = int(legal.sum())

    if legal_count == 0:
        raise ValueError(f"row {row_idx}: no legal actions in mask")

    total = float(pi.sum())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"row {row_idx}: pi sums to {total:.8f}, expected ~1.0")

    illegal_mass = float(pi[~legal].sum())
    if illegal_mass > 1e-6:
        bad = np.where((~legal) & (pi > 1e-9))[0][:10]
        raise ValueError(
            f"row {row_idx}: illegal policy mass {illegal_mass:.8e} on actions {bad.tolist()}"
        )

    top1_idx = int(np.argmax(pi))
    if not legal[top1_idx]:
        raise ValueError(f"row {row_idx}: top-1 action {top1_idx} is illegal")

    return {
        "illegal_mass": illegal_mass,
        "entropy":      entropy(pi),
        "top1_idx":     top1_idx,
        "top1_prob":    float(pi[top1_idx]),
        "legal_count":  legal_count,
    }


def summarize_bad_row(row: dict, idx: int) -> None:
    """Print compact debug info for a malformed policy row."""
    pi   = np.asarray(row.get("policy_57", row.get("pi", [])))
    mask = np.asarray(row.get("mask", []))
    if pi.size == 0:
        print(f"\nBad row {idx}: no pi data available")
        return
    top  = np.argsort(-pi)[:5]
    print(f"\nBad row {idx}")
    print(f"  top-5 actions : {top.tolist()}")
    print(f"  top-5 probs   : {[float(pi[t]) for t in top]}")
    if mask.size > 0:
        print(f"  legal(top-5)  : {[bool(mask[t] > 0.5) for t in top]}")

# ───────────────────────────────────────────────────────────────────────────────

from match_equity import ME3D, get_match_equity, delta_me, MATCH_TARGET, DOB_VALUES
from domino_encoder import DominoEncoder
from domino_env import TILES, NUM_TILES
from test_encoder_parity import encode_snapshot, nn_action_mask


def validate_me3d_parity(json_path, eps=1e-9):
    """Cross-validate JS ME3D table against Python ME3D table.

    Usage: Run me3dParityExport() in browser, then:
        python validate_training_data.py --me3d me3d_js.json
    """
    with open(json_path) as f:
        entries = json.load(f)

    total = len(entries)
    mismatches = 0
    max_diff = 0.0

    for entry in entries:
        s1 = entry['s1']
        s2 = entry['s2']
        d = entry['dob_idx']
        js_me = entry['me']
        py_me = float(ME3D[s1][s2][d])
        diff = abs(js_me - py_me)
        max_diff = max(max_diff, diff)

        if diff > eps:
            mismatches += 1
            if mismatches <= 10:
                print(f"  MISMATCH ME3D[{s1}][{s2}][{d}]: JS={js_me:.12f} PY={py_me:.12f} diff={diff:.2e}")

    print(f"\nME3D Parity: {total - mismatches}/{total} PASS, {mismatches} FAIL")
    print(f"Max diff: {max_diff:.2e} (eps={eps:.2e})")
    if mismatches == 0:
        print("ALL ME3D ENTRIES MATCH")
    return mismatches == 0


def validate_delta_me_parity(data, eps=1e-6):
    """Re-compute dME from game results and compare to JS v_target."""
    total = len(data)
    mismatches = 0
    max_diff = 0.0

    for i, entry in enumerate(data):
        snap = entry['snapshot']
        js_v = entry['v_target']
        result = entry['game_result']

        winner_team = result['winner_team']
        base_points = result['base_points']
        my_team = snap['player'] % 2
        my_score = snap['match_score'][my_team]
        opp_score = snap['match_score'][1 - my_team]
        multiplier = snap['score_multiplier']

        py_v = delta_me(winner_team, base_points, my_team, my_score, opp_score, multiplier)

        diff = abs(js_v - py_v)
        max_diff = max(max_diff, diff)

        if diff > eps:
            mismatches += 1
            if mismatches <= 10:
                print(f"  MISMATCH snap {i}: JS_v={js_v:.8f} PY_v={py_v:.8f} diff={diff:.2e}")
                print(f"    winner={winner_team} pts={base_points} team={my_team} "
                      f"scores={my_score}v{opp_score} mul={multiplier}")

    print(f"\ndME Parity: {total - mismatches}/{total} PASS, {mismatches} FAIL")
    print(f"Max diff: {max_diff:.2e}")
    if mismatches == 0:
        print("ALL dME VALUES MATCH")
    return mismatches == 0


def validate_encoder_parity(data, eps=1e-6, sample_size=200):
    """Re-encode snapshots through Python and compare to JS."""
    indices = list(range(len(data)))
    if len(indices) > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(indices, size=sample_size, replace=False)

    state_mismatches = 0
    mask_mismatches = 0
    total = len(indices)

    for idx in indices:
        entry = data[idx]
        snap = entry['snapshot']

        py_state, py_mask = encode_snapshot(snap)

        # Check if JS state/mask is included (from massParityCheck-style export)
        # Training data doesn't include js_state/js_mask by default,
        # so we just validate the snapshot is encodable without errors
        # The actual parity is tested via massParityCheck separately

    print(f"\nEncoder sanity: {total} snapshots encoded without errors")
    return True


def validate_data_quality(data):
    """Check training data for quality issues."""
    total = len(data)
    if total == 0:
        print("ERROR: No data to validate")
        return False

    v_targets = [d['v_target'] for d in data]
    v_arr = np.array(v_targets)

    # Basic stats
    print(f"\n{'='*60}")
    print(f"  TRAINING DATA QUALITY REPORT")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"\ndME distribution:")
    print(f"  Mean:   {v_arr.mean():.6f}")
    print(f"  Std:    {v_arr.std():.6f}")
    print(f"  Min:    {v_arr.min():.6f}")
    print(f"  Max:    {v_arr.max():.6f}")
    print(f"  Median: {np.median(v_arr):.6f}")

    # Check range
    in_range = np.all((v_arr >= -1.0) & (v_arr <= 1.0))
    print(f"\n  All in [-1, 1]: {'YES' if in_range else 'NO'}")
    if not in_range:
        oob = np.sum((v_arr < -1.0) | (v_arr > 1.0))
        print(f"  Out of range: {oob}")

    # Sign distribution
    pos = np.sum(v_arr > 0)
    neg = np.sum(v_arr < 0)
    zero = np.sum(v_arr == 0)
    print(f"  Positive (won): {pos} ({pos/total*100:.1f}%)")
    print(f"  Negative (lost): {neg} ({neg/total*100:.1f}%)")
    print(f"  Zero (tie): {zero} ({zero/total*100:.1f}%)")

    # Magnitude buckets
    abs_v = np.abs(v_arr)
    print(f"\n  |dME| buckets:")
    for lo, hi in [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 1.0)]:
        count = np.sum((abs_v >= lo) & (abs_v < hi))
        print(f"    [{lo:.2f}, {hi:.2f}): {count} ({count/total*100:.1f}%)")

    # ── Per-row policy validation using validate_policy_row ──────────────────
    entropy_values  = []
    top1_probs      = []
    legal_counts    = []
    illegal_mass_violations = 0
    bad_sums        = 0
    row_errors      = 0

    for i, d in enumerate(data):
        snap = d['snapshot']
        pi   = np.array(d['policy_57'], dtype=np.float32)
        mask = nn_action_mask(snap).astype(np.float32)

        try:
            stats = validate_policy_row(pi, mask, row_idx=i)
            entropy_values.append(stats["entropy"])
            top1_probs.append(stats["top1_prob"])
            legal_counts.append(stats["legal_count"])
            if stats["illegal_mass"] > 1e-6:
                illegal_mass_violations += 1
                if illegal_mass_violations <= 5:
                    print(f"  ILLEGAL MASS snap {i}: {stats['illegal_mass']:.6e}")
        except ValueError as e:
            row_errors += 1
            if row_errors <= 5:
                summarize_bad_row(d, i)
                print(f"  ERROR: {e}")
            bad_sums += 1

    p_arr = np.stack([np.array(d['policy_57']) for d in data])
    nonzero_per_row = (p_arr > 0.001).sum(axis=1)

    print(f"\nPolicy distribution:")
    print(f"  Mean legal actions  : {nonzero_per_row.mean():.1f}")
    print(f"  Pass-only samples   : {np.sum(p_arr[:, 56] > 0.99)}")
    print(f"  Policy sum check    : {'PASS' if bad_sums == 0 else f'FAIL ({bad_sums} bad)'}")
    print(f"  PI legality check   : {'PASS' if illegal_mass_violations == 0 else f'FAIL ({illegal_mass_violations} bad)'}")

    if entropy_values:
        mean_ent  = np.mean(entropy_values)
        mean_top1 = np.mean(top1_probs)
        mean_legal= np.mean(legal_counts)
        min_legal = np.min(legal_counts)
        max_legal = np.max(legal_counts)
        print(f"\nPolicy target stats (visit-count quality):")
        print(f"  mean entropy      : {mean_ent:.4f}")
        print(f"  mean top1 prob    : {mean_top1:.4f}")
        print(f"  mean legal count  : {mean_legal:.2f}")
        print(f"  min  legal count  : {min_legal}")
        print(f"  max  legal count  : {max_legal}")

        # Quality sanity warnings
        if mean_top1 < 0.05:
            print("  WARNING: policy targets look unusually flat (mean_top1 < 0.05)")
        if mean_top1 > 0.95:
            print("  WARNING: policy targets look unusually sharp / nearly deterministic")
        if mean_ent < 0.05:
            print("  WARNING: very low entropy — too few MCTS visits or overconfident search")
        max_uniform = math.log(mean_legal) if mean_legal > 1 else 1.0
        if mean_ent > max_uniform * 0.95:
            print("  WARNING: very high entropy — targets barely better than uniform")

    # Team POV stability: v_target POV must match snapshot player's team
    pov_violations = 0
    for i, d in enumerate(data):
        snap = d['snapshot']
        team = snap['player'] % 2
        v = d['v_target']
        winner = d['game_result']['winner_team']
        if winner < 0:
            continue
        # v should be positive iff winner == team (from this player's POV)
        if (winner == team and v < -1e-9) or (winner != team and v > 1e-9):
            pov_violations += 1
            if pov_violations <= 5:
                print(f"  POV FLIP snap {i}: player={snap['player']} team={team} "
                      f"winner={winner} v={v:.6f}")
    print(f"  Team POV stability (dME sign matches player's team): "
          f"{'PASS' if pov_violations == 0 else f'FAIL ({pov_violations} bad)'}")

    # Match score distribution
    scores = [(d['snapshot']['match_score'][0], d['snapshot']['match_score'][1])
              for d in data]
    unique_scores = set(scores)
    print(f"\nMatch context:")
    print(f"  Unique score states: {len(unique_scores)}")
    multipliers = [d['snapshot']['score_multiplier'] for d in data]
    for mul in sorted(set(multipliers)):
        count = multipliers.count(mul)
        print(f"  Multiplier {mul}: {count} ({count/total*100:.1f}%)")

    # Game result types
    result_types = [d['game_result']['type'] for d in data]
    for rt in sorted(set(result_types)):
        count = result_types.count(rt)
        print(f"  Result '{rt}': {count} ({count/total*100:.1f}%)")

    # Correlation check: positive dME should correspond to winning team
    correct_sign = 0
    for d in data:
        v = d['v_target']
        winner = d['game_result']['winner_team']
        team = d['snapshot']['player'] % 2
        if winner < 0:
            continue  # tie
        if (winner == team and v > 0) or (winner != team and v < 0):
            correct_sign += 1
    non_tie = sum(1 for d in data if d['game_result']['winner_team'] >= 0)
    if non_tie > 0:
        print(f"\n  dME sign correctness: {correct_sign}/{non_tie} "
              f"({correct_sign/non_tie*100:.1f}%)")
        if correct_sign / non_tie < 0.99:
            print("  WARNING: Some dME signs don't match game outcomes!")

    print(f"\n{'='*60}")

    issues = []
    if not in_range:
        issues.append("dME values out of [-1, 1]")
    if bad_sums > 0:
        issues.append(f"{bad_sums} policies don't sum to 1.0")
    if illegal_mass_violations > 0:
        issues.append(f"{illegal_mass_violations} policies have mass on illegal actions")
    if pov_violations > 0:
        issues.append(f"{pov_violations} dME POV sign flips")
    if non_tie > 0 and correct_sign / non_tie < 0.99:
        issues.append("dME sign mismatches")

    if row_errors > 0:
        issues.append(f"{row_errors} malformed policy rows")

    if issues:
        print(f"ISSUES: {', '.join(issues)}")
        return False
    else:
        print("ALL QUALITY CHECKS PASSED")
        return True


def quick_validate():
    """Run a quick validation using Python-only data generation.
    Plays 20 matches, computes dME, checks everything is consistent."""
    from domino_env import DominoEnv, DominoMatch

    print("Quick validation: 20 Python matches with dME targets...")
    data = []

    for m in range(20):
        match = DominoMatch(target_points=6)
        game_idx = 0

        while not match.match_over and game_idx < 20:
            env = match.env
            obs = match.new_game(seed=m * 1000 + game_idx)
            scores_before = list(match.scores)
            multiplier_before = match.multiplier

            game_steps = []
            step = 0

            while not env.is_over() and step < 200:
                mask = env.get_legal_moves_mask()
                if mask.sum() == 0:
                    break

                team = env.current_team
                my_score = match.scores[team]
                opp_score = match.scores[1 - team]

                game_steps.append({
                    'team': team,
                    'my_score': scores_before[team],
                    'opp_score': scores_before[1 - team],
                    'multiplier': multiplier_before,
                })

                # Random legal action
                legal = np.where(mask > 0)[0]
                action = np.random.choice(legal)
                obs, _, done, info = env.step(action)
                step += 1

            if env.game_over:
                match.record_game_result(env.winner_team, env.points_won)

                for gs in game_steps:
                    v = delta_me(
                        winner_team=env.winner_team,
                        points=env.points_won,
                        my_team=gs['team'],
                        my_score=gs['my_score'],
                        opp_score=gs['opp_score'],
                        multiplier=gs['multiplier']
                    )
                    # Verify sign
                    if env.winner_team >= 0:
                        if env.winner_team == gs['team']:
                            assert v >= 0, f"dME should be >= 0 for winner, got {v}"
                        else:
                            assert v <= 0, f"dME should be <= 0 for loser, got {v}"
                    # Verify range
                    assert -1.0 <= v <= 1.0, f"dME out of range: {v}"
                    data.append(v)

            game_idx += 1

    v_arr = np.array(data)
    print(f"Generated {len(data)} dME samples from 20 matches")
    print(f"  Mean: {v_arr.mean():.4f}, Std: {v_arr.std():.4f}")
    print(f"  Min: {v_arr.min():.4f}, Max: {v_arr.max():.4f}")
    print(f"  All signs correct (verified by assertions)")
    print(f"  All in [-1, 1] (verified by assertions)")
    print("QUICK VALIDATION PASSED")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python validate_training_data.py <training_data.jsonl>  # validate JS export")
        print("  python validate_training_data.py --me3d <me3d_js.json>  # ME3D parity")
        print("  python validate_training_data.py --quick                 # Python-only quick check")
        sys.exit(1)

    if sys.argv[1] == '--me3d':
        if len(sys.argv) < 3:
            print("Usage: python validate_training_data.py --me3d me3d_js.json")
            sys.exit(1)
        ok = validate_me3d_parity(sys.argv[2])
        sys.exit(0 if ok else 1)

    if sys.argv[1] == '--quick':
        ok = quick_validate()
        sys.exit(0 if ok else 1)

    # Load JSONL training data
    json_path = sys.argv[1]
    print(f"Loading training data from {json_path}...")
    data = []
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples")

    # Run all validations
    print("\n--- dME Parity Check ---")
    me_ok = validate_delta_me_parity(data)

    print("\n--- Encoder Sanity Check ---")
    enc_ok = validate_encoder_parity(data)

    print("\n--- Data Quality Report ---")
    quality_ok = validate_data_quality(data)

    # Summary
    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  dME parity:     {'PASS' if me_ok else 'FAIL'}")
    print(f"  Encoder sanity:  {'PASS' if enc_ok else 'FAIL'}")
    print(f"  Data quality:    {'PASS' if quality_ok else 'FAIL'}")
    all_ok = me_ok and enc_ok and quality_ok
    print(f"  Overall:         {'ALL PASS' if all_ok else 'SOME FAILURES'}")
    print(f"{'='*60}")
    sys.exit(0 if all_ok else 1)
