"""
Match Equity Table for Pernambuco Domino (match to 6 with dobrada).

Ports the ME3D 3D DP table from simulator.html.
ME3D[s1][s2][dob_idx] = P(team1 wins match | scores s1 vs s2, dobrada level)

Scoring:
  - Batida (normal + blocked): 1 point, ~70%
  - Carroca (double going out): 2 points, ~16%
  - La-e-lo (plays both ends): 3 points, ~10%
  - Cruzada (double + both ends): 4 points, ~4%
  - Dobrada: multiplier [1, 2, 4, 8] applied to all points
  - Match first to 6 points

Usage:
    from match_equity import ME3D, get_match_equity, delta_me
    me = get_match_equity(3, 2, multiplier=2)
    reward = delta_me(winner_team=0, points=2, my_team=0, my_score=3, opp_score=2, multiplier=1)
"""

import numpy as np

MATCH_TARGET = 6
POINT_DIST = [
    (1, 0.70),  # batida: normal + blocked
    (2, 0.16),  # carroca
    (3, 0.10),  # la-e-lo
    (4, 0.04),  # cruzada
]
DOB_VALUES = [1, 2, 4, 8]
TIE_PROB = 0.03  # ~3% of rounds are ties (blocked with dobrada)


def _build_me3d():
    """Build the 3D match equity table via backward induction."""
    T = MATCH_TARGET
    S = T + 5  # buffer for overflow scores

    me = np.zeros((S, S, len(DOB_VALUES)), dtype=np.float64)

    # Base cases
    for d in range(len(DOB_VALUES)):
        for s1 in range(T, S):
            for s2 in range(S):
                me[s1][s2][d] = 0.5 if s2 >= T else 1.0
        for s1 in range(T):
            for s2 in range(T, S):
                me[s1][s2][d] = 0.0

    # Fill bottom-up: iterate d from highest to lowest
    for s1 in range(T - 1, -1, -1):
        for s2 in range(T - 1, -1, -1):
            for d in range(len(DOB_VALUES) - 1, -1, -1):
                dob = DOB_VALUES[d]
                decisive = 0.0
                for base_pts, prob in POINT_DIST:
                    pts = base_pts * dob
                    s1w = min(s1 + pts, T + 4)
                    s2w = min(s2 + pts, T + 4)
                    decisive += 0.5 * prob * me[s1w][s2][0] + 0.5 * prob * me[s1][s2w][0]

                next_dob_idx = min(d + 1, len(DOB_VALUES) - 1)
                if d == len(DOB_VALUES) - 1:
                    # Max dobrada: tie maps to self → val = decisive
                    me[s1][s2][d] = decisive
                else:
                    me[s1][s2][d] = decisive * (1 - TIE_PROB) + TIE_PROB * me[s1][s2][next_dob_idx]

    return me


# Pre-computed table (computed once at import time)
ME3D = _build_me3d()


def get_match_equity(s1, s2, multiplier=1):
    """Get match equity P(team1 wins) given scores and dobrada multiplier.

    Args:
        s1: team 1 score
        s2: team 2 score
        multiplier: dobrada multiplier (1, 2, 4, or 8)

    Returns:
        float in [0, 1]
    """
    d_idx = DOB_VALUES.index(multiplier) if multiplier in DOB_VALUES else 0
    c1 = min(s1, MATCH_TARGET + 4)
    c2 = min(s2, MATCH_TARGET + 4)
    return float(ME3D[c1][c2][d_idx])


def delta_me(winner_team, points, my_team, my_score, opp_score, multiplier=1):
    """Compute match equity change from a game result.

    Args:
        winner_team: 0 or 1 (which team won)
        points: base points won (1-4, before multiplier)
        my_team: 0 or 1 (which team we're evaluating for)
        my_score: our current score before this game
        opp_score: opponent's current score before this game
        multiplier: dobrada multiplier at time of game

    Returns:
        float: ΔME (positive = good for my_team)
    """
    if winner_team < 0:
        return 0.0  # tie

    dob = multiplier if multiplier in DOB_VALUES else 1
    current_me = get_match_equity(my_score, opp_score, dob)

    pts = points * dob

    if winner_team == my_team:
        new_me = get_match_equity(min(my_score + pts, MATCH_TARGET + 4), opp_score, 1)
    else:
        new_me = get_match_equity(my_score, min(opp_score + pts, MATCH_TARGET + 4), 1)

    return new_me - current_me
