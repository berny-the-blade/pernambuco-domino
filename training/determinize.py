"""
MRV backtracking determinization sampler for Pernambuco Domino.

Assigns unknown tiles to {partner, LHO, RHO, dorme} respecting:
  - Hard constraints (cantHave)
  - Exact slot counts (hand sizes + dorme)
  - Belief-weighted sampling via Gumbel-top-k

Usage:
    from determinize import determinize_mrv
    assign = determinize_mrv(obs, rng, belief4)
    # assign: dict { tile_idx: loc } where loc in {0,1,2,3}
"""

import numpy as np
from domino_env import TILES, NUM_TILES

LOC_P = 0    # partner
LOC_LHO = 1  # left-hand opponent
LOC_RHO = 2  # right-hand opponent
LOC_D = 3    # dorme (removed tiles)


def _gumbel_shuffle(items, weights, rng):
    """Order items by Gumbel-top-k trick for weighted random permutation."""
    g = -np.log(-np.log(rng.random(len(items)) + 1e-12) + 1e-12)
    scores = np.log(np.maximum(weights, 1e-12)) + g
    order = np.argsort(-scores)
    return [items[i] for i in order]


def determinize_mrv(obs, rng, belief4=None, max_backtracks=5000):
    """
    Assign unknown tiles to locations using MRV backtracking with belief weights.

    Args:
        obs: dict from DominoEnv.get_obs() with keys:
             player, hand, played, cant_have, hand_sizes
        rng: numpy RandomState for reproducibility
        belief4: optional (28, 4) array of probabilities.
                 Columns: [partner, LHO, RHO, dorme].
                 If None, uses uniform priors.
        max_backtracks: limit before falling back to random deal

    Returns:
        dict { tile_idx: loc } for all unknown tiles
        loc in {LOC_P=0, LOC_LHO=1, LOC_RHO=2, LOC_D=3}
    """
    me = obs['player']
    partner = (me + 2) % 4
    lho = (me + 1) % 4
    rho = (me + 3) % 4
    others = [partner, lho, rho]  # index 0,1,2 = LOC_P, LOC_LHO, LOC_RHO

    my_hand = set(obs['hand'])
    played = obs['played'] if isinstance(obs['played'], set) else set(obs['played'])

    unknown = [t for t in range(NUM_TILES) if t not in my_hand and t not in played]

    # Slot capacities: how many tiles each location needs
    slots = {
        LOC_P: obs['hand_sizes'][partner],
        LOC_LHO: obs['hand_sizes'][lho],
        LOC_RHO: obs['hand_sizes'][rho],
        LOC_D: len(unknown) - obs['hand_sizes'][partner] - obs['hand_sizes'][lho] - obs['hand_sizes'][rho],
    }

    # Safety: ensure dorme slots >= 0
    if slots[LOC_D] < 0:
        slots[LOC_D] = 0

    # Use uniform belief if none provided
    if belief4 is None:
        belief4 = np.ones((NUM_TILES, 4), dtype=np.float64) * 0.25

    # Build allowed locations per tile
    def is_allowed(tile, loc):
        if loc == LOC_D:
            return True
        p_abs = others[loc]
        left, right = TILES[tile]
        cant = obs['cant_have'][p_abs] if isinstance(obs['cant_have'], dict) else obs['cant_have'][p_abs]
        return left not in cant and right not in cant

    cands = {}
    for t in unknown:
        locs = [loc for loc in (LOC_P, LOC_LHO, LOC_RHO, LOC_D) if is_allowed(t, loc)]
        if not locs:
            locs = [LOC_D]  # fallback: force to dorme
        cands[t] = locs

    # MRV ordering: tiles with fewest options first (most constrained)
    ordered = sorted(unknown, key=lambda t: len(cands[t]))

    assign = {}
    remaining_slots = dict(slots)
    backtracks = [0]

    def dfs(i):
        if i == len(ordered):
            return True
        if backtracks[0] > max_backtracks:
            return False

        t = ordered[i]
        locs = [loc for loc in cands[t] if remaining_slots[loc] > 0]
        if not locs:
            return False

        # Weight by belief for diversity
        ws = np.array([max(belief4[t, loc], 1e-9) for loc in locs], dtype=np.float64)
        locs_order = _gumbel_shuffle(locs, ws, rng)

        for loc in locs_order:
            assign[t] = loc
            remaining_slots[loc] -= 1
            if dfs(i + 1):
                return True
            remaining_slots[loc] += 1
            del assign[t]
            backtracks[0] += 1

        return False

    ok = dfs(0)

    if not ok:
        # Fallback: random feasible fill (prevents deadlock)
        assign.clear()
        remaining_slots = dict(slots)
        rng.shuffle(ordered)
        for t in ordered:
            locs = [loc for loc in cands[t] if remaining_slots[loc] > 0]
            if not locs:
                # Force into any location with space
                locs = [loc for loc in (LOC_D, LOC_P, LOC_LHO, LOC_RHO)
                        if remaining_slots[loc] > 0]
            if locs:
                loc = locs[rng.randint(len(locs))]
            else:
                loc = LOC_D  # absolute fallback
            assign[t] = loc
            remaining_slots[loc] -= 1

    return assign


def apply_determinization(env, assign):
    """Apply a determinization assignment to a DominoEnv clone.

    Args:
        env: cloned DominoEnv (will be mutated)
        assign: dict { tile_idx: loc } from determinize_mrv

    Modifies env.hands and env.dorme in place.
    """
    me = env.current_player
    partner = (me + 2) % 4
    lho = (me + 1) % 4
    rho = (me + 3) % 4

    loc_to_player = {LOC_P: partner, LOC_LHO: lho, LOC_RHO: rho}

    new_hands = {partner: [], lho: [], rho: []}
    new_dorme = []

    for tile, loc in assign.items():
        if loc == LOC_D:
            new_dorme.append(tile)
        else:
            p = loc_to_player[loc]
            new_hands[p].append(tile)

    env.hands[partner] = new_hands[partner]
    env.hands[lho] = new_hands[lho]
    env.hands[rho] = new_hands[rho]
    env.dorme = new_dorme
