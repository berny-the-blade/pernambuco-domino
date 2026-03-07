"""
Belief constraints derived from public history.

Extracts conservative hidden-information constraints from:
  - pass events (player cannot hold tiles matching board ends when they passed)
  - cantHave sets (from game state tracking)

These constraints are used by the belief sampler for determinization
in IS-MCTS. They never exclude a world still possible under public history.
"""

from domino_env import TILES, NUM_TILES, tile_has_number


class PlayerConstraint:
    __slots__ = ('forbidden_tiles',)

    def __init__(self):
        self.forbidden_tiles = set()


def derive_constraints_from_state(env, viewer):
    """Extract forbidden tile sets for each hidden player from env state.

    Uses cantHave sets (which track pass-derived info during gameplay)
    to build per-player forbidden tile sets.

    Args:
        env: DominoEnv instance
        viewer: player index (0-3) whose perspective we're sampling from

    Returns:
        dict[int, PlayerConstraint] for players != viewer
    """
    constraints = {}
    for p in range(4):
        if p == viewer:
            continue
        pc = PlayerConstraint()
        # cantHave[p] contains pip values the player cannot hold
        # (derived from passes during gameplay)
        for t in range(NUM_TILES):
            left, right = TILES[t]
            if left in env.cant_have[p] or right in env.cant_have[p]:
                pc.forbidden_tiles.add(t)
        constraints[p] = pc
    return constraints
