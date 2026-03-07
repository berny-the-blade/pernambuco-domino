"""
Belief sampler for IS-MCTS determinization in Pernambuco Domino.

Replaces the belief-matrix weighted sampling with constraint-aware
particle-based sampling. Features:
  - Respects exact hidden hand sizes
  - Respects dormant tile count
  - Conditions on pass-derived forbidden-tile constraints
  - Tracks constraint fallback rate for diagnostics
  - Supports multiple particles per root for IS-MCTS

Integrates with existing DominoEnv and DominoEncoder.
"""

import random
import numpy as np
from belief_constraints import derive_constraints_from_state
from domino_env import NUM_TILES


class Particle:
    __slots__ = ('particle_id', 'hands', 'dorme', 'viewer', 'is_fallback')

    def __init__(self, particle_id, hands, dorme, viewer, is_fallback=False):
        self.particle_id = particle_id
        self.hands = hands        # dict[int, list[int]] — player -> tiles
        self.dorme = dorme        # list[int]
        self.viewer = viewer      # int
        self.is_fallback = is_fallback


class BeliefSampler:
    """Constraint-aware belief sampler for IS-MCTS determinization."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self._next_id = 1
        self.total_samples = 0
        self.fallback_count = 0

    def sample_particle(self, env, viewer=None):
        """Sample a single particle (hidden assignment) from the current state.

        Args:
            env: DominoEnv instance
            viewer: player whose hand is known (default: env.current_player)

        Returns:
            Particle with hidden hand assignments
        """
        if viewer is None:
            viewer = env.current_player

        my_hand = set(env.hands[viewer])
        unknown = [t for t in range(NUM_TILES) if t not in my_hand and t not in env.played]

        hand_sizes = {}
        for p in range(4):
            if p != viewer:
                hand_sizes[p] = len(env.hands[p])
        dorme_count = len(env.dorme)

        constraints = derive_constraints_from_state(env, viewer)

        self.total_samples += 1
        result = self._allocate_constrained(unknown, hand_sizes, dorme_count, constraints)

        if result is None:
            # Fallback: random unconstrained assignment
            self.fallback_count += 1
            result = self._allocate_unconstrained(unknown, hand_sizes, dorme_count)
            is_fallback = True
        else:
            is_fallback = False

        pid = self._next_id
        self._next_id += 1

        return Particle(
            particle_id=pid,
            hands=result['hands'],
            dorme=result['dorme'],
            viewer=viewer,
            is_fallback=is_fallback,
        )

    def sample_particles(self, env, k, viewer=None):
        """Sample k particles."""
        return [self.sample_particle(env, viewer) for _ in range(k)]

    def determinize_env(self, env, particle):
        """Apply a particle's hidden assignment to a cloned env.

        Args:
            env: DominoEnv to clone and modify
            particle: Particle with hidden assignment

        Returns:
            New DominoEnv with determinized hidden hands
        """
        det = env.clone()
        viewer = particle.viewer
        for p, tiles in particle.hands.items():
            if p != viewer:
                det.hands[p] = list(tiles)
        det.dorme = list(particle.dorme)
        return det

    def stats(self):
        """Return diagnostic statistics."""
        rate = self.fallback_count / self.total_samples if self.total_samples > 0 else 0.0
        return {
            'total_samples': self.total_samples,
            'fallback_count': self.fallback_count,
            'fallback_rate': rate,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _allocate_constrained(self, unknown, hand_sizes, dorme_count, constraints):
        """Try constrained allocation with rejection sampling (up to 100 attempts)."""
        for _ in range(100):
            pool = list(unknown)
            self.rng.shuffle(pool)
            remaining = set(pool)
            hands = {}
            ok = True

            for p in sorted(hand_sizes.keys()):
                need = hand_sizes[p]
                forbidden = constraints[p].forbidden_tiles if p in constraints else set()
                candidates = [t for t in remaining if t not in forbidden]
                if len(candidates) < need:
                    ok = False
                    break
                chosen = self.rng.sample(candidates, need)
                hands[p] = sorted(chosen)
                for t in chosen:
                    remaining.remove(t)

            if not ok:
                continue

            remaining_list = list(remaining)
            if len(remaining_list) < dorme_count:
                continue

            self.rng.shuffle(remaining_list)
            dorme = sorted(remaining_list[:dorme_count])

            return {'hands': hands, 'dorme': dorme}

        return None  # all attempts failed

    def _allocate_unconstrained(self, unknown, hand_sizes, dorme_count):
        """Fallback: random allocation ignoring constraints."""
        pool = list(unknown)
        self.rng.shuffle(pool)
        hands = {}
        cursor = 0
        for p in sorted(hand_sizes.keys()):
            n = hand_sizes[p]
            hands[p] = sorted(pool[cursor:cursor + n])
            cursor += n
        dorme = sorted(pool[cursor:cursor + dorme_count])
        return {'hands': hands, 'dorme': dorme}
