"""
State encoder for Pernambuco Domino.
Converts game observations into a 213-dimensional tensor for the neural network.

Dimension breakdown:
  [0:28]    My hand (binary: do I hold tile t?)
  [28:56]   Played tiles (binary: has tile t been played?)
  [56:63]   Left end one-hot (7 values for 0-6, all zeros if empty)
  [63:70]   Right end one-hot
  [70:91]   CantHave: 3 players x 7 numbers = 21 binary flags
  [91:119]  Partner play history (binary: has partner played tile t?) -- NEW
  [119:203] Belief matrix: 3 players x 28 tiles = 84 probabilities
  [203:207] Hand sizes (4 players, normalized by /6)
  [207:209] Match scores (my_score/6, opp_score/6)
  [209:210] Score multiplier (normalized)
  [210:211] Board length (normalized by /24)
  [211:212] Game phase (tiles_played / 24)
  [212:213] My team (0 or 1)
  Total: 213
"""

import numpy as np
from domino_env import TILES, NUM_TILES, tile_has_number


class DominoEncoder:
    """Encodes game state into a fixed 213-dim numpy vector."""

    STATE_DIM = 213

    def __init__(self):
        self.belief = np.ones((NUM_TILES, 4), dtype=np.float64) * 0.25
        self._known_locations = {}

    def reset(self):
        """Reset belief state for a new game."""
        self.belief = np.ones((NUM_TILES, 4), dtype=np.float64) * 0.25
        self._known_locations = {}

    def clone(self):
        """Deep copy for MCTS simulations."""
        enc = DominoEncoder()
        enc.belief = self.belief.copy()
        enc._known_locations = dict(self._known_locations)
        return enc

    def encode(self, obs, my_score=0, opp_score=0, multiplier=1):
        """
        Encode an observation dict into a 213-dim numpy array.

        Args:
            obs: dict from DominoEnv.get_obs()
            my_score, opp_score: match scores for the current player's team
            multiplier: current game multiplier (dobrada)

        Returns:
            np.ndarray of shape (213,), dtype float32
        """
        state = np.zeros(self.STATE_DIM, dtype=np.float32)
        me = obs['player']
        partner = (me + 2) % 4
        lho = (me + 1) % 4
        rho = (me + 3) % 4

        # [0:28] My hand
        for t in obs['hand']:
            state[t] = 1.0

        # [28:56] Played tiles
        for t in obs['played']:
            state[28 + t] = 1.0

        # [56:63] Left end one-hot
        le = obs['left_end']
        if 0 <= le <= 6:
            state[56 + le] = 1.0

        # [63:70] Right end one-hot
        re = obs['right_end']
        if 0 <= re <= 6:
            state[63 + re] = 1.0

        # [70:91] CantHave: 3 other players x 7 numbers
        other_players = [partner, lho, rho]
        for i, p in enumerate(other_players):
            for n in obs['cant_have'][p]:
                if 0 <= n <= 6:
                    state[70 + i * 7 + n] = 1.0

        # [91:119] Partner play history (binary: has partner played tile t?)
        for t in obs['plays_by'][partner]:
            state[91 + t] = 1.0

        # [119:203] Belief probabilities: 3 zones x 28 tiles
        self._sync_belief(obs, me)
        belief = self.export_conditional_belief()
        for i in range(3):
            state[119 + i * 28: 119 + (i + 1) * 28] = belief[:, i]

        # [203:207] Hand sizes (normalized)
        for i, p in enumerate([me, partner, lho, rho]):
            state[203 + i] = obs['hand_sizes'][p] / 6.0

        # [207:209] Match scores
        state[207] = my_score / 6.0
        state[208] = opp_score / 6.0

        # [209:210] Multiplier
        state[209] = min(multiplier, 4) / 4.0

        # [210:211] Board length
        state[210] = obs['board_length'] / 24.0

        # [211:212] Game phase
        state[211] = len(obs['played']) / 24.0

        # [212:213] My team
        state[212] = float(me % 2)

        # Safety assertion
        assert state.shape == (213,), f"Encoder shape mismatch: {state.shape}"
        assert np.isfinite(state).all(), "NaN/Inf in encoded state"

        return state

    def _sync_belief(self, obs, me):
        """Update belief matrix from observation knowledge."""
        partner = (me + 2) % 4
        lho = (me + 1) % 4
        rho = (me + 3) % 4
        other = [partner, lho, rho]
        my_hand = set(obs['hand'])

        for t in range(NUM_TILES):
            if t in my_hand:
                self.belief[t, :] = 0.0
                continue
            if t in obs['played']:
                self.belief[t, :] = 0.0
                continue

            left, right = TILES[t]

            for i, p in enumerate(other):
                if left in obs['cant_have'][p] or right in obs['cant_have'][p]:
                    self.belief[t, i] = 0.0

            total = self.belief[t, :].sum()
            if total > 0:
                self.belief[t, :] /= total
            else:
                self.belief[t, :] = 0.0
                self.belief[t, 3] = 1.0

    def update_on_pass(self, passer_relative, left_end, right_end):
        """Update beliefs when a player passes."""
        if passer_relative < 0 or passer_relative > 2:
            return
        for t in range(NUM_TILES):
            if tile_has_number(t, left_end) or tile_has_number(t, right_end):
                self.belief[t, passer_relative] = 0.0
        for t in range(NUM_TILES):
            total = self.belief[t, :].sum()
            if total > 0:
                self.belief[t, :] /= total

    def update_on_play(self, player_relative, tile_idx):
        """Update beliefs when a player plays a tile."""
        self.belief[tile_idx, :] = 0.0

    def export_conditional_belief(self):
        """Export belief as conditional distribution P(zone | not dorme)."""
        out = np.zeros((NUM_TILES, 3), dtype=np.float32)
        for t in range(NUM_TILES):
            row = self.belief[t, :3]
            s = row.sum()
            if s > 0:
                out[t] = row / s
            else:
                out[t] = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        return out

    @property
    def belief_state(self):
        """Return the current belief matrix for determinization."""
        return self.belief.copy()
