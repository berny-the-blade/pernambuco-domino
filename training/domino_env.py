"""
Pernambuco Domino game environment for neural network training.
Ports the core game logic from simulator.html to Python.

Rules: 28 tiles (0-0 through 6-6), 4 players in 2 teams (P0+P2 vs P1+P3),
4 tiles removed to dorme, highest double opens.
Scoring: Batida=1, Carroca=2, La-e-lo=3, Cruzada=4.
Blocked: individual lowest pips wins (1pt), ties go to opener's team.

Action space (57):
  0-27:  play tile[action] on LEFT side
  28-55: play tile[action-28] on RIGHT side
  56:    pass
"""

import numpy as np
from copy import deepcopy

# Tile lookup: index -> (left, right). 28 tiles total.
TILES = []
TILE_ID = {}  # "i-j" string -> index
for _i in range(7):
    for _j in range(_i, 7):
        idx = len(TILES)
        TILES.append((_i, _j))
        TILE_ID[f"{_i}-{_j}"] = idx
NUM_TILES = 28


def tile_pips(t):
    return TILES[t][0] + TILES[t][1]


def tile_is_double(t):
    return TILES[t][0] == TILES[t][1]


def tile_has_number(t, n):
    return TILES[t][0] == n or TILES[t][1] == n


def double_index(n):
    """Index of the [n|n] double tile."""
    # Doubles are at positions: 0=[0|0], 7=[1|1], 13=[2|2], 18=[3|3], 22=[4|4], 25=[5|5], 27=[6|6]
    for i, (l, r) in enumerate(TILES):
        if l == n and r == n:
            return i
    return -1


class DominoEnv:
    """Full Pernambuco Domino game environment."""

    def __init__(self):
        self.hands = [[], [], [], []]
        self.dorme = []
        self.board = []
        self.left_end = -1
        self.right_end = -1
        self.played = set()
        self.current_player = 0
        self.pass_count = 0
        self.game_over = False
        self.winner_team = -1
        self.points_won = 0
        self.result_type = ''
        self.opener = 0
        self.cant_have = [set() for _ in range(4)]
        self.plays_by = [[] for _ in range(4)]

    def reset(self, seed=None):
        """Deal a new game. Returns initial observation."""
        rng = np.random.RandomState(seed)
        deck = list(range(NUM_TILES))
        rng.shuffle(deck)

        self.hands = [deck[i * 6:(i + 1) * 6] for i in range(4)]
        self.dorme = deck[24:28]
        self.board = []
        self.left_end = -1
        self.right_end = -1
        self.played = set()
        self.current_player = 0
        self.pass_count = 0
        self.game_over = False
        self.winner_team = -1
        self.points_won = 0
        self.result_type = ''
        self.cant_have = [set() for _ in range(4)]
        self.plays_by = [[] for _ in range(4)]

        # Find and auto-play opener's highest double
        opener, opener_tile = self._find_opener()
        self.opener = opener
        self.current_player = opener

        if opener_tile is not None:
            self._execute_play(opener, opener_tile, None)
            self.current_player = (opener + 1) % 4

        return self.get_obs()

    def _find_opener(self):
        """Find player with highest double. Returns (player_idx, tile_idx)."""
        for d in range(6, -1, -1):
            di = double_index(d)
            for p in range(4):
                if di in self.hands[p]:
                    return p, di
        # No doubles dealt (extremely rare) — player 0 plays highest pip
        best = max(self.hands[0], key=tile_pips)
        return 0, best

    def _can_play_tile(self, tile_idx):
        """Check if tile can be played on the current board."""
        if len(self.board) == 0:
            return True
        left, right = TILES[tile_idx]
        return (left == self.left_end or right == self.left_end or
                left == self.right_end or right == self.right_end)

    def _can_play_on_side(self, tile_idx, side):
        """Check if tile can play on a specific side."""
        if len(self.board) == 0:
            return True
        left, right = TILES[tile_idx]
        if side == 'left':
            return left == self.left_end or right == self.left_end
        else:
            return left == self.right_end or right == self.right_end

    def _could_play_both_ends(self, tile_idx, le, re):
        """Check if tile matches both board ends (la-e-lo potential).
        Must check against the board ends BEFORE the tile was played."""
        if le < 0 or re < 0:
            return False
        left, right = TILES[tile_idx]
        if le == re:
            # Both ends same number: only a double of that number qualifies
            return left == le and right == le
        matches_l = (left == le or right == le)
        matches_r = (left == re or right == re)
        return matches_l and matches_r

    def get_legal_moves_mask(self):
        """Return 57-dim binary mask of legal actions for current player."""
        mask = np.zeros(57, dtype=np.float32)
        if self.game_over:
            return mask

        hand = self.hands[self.current_player]
        has_play = False

        symmetric = (len(self.board) > 0 and self.left_end == self.right_end)

        for tile_idx in hand:
            if not self._can_play_tile(tile_idx):
                continue
            has_play = True
            if len(self.board) == 0:
                # First play after opener (or rare no-double case)
                mask[tile_idx] = 1.0
            else:
                if self._can_play_on_side(tile_idx, 'left'):
                    mask[tile_idx] = 1.0
                if self._can_play_on_side(tile_idx, 'right') and not symmetric:
                    mask[28 + tile_idx] = 1.0

        if not has_play:
            mask[56] = 1.0  # Pass

        return mask

    def step(self, action):
        """Execute an action. Returns (obs, reward, done, info).
        Reward is from the perspective of the player who just moved."""
        if self.game_over:
            return self.get_obs(), 0.0, True, {'winner_team': self.winner_team,
                                                'points': self.points_won, 'type': self.result_type}

        player = self.current_player

        if action == 56:
            # Pass
            self.cant_have[player].add(self.left_end)
            if self.right_end != self.left_end:
                self.cant_have[player].add(self.right_end)
            self.pass_count += 1
            self.plays_by[player]  # no append for pass
            if self.pass_count >= 4:
                self._resolve_block()
            else:
                self.current_player = (self.current_player + 1) % 4
        else:
            tile_idx = action if action < 28 else action - 28
            side = 'left' if action < 28 else 'right'

            # Save previous ends for la-e-lo/cruzada scoring
            prev_le, prev_re = self.left_end, self.right_end

            self._execute_play(player, tile_idx, side)
            self.pass_count = 0

            if len(self.hands[player]) == 0:
                self._resolve_win(player, tile_idx, prev_le, prev_re)
            else:
                self.current_player = (self.current_player + 1) % 4

        # Compute reward from perspective of the player who acted
        reward = 0.0
        if self.game_over:
            team = player % 2
            if self.winner_team == team:
                reward = self.points_won / 4.0
            elif self.winner_team >= 0:
                reward = -self.points_won / 4.0

        info = {'winner_team': self.winner_team, 'points': self.points_won,
                'type': self.result_type}
        return self.get_obs(), reward, self.game_over, info

    def _execute_play(self, player, tile_idx, side):
        """Play a tile onto the board. Updates hands, board, ends, knowledge."""
        if tile_idx in self.hands[player]:
            self.hands[player].remove(tile_idx)
        self.played.add(tile_idx)
        self.plays_by[player].append(tile_idx)
        self.board.append(tile_idx)

        left, right = TILES[tile_idx]

        if len(self.board) == 1:
            # First tile on board
            self.left_end = left
            self.right_end = right
        elif side == 'left':
            self.left_end = right if left == self.left_end else left
        else:
            self.right_end = left if right == self.right_end else right

    def _resolve_win(self, player, last_tile, prev_le, prev_re):
        """Score the win for player who went out."""
        self.game_over = True
        self.winner_team = player % 2

        is_double = tile_is_double(last_tile)
        both_ends = self._could_play_both_ends(last_tile, prev_le, prev_re)

        if is_double and both_ends:
            self.points_won = 4
            self.result_type = 'cruzada'
        elif both_ends:
            self.points_won = 3
            self.result_type = 'laelo'
        elif is_double:
            self.points_won = 2
            self.result_type = 'carroca'
        else:
            self.points_won = 1
            self.result_type = 'batida'

    def _resolve_block(self):
        """Resolve blocked game: individual lowest pips wins."""
        self.game_over = True
        pip_counts = [(sum(tile_pips(t) for t in self.hands[p]), p) for p in range(4)]
        pip_counts.sort()

        lowest_pips = pip_counts[0][0]
        lowest_player = pip_counts[0][1]

        # Check for tie at the lowest level
        tied = [p for pips, p in pip_counts if pips == lowest_pips]

        if len(tied) == 1:
            self.winner_team = lowest_player % 2
            self.points_won = 1
            self.result_type = 'blocked'
        else:
            # Tie: check if any tied player is on opener's team
            opener_team = self.opener % 2
            tied_teams = set(p % 2 for p in tied)
            if len(tied_teams) == 1:
                # All tied players on same team
                self.winner_team = tied[0] % 2
                self.points_won = 1
                self.result_type = 'blocked'
            else:
                # Cross-team tie: opener's team wins (dobrada)
                self.winner_team = opener_team
                self.points_won = 1
                self.result_type = 'dobrada'

    def get_obs(self):
        """Get observation dict for current player."""
        return {
            'player': self.current_player,
            'team': self.current_player % 2,
            'hand': list(self.hands[self.current_player]),
            'played': set(self.played),
            'left_end': self.left_end,
            'right_end': self.right_end,
            'board_length': len(self.board),
            'cant_have': [set(s) for s in self.cant_have],
            'plays_by': [list(p) for p in self.plays_by],
            'hand_sizes': [len(h) for h in self.hands],
        }

    def get_observable_state(self):
        """Returns (my_hand, played_tiles, board_ends) for encoder."""
        return (
            list(self.hands[self.current_player]),
            set(self.played),
            (self.left_end, self.right_end)
        )

    def get_scores(self, team):
        """Returns (my_score, opp_score, points_to_win) for match context.
        Placeholder for single-game mode — override for match play."""
        return 0, 0, 6

    @property
    def current_team(self):
        return self.current_player % 2

    def is_over(self):
        return self.game_over

    def get_match_results(self):
        return self.winner_team, self.points_won

    def clone(self):
        """Deep copy for MCTS simulations."""
        return deepcopy(self)

    def determinize_hidden_hands(self, belief_matrix):
        """Determinize unknown hands based on Bayesian belief matrix.
        belief_matrix: shape (28, 4) — probability tile t is in zone z.
        Zones: 0=partner, 1=LHO, 2=RHO, 3=dorme (relative to current player).

        Uses rejection sampling: if greedy assignment hits a contradiction
        (prob_sum == 0 for a tile), retry from scratch up to 100 times.
        Falls back to constraint-ignoring random deal on failure.
        """
        me = self.current_player
        partner = (me + 2) % 4
        lho = (me + 1) % 4
        rho = (me + 3) % 4
        zone_to_player = {0: partner, 1: lho, 2: rho, 3: None}

        my_hand = set(self.hands[me])
        unknown = [t for t in range(NUM_TILES) if t not in my_hand and t not in self.played]

        targets = [len(self.hands[partner]), len(self.hands[lho]),
                   len(self.hands[rho]), len(self.dorme)]

        for attempt in range(100):
            assigned = [[] for _ in range(4)]
            order = list(unknown)
            np.random.shuffle(order)
            contradiction = False

            for tile in order:
                probs = belief_matrix[tile].copy().astype(np.float64)

                # Mask full zones
                for i in range(4):
                    if len(assigned[i]) >= targets[i]:
                        probs[i] = 0.0

                # Respect cantHave constraints
                left, right = TILES[tile]
                for i in range(3):  # zones 0-2 are players
                    p = zone_to_player[i]
                    if p is not None:
                        if left in self.cant_have[p] or right in self.cant_have[p]:
                            probs[i] = 0.0

                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs /= prob_sum
                    owner = np.random.choice(4, p=probs)
                    assigned[owner].append(tile)
                else:
                    # Contradiction — this deal is impossible, retry
                    contradiction = True
                    break

            if not contradiction:
                # Verify all zones filled exactly
                valid = all(len(assigned[i]) == targets[i] for i in range(4))
                if valid:
                    self.hands[partner] = assigned[0]
                    self.hands[lho] = assigned[1]
                    self.hands[rho] = assigned[2]
                    self.dorme = assigned[3]
                    return

        # Fallback: deal randomly ignoring beliefs (prevents deadlock)
        order = list(unknown)
        np.random.shuffle(order)
        idx = 0
        for zone, count in enumerate(targets):
            zone_tiles = order[idx:idx + count]
            idx += count
            if zone == 0:
                self.hands[partner] = zone_tiles
            elif zone == 1:
                self.hands[lho] = zone_tiles
            elif zone == 2:
                self.hands[rho] = zone_tiles
            else:
                self.dorme = zone_tiles


class DominoMatch:
    """Plays a full match to target_points using DominoEnv for individual games."""

    def __init__(self, target_points=6):
        self.target_points = target_points
        self.scores = [0, 0]  # team 0, team 1
        self.env = DominoEnv()
        self.match_over = False
        self.match_winner = -1
        self.multiplier = 1  # Doubles on ties

    def new_game(self, seed=None):
        """Start a new game within the match."""
        return self.env.reset(seed)

    def record_game_result(self, winner_team, points):
        """Record game result and check for match end."""
        actual_pts = points * self.multiplier
        if winner_team >= 0:
            self.scores[winner_team] += actual_pts
            self.multiplier = 1
        else:
            self.multiplier *= 2  # Dobrada

        if self.scores[0] >= self.target_points:
            self.match_over = True
            self.match_winner = 0
        elif self.scores[1] >= self.target_points:
            self.match_over = True
            self.match_winner = 1

    def get_scores(self, team):
        my_score = self.scores[team]
        opp_score = self.scores[1 - team]
        return my_score, opp_score, self.target_points
