"""
Vectorized POMCTS (Observable-State MCTS) for Pernambuco Domino.

Key design: N independent games run in lockstep. No determinization — uses the
213-dim observable state directly (POMCTS). All leaf evaluations are batched into
a single GPU forward pass, eliminating the GIL bottleneck of the IPC/threading
approach.

Tree notes
----------
- One UCT tree per slot, rebuilt from scratch each move.
- Values stored from Team 0's perspective throughout the tree.
- UCB selection flips sign for Team 1 (minimiser).
- Temperature 1.0 for moves 0–13, 0.1 after.
- Dirichlet noise at root: alpha=0.3, weight=0.25.

Usage
-----
    vmcts = VectorizedMCTS(model, device, num_games=64, sims_per_move=800)
    training_data = vmcts.run_generation(games_per_batch=256)
    # training_data: list of (state_np, mask_np, policy_np, value_float)
"""

import math
import numpy as np
import torch

from domino_env import DominoEnv, DominoMatch
from domino_encoder import DominoEncoder
from match_equity import delta_me

# Tile lookup (28 tiles: (a, b) with a <= b)
TILES = [(a, b) for a in range(7) for b in range(a, 7)]


def _build_belief_target(hidden_hands_by_player, me):
    """21-dim: partner/LHO/RHO each have pip 0-6 (1 if any tile with that pip)."""
    target = np.zeros(21, dtype=np.float32)
    for rel_idx, abs_player in enumerate([(me + 2) % 4, (me + 1) % 4, (me + 3) % 4]):
        seen = np.zeros(7, dtype=np.float32)
        for tile_idx in hidden_hands_by_player[abs_player]:
            a, b = TILES[tile_idx]
            seen[a] = 1.0
            seen[b] = 1.0
        target[rel_idx * 7: rel_idx * 7 + 7] = seen
    return target


def _build_support_target(hidden_hands_by_player, me, left_end, right_end):
    """6-dim: [partner_L, partner_R, lho_L, lho_R, rho_L, rho_R]."""
    target = np.zeros(6, dtype=np.float32)
    for rel_idx, abs_player in enumerate([(me + 2) % 4, (me + 1) % 4, (me + 3) % 4]):
        tiles = hidden_hands_by_player[abs_player]
        can_left  = any(TILES[t][0] == left_end  or TILES[t][1] == left_end  for t in tiles)
        can_right = any(TILES[t][0] == right_end or TILES[t][1] == right_end for t in tiles)
        target[rel_idx * 2]     = float(can_left)
        target[rel_idx * 2 + 1] = float(can_right)
    return target


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ('prior', 'visits', 'value_sum', 'children')

    def __init__(self, prior: float):
        self.prior = prior        # P(a) from policy network
        self.visits = 0           # N
        self.value_sum = 0.0      # W  (Team 0 perspective)
        self.children: dict = {}  # action_index -> MCTSNode

    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.q_value() + u


# ---------------------------------------------------------------------------
# Per-game slot
# ---------------------------------------------------------------------------

class _Slot:
    """Holds all state for one parallel game/match."""

    def __init__(self, use_belief_head=False, use_support_head=False):
        self.use_belief_head = use_belief_head
        self.use_support_head = use_support_head
        self.match = DominoMatch(target_points=6)
        self.encoder = DominoEncoder()

        # Current position MCTS state
        self.root: MCTSNode | None = None
        self.sims_done = 0
        self.move_count = 0          # moves played in current game
        self.game_history = []       # steps recorded before game outcome known

        # Saved for ME computation at game end
        self.scores_before = [0, 0]
        self.mult_before = 1

        # Lifecycle flags
        self.state = 'init'          # 'init' | 'needs_root' | 'simulating' | 'done'

        self._start_new_game()

    # ------------------------------------------------------------------ #
    # Lifecycle helpers
    # ------------------------------------------------------------------ #

    def _start_new_game(self):
        """Begin a new individual game within the current match."""
        self.encoder.reset()
        self.match.new_game()            # resets match.env in-place, returns obs

        self.scores_before = list(self.match.scores)
        self.mult_before = self.match.multiplier

        self.root = None
        self.sims_done = 0
        self.move_count = 0
        self.game_history = []
        self.state = 'needs_root'

    @property
    def env(self) -> DominoEnv:
        return self.match.env

    def current_obs_and_scores(self):
        """Return (obs, my_score, opp_score, multiplier) for NN encoding."""
        env = self.env
        obs = env.get_obs()
        team = env.current_team
        return obs, self.match.scores[team], self.match.scores[1 - team], self.match.multiplier

    def encode_state(self) -> tuple:
        """Return (state_np, mask_np) for the current real position."""
        obs, my_s, opp_s, mult = self.current_obs_and_scores()
        state = self.encoder.encode(obs, my_s, opp_s, mult)
        mask = self.env.get_legal_moves_mask()
        return state, mask

    def record_move(self, state_np, mask_np, pi_np):
        """Append a move to the game history (value filled in at game end)."""
        env = self.env
        me = env.current_player
        step = {
            'state': state_np,
            'mask': mask_np,
            'pi': pi_np,
            'team': env.current_team,
        }
        if self.use_belief_head or self.use_support_head:
            step['belief_target'] = _build_belief_target(env.hands, me)
        if self.use_support_head:
            if len(env.board) == 0:
                step['support_target'] = np.zeros(6, dtype=np.float32)
            else:
                step['support_target'] = _build_support_target(
                    env.hands, me, env.left_end, env.right_end)
        self.game_history.append(step)

    def flush_game_data(self, training_list: list):
        """
        After env.game_over is True: assign ME-delta targets and append to
        training_list.  Returns number of samples added.
        """
        env = self.env
        assert env.game_over, "flush_game_data called on unfinished game"

        n = 0
        for step in self.game_history:
            team = step['team']
            v = delta_me(
                winner_team=env.winner_team,
                points=env.points_won,
                my_team=team,
                my_score=self.scores_before[team],
                opp_score=self.scores_before[1 - team],
                multiplier=self.mult_before,
            )
            if self.use_support_head:
                training_list.append((
                    step['state'], step['mask'], step['pi'], float(v),
                    step['belief_target'], step['support_target'],
                ))
            elif self.use_belief_head:
                training_list.append((
                    step['state'], step['mask'], step['pi'], float(v),
                    step['belief_target'],
                ))
            else:
                training_list.append((step['state'], step['mask'], step['pi'], float(v)))
            n += 1
        return n


# ---------------------------------------------------------------------------
# Vectorised MCTS
# ---------------------------------------------------------------------------

class VectorizedMCTS:
    """
    Manages N independent games in lockstep.

    All games advance to their leaf node, then ONE batched GPU forward pass
    evaluates all N states simultaneously.  No IPC, no threading, no GIL issues.

    The MCTS tree for each game is a standard UCT tree operating on the full
    (known) game state.  The neural network is evaluated only on the 213-dim
    *observable* state (POMCTS style — no determinization needed).

    Parameters
    ----------
    model       : DominoNet — already on `device`
    device      : torch.device
    num_games   : number of parallel game slots (default 64)
    sims_per_move : MCTS simulations per move decision (default 800)
    c_puct      : UCB exploration constant (default 1.5)
    """

    def __init__(self, model, device,
                 num_games: int = 64,
                 sims_per_move: int = 800,
                 c_puct: float = 1.5,
                 use_belief_head: bool = False,
                 use_support_head: bool = False):
        self.model = model
        self.device = device
        self.num_games = num_games
        self.sims_per_move = sims_per_move
        self.c_puct = c_puct
        self.use_belief_head = use_belief_head
        self.use_support_head = use_support_head

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run_generation(self, games_per_batch: int = 256) -> list:
        """
        Play `games_per_batch` full matches using vectorised batched inference.

        Returns
        -------
        list of (state_np, mask_np, policy_np, value_float) training tuples.
        """
        training_data: list = []
        matches_target = games_per_batch
        matches_done = 0

        # Create N parallel slots — no more than the target number.
        n = min(self.num_games, matches_target)
        slots = [_Slot(use_belief_head=self.use_belief_head,
                       use_support_head=self.use_support_head) for _ in range(n)]
        # Track how many matches each slot has completed.
        slot_matches = [0] * n

        self.model.eval()

        while matches_done < matches_target:
            # ── Phase 1: batch-initialise any roots that need it ──────────
            uninit = [s for s in slots if s.state == 'needs_root' and not s.state == 'done']
            # (filter again cleanly)
            uninit = [s for s in slots if s.state == 'needs_root']
            if uninit:
                self._batch_init_roots(uninit)
                for s in uninit:
                    s.state = 'simulating'

            # ── Phase 2: one simulation step for all simulating slots ─────
            active = [s for s in slots if s.state == 'simulating']
            if active:
                self._batch_sim_step(active)

            # ── Phase 3: pick actions for slots that have hit sim budget ──
            ready = [s for s in slots if s.state == 'simulating'
                     and s.sims_done >= self.sims_per_move]
            for s in ready:
                self._pick_and_advance(s, training_data)

                if s.env.is_over():
                    n_samples = s.flush_game_data(training_data)
                    s.match.record_game_result(s.env.winner_team, s.env.points_won)

                    if s.match.match_over:
                        matches_done += 1
                        if matches_done < matches_target:
                            # Recycle: start a brand-new match
                            s.match = DominoMatch(target_points=6)
                            s._start_new_game()
                        else:
                            s.state = 'done'
                    else:
                        # Continue same match, new game
                        s._start_new_game()
                else:
                    # Game still ongoing: init MCTS for new position
                    s.root = None
                    s.sims_done = 0
                    s.state = 'needs_root'

        return training_data

    # ------------------------------------------------------------------ #
    # Root initialisation (batch)
    # ------------------------------------------------------------------ #

    def _batch_init_roots(self, slots: list):
        """
        One batched NN call to get root priors for all slots needing root init.
        Expands root nodes and applies Dirichlet noise.
        """
        states = []
        masks = []
        for s in slots:
            st, mask = s.encode_state()
            states.append(st)
            masks.append(mask)

        policies, _ = self._batch_infer(
            np.stack(states, axis=0),
            np.stack(masks, axis=0),
        )

        for s, policy, mask in zip(slots, policies, masks):
            root = MCTSNode(prior=1.0)
            valid_actions = np.where(mask > 0)[0]

            for act in valid_actions:
                root.children[act] = MCTSNode(prior=float(policy[act]))

            # Dirichlet noise at root (forced exploration)
            if len(valid_actions) > 1:
                noise = np.random.dirichlet([0.3] * len(valid_actions))
                for i, act in enumerate(valid_actions):
                    child = root.children[act]
                    child.prior = 0.75 * child.prior + 0.25 * noise[i]

            s.root = root
            s.sims_done = 0

    # ------------------------------------------------------------------ #
    # One simulation step (batch)
    # ------------------------------------------------------------------ #

    def _batch_sim_step(self, slots: list):
        """
        For each slot: clone env, traverse tree to a leaf or terminal.
        Batch-evaluate all non-terminal leaves.  Backprop all results.
        """
        leaf_infos = []   # (slot_idx, path, current_team_at_leaf, state_np, mask_np)
        term_infos = []   # (path, value_team0) — already know the value

        for idx, s in enumerate(slots):
            path, sim_env = self._traverse(s)

            if sim_env.is_over():
                # Terminal: use exact game result as value
                winner_team, points_won = sim_env.get_match_results()
                # Convert to match-equity scale (points/4 as in original)
                raw_val = points_won / 4.0
                v_team0 = raw_val if winner_team == 0 else -raw_val
                term_infos.append((path, v_team0))
            else:
                # Leaf: need NN evaluation
                obs = sim_env.get_obs()
                enc = DominoEncoder()  # fresh encoder — beliefs recomputed from obs
                team = sim_env.current_team
                my_score = s.match.scores[team]
                opp_score = s.match.scores[1 - team]
                state_np = enc.encode(obs, my_score, opp_score, s.match.multiplier)
                mask_np = sim_env.get_legal_moves_mask()
                leaf_infos.append((idx, path, team, state_np, mask_np))

        # Backprop terminals immediately
        for path, v_team0 in term_infos:
            _backprop(path, v_team0)

        # Batch-evaluate leaves
        if leaf_infos:
            batch_states = np.stack([li[3] for li in leaf_infos], axis=0)
            batch_masks = np.stack([li[4] for li in leaf_infos], axis=0)
            policies, values = self._batch_infer(batch_states, batch_masks)

            for (idx, path, team, state_np, mask_np), policy, value in zip(
                    leaf_infos, policies, values):
                # Expand the leaf node (last in path)
                leaf_node = path[-1]
                for act in np.where(mask_np > 0)[0]:
                    if act not in leaf_node.children:
                        leaf_node.children[act] = MCTSNode(prior=float(policy[act]))

                # Convert value to Team 0 perspective
                v_team0 = float(value) if team == 0 else -float(value)
                _backprop(path, v_team0)

        # Increment sim counter for every slot
        for s in slots:
            s.sims_done += 1

    # ------------------------------------------------------------------ #
    # Tree traversal (per slot)
    # ------------------------------------------------------------------ #

    def _traverse(self, slot: _Slot):
        """
        Traverse the MCTS tree from root to a leaf or terminal node.

        Returns
        -------
        path     : list of MCTSNode visited (root → leaf)
        sim_env  : cloned DominoEnv after traversal
        """
        sim_env = slot.env.clone()  # deep copy — full game state visible
        node = slot.root
        path = [node]

        while node.children and not sim_env.is_over():
            mask = sim_env.get_legal_moves_mask()
            action = self._select_action(node, mask, sim_env.current_team)
            if action is None:
                break
            sim_env.step(action)
            node = node.children[action]
            path.append(node)

        return path, sim_env

    def _select_action(self, node: MCTSNode, mask: np.ndarray, current_team: int):
        """PUCT selection. Team 0 maximises, Team 1 minimises."""
        best_score = -math.inf
        best_action = None

        parent_visits = node.visits

        for action, child in node.children.items():
            if mask[action] == 0.0:
                continue
            q = child.q_value() if current_team == 0 else -child.q_value()
            u = self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # ------------------------------------------------------------------ #
    # Action selection (real game)
    # ------------------------------------------------------------------ #

    def _pick_and_advance(self, slot: _Slot, training_data: list):
        """
        Extract improved policy from MCTS visit counts, sample action,
        record the move in training history, and step the real environment.
        """
        env = slot.env
        root = slot.root
        move = slot.move_count

        temperature = 1.0 if move < 14 else 0.1

        # Compute policy from visit counts
        visit_counts = np.zeros(57, dtype=np.float32)
        for act, child in root.children.items():
            visit_counts[act] = child.visits

        if temperature < 0.01 or visit_counts.max() == 0:
            best = int(np.argmax(visit_counts))
            pi = np.zeros(57, dtype=np.float32)
            pi[best] = 1.0
        else:
            visits_temp = visit_counts ** (1.0 / temperature)
            total = visits_temp.sum()
            pi = visits_temp / total if total > 0 else visit_counts

        # Sample action
        action = int(np.random.choice(57, p=pi / pi.sum()))

        # Record BEFORE stepping (need current state encoding)
        state_np, mask_np = slot.encode_state()
        slot.record_move(state_np, mask_np, pi)

        # Step real env
        env.step(action)
        slot.move_count += 1

        # Clear MCTS state for next position
        slot.root = None
        slot.sims_done = 0

    # ------------------------------------------------------------------ #
    # Batched NN inference
    # ------------------------------------------------------------------ #

    def _batch_infer(self, states_np: np.ndarray, masks_np: np.ndarray):
        """
        Run the model on a batch.

        Parameters
        ----------
        states_np : (N, 213) float32
        masks_np  : (N, 57) float32

        Returns
        -------
        policies  : (N, 57) numpy float32
        values    : (N,)    numpy float32
        """
        with torch.no_grad():
            x = torch.tensor(states_np, dtype=torch.float32, device=self.device)
            m = torch.tensor(masks_np, dtype=torch.float32, device=self.device)
            out = self.model(x, valid_actions_mask=m)
            # Model may return (policy, value) or (policy, value, belief, support)
            policy_t, value_t = out[0], out[1]
        return (
            policy_t.cpu().numpy(),
            value_t.squeeze(-1).cpu().numpy(),
        )


# ---------------------------------------------------------------------------
# Backpropagation helper (module-level for clarity)
# ---------------------------------------------------------------------------

def _backprop(path: list, v_team0: float):
    """Increment visit counts and accumulate value (Team 0 perspective) along path."""
    for node in path:
        node.visits += 1
        node.value_sum += v_team0
