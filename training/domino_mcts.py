"""
Information-Set Monte Carlo Tree Search (IS-MCTS) with PUCT selection.
Designed for 2v2 Pernambuco Domino with imperfect information.

Key design: All values stored from Team 0's perspective.
Before each simulation, hidden hands are determinized from Bayesian beliefs.
"""

import math
import numpy as np
import torch


class MCTSNode:
    __slots__ = ('prior', 'visits', 'value_sum', 'children')

    def __init__(self, prior):
        self.prior = prior        # P: neural net's initial probability for this action
        self.visits = 0           # N: how many times this node was visited
        self.value_sum = 0.0      # W: total accumulated value (Team 0 perspective)
        self.children = {}        # action_index -> MCTSNode

    @property
    def q_value(self):
        """Q: average value of this node for Team 0."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class DominoMCTS:
    """IS-MCTS with PUCT selection for imperfect-information domino."""

    def __init__(self, model, num_simulations=100, c_puct=1.5):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = torch.device("cpu")  # CPU for parallel workers

    def get_action_probs(self, env, encoder, temperature=1.0):
        """
        Build the MCTS tree and return improved policy (pi).

        Args:
            env: DominoEnv instance (current game state)
            encoder: DominoEncoder instance (current beliefs)
            temperature: exploration temperature (1.0=explore, 0.1=exploit)

        Returns:
            pi: np.ndarray of shape (57,) — improved action probabilities
        """
        obs = env.get_obs()
        valid_mask = env.get_legal_moves_mask()
        state_np = encoder.encode(obs)

        # Get root policy from neural net
        root_probs, _ = self.model.predict(state_np, valid_mask, self.device)

        root = MCTSNode(prior=1.0)
        valid_actions = np.where(valid_mask > 0)[0]

        # Expand root node
        for act in valid_actions:
            root.children[act] = MCTSNode(prior=root_probs[act])

        # Add Dirichlet noise to root for exploration
        if len(valid_actions) > 1:
            noise = np.random.dirichlet([0.3] * len(valid_actions))
            for i, act in enumerate(valid_actions):
                root.children[act].prior = (
                    0.75 * root.children[act].prior + 0.25 * noise[i]
                )

        # Run simulations
        for _ in range(self.num_simulations):
            # Determinize: hallucinate hidden hands from beliefs
            sim_env = env.clone()
            sim_env.determinize_hidden_hands(encoder.belief_state)
            sim_enc = encoder.clone()

            node = root
            search_path = [node]

            # === SELECTION: traverse tree using PUCT ===
            while node.children and not sim_env.is_over():
                sim_mask = sim_env.get_legal_moves_mask()
                legal_children = [a for a in node.children if sim_mask[a] > 0]

                if not legal_children:
                    break  # Need to expand

                action, child = self._select_child(
                    node, sim_mask, sim_env.current_team
                )
                if action is None:
                    break

                # Execute action in simulation
                sim_env.step(action)

                # Update beliefs if pass
                if action == 56:
                    passer = (sim_env.current_player - 1) % 4
                    me = env.current_player
                    if passer != me:
                        rel = self._player_to_relative(passer, me)
                        if rel is not None:
                            obs_sim = sim_env.get_obs()
                            sim_enc.update_on_pass(
                                rel, obs_sim['left_end'], obs_sim['right_end']
                            )
                else:
                    tile = action if action < 28 else action - 28
                    player = (sim_env.current_player - 1) % 4
                    me = env.current_player
                    if player != me:
                        rel = self._player_to_relative(player, me)
                        if rel is not None:
                            sim_enc.update_on_play(rel, tile)

                node = child
                search_path.append(node)

            # === EXPANSION & EVALUATION ===
            if not sim_env.is_over():
                s_obs = sim_env.get_obs()
                s_mask = sim_env.get_legal_moves_mask()
                s_state = sim_enc.encode(s_obs)

                leaf_probs, leaf_value = self.model.predict(
                    s_state, s_mask, self.device
                )
                eval_team = sim_env.current_team

                # Expand new children
                for act in np.where(s_mask > 0)[0]:
                    if act not in node.children:
                        node.children[act] = MCTSNode(prior=leaf_probs[act])
            else:
                # Terminal: use exact result
                winner_team, points_won = sim_env.get_match_results()
                leaf_value = points_won / 4.0
                eval_team = winner_team

            # === BACKPROPAGATION ===
            # Normalize to Team 0's perspective
            v_team0 = leaf_value if eval_team == 0 else -leaf_value

            for path_node in search_path:
                path_node.visits += 1
                path_node.value_sum += v_team0

        # === BUILD FINAL POLICY (pi) ===
        action_visits = np.zeros(57, dtype=np.float32)
        for act, child in root.children.items():
            action_visits[act] = child.visits

        if temperature < 0.01:
            # Greedy: most-visited action, tie-break by Q-value
            max_visits = action_visits.max()
            candidates = np.where(action_visits == max_visits)[0]
            if len(candidates) == 1:
                best = candidates[0]
            else:
                best = max(candidates,
                           key=lambda a: root.children[a].q_value
                           if a in root.children else 0.0)
            pi = np.zeros(57, dtype=np.float32)
            pi[best] = 1.0
        else:
            visits_temp = action_visits ** (1.0 / temperature)
            total = visits_temp.sum()
            pi = visits_temp / total if total > 0 else valid_mask / valid_mask.sum()

        return pi

    def _select_child(self, node, valid_mask, current_team):
        """Select child using PUCT formula."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            if valid_mask[action] == 0.0:
                continue

            # Q-value flipped for Team 1 (they minimize Team 0's value)
            q = child.q_value if current_team == 0 else -child.q_value

            # PUCT exploration bonus
            u = (self.c_puct * child.prior *
                 math.sqrt(node.visits) / (1 + child.visits))

            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    @staticmethod
    def _player_to_relative(player, me):
        """Convert absolute player index to relative (0=partner, 1=LHO, 2=RHO)."""
        diff = (player - me) % 4
        if diff == 2:
            return 0  # partner
        elif diff == 1:
            return 1  # LHO
        elif diff == 3:
            return 2  # RHO
        return None  # self
