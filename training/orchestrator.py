"""
Self-Play Orchestrator for Pernambuco Domino AI training.
Uses torch.multiprocessing to spawn parallel CPU workers for data generation,
then trains the master network on GPU with the collected replay buffer.

Value targets: ΔME (match equity delta) instead of raw points/4.
Arena: duplicate deals with side-swap for fair evaluation.

Usage:
    python orchestrator.py                    # Default: 4 workers, 100 generations
    python orchestrator.py --workers 8 --generations 200 --games-per-worker 500
    python orchestrator.py --mcts --mcts-sims 100 --value-target me
"""

import os
import sys
import time
import argparse
import subprocess
from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Add parent dir for imports when running from training/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domino_env import DominoEnv, DominoMatch, TILES
from domino_net import DominoNet
from domino_encoder import DominoEncoder
from domino_mcts import DominoMCTS
from domino_trainer import Trainer, ReplayDataset
from match_equity import get_match_equity, delta_me, DOB_VALUES


def build_support_target(hidden_hands_by_player, me, left_end, right_end):
    """
    6-dim end-support target:
      [partner_can_left, partner_can_right,
       lho_can_left,     lho_can_right,
       rho_can_left,     rho_can_right]

    For each non-self player: can they play the current left/right board end?
    hidden_hands_by_player: dict/list keyed by absolute player id (tile indices)
    """
    target = np.zeros(6, dtype=np.float32)
    partner = (me + 2) % 4
    lho     = (me + 1) % 4
    rho     = (me + 3) % 4
    for rel_idx, abs_player in enumerate([partner, lho, rho]):
        tiles = hidden_hands_by_player[abs_player]
        can_left  = any(TILES[t][0] == left_end  or TILES[t][1] == left_end  for t in tiles)
        can_right = any(TILES[t][0] == right_end or TILES[t][1] == right_end for t in tiles)
        target[rel_idx * 2]     = float(can_left)
        target[rel_idx * 2 + 1] = float(can_right)
    return target


def build_belief_target(hidden_hands_by_player, me):
    """
    21-dim target:
      0..6  = partner has pip 0..6
      7..13 = LHO has pip 0..6
      14..20 = RHO has pip 0..6

    hidden_hands_by_player: dict/list keyed by absolute player id
    """
    target = np.zeros(21, dtype=np.float32)
    partner = (me + 2) % 4
    lho     = (me + 1) % 4
    rho     = (me + 3) % 4
    for rel_idx, abs_player in enumerate([partner, lho, rho]):
        seen = np.zeros(7, dtype=np.float32)
        for tile_idx in hidden_hands_by_player[abs_player]:
            a, b = TILES[tile_idx]
            seen[a] = 1.0
            seen[b] = 1.0
        target[rel_idx * 7 : rel_idx * 7 + 7] = seen
    return target


def self_play_worker(worker_id, model_state_dict, num_games, use_mcts,
                     mcts_sims, result_queue, value_target='me',
                     policy_target='visits',
                     high_sim_fraction=0.1, high_sim_multiplier=4,
                     use_belief_head=False, use_support_head=False):
    """
    Isolated CPU process. Plays games against itself and pipes training data back.

    Each data point: (state_np, mask_np, pi_np, v_target)

    value_target: 'me' for ΔME (match equity), 'points' for points/4 (legacy)
    high_sim_fraction: fraction of games using high sim count (default 10%)
    high_sim_multiplier: multiplier for high-quality games (default 4×)
    """
    device = torch.device("cpu")
    model = DominoNet().to(device)
    incompat = model.load_state_dict(model_state_dict, strict=False)
    if incompat.missing_keys:
        print(f"[worker {worker_id}] Missing keys: {incompat.missing_keys}")
    if incompat.unexpected_keys:
        print(f"[worker {worker_id}] Unexpected keys: {incompat.unexpected_keys}")
    model.eval()
    torch.set_grad_enabled(False)

    # Unique seed per worker
    np.random.seed(int(time.time() * 1000) % (2**31) + worker_id * 1000)

    if policy_target == 'visits' and not use_mcts:
        raise ValueError("policy_target='visits' requires use_mcts=True")

    # Mixed sim budget: 90% at base sims, 10% at 4× base sims
    mcts_base = DominoMCTS(model, num_simulations=mcts_sims) if use_mcts else None
    mcts_high = DominoMCTS(model, num_simulations=mcts_sims * high_sim_multiplier) if use_mcts else None
    worker_data = []

    for game_idx in range(num_games):
        # Mixed sim budget: this match uses high sims if in the top fraction
        use_high_sims = use_mcts and (np.random.random() < high_sim_fraction)
        mcts = mcts_high if use_high_sims else mcts_base

        # Play a full match (first to 6) for ME context
        match = DominoMatch(target_points=6)
        match_history = []  # list of (game_history, match_state_before)

        while not match.match_over:
            env = match.env
            encoder = DominoEncoder()
            obs = match.new_game()
            encoder.reset()

            # Capture match state BEFORE this game (for ΔME)
            scores_before = list(match.scores)
            multiplier_before = match.multiplier

            game_history = []
            step_count = 0

            while not env.is_over() and step_count < 200:
                current_team = env.current_team
                valid_mask = env.get_legal_moves_mask()

                if valid_mask.sum() == 0:
                    break

                # Encode with match context
                my_team = current_team
                my_score = match.scores[my_team]
                opp_score = match.scores[1 - my_team]
                state_np = encoder.encode(obs, my_score=my_score,
                                          opp_score=opp_score,
                                          multiplier=match.multiplier)

                if use_mcts and valid_mask.sum() > 1 and policy_target == 'visits':
                    # AlphaZero-style: train on MCTS visit-count pi
                    temp = 1.0 if step_count < 14 else 0.1
                    target_pi = mcts.get_action_probs(env, encoder, temperature=temp)
                else:
                    # Legacy heuristic: network policy + Dirichlet noise
                    policy_probs, _ = model.predict(state_np, valid_mask, device)

                    valid_indices = np.where(valid_mask > 0)[0]
                    target_pi = policy_probs.copy()

                    if len(valid_indices) > 1:
                        alpha = min(1.0, 10.0 / max(len(valid_indices), 1))
                        noise = np.random.dirichlet([alpha] * len(valid_indices))
                        target_pi[valid_indices] = (
                            0.75 * target_pi[valid_indices] + 0.25 * noise
                        )
                        total = target_pi.sum()
                        if total > 0:
                            target_pi /= total

                chosen_action = np.random.choice(57, p=target_pi)

                record = {
                    'state': state_np,
                    'mask': valid_mask,
                    'pi': target_pi,
                    'team': current_team,
                }
                if use_belief_head:
                    record['belief_target'] = build_belief_target(env.hands, obs['player'])
                if use_support_head:
                    record['support_target'] = build_support_target(
                        env.hands, obs['player'],
                        obs.get('left_end', env.left_end),
                        obs.get('right_end', env.right_end),
                    )
                game_history.append(record)

                # Execute action
                obs, _, done, info = env.step(chosen_action)

                # Update encoder beliefs for played tiles
                if chosen_action < 56:
                    tile = chosen_action if chosen_action < 28 else chosen_action - 28
                    encoder.update_on_play(0, tile)

                # Update encoder beliefs for passes
                if chosen_action == 56:
                    passer = (env.current_player - 1) % 4
                    me_player = game_history[-1].get('_me_player', obs['player'])
                    # Pass updates are handled by _sync_belief via env.cant_have

                step_count += 1

            # Record game result in match
            if env.game_over:
                match.record_game_result(env.winner_team, env.points_won)

                # Compute value targets
                if value_target == 'me':
                    # ΔME: match equity change from this game
                    for step in game_history:
                        team = step['team']
                        my_s = scores_before[team]
                        opp_s = scores_before[1 - team]
                        v = delta_me(
                            winner_team=env.winner_team,
                            points=env.points_won,
                            my_team=team,
                            my_score=my_s,
                            opp_score=opp_s,
                            multiplier=multiplier_before
                        )
                        if use_support_head:
                            worker_data.append((
                                step['state'], step['mask'], step['pi'], v,
                                step['belief_target'], step['support_target'],
                            ))
                        elif use_belief_head:
                            worker_data.append((
                                step['state'], step['mask'], step['pi'], v,
                                step['belief_target'],
                            ))
                        else:
                            worker_data.append((
                                step['state'], step['mask'], step['pi'], v,
                            ))
                else:
                    # Legacy: points/4
                    winner_team = env.winner_team
                    reward_magnitude = env.points_won / 4.0
                    for step in game_history:
                        if step['team'] == winner_team:
                            v_target = reward_magnitude
                        else:
                            v_target = -reward_magnitude
                        if use_support_head:
                            worker_data.append((
                                step['state'], step['mask'], step['pi'], v_target,
                                step['belief_target'], step['support_target'],
                            ))
                        elif use_belief_head:
                            worker_data.append((
                                step['state'], step['mask'], step['pi'], v_target,
                                step['belief_target'],
                            ))
                        else:
                            worker_data.append((
                                step['state'], step['mask'], step['pi'], v_target,
                            ))

    result_queue.put(worker_data)


def arena_match(champion_weights, challenger_weights, seed, challenger_team=0):
    """
    Play one deterministic match (first to 6): challenger vs champion.
    No noise, greedy policy (argmax). Returns (match_winner, games_played).
    """
    device = torch.device("cpu")
    champion = DominoNet().to(device)
    champion.load_state_dict(champion_weights, strict=False)
    champion.eval()

    challenger = DominoNet().to(device)
    challenger.load_state_dict(challenger_weights, strict=False)
    challenger.eval()

    match = DominoMatch(target_points=6)
    game_seed = seed
    games_played = 0

    with torch.no_grad():
        while not match.match_over and games_played < 20:  # safety limit
            env = match.env
            encoder = DominoEncoder()
            obs = match.new_game(seed=game_seed + games_played)
            encoder.reset()
            step = 0

            while not env.is_over() and step < 200:
                mask = env.get_legal_moves_mask()
                if mask.sum() == 0:
                    break

                team = env.current_team
                my_score = match.scores[team]
                opp_score = match.scores[1 - team]
                state = encoder.encode(obs, my_score=my_score,
                                       opp_score=opp_score,
                                       multiplier=match.multiplier)

                # Select model based on team
                model = challenger if team == challenger_team else champion
                policy, _ = model.predict(state, mask, device)

                # Greedy: argmax (no noise, no temperature)
                action = int(np.argmax(policy))

                obs, _, done, info = env.step(action)
                if action < 56:
                    tile = action if action < 28 else action - 28
                    encoder.update_on_play(0, tile)
                step += 1

            if env.game_over:
                match.record_game_result(env.winner_team, env.points_won)
            games_played += 1

    return match.match_winner, games_played


def arena_worker(worker_id, champion_weights, challenger_weights,
                 game_seeds, result_queue):
    """Run duplicate-deal arena matches in a subprocess.
    Each seed is played twice (challenger as team 0 and team 1)."""
    challenger_wins = 0
    champion_wins = 0
    total_matches = 0

    for seed in game_seeds:
        for c_team in [0, 1]:
            winner, _ = arena_match(champion_weights, challenger_weights,
                                    seed, challenger_team=c_team)
            total_matches += 1
            if winner == c_team:
                challenger_wins += 1
            elif winner >= 0:
                champion_wins += 1

    result_queue.put((challenger_wins, champion_wins, total_matches))


class Orchestrator:
    """Manages the generational self-play + training loop with arena gatekeeper.

    Value targets: ΔME (match equity delta) or legacy points/4.
    Arena: sequential SPRT-style duplicate-deal matches with side-swap.
    Starts with 100 seeds, extends to 400 if inconclusive.

    Budget-specific checkpoint tracking:
      After each generation, a quick eval is run at each deployment budget.
      The best checkpoint per budget is saved as checkpoints/best_{N}sims.pt.
      This decouples "best for deployment" from "latest" (which auto-promotes).

    Live browser context:
      index.html ismctsEval uses MAX_ITERATIONS=600 with TIME_LIMIT=300ms.
      This is wall-clock limited, equivalent to ~100-200 Python fixed-count sims
      depending on hardware. Deploy best_100sims.pt or best_200sims.pt.
    """

    ARENA_SEEDS_MIN = 100    # start with 100 seeds (200 matches)
    ARENA_SEEDS_MAX = 600    # extend up to 600 seeds (1200 matches)
    ARENA_SEEDS_STEP = 100   # add 100 seeds per extension

    # Sim budgets to track for deployment-specific best checkpoints.
    # Browser equivalent: ~100-200 sims (TIME_LIMIT=300ms, MAX_ITERATIONS=600).
    BUDGET_EVAL_SIMS = [50, 100, 200]
    BUDGET_EVAL_PAIRS = 50   # 50 pairs = 100 games, fast enough per-gen

    # Graduated gating: after N consecutive rejections, relax the promote threshold.
    # This prevents stalling when the model improves by small increments (52-53%).
    # Format: (consecutive_rejections_threshold, min_win_rate_for_final_promote)
    GATING_SCHEDULE = [
        (0, 0.520),   # default: need 52% (CI lower > 50% still preferred)
        (3, 0.515),   # after 3 rejections: accept 51.5%
        (6, 0.510),   # after 6 rejections: accept 51.0%
        (10, 0.505),  # after 10 rejections: accept 50.5%
    ]

    def __init__(self, num_workers=4, buffer_size=200000, use_mcts=True,
                 mcts_sims=200, value_target='me', policy_target='visits',
                 high_sim_fraction=0.1, high_sim_multiplier=4,
                 use_belief_head=False, belief_weight=0.1,
                 use_support_head=False, support_weight=0.1,
                 aux_detach=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.high_sim_fraction = high_sim_fraction
        self.high_sim_multiplier = high_sim_multiplier
        self.policy_target = policy_target
        print(f"Orchestrator device: {self.device}")
        print(f"Workers: {num_workers}, Buffer: {buffer_size}, "
              f"MCTS: {use_mcts} ({mcts_sims} sims, "
              f"{high_sim_fraction:.0%} at {mcts_sims * high_sim_multiplier}), "
              f"Value: {value_target}, Policy: {policy_target}, "
              f"BeliefHead: {use_belief_head}, BeliefWeight: {belief_weight}, "
              f"SupportHead: {use_support_head}, SupportWeight: {support_weight}, "
              f"AuxDetach: {aux_detach}")

        self.model = DominoNet().to(self.device)
        self.champion_weights = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        self.use_belief_head  = use_belief_head
        self.use_support_head = use_support_head
        self.belief_weight    = belief_weight
        self.support_weight   = support_weight
        self.aux_detach       = aux_detach
        self.trainer = Trainer(self.model, lr=1e-3,
                               belief_weight=belief_weight,
                               support_weight=support_weight,
                               aux_detach=aux_detach)

        self.num_workers = num_workers
        self.use_mcts = use_mcts
        self.mcts_sims = mcts_sims
        self.value_target = value_target
        self.replay_buffer = deque(maxlen=buffer_size)
        self.generation = 0
        self.rejections = 0  # count of failed arena challenges
        self.consecutive_rejections = 0  # resets on promotion

        # Budget-specific best checkpoint tracking.
        # Maps: sim_budget -> {"gen": int, "win_rate": float, "path": str}
        self.budget_best: dict = {}
        self._enable_budget_tracking = True  # set False to skip budget evals

    def _get_promote_threshold(self):
        """Get the current minimum win rate for promotion based on consecutive rejections."""
        threshold = self.GATING_SCHEDULE[0][1]  # default
        for min_rejections, min_wr in self.GATING_SCHEDULE:
            if self.consecutive_rejections >= min_rejections:
                threshold = min_wr
        return threshold

    # ── Budget-specific checkpoint tracking ──────────────────────────────────

    def _budget_eval_at(self, challenger_weights, ref_weights, num_sims, num_pairs, seed_base):
        """Quick duplicate-deal eval at a fixed sim budget. Returns challenger win rate."""
        from domino_mcts import DominoMCTS as _MCTS
        device = torch.device("cpu")

        chall_model = DominoNet().to(device)
        chall_model.load_state_dict(challenger_weights, strict=False)
        chall_model.eval()

        ref_model = DominoNet().to(device)
        ref_model.load_state_dict(ref_weights, strict=False)
        ref_model.eval()

        mcts_c = _MCTS(chall_model, num_simulations=num_sims)
        mcts_r = _MCTS(ref_model,   num_simulations=num_sims)

        wins_c = 0
        total  = 0
        enc_c = DominoEncoder()
        enc_r = DominoEncoder()

        with torch.no_grad():
            for i in range(num_pairs):
                seed = seed_base + i
                for c_side in [0, 1]:
                    env = DominoEnv()
                    enc_c.reset(); enc_r.reset()
                    obs = env.reset(seed=seed)
                    step = 0
                    while not env.is_over() and step < 200:
                        mask = env.get_legal_moves_mask()
                        if mask.sum() == 0:
                            break
                        team = env.current_player % 2
                        # c_side=0: challenger plays team0; c_side=1: challenger plays team1
                        if team == c_side:
                            state = enc_c.encode(obs)
                            pi = mcts_c.get_action_probs(env, enc_c, temperature=0.1)
                        else:
                            state = enc_r.encode(obs)
                            pi = mcts_r.get_action_probs(env, enc_r, temperature=0.1)
                        action = int(np.argmax(pi * mask))
                        obs, _, done, _ = env.step(action)
                        step += 1
                    if env.game_over:
                        if env.winner_team == c_side:
                            wins_c += 1
                    total += 1
        return wins_c / total if total > 0 else 0.5

    def _track_budget_checkpoints(self, gen, ckpt_path_current):
        """
        Run quick evals at each BUDGET_EVAL_SIMS budget.
        Update checkpoints/best_{N}sims.pt if the current gen wins.

        Uses the previous best as reference (or gen1 if no prior best).
        Runs in-process (no subprocess) since it's a small eval.
        """
        if not self._enable_budget_tracking:
            return
        import shutil

        current_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        seed_base = 90000 + gen * 100  # dedicated seed range, away from arena/training

        for num_sims in self.BUDGET_EVAL_SIMS:
            best_info = self.budget_best.get(num_sims)
            if best_info is None:
                # No prior best → current gen is automatically the best at this budget
                best_path = os.path.join("checkpoints", f"best_{num_sims}sims.pt")
                shutil.copy2(ckpt_path_current, best_path)
                self.budget_best[num_sims] = {
                    "gen": gen, "win_rate": 0.5, "path": best_path
                }
                print(f"  [Budget {num_sims}sims] Initialized best → gen {gen}")
                continue

            # Load reference (previous best at this budget)
            ref_ckpt = torch.load(best_info["path"], map_location="cpu", weights_only=True)
            ref_weights = ref_ckpt.get("model_state_dict", ref_ckpt)

            wr = self._budget_eval_at(
                current_weights, ref_weights,
                num_sims=num_sims,
                num_pairs=self.BUDGET_EVAL_PAIRS,
                seed_base=seed_base + num_sims,
            )
            if wr >= 0.50:
                best_path = os.path.join("checkpoints", f"best_{num_sims}sims.pt")
                shutil.copy2(ckpt_path_current, best_path)
                prev_gen = best_info["gen"]
                self.budget_best[num_sims] = {
                    "gen": gen, "win_rate": wr, "path": best_path
                }
                print(f"  [Budget {num_sims}sims] New best: gen {gen} "
                      f"(wr={wr:.1%} vs gen {prev_gen})")
            else:
                print(f"  [Budget {num_sims}sims] No improvement: gen {gen} "
                      f"wr={wr:.1%} vs gen {best_info['gen']} — keeping existing best")

    # ── Main training loop ────────────────────────────────────────────────────

    def run(self, total_generations=100, games_per_worker=250):
        """Run the full generational training loop."""
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        os.makedirs("checkpoints", exist_ok=True)

        for gen in range(1, total_generations + 1):
            self.generation = gen
            print(f"\n{'='*50}")
            print(f"  GENERATION {gen}/{total_generations}")
            print(f"{'='*50}")

            # === PHASE 1: PARALLEL DATA GENERATION ===
            t0 = time.time()
            print(f"Spawning {self.num_workers} workers "
                  f"({games_per_worker} games each)...", flush=True)

            # Extract weights for CPU workers
            shared_weights = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }

            ctx = mp.get_context('spawn')
            result_queue = ctx.Queue()
            processes = []

            for w_id in range(self.num_workers):
                p = ctx.Process(
                    target=self_play_worker,
                    args=(w_id, shared_weights, games_per_worker,
                          self.use_mcts, self.mcts_sims, result_queue,
                          self.value_target, self.policy_target,
                          self.high_sim_fraction, self.high_sim_multiplier,
                          self.use_belief_head, self.use_support_head)
                )
                p.start()
                processes.append(p)

            # Collect results — timeout scales with games and MCTS sims
            # ~60s per MCTS game (50 sims), ~1s per no-MCTS game
            per_game_s = 90 if self.use_mcts else 3  # generous estimate
            worker_timeout = max(3600, games_per_worker * per_game_s)
            collected = 0
            total_samples = 0
            while collected < self.num_workers:
                try:
                    worker_data = result_queue.get(timeout=worker_timeout)
                    self.replay_buffer.extend(worker_data)
                    total_samples += len(worker_data)
                    collected += 1
                    print(f"  Worker {collected}/{self.num_workers} done: "
                          f"{len(worker_data)} samples", flush=True)
                except Exception as e:
                    alive = sum(1 for p in processes if p.is_alive())
                    print(f"  Worker collection error: {type(e).__name__}: {e} "
                          f"({alive} workers still alive)", flush=True)
                    if alive == 0:
                        print("  All workers dead, stopping collection.",
                              flush=True)
                        break
                    collected += 1

            for p in processes:
                p.join(timeout=30)

            elapsed = time.time() - t0
            total_games = self.num_workers * games_per_worker
            print(f"Self-play: {total_games} games in {elapsed:.1f}s "
                  f"({total_games/elapsed:.0f} games/s)")
            print(f"Buffer: {len(self.replay_buffer)} samples "
                  f"(+{total_samples} new)")

            # === PHASE 2: TRAINING ===
            min_buffer = 2000
            if len(self.replay_buffer) >= min_buffer:
                print("Training neural network...")
                dataset = ReplayDataset(list(self.replay_buffer))
                dataloader = DataLoader(
                    dataset, batch_size=256, shuffle=True,
                    num_workers=0, pin_memory=(self.device.type == 'cuda')
                )

                for epoch in range(5):
                    loss, v_loss, p_loss, b_loss, s_loss = self.trainer.train_epoch(dataloader)
                    print(f"  Epoch {epoch+1}/5 | "
                          f"Loss: {loss:.4f} "
                          f"(V: {v_loss:.4f}, P: {p_loss:.4f}, "
                          f"B: {b_loss:.4f}, S: {s_loss:.4f})")

                # === PHASE 3: ARENA EVALUATION ===
                challenger_weights = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

                # Arena gate REMOVED (per Gemini recommendation):
                # 50-game SPRT has 40% false rejection rate at 52% true edge.
                # Always promote latest checkpoint — unfreezes training.

                # === PARTNERSHIP SUITE REGRESSION GUARD ===
                # Soft check only — does not block promotion.
                # Logs a warning if partnership score drops vs baseline.
                partnership_score = None
                try:
                    import sys as _sys
                    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
                    from test_partnership_suite import evaluate_suite, make_engine_fn, SUITE_PATH
                    _eng = make_engine_fn(self.model, sims=0, device=str(self.device))
                    _report = evaluate_suite(_eng, SUITE_PATH)
                    partnership_score = _report["avg_score"]
                    PARTNER_BASELINE = 0.462  # gen50 greedy baseline
                    if partnership_score < PARTNER_BASELINE - 0.05:
                        print(f"  [PARTNERSHIP WARNING] score={partnership_score:.3f} "
                              f"< baseline {PARTNER_BASELINE:.3f} - 0.05 "
                              f"(regression detected — consider rejecting)")
                    else:
                        print(f"  [PARTNERSHIP] score={partnership_score:.3f} "
                              f"(baseline={PARTNER_BASELINE:.3f})")
                except Exception as _e:
                    print(f"  [PARTNERSHIP] suite check skipped: {_e}")

                self.champion_weights = challenger_weights
                ckpt_path = f"checkpoints/domino_gen_{gen:04d}.pt"
                torch.save({
                    'generation': gen,
                    'model_state_dict': self.model.state_dict(),
                    'buffer_size': len(self.replay_buffer),
                    'consecutive_rejections': 0,
                    'partnership_score': partnership_score,
                }, ckpt_path)
                print(f"Auto-promoted gen {gen}. Saved: {ckpt_path}")

                # Track budget-specific best checkpoints.
                # best_{N}sims.pt = best checkpoint at that deployment sim budget.
                # Browser uses TIME_LIMIT=300ms ≈ 100-200 Python sims.
                if self._enable_budget_tracking and self.use_mcts:
                    self._track_budget_checkpoints(gen, ckpt_path)
            else:
                print(f"Buffer too small ({len(self.replay_buffer)}/{min_buffer}). "
                      f"Gathering more data...")

            # === LR DECAY SCHEDULE (relative to this run) ===
            if gen in (50, 100):
                for g in self.trainer.optimizer.param_groups:
                    g['lr'] *= 0.1
                print(f"LR decay at gen {gen}: "
                      f"new LR = {self.trainer.optimizer.param_groups[0]['lr']:.2e}")

        print(f"\nTraining complete. {total_generations} generations. "
              f"Rejections: {self.rejections}")

    def _run_arena_batch(self, challenger_weights, seeds):
        """Run a batch of arena seeds in parallel. Returns (c_wins, ch_wins, total)."""
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        n_workers = min(self.num_workers, len(seeds))
        chunk = max(1, len(seeds) // n_workers)
        processes = []

        for w_id in range(n_workers):
            start = w_id * chunk
            end = start + chunk if w_id < n_workers - 1 else len(seeds)
            if start >= len(seeds):
                break
            p = ctx.Process(
                target=arena_worker,
                args=(w_id, self.champion_weights, challenger_weights,
                      seeds[start:end], result_queue)
            )
            p.start()
            processes.append(p)

        c_wins, ch_wins, total = 0, 0, 0
        for _ in range(len(processes)):
            try:
                cw, chw, tm = result_queue.get(timeout=300)
                c_wins += cw
                ch_wins += chw
                total += tm
            except Exception as e:
                print(f"  Arena worker error: {e}")

        for p in processes:
            p.join(timeout=30)

        return c_wins, ch_wins, total

    def _arena_evaluate(self, challenger_weights):
        """Sequential SPRT-style arena with graduated gating.

        Protocol:
          1. Run 100 seeds (200 matches)
          2. If win_rate < 47% → early reject (clearly worse)
          3. If win_rate > 55% AND CI lower > 50% → early promote
          4. If CI lower > 50% → promote (strong statistical evidence)
          5. Otherwise → extend by 100 seeds, repeat up to ARENA_SEEDS_MAX
          6. Final decision: graduated threshold based on consecutive rejections
             (relaxes from 52% → 51.5% → 51% → 50.5% as rejections accumulate)
        """
        t0 = time.time()
        challenger_wins = 0
        champion_wins = 0
        matches_played = 0
        seeds_used = 0

        promote_threshold = self._get_promote_threshold()
        print(f"Arena: sequential SPRT (min {self.ARENA_SEEDS_MIN}, "
              f"max {self.ARENA_SEEDS_MAX} seeds, ×2 sides each, "
              f"gate={promote_threshold:.1%} after {self.consecutive_rejections} "
              f"consecutive rejections)...")

        while seeds_used < self.ARENA_SEEDS_MAX:
            batch_size = min(self.ARENA_SEEDS_STEP,
                             self.ARENA_SEEDS_MAX - seeds_used)
            seed_base = self.generation * 10000 + seeds_used
            seeds = list(range(seed_base, seed_base + batch_size))

            cw, chw, tm = self._run_arena_batch(challenger_weights, seeds)
            challenger_wins += cw
            champion_wins += chw
            matches_played += tm
            seeds_used += batch_size

            if matches_played == 0:
                win_rate = 0.5
            else:
                win_rate = challenger_wins / matches_played

            ci_half = 1.96 * np.sqrt(
                win_rate * (1 - win_rate) / max(matches_played, 1))
            ci_lower = win_rate - ci_half
            ci_upper = win_rate + ci_half

            print(f"  [{seeds_used} seeds / {matches_played} matches] "
                  f"Challenger {challenger_wins}W vs Champion {champion_wins}W "
                  f"({win_rate:.1%}, CI: [{ci_lower:.1%}, {ci_upper:.1%}])")

            # Early decisions (only after minimum seeds)
            if seeds_used >= self.ARENA_SEEDS_MIN:
                # Early reject: clearly worse
                if win_rate < 0.47:
                    print(f"  EARLY REJECT (win_rate {win_rate:.1%} < 47%)")
                    return False
                # Early promote: clearly better
                if win_rate > 0.55 and ci_lower > 0.50:
                    elapsed = time.time() - t0
                    print(f"  EARLY PROMOTE! ({win_rate:.1%}, "
                          f"CI lower {ci_lower:.1%} > 50%) in {elapsed:.1f}s")
                    return True
                # Conclusive: CI entirely above or below 50%
                if ci_lower > 0.50:
                    elapsed = time.time() - t0
                    print(f"  PROMOTED! (CI lower {ci_lower:.1%} > 50%) "
                          f"in {elapsed:.1f}s")
                    return True
                if ci_upper < 0.50:
                    print(f"  REJECTED (CI upper {ci_upper:.1%} < 50%)")
                    return False

            # If more seeds available and result is inconclusive, extend
            if seeds_used < self.ARENA_SEEDS_MAX:
                print(f"  Inconclusive — extending arena...")

        # Final decision after max seeds — use graduated threshold
        elapsed = time.time() - t0
        if matches_played > 0:
            win_rate = challenger_wins / matches_played
            ci_half = 1.96 * np.sqrt(
                win_rate * (1 - win_rate) / matches_played)
            ci_lower = win_rate - ci_half
        else:
            win_rate, ci_lower = 0.5, 0.0

        # Strong statistical evidence always promotes
        if ci_lower > 0.50:
            print(f"  PROMOTED (final: {win_rate:.1%}, "
                  f"CI lower {ci_lower:.1%} > 50%) in {elapsed:.1f}s")
            return True

        # Graduated gate: accept modest improvements after repeated rejections
        if win_rate >= promote_threshold and matches_played >= self.ARENA_SEEDS_MAX * 2 * 0.8:
            print(f"  GRADUATED PROMOTE (final: {win_rate:.1%} >= "
                  f"{promote_threshold:.1%} gate, {self.consecutive_rejections} "
                  f"prior rejections) in {elapsed:.1f}s")
            return True
        else:
            print(f"  REJECTED (final: {win_rate:.1%}, "
                  f"CI lower {ci_lower:.1%}, gate {promote_threshold:.1%}) "
                  f"in {elapsed:.1f}s")
            return False

    def _notify_telegram(self, message):
        """Send a Telegram notification via OpenClaw CLI (fire and forget)."""
        try:
            subprocess.Popen(
                ['openclaw', 'message', 'send',
                 '--channel', 'telegram',
                 '--target', '1570780899',
                 '--message', message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"  [Telegram notify failed: {e}]")

    def load_checkpoint(self, path):
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        incompat = self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if incompat.missing_keys:
            print(f"[load_checkpoint] Missing keys: {incompat.missing_keys}")
        if incompat.unexpected_keys:
            print(f"[load_checkpoint] Unexpected keys: {incompat.unexpected_keys}")
        self.champion_weights = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        self.generation = ckpt.get('generation', 0)
        self.consecutive_rejections = ckpt.get('consecutive_rejections', 0)
        print(f"Loaded checkpoint: {path} (gen {self.generation}, "
              f"consec_rej: {self.consecutive_rejections}, "
              f"gate: {self._get_promote_threshold():.1%})")


def main():
    parser = argparse.ArgumentParser(description='Pernambuco Domino AI Trainer')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1),
                        help='Number of parallel self-play workers')
    parser.add_argument('--generations', type=int, default=100,
                        help='Total training generations')
    parser.add_argument('--games-per-worker', type=int, default=250,
                        help='Games each worker plays per generation')
    parser.add_argument('--buffer-size', type=int, default=200000,
                        help='Replay buffer capacity')
    parser.add_argument('--mcts', action='store_true', default=True,
                        help='Use IS-MCTS (slower but stronger, default ON)')
    parser.add_argument('--no-mcts', dest='mcts', action='store_false',
                        help='Disable IS-MCTS (fast, weaker)')
    parser.add_argument('--mcts-sims', type=int, default=200,
                        help='MCTS simulations per move (base budget, default 200)')
    parser.add_argument('--high-sim-fraction', type=float, default=0.1,
                        help='Fraction of games using high sim count (default 0.1)')
    parser.add_argument('--high-sim-multiplier', type=int, default=4,
                        help='Multiplier for high-quality games (default 4x)')
    parser.add_argument('--value-target', type=str, default='me',
                        choices=['me', 'points'],
                        help='Value target: "me" for ΔME (default), '
                             '"points" for legacy points/4')
    parser.add_argument('--policy-target', type=str, default='visits',
                        choices=['visits', 'heuristic'],
                        help='Policy target: "visits" = MCTS visit-count pi (default, AlphaZero-style), '
                             '"heuristic" = network policy + Dirichlet noise (legacy)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--belief-head', action='store_true',
                        help='Enable 21-output auxiliary pip-belief head')
    parser.add_argument('--belief-weight', type=float, default=0.1,
                        help='Auxiliary belief loss weight (default 0.1)')
    parser.add_argument('--support-head', action='store_true',
                        help='Enable 6-output end-support auxiliary head (Phase 6.5)')
    parser.add_argument('--support-weight', type=float, default=0.1,
                        help='Auxiliary support loss weight (default 0.1)')
    parser.add_argument('--aux-detach', action='store_true', default=True,
                        help='Stop-gradient through aux probs before conditioning '
                             'policy/value heads (default: True, safer for Phase 6.5 probe)')
    parser.add_argument('--no-aux-detach', dest='aux_detach', action='store_false',
                        help='Allow full gradients through aux conditioning path')
    parser.add_argument('--no-budget-tracking', action='store_true',
                        help='Disable per-gen budget-specific checkpoint tracking')
    args = parser.parse_args()

    orch = Orchestrator(
        num_workers=args.workers,
        buffer_size=args.buffer_size,
        use_mcts=args.mcts,
        mcts_sims=args.mcts_sims,
        value_target=args.value_target,
        policy_target=args.policy_target,
        high_sim_fraction=args.high_sim_fraction,
        high_sim_multiplier=args.high_sim_multiplier,
        use_belief_head=args.belief_head,
        belief_weight=args.belief_weight,
        use_support_head=args.support_head,
        support_weight=args.support_weight,
        aux_detach=args.aux_detach,
    )

    if args.resume:
        orch.load_checkpoint(args.resume)

    if args.no_budget_tracking:
        orch._enable_budget_tracking = False

    orch.run(
        total_generations=args.generations,
        games_per_worker=args.games_per_worker,
    )


if __name__ == "__main__":
    main()
