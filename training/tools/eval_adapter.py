"""
tools/eval_adapter.py -- Thin adapter layer for evaluation scripts.

Exposes a clean interface between eval scripts (search_scaling_eval.py,
particle_disagreement_eval.py) and the training pipeline internals.

All pipeline-specific glue lives here. Eval scripts import from this module
and should not call DominoNet / DominoMCTS / DominoEnv directly.

Public API:
    load_checkpoint_model(checkpoint_path, device) -> LoadedModel
    build_eval_agent(loaded, sims, temperature)    -> EvalAgent
    run_duplicate_pair(agent_a, agent_b, seed)     -> DuplicatePairResult
    sample_public_states(...)                      -> list[PublicStateSample]
    run_particle_search(agent, public_state, ...)  -> ParticleSearchResult
"""

from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# Resolve training/ root so imports work regardless of cwd
_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

from domino_env import DominoEnv, TILES, NUM_TILES
from domino_encoder import DominoEncoder
from domino_net import DominoNet
from domino_mcts import DominoMCTS


# ============================================================
# Data contracts
# ============================================================

@dataclass
class LoadedModel:
    checkpoint_path: str
    model: DominoNet
    generation: int | None
    device: str
    extra: dict = field(default_factory=dict)


@dataclass
class EvalAgent:
    loaded: LoadedModel
    sims: int
    temperature: float
    extra: dict = field(default_factory=dict)

    @property
    def model(self) -> DominoNet:
        return self.loaded.model

    def make_mcts(self) -> DominoMCTS:
        return DominoMCTS(self.loaded.model, num_simulations=self.sims)


@dataclass
class DuplicatePairGameResult:
    margin_a: float          # teamA_points - teamB_points (from A's perspective)
    winner: int | None       # 0 = A's team won, 1 = B's team won, None = draw
    root_entropies: list[float]
    root_top1_masses: list[float]
    root_top2_gaps: list[float]
    forced_move_flags: list[bool]
    game_length: int
    extra: dict = field(default_factory=dict)


@dataclass
class DuplicatePairResult:
    seed: int
    game1: DuplicatePairGameResult   # A = team 0
    game2: DuplicatePairGameResult   # A = team 1 (seats swapped)

    @property
    def pair_margin_a(self) -> float:
        return 0.5 * (self.game1.margin_a + self.game2.margin_a)

    @property
    def wins_a(self) -> int:
        return int(self.game1.winner == 0) + int(self.game2.winner == 0)

    @property
    def wins_b(self) -> int:
        return int(self.game1.winner == 1) + int(self.game2.winner == 1)


@dataclass
class PublicStateSample:
    state_id: int
    public_state: Any        # DominoEnv snapshot (cloned)
    phase: str               # "early" | "mid" | "late"
    forced: bool
    legal_count: int
    obs: dict = field(default_factory=dict)
    mask: Any = None         # np.ndarray of shape (57,)
    extra: dict = field(default_factory=dict)


@dataclass
class ParticleSearchResult:
    top_action: int
    root_value: float
    root_policy: list[float]
    root_entropy: float
    forced: bool
    legal_count: int
    extra: dict = field(default_factory=dict)


# ============================================================
# 1. load_checkpoint_model
# ============================================================

def load_checkpoint_model(checkpoint_path: str, device: str = "cpu") -> LoadedModel:
    """
    Load a DominoNet checkpoint and return a normalized handle.

    Safe to call repeatedly. Does not mutate trainer/global state.
    Works for old (input_dim=185) and current (input_dim=213) checkpoints.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        generation = ckpt.get("generation", None)
    else:
        state_dict = ckpt
        generation = None

    # Auto-detect generation from filename if not in checkpoint
    if generation is None:
        try:
            bn = os.path.basename(checkpoint_path)
            generation = int(bn.replace("domino_gen_", "").replace(".pt", ""))
        except ValueError:
            pass

    input_dim = state_dict["input_fc.weight"].shape[1]
    model = DominoNet(input_dim=input_dim, hidden_dim=256, num_actions=57, num_blocks=4)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return LoadedModel(
        checkpoint_path=checkpoint_path,
        model=model,
        generation=generation,
        device=device,
        extra={"input_dim": input_dim},
    )


# ============================================================
# 2. build_eval_agent
# ============================================================

def build_eval_agent(loaded: LoadedModel, sims: int,
                      temperature: float = 0.1) -> EvalAgent:
    """
    Build a search-capable eval agent from a loaded checkpoint.
    Temperature 0.1 = mostly greedy (better for evaluation than 0.0 exact tie-breaking).
    """
    return EvalAgent(loaded=loaded, sims=sims, temperature=temperature)


# ============================================================
# 3. run_duplicate_pair
# ============================================================

def _play_one_game(agent_team0: EvalAgent, agent_team1: EvalAgent,
                    seed: int) -> DuplicatePairGameResult:
    """
    Play one full game. agent_team0 controls players 0 & 2, agent_team1 controls 1 & 3.
    Returns result from team0's perspective.

    DominoEncoder.encode(obs) calls _sync_belief(obs) internally on every call,
    so beliefs stay current without manual update_on_pass/play in the outer loop.
    """
    env = DominoEnv()
    enc0 = DominoEncoder(); enc0.reset()
    enc1 = DominoEncoder(); enc1.reset()
    env.reset(seed=seed)

    mcts0 = agent_team0.make_mcts()
    mcts1 = agent_team1.make_mcts()

    root_entropies: list[float] = []
    root_top1_masses: list[float] = []
    root_top2_gaps: list[float] = []
    forced_flags: list[bool] = []
    move_count = 0

    while not env.is_over():
        mask = env.get_legal_moves_mask()
        legal = np.where(mask > 0)[0]
        forced = len(legal) == 1
        forced_flags.append(forced)
        move_count += 1

        team = env.current_player % 2
        if team == 0:
            pi = mcts0.get_action_probs(env, enc0, temperature=agent_team0.temperature)
        else:
            pi = mcts1.get_action_probs(env, enc1, temperature=agent_team1.temperature)

        # Collect root diagnostics (from whichever agent just moved)
        vis = pi[pi > 1e-9]
        if vis.size > 0:
            ent = float(-np.sum(vis * np.log(vis + 1e-10)))
            sv = np.sort(vis)[::-1]
            t1 = float(sv[0])
            t2g = float(sv[0] - sv[1]) if len(sv) > 1 else 1.0
        else:
            ent, t1, t2g = 0.0, 1.0, 1.0

        root_entropies.append(ent)
        root_top1_masses.append(t1)
        root_top2_gaps.append(t2g)

        action = int(np.argmax(pi * mask))
        env.step(action)

    winner_team = env.winner_team
    points = env.points_won
    # margin from team0's perspective: positive = team0 won
    margin = float(points if winner_team == 0 else -points)

    return DuplicatePairGameResult(
        margin_a=margin,
        winner=winner_team if winner_team >= 0 else None,
        root_entropies=root_entropies,
        root_top1_masses=root_top1_masses,
        root_top2_gaps=root_top2_gaps,
        forced_move_flags=forced_flags,
        game_length=move_count,
    )


def run_duplicate_pair(agent_a: EvalAgent, agent_b: EvalAgent,
                        seed: int) -> DuplicatePairResult:
    """
    Run a duplicate-deal pair with seat swap.

    Game 1: A controls team 0, B controls team 1.
    Game 2: same deal (seed), B controls team 0, A controls team 1.

    Margin convention: margin_a = teamA_points - teamB_points.
    Continuous (1-4 point range). Positive = A won, negative = B won.

    pair_margin_a = 0.5 * (game1.margin_a + game2.margin_a)
    """
    game1 = _play_one_game(agent_a, agent_b, seed)

    # Game 2: swap seats. B is now team0, A is team1.
    # game2_result is from team0 (B's) perspective, so flip for A.
    game2_raw = _play_one_game(agent_b, agent_a, seed)
    game2 = DuplicatePairGameResult(
        margin_a=-game2_raw.margin_a,   # flip to A's perspective
        winner=(1 - game2_raw.winner) if game2_raw.winner is not None else None,
        root_entropies=game2_raw.root_entropies,
        root_top1_masses=game2_raw.root_top1_masses,
        root_top2_gaps=game2_raw.root_top2_gaps,
        forced_move_flags=game2_raw.forced_move_flags,
        game_length=game2_raw.game_length,
    )

    return DuplicatePairResult(seed=seed, game1=game1, game2=game2)


# ============================================================
# 4. sample_public_states
# ============================================================

def _game_phase(obs: dict) -> str:
    n = len(obs.get("played", []))
    return "early" if n <= 7 else ("mid" if n <= 15 else "late")


def sample_public_states(state_source: str, state_count: int,
                          phase_buckets: list[str],
                          non_forced_only: bool = False,
                          state_file: str | None = None,
                          seed_base: int = 9000) -> list[PublicStateSample]:
    """
    Sample non-terminal game states via random self-play, stratified by phase.

    state_source: "replay" | "random" (both use random play for now)
    state_file: not yet used; reserved for future file-based sampling

    Returns list of PublicStateSample, one per state.
    """
    rng = np.random.default_rng(42)
    per_phase = {p: state_count // len(phase_buckets) for p in phase_buckets}
    remainder = state_count - sum(per_phase.values())
    for p in list(per_phase.keys())[:remainder]:
        per_phase[p] += 1

    collected: dict[str, list] = {p: [] for p in phase_buckets}
    total_needed = state_count
    game_idx = 0

    while sum(len(v) for v in collected.values()) < total_needed:
        env = DominoEnv()
        env.reset(seed=seed_base + game_idx)
        game_idx += 1

        while not env.is_over():
            mask = env.get_legal_moves_mask()
            legal = np.where(mask > 0)[0]
            obs = env.get_obs()
            ph = _game_phase(obs)
            forced = len(legal) == 1

            if ph in phase_buckets and len(collected[ph]) < per_phase[ph]:
                if (not non_forced_only) or (not forced):
                    collected[ph].append((env.clone(), obs, mask.copy()))

            env.step(int(rng.choice(legal)))

    samples = []
    sid = 0
    for ph in phase_buckets:
        for env_snap, obs, mask in collected[ph]:
            samples.append(PublicStateSample(
                state_id=sid,
                public_state=env_snap,
                phase=ph,
                forced=(int(mask.sum()) == 1),
                legal_count=int(mask.sum()),
                obs=obs,
                mask=mask,
            ))
            sid += 1

    return samples


# ============================================================
# 5. run_particle_search
# ============================================================

def _sample_one_particle(obs: dict, rng) -> dict:
    """
    Sample one plausible hand assignment consistent with observations.
    Returns {partner: set, lho: set, rho: set} of tile indices.
    """
    me = obs["player"]
    partner, lho, rho = (me + 2) % 4, (me + 1) % 4, (me + 3) % 4
    my_hand  = set(obs["hand"])
    played   = set(obs["played"])
    cant_have  = obs["cant_have"]
    hand_sizes = obs["hand_sizes"]
    unknown = [t for t in range(NUM_TILES) if t not in my_hand and t not in played]
    player_names = ["partner", "lho", "rho"]
    player_abs   = [partner, lho, rho]
    targets = {"partner": hand_sizes[partner], "lho": hand_sizes[lho], "rho": hand_sizes[rho]}

    for _attempt in range(40):
        asgn = {"partner": set(), "lho": set(), "rho": set()}
        tiles = list(unknown); rng.shuffle(tiles)
        ok = True
        for tile in tiles:
            left, right = TILES[tile]
            eligible = [n for n, p in zip(player_names, player_abs)
                        if left  not in cant_have[p] and
                           right not in cant_have[p] and
                           len(asgn[n]) < targets[n]]
            eligible.append("dorme")
            weights = [max(0.0, float(targets[n] - len(asgn[n])))
                       if n != "dorme" else 0.5 for n in eligible]
            total_w = sum(weights)
            if total_w == 0:
                ok = False; break
            weights = [w / total_w for w in weights]
            chosen = rng.choice(eligible, p=weights)
            if chosen != "dorme":
                asgn[chosen].add(tile)
        if ok:
            return asgn
    return {"partner": set(), "lho": set(), "rho": set()}   # fallback: empty


def _encoder_for_particle(obs: dict, assignment: dict) -> DominoEncoder:
    enc = DominoEncoder(); enc.reset()
    enc.belief[:, :] = 0.0
    my_hand = set(obs["hand"]); played = set(obs["played"])
    for t in range(NUM_TILES):
        if t in my_hand or t in played:
            continue
        if   t in assignment.get("partner", set()): enc.belief[t, 0] = 1.0
        elif t in assignment.get("lho",     set()): enc.belief[t, 1] = 1.0
        elif t in assignment.get("rho",     set()): enc.belief[t, 2] = 1.0
        else:                                        enc.belief[t, 3] = 1.0
    return enc


def run_particle_search(agent: EvalAgent, public_state: PublicStateSample,
                         particle_idx: int, seed: int) -> ParticleSearchResult:
    """
    Run one search from one state under one sampled particle.

    Particle is drawn deterministically from (seed, particle_idx) so results
    are reproducible. Same public_state across all particles for a given row.
    """
    # Deterministic RNG per (seed, particle_idx)
    # seed should be: seed_base + state_id * 1000 (caller's responsibility)
    rng = np.random.default_rng(seed + particle_idx)

    obs  = public_state.obs
    mask = public_state.mask
    env_snap = public_state.public_state

    assignment = _sample_one_particle(obs, rng)
    enc = _encoder_for_particle(obs, assignment)
    state_np = enc.encode(obs)

    if agent.sims > 0:
        mcts = agent.make_mcts()
        pi = mcts.get_action_probs(env_snap.clone(), enc, temperature=1.0)
        _, raw_v = agent.model.predict(state_np, mask)
        root_value = float(raw_v)
    else:
        pi, raw_v = agent.model.predict(state_np, mask)
        root_value = float(raw_v)

    vis = pi[pi > 1e-9]
    ent = float(-np.sum(vis * np.log(vis + 1e-10))) if vis.size > 0 else 0.0

    return ParticleSearchResult(
        top_action=int(np.argmax(pi)),
        root_value=root_value,
        root_policy=pi.tolist(),
        root_entropy=ent,
        forced=public_state.forced,
        legal_count=public_state.legal_count,
    )
