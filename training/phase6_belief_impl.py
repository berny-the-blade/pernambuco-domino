"""
Phase 6 MVP — Belief-Aware Representation Learning
Pseudocode / implementation reference
Source: Bernie analysis 2026-03-08

Design intent:
  - Train the belief head jointly with policy/value
  - Do NOT wire belief into search/inference yet
  - Let it reshape the trunk first
  - Measure whether that alone improves partnership suite + search scaling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. TILE / PLAYER HELPERS
# ---------------------------------------------------------------------------

TILES = [
    (0, 0),
    (0, 1), (1, 1),
    (0, 2), (1, 2), (2, 2),
    (0, 3), (1, 3), (2, 3), (3, 3),
    (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),
    (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5),
    (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6),
]


def other_players(me: int):
    """
    Relative order (fixed for all label indexing):
      0 = partner
      1 = LHO
      2 = RHO
    """
    partner = (me + 2) % 4
    lho     = (me + 1) % 4
    rho     = (me + 3) % 4
    return [partner, lho, rho]


def tile_to_pips(tile):
    """
    Accepts tile-index (int) or tile tuple/list [a, b].
    Returns (a, b) pip tuple.
    """
    if isinstance(tile, int):
        return TILES[tile]
    return tuple(tile)


# ---------------------------------------------------------------------------
# 2. LABEL GENERATION
# ---------------------------------------------------------------------------

def build_belief_target(hidden_hands_by_player: dict, me: int) -> np.ndarray:
    """
    Build the 21-dim binary belief target for a training record.

    hidden_hands_by_player: dict[int, list[tile]]
      where tile is either a tile-index (int) or a [a, b] pair.
      Keys are absolute player indices (0..3).

    me: absolute index of the player to move.

    Returns: np.ndarray shape (21,), float32

    Target layout:
      Index  0..6  = partner has pip 0..6
      Index  7..13 = LHO     has pip 0..6
      Index 14..20 = RHO     has pip 0..6

    Each value is 1.0 if that player holds ANY tile containing that pip,
    0.0 otherwise.
    """
    target = np.zeros(21, dtype=np.float32)

    for rel_idx, abs_player in enumerate(other_players(me)):
        seen = np.zeros(7, dtype=np.float32)
        for tile in hidden_hands_by_player[abs_player]:
            a, b = tile_to_pips(tile)
            seen[a] = 1.0
            seen[b] = 1.0
        start = rel_idx * 7
        target[start:start + 7] = seen

    return target


# Example usage:
# hidden = {
#     0: [(5,5), (1,4), (0,3)],
#     1: [(5,1), (2,2)],
#     2: [(1,6), (3,4)],
#     3: [(0,6), (4,5)],
# }
# me = 0
# belief_target = build_belief_target(hidden, me)  # shape (21,)


# ---------------------------------------------------------------------------
# 3. TRAINING RECORD SCHEMA
# ---------------------------------------------------------------------------

# When exporting a self-play row, extend record like this:
#
# record = {
#     "x":             encoded_state,                              # list[float], len 185 (or 213)
#     "pi":            policy_target,                              # list[float], len 57
#     "v":             value_target_dme,                           # scalar float
#     "mask":          legal_mask,                                 # list[float], len 57
#     "belief_target": build_belief_target(hidden_hands, me).tolist(),  # NEW: list[float], len 21
#     "meta": {
#         "player":        me,
#         "team_pov":      me % 2,
#         "scores_before": [score0, score1],
#         "dobrada_before": dobrada_idx,
#     }
# }


# ---------------------------------------------------------------------------
# 4. MODEL DEFINITION
# ---------------------------------------------------------------------------

class DominoNet(nn.Module):
    def __init__(self, state_dim=185, hidden_dim=256):
        super().__init__()

        # Replace this trunk with your real trunk.
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Existing heads (unchanged)
        self.policy_head = nn.Linear(hidden_dim, 57)
        self.value_head  = nn.Linear(hidden_dim, 1)

        # NEW: auxiliary belief head
        # Small 2-layer head; no need to be large.
        self.belief_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 21),
        )

    def forward(self, x, return_belief=True):
        h = self.trunk(x)

        policy_logits = self.policy_head(h)
        value         = torch.tanh(self.value_head(h)).squeeze(-1)

        if return_belief:
            belief_logits = self.belief_head(h)
            return {
                "policy_logits": policy_logits,
                "value":         value,
                "belief_logits": belief_logits,
            }

        return {
            "policy_logits": policy_logits,
            "value":         value,
        }


# ---------------------------------------------------------------------------
# 5. TRAINING LOSSES
# ---------------------------------------------------------------------------

def masked_policy_ce(policy_logits, pi_target, legal_mask):
    """
    policy_logits : [B, 57]
    pi_target     : [B, 57]  (visit-count distribution, sums to 1)
    legal_mask    : [B, 57]  (1 = legal, 0 = illegal)
    """
    masked_logits = policy_logits.masked_fill(legal_mask <= 0, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    loss = -(pi_target * log_probs).sum(dim=-1).mean()
    return loss


def training_step(model, batch, lambda_value=1.0, lambda_belief=0.2):
    """
    batch keys:
      x              : [B, state_dim]
      pi             : [B, 57]
      v              : [B]
      mask           : [B, 57]
      belief_target  : [B, 21]  float32 binary
    """
    x              = batch["x"]
    pi             = batch["pi"]
    v              = batch["v"]
    mask           = batch["mask"]
    belief_target  = batch["belief_target"]

    out = model(x, return_belief=True)

    policy_loss = masked_policy_ce(out["policy_logits"], pi, mask)
    value_loss  = F.mse_loss(out["value"], v)
    belief_loss = F.binary_cross_entropy_with_logits(
        out["belief_logits"],
        belief_target,
    )

    total_loss = policy_loss + lambda_value * value_loss + lambda_belief * belief_loss

    stats = {
        "loss":         total_loss.item(),
        "policy_loss":  policy_loss.item(),
        "value_loss":   value_loss.item(),
        "belief_loss":  belief_loss.item(),
    }
    return total_loss, stats


# Why BCEWithLogitsLoss and NOT CrossEntropyLoss:
#   Each pip label is independent (multi-label binary classification).
#   Partner can have BOTH pip 1 and pip 4 simultaneously.
#   Softmax/CE is wrong here. BCE is correct.


# ---------------------------------------------------------------------------
# 6. EVALUATION METRICS FOR THE BELIEF HEAD
# ---------------------------------------------------------------------------

@torch.no_grad()
def belief_metrics(model, batch):
    """
    Quick scalar belief evaluation.
    For better diagnostics, split by player (partner/LHO/RHO) and by pip.
    """
    out   = model(batch["x"], return_belief=True)
    probs  = torch.sigmoid(out["belief_logits"])
    target = batch["belief_target"]

    preds = (probs > 0.5).float()
    acc   = (preds == target).float().mean().item()
    bce   = F.binary_cross_entropy_with_logits(out["belief_logits"], target).item()

    return {
        "belief_acc": acc,
        "belief_bce": bce,
    }


# Optional: per-player breakdown during validation
def belief_metrics_verbose(model, batch):
    out    = model(batch["x"], return_belief=True)
    probs  = torch.sigmoid(out["belief_logits"])  # [B, 21]
    target = batch["belief_target"]               # [B, 21]

    results = {}
    for rel_idx, name in enumerate(["partner", "lho", "rho"]):
        start = rel_idx * 7
        end   = start + 7
        p = probs[:, start:end]
        t = target[:, start:end]
        preds = (p > 0.5).float()
        results[f"{name}_acc"] = (preds == t).float().mean().item()
        results[f"{name}_bce"] = F.binary_cross_entropy_with_logits(p, t).item()

    return results


# Training-time diagnostic — dump a single record during validation:
# probs = torch.sigmoid(out["belief_logits"][0]).cpu().numpy()
# print("Partner pips 0..6:", probs[0:7])
# print("LHO pips 0..6:",    probs[7:14])
# print("RHO pips 0..6:",    probs[14:21])


# ---------------------------------------------------------------------------
# 7. EXPORT — STRIP BELIEF HEAD FOR BROWSER/MOBILE
# ---------------------------------------------------------------------------

def export_inference_state_dict(model):
    """
    Return a state_dict with belief_head.* keys removed.
    Use this for browser/mobile export — keeps loader unchanged.

    The trunk/policy/value weights are IDENTICAL to a jointly-trained model.
    The representation benefit is baked into the trunk already.
    """
    sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if not k.startswith("belief_head.")}
    return filtered


# If your exporter writes raw arrays, simply skip belief_head.* keys entirely.
# No browser/mobile inference changes required in Phase 6.


# ---------------------------------------------------------------------------
# 8. HYPERPARAMETER STARTING POINTS
# ---------------------------------------------------------------------------

# Recommended starting config:
LAMBDA_VALUE  = 1.0
LAMBDA_BELIEF = 0.2   # try 0.1 if policy/value regress; 0.3 if belief trains too slowly

# Experiment matrix:
#   Baseline:  no belief head
#   Variant A: λ_belief = 0.2
#   Variant B: λ_belief = 0.1
#   Variant C: λ_belief = 0.3

# 5–10 generations is enough to evaluate. No need for 50.


# ---------------------------------------------------------------------------
# 9. PHASE 6.5 PREVIEW (not yet — measure Phase 6 first)
# ---------------------------------------------------------------------------

# If Phase 6 MVP works, next step is to feed belief outputs back into
# policy/value heads explicitly:
#
# class DominoNetV2(nn.Module):
#     def forward(self, x):
#         h = self.trunk(x)
#         belief_logits = self.belief_head(h)
#         belief_probs  = torch.sigmoid(belief_logits)         # [B, 21]
#         h_aug         = torch.cat([h, belief_probs], dim=-1) # [B, hidden+21]
#         policy_logits = self.policy_head_v2(h_aug)
#         value         = self.value_head_v2(h_aug)
#         ...
#
# But do NOT do this in Phase 6. Trunk reshaping first. Measure. Then condition.
