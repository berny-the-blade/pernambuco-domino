"""
Two-headed ResNet for Pernambuco Domino — Phase 6.5

Input:  213-dim state tensor
Output: Policy (57-dim probability distribution over actions)
        Value  (scalar in [-1, 1], Team 0's expected match equity)

Phase 6.5 changes:
  - belief_head: 21-output pip-presence auxiliary (partner/LHO/RHO × 7 pips)
  - support_head: 6-output end-playability auxiliary (partner/LHO/RHO × left/right)
  - aux_proj: projects 27-dim concat(belief_probs, support_probs) back to hidden_dim
  - aux_gate: learned scalar init=0.0 — starts as identity (old behavior), learns to use aux
  - h_cond = h + tanh(gate) * gelu(aux_proj(aux)) fed into policy + value heads

Old checkpoints load cleanly with strict=False:
  - New layers random-init, aux_gate=0.0 → h_cond == h at start → identical to Phase 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Pre-activation residual block for 1D input."""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return F.relu(out + residual)


class DominoNet(nn.Module):
    """
    Two-headed network with belief-conditioned policy + value.

    Architecture:
      trunk → h
      h → belief_head → 21 belief logits   (pip presence per opponent)
      h → support_head → 6 support logits  (can play left/right per opponent)
      h_cond = h + tanh(aux_gate) * gelu(aux_proj(concat(sigmoid(b), sigmoid(s))))
      h_cond → policy_head → 57 action logits
      h_cond → value_head  → scalar
    """

    def __init__(self, input_dim=213, hidden_dim=256, num_actions=57, num_blocks=4):
        super().__init__()

        # ── Trunk ────────────────────────────────────────────────────
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])

        # ── Policy head ──────────────────────────────────────────────
        self.policy_fc1 = nn.Linear(hidden_dim, 128)
        self.policy_bn  = nn.BatchNorm1d(128)
        self.policy_fc2 = nn.Linear(128, num_actions)

        # ── Value head ───────────────────────────────────────────────
        self.value_fc1 = nn.Linear(hidden_dim, 64)
        self.value_bn  = nn.BatchNorm1d(64)
        self.value_fc2 = nn.Linear(64, 1)

        # ── Belief head (21 outputs: partner/LHO/RHO × 7 pips) ──────
        self.belief_fc1 = nn.Linear(hidden_dim, 128)
        self.belief_bn  = nn.BatchNorm1d(128)
        self.belief_fc2 = nn.Linear(128, 21)

        # ── End-support head (6 outputs: partner/LHO/RHO × left/right) ──
        self.support_fc1 = nn.Linear(hidden_dim, 64)
        self.support_bn  = nn.BatchNorm1d(64)
        self.support_fc2 = nn.Linear(64, 6)

        # ── Auxiliary conditioning path ───────────────────────────────
        # Projects 27-dim aux probs back to hidden_dim and adds to trunk repr
        self.aux_proj = nn.Linear(27, hidden_dim)

        # Learned gate: init=0.0 so tanh(0)=0 → h_cond==h at start
        # Gradually learns to open the gate as aux heads improve
        self.aux_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, valid_actions_mask=None, return_belief=False):
        """
        Args:
            x:                  (B, input_dim) state tensor
            valid_actions_mask: (B, num_actions) binary mask of legal actions
            return_belief:      if True, also return (belief_logits, support_logits)

        Returns:
            policy:  (B, num_actions) masked softmax probabilities
            value:   (B, 1) scalar in [-1, 1]
        or (if return_belief=True):
            policy, value, belief_logits (B,21), support_logits (B,6)
        """
        # BatchNorm requires B>1 in train mode — switch to eval for single samples
        if x.shape[0] == 1:
            self.eval()

        # ── Shared trunk ─────────────────────────────────────────────
        h = F.relu(self.input_bn(self.input_fc(x)))
        for block in self.res_blocks:
            h = block(h)

        # ── Auxiliary heads ──────────────────────────────────────────
        b = F.relu(self.belief_bn(self.belief_fc1(h)))
        belief_logits = self.belief_fc2(b)                      # (B, 21)

        s = F.relu(self.support_bn(self.support_fc1(h)))
        support_logits = self.support_fc2(s)                    # (B, 6)

        # ── Auxiliary conditioning ────────────────────────────────────
        # Combine aux predictions and project back into trunk space
        aux = torch.cat(
            [torch.sigmoid(belief_logits), torch.sigmoid(support_logits)],
            dim=-1
        )                                                        # (B, 27)
        h_cond = h + torch.tanh(self.aux_gate) * F.gelu(self.aux_proj(aux))

        # ── Policy head (conditioned) ────────────────────────────────
        p = F.relu(self.policy_bn(self.policy_fc1(h_cond)))
        p = self.policy_fc2(p)                                   # raw logits
        if valid_actions_mask is not None:
            p = p + (1.0 - valid_actions_mask) * (-1e9)
        policy = F.softmax(p, dim=-1)

        # ── Value head (conditioned) ─────────────────────────────────
        v = F.relu(self.value_bn(self.value_fc1(h_cond)))
        v = torch.tanh(self.value_fc2(v))

        if return_belief:
            return policy, v, belief_logits, support_logits

        return policy, v

    def predict(self, state_np, mask_np, device=None):
        """Convenience: numpy in, numpy out. For single-state inference."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            state_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t  = torch.tensor(mask_np,  dtype=torch.float32).unsqueeze(0).to(device)
            policy, value = self(state_t, valid_actions_mask=mask_t)
        return policy.squeeze(0).cpu().numpy(), value.item()
