"""
Two-headed ResNet for Pernambuco Domino — Phase 6.5 / support-summary update

Input:  213-dim state tensor
Output: Policy (57-dim probability distribution over actions)
        Value  (scalar in [-1, 1], Team 0's expected match equity)

Phase 6.5 changes:
  - belief_head: 21-output pip-presence auxiliary (partner/LHO/RHO × 7 pips)
  - support_head: 6-output end-playability auxiliary (partner/LHO/RHO × left/right)
  - aux_proj: projects concat(belief_probs, support_summary) back to hidden_dim
  - aux_gate: learned scalar init=0.0 — starts as identity (old behavior), learns to use aux
  - h_cond = h + tanh(gate) * gelu(aux_proj(aux)) fed into policy + value heads

Support-summary update:
  - Raw support probs (6) are extended with 4 engineered relative features:
      left_team_adv   = partner_left  - max(lho_left,  rho_left)
      right_team_adv  = partner_right - max(lho_right, rho_right)
      left_opp_press  = 1 - max(lho_left,  rho_left)
      right_opp_press = 1 - max(lho_right, rho_right)
  - support_summary: 10 dims (6 raw + 4 derived)
  - aux total: belief 21 + support_summary 10 = 31 dims  (was 27)

Old checkpoints load cleanly with strict=False:
  - aux_proj reinits (shape change 27→31) but aux_gate is preserved
  - A 3-5 gen warm-up probe is recommended after loading
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
      h → support_head → 6 support logits   (can play left/right per opponent)
      support_summary  = [support_probs(6), team_adv(2), opp_pressure(2)] = 10 dims
      aux              = concat(belief_probs(21), support_summary(10)) = 31 dims
      h_cond = h + tanh(aux_gate) * relu(aux_proj(aux))
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
        # Projects 31-dim aux (belief 21 + support_summary 10) to hidden_dim
        # support_summary = raw support probs (6) + 4 derived relative features
        self.aux_proj = nn.Linear(31, hidden_dim)

        # Learned gate: init=0.0 so tanh(0)=0 → h_cond==h at start
        # Gradually learns to open the gate as aux heads improve
        self.aux_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, valid_actions_mask=None, return_aux=False, aux_detach=True):
        """
        Args:
            x:                  (B, input_dim) state tensor
            valid_actions_mask: (B, num_actions) binary mask of legal actions
            return_aux:         if True, also return belief_logits and support_logits
            aux_detach:         if True (default), stop-gradient through aux features
                                before conditioning policy/value heads.
                                Safe for first probe: aux heads learn from their own
                                supervised loss; policy/value can USE aux predictions
                                without distorting them through their gradients.

        Returns:
            policy:  (B, num_actions) masked softmax probabilities
            value:   (B, 1) scalar in [-1, 1]
        or (if return_aux=True):
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
        belief_probs  = torch.sigmoid(belief_logits)            # (B, 21)
        support_probs = torch.sigmoid(support_logits)           # (B, 6)

        # Support summary: raw probs + relative end-advantage / pressure features
        # support_probs column order: [partner_left, partner_right,
        #                              lho_left, lho_right, rho_left, rho_right]
        p_left   = support_probs[:, 0:1]   # partner left
        p_right  = support_probs[:, 1:2]   # partner right
        lho_l    = support_probs[:, 2:3]   # LHO left
        lho_r    = support_probs[:, 3:4]   # LHO right
        rho_l    = support_probs[:, 4:5]   # RHO left
        rho_r    = support_probs[:, 5:6]   # RHO right

        opp_left_max  = torch.maximum(lho_l, rho_l)  # strongest opponent at left end
        opp_right_max = torch.maximum(lho_r, rho_r)  # strongest opponent at right end

        left_team_adv   = p_left  - opp_left_max          # +: we own left, -: they do
        right_team_adv  = p_right - opp_right_max         # same for right
        left_opp_press  = 1.0 - opp_left_max              # high: left end is locked for opponents
        right_opp_press = 1.0 - opp_right_max             # same for right

        support_summary = torch.cat([
            support_probs,    # (B, 6)  raw probabilities
            left_team_adv,    # (B, 1)
            right_team_adv,   # (B, 1)
            left_opp_press,   # (B, 1)
            right_opp_press,  # (B, 1)
        ], dim=-1)            # (B, 10)

        aux = torch.cat([belief_probs, support_summary], dim=-1)  # (B, 31)

        # Stop-gradient: aux probs inform policy/value but don't receive
        # gradients from policy/value loss — auxiliary heads train cleanly
        if aux_detach:
            aux = aux.detach()

        aux_feat = F.relu(self.aux_proj(aux))
        h_cond = h + torch.tanh(self.aux_gate) * aux_feat

        # ── Policy head (conditioned) ────────────────────────────────
        p = F.relu(self.policy_bn(self.policy_fc1(h_cond)))
        p = self.policy_fc2(p)                                   # raw logits
        if valid_actions_mask is not None:
            p = p + (1.0 - valid_actions_mask) * (-1e9)
        policy = F.softmax(p, dim=-1)

        # ── Value head (conditioned) ─────────────────────────────────
        v = F.relu(self.value_bn(self.value_fc1(h_cond)))
        value = torch.tanh(self.value_fc2(v))

        if return_aux:
            return policy, value, belief_logits, support_logits

        return policy, value

    def predict(self, state_np, mask_np, device=None):
        """Convenience: numpy in, numpy out. For single-state inference."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            state_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t  = torch.tensor(mask_np,  dtype=torch.float32).unsqueeze(0).to(device)
            out = self(state_t, valid_actions_mask=mask_t)
            policy, value = out[0], out[1]  # ignore aux heads (belief, support)
        return policy.squeeze(0).cpu().numpy(), value.item()
