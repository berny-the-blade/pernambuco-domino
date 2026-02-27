"""
Two-headed ResNet for Pernambuco Domino.

Input:  185-dim state tensor
Output: Policy (57-dim probability distribution over actions)
        Value  (scalar in [-1, 1], Team 0's expected match equity)

Architecture: 4 residual blocks (256 hidden) + separate policy/value heads.
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
    """Two-headed network: policy + value."""

    def __init__(self, input_dim=185, hidden_dim=256, num_actions=57, num_blocks=4):
        super().__init__()
        # Input projection
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # Residual tower
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])

        # Policy head
        self.policy_fc1 = nn.Linear(hidden_dim, 128)
        self.policy_bn = nn.BatchNorm1d(128)
        self.policy_fc2 = nn.Linear(128, num_actions)

        # Value head
        self.value_fc1 = nn.Linear(hidden_dim, 64)
        self.value_bn = nn.BatchNorm1d(64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x, valid_actions_mask=None):
        """
        Args:
            x: (batch, 185) state tensor
            valid_actions_mask: (batch, 57) binary mask of legal actions

        Returns:
            policy: (batch, 57) probability distribution (masked + softmax)
            value:  (batch, 1) scalar in [-1, 1]
        """
        # Handle single-sample BatchNorm edge case
        if x.shape[0] == 1:
            self.eval()

        # Shared trunk
        h = F.relu(self.input_bn(self.input_fc(x)))
        for block in self.res_blocks:
            h = block(h)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_fc1(h)))
        p = self.policy_fc2(p)  # raw logits

        if valid_actions_mask is not None:
            # Mask illegal actions with large negative value before softmax
            p = p + (1 - valid_actions_mask) * (-1e9)

        policy = F.softmax(p, dim=-1)

        # Value head
        v = F.relu(self.value_bn(self.value_fc1(h)))
        v = torch.tanh(self.value_fc2(v))

        return policy, v

    def predict(self, state_np, mask_np, device=None):
        """Convenience: numpy in, numpy out. For single-state inference."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            state_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).to(device)
            policy, value = self(state_t, valid_actions_mask=mask_t)
        return policy.squeeze(0).cpu().numpy(), value.item()
