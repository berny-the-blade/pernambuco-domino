"""
Training loop for Pernambuco Domino neural network.
Composite loss: MSE(value) + CrossEntropy(policy).
Uses a replay buffer stored as list of (state, mask, pi, value_target) tuples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ReplayDataset(Dataset):
    """PyTorch dataset wrapping a replay buffer of (state, mask, pi, v_target) tuples."""

    def __init__(self, buffer):
        """
        Args:
            buffer: list of (state_np, mask_np, pi_np, v_target_float) tuples
        """
        self.states = np.array([b[0] for b in buffer], dtype=np.float32)
        self.masks = np.array([b[1] for b in buffer], dtype=np.float32)
        self.pis = np.array([b[2] for b in buffer], dtype=np.float32)
        self.values = np.array([b[3] for b in buffer], dtype=np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (torch.tensor(self.states[idx]),
                torch.tensor(self.masks[idx]),
                torch.tensor(self.pis[idx]),
                torch.tensor(self.values[idx]))


class Trainer:
    """AlphaZero-style trainer with composite loss."""

    def __init__(self, model, lr=1e-3, weight_decay=1e-4):
        self.model = model
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_epoch(self, dataloader):
        """Train for one epoch. Returns (total_loss, value_loss, policy_loss)."""
        self.model.train()
        total_loss = 0.0
        total_v_loss = 0.0
        total_p_loss = 0.0
        n_batches = 0

        for states, masks, target_pis, target_vs in dataloader:
            states = states.to(self.device)
            masks = masks.to(self.device)
            target_pis = target_pis.to(self.device)
            target_vs = target_vs.to(self.device)

            # Forward pass
            pred_policy, pred_value = self.model(states, valid_actions_mask=masks)

            # Value loss: MSE
            v_loss = F.mse_loss(pred_value, target_vs)

            # Policy loss: cross-entropy between predicted and target distributions
            # Use KL divergence (equivalent to cross-entropy when target is fixed)
            # Add small epsilon to avoid log(0)
            log_pred = torch.log(pred_policy + 1e-8)
            p_loss = -torch.mean(torch.sum(target_pis * log_pred, dim=1))

            # Composite loss
            loss = v_loss + p_loss

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_v_loss += v_loss.item()
            total_p_loss += p_loss.item()
            n_batches += 1

        if n_batches == 0:
            return 0.0, 0.0, 0.0

        return (total_loss / n_batches,
                total_v_loss / n_batches,
                total_p_loss / n_batches)
