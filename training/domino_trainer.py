"""
Training loop for Pernambuco Domino neural network.
Composite loss: policy + value + optional auxiliary belief loss.
Supports replay tuples:
    (state, mask, pi, value_target)
or
    (state, mask, pi, value_target, belief_target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ReplayDataset(Dataset):
    """PyTorch dataset wrapping replay buffer.

    Supports tuples of:
        (state, mask, pi, value)
    or
        (state, mask, pi, value, belief_target)
    """

    def __init__(self, buffer):
        self.states = np.array([b[0] for b in buffer], dtype=np.float32)
        self.masks  = np.array([b[1] for b in buffer], dtype=np.float32)
        self.pis    = np.array([b[2] for b in buffer], dtype=np.float32)
        self.values = np.array([b[3] for b in buffer], dtype=np.float32).reshape(-1, 1)

        self.has_belief = len(buffer[0]) >= 5
        if self.has_belief:
            self.beliefs = np.array([b[4] for b in buffer], dtype=np.float32)
        else:
            self.beliefs = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        if self.has_belief:
            return (
                torch.tensor(self.states[idx]),
                torch.tensor(self.masks[idx]),
                torch.tensor(self.pis[idx]),
                torch.tensor(self.values[idx]),
                torch.tensor(self.beliefs[idx]),
            )
        return (
            torch.tensor(self.states[idx]),
            torch.tensor(self.masks[idx]),
            torch.tensor(self.pis[idx]),
            torch.tensor(self.values[idx]),
        )


class Trainer:
    """AlphaZero-style trainer with composite loss."""

    def __init__(self, model, lr=1e-3, weight_decay=1e-4, belief_weight=0.2):
        self.model = model
        self.device = next(model.parameters()).device
        self.belief_weight = belief_weight
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_epoch(self, dataloader):
        """Train for one epoch.

        Returns:
            total_loss, value_loss, policy_loss, belief_loss
        """
        self.model.train()
        total_loss = total_v = total_p = total_b = 0.0
        n_batches = 0

        for batch in dataloader:
            if len(batch) == 5:
                states, masks, target_pis, target_vs, target_beliefs = batch
                target_beliefs = target_beliefs.to(self.device)
                use_belief = True
            else:
                states, masks, target_pis, target_vs = batch
                target_beliefs = None
                use_belief = False

            states     = states.to(self.device)
            masks      = masks.to(self.device)
            target_pis = target_pis.to(self.device)
            target_vs  = target_vs.to(self.device)

            if use_belief:
                pred_policy, pred_value, pred_belief_logits = self.model(
                    states, valid_actions_mask=masks, return_belief=True
                )
            else:
                pred_policy, pred_value = self.model(states, valid_actions_mask=masks)
                pred_belief_logits = None

            v_loss = F.mse_loss(pred_value, target_vs)
            log_pred = torch.log(pred_policy + 1e-8)
            p_loss = -torch.mean(torch.sum(target_pis * log_pred, dim=1))

            if use_belief:
                b_loss = F.binary_cross_entropy_with_logits(
                    pred_belief_logits, target_beliefs
                )
            else:
                b_loss = torch.tensor(0.0, device=self.device)

            loss = v_loss + p_loss + self.belief_weight * b_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item(); total_v += v_loss.item()
            total_p += p_loss.item();  total_b += b_loss.item()
            n_batches += 1

        if n_batches == 0:
            return 0.0, 0.0, 0.0, 0.0

        return total_loss/n_batches, total_v/n_batches, total_p/n_batches, total_b/n_batches
