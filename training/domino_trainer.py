"""
Training loop for Pernambuco Domino neural network — Phase 6.5

Composite loss: policy + value + belief (pip-presence) + support (end-playability)

Replay tuple formats accepted:
    4-elem: (state, mask, pi, value)
    5-elem: (state, mask, pi, value, belief_target)
    6-elem: (state, mask, pi, value, belief_target, support_target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


def _validate_policy_targets_np(pis, masks, atol=1e-4):
    """
    Fast offline validation for replay-buffer rows before training starts.
    Raises ValueError immediately if anything is malformed.
    """
    pis   = np.asarray(pis,   dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)

    if pis.ndim != 2 or pis.shape[1] != 57:
        raise ValueError(f"ReplayDataset: expected pis shape [N,57], got {pis.shape}")
    if masks.ndim != 2 or masks.shape[1] != 57:
        raise ValueError(f"ReplayDataset: expected masks shape [N,57], got {masks.shape}")

    if not np.isfinite(pis).all():
        raise ValueError("ReplayDataset: pi contains NaN/Inf")
    if not np.isfinite(masks).all():
        raise ValueError("ReplayDataset: mask contains NaN/Inf")

    if (pis < -1e-6).any():
        bad = np.argwhere(pis < -1e-6)[:10]
        raise ValueError(f"ReplayDataset: negative pi mass at {bad.tolist()}")

    row_sums = pis.sum(axis=1)
    bad_sum  = np.where(np.abs(row_sums - 1.0) > atol)[0]
    if len(bad_sum) > 0:
        idx = int(bad_sum[0])
        raise ValueError(
            f"ReplayDataset: pi row {idx} sums to {row_sums[idx]:.8f}, expected ~1.0"
        )

    illegal_mass = (pis * (masks <= 0.5)).sum(axis=1)
    bad_illegal  = np.where(illegal_mass > 1e-6)[0]
    if len(bad_illegal) > 0:
        idx         = int(bad_illegal[0])
        bad_actions = np.where((masks[idx] <= 0.5) & (pis[idx] > 1e-9))[0][:10]
        raise ValueError(
            f"ReplayDataset: row {idx} has illegal pi mass {illegal_mass[idx]:.8e} "
            f"on actions {bad_actions.tolist()}"
        )

    legal_counts = (masks > 0.5).sum(axis=1)
    if (legal_counts == 0).any():
        idx = int(np.where(legal_counts == 0)[0][0])
        raise ValueError(f"ReplayDataset: row {idx} has zero legal actions")


def _validate_policy_targets_torch(target_pis, masks, atol=1e-4):
    """Cheap batch-time safety check. Call only on the first batch or debug runs."""
    if target_pis.ndim != 2 or target_pis.shape[1] != 57:
        raise ValueError(f"train_epoch: target_pis shape {tuple(target_pis.shape)} != [B,57]")
    if masks.ndim != 2 or masks.shape[1] != 57:
        raise ValueError(f"train_epoch: masks shape {tuple(masks.shape)} != [B,57]")

    if not torch.isfinite(target_pis).all():
        raise ValueError("train_epoch: target_pis contains NaN/Inf")
    if not torch.isfinite(masks).all():
        raise ValueError("train_epoch: masks contains NaN/Inf")

    if (target_pis < -1e-6).any():
        raise ValueError("train_epoch: target_pis contains negative mass")

    row_sums = target_pis.sum(dim=1)
    if torch.max(torch.abs(row_sums - 1.0)) > atol:
        bad_idx = int(torch.argmax(torch.abs(row_sums - 1.0)).item())
        summary = _summarize_bad_policy_row(target_pis[bad_idx], masks[bad_idx])
        print(f"  [DEBUG] bad row {bad_idx}: {summary}")
        raise ValueError(
            f"train_epoch: pi row {bad_idx} sums to {row_sums[bad_idx].item():.8f}, expected ~1.0"
        )

    illegal_mass = (target_pis * (masks <= 0.5)).sum(dim=1)
    if torch.max(illegal_mass) > 1e-6:
        bad_idx = int(torch.argmax(illegal_mass).item())
        summary = _summarize_bad_policy_row(target_pis[bad_idx], masks[bad_idx])
        print(f"  [DEBUG] bad row {bad_idx}: {summary}")
        raise ValueError(
            f"train_epoch: row {bad_idx} has illegal pi mass {illegal_mass[bad_idx].item():.8e}"
        )


def _summarize_bad_policy_row(pi_row, mask_row) -> dict:
    """Return compact debug info for a bad policy row (tensor inputs)."""
    pi   = pi_row.detach().cpu().numpy()
    mask = mask_row.detach().cpu().numpy()
    top  = np.argsort(-pi)[:5]
    return {
        "top_actions":  top.tolist(),
        "top_probs":    [float(pi[t]) for t in top],
        "top_legal":    [bool(mask[t] > 0.5) for t in top],
        "sum":          float(pi.sum()),
        "illegal_mass": float((pi * (mask <= 0.5)).sum()),
    }


class ReplayDataset(Dataset):
    """PyTorch dataset wrapping replay buffer.

    Supports tuples of length 4, 5, or 6:
        (state, mask, pi, value)
        (state, mask, pi, value, belief_target)
        (state, mask, pi, value, belief_target, support_target)
    """

    def __init__(self, buffer):
        self.states = np.array([b[0] for b in buffer], dtype=np.float32)
        self.masks  = np.array([b[1] for b in buffer], dtype=np.float32)
        self.pis    = np.array([b[2] for b in buffer], dtype=np.float32)

        _validate_policy_targets_np(self.pis, self.masks)

        self.values = np.array([b[3] for b in buffer], dtype=np.float32).reshape(-1, 1)

        self.has_belief  = len(buffer[0]) >= 5
        self.has_support = len(buffer[0]) >= 6

        if self.has_belief:
            self.beliefs = np.array([b[4] for b in buffer], dtype=np.float32)
            if self.beliefs.shape[1] != 21:
                raise ValueError(
                    f"ReplayDataset: expected belief shape [N,21], got {self.beliefs.shape}"
                )
            if not np.isfinite(self.beliefs).all():
                raise ValueError("ReplayDataset: belief_target contains NaN/Inf")
            if ((self.beliefs < -1e-6) | (self.beliefs > 1.0 + 1e-6)).any():
                raise ValueError("ReplayDataset: belief_target outside [0,1]")
        else:
            self.beliefs = None

        if self.has_support:
            self.supports = np.array([b[5] for b in buffer], dtype=np.float32)
            if self.supports.shape[1] != 6:
                raise ValueError(
                    f"ReplayDataset: expected support shape [N,6], got {self.supports.shape}"
                )
            if not np.isfinite(self.supports).all():
                raise ValueError("ReplayDataset: support_target contains NaN/Inf")
            if ((self.supports < -1e-6) | (self.supports > 1.0 + 1e-6)).any():
                raise ValueError("ReplayDataset: support_target outside [0,1]")
        else:
            self.supports = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        base = (
            torch.tensor(self.states[idx]),
            torch.tensor(self.masks[idx]),
            torch.tensor(self.pis[idx]),
            torch.tensor(self.values[idx]),
        )
        if self.has_support:
            return base + (
                torch.tensor(self.beliefs[idx]),
                torch.tensor(self.supports[idx]),
            )
        if self.has_belief:
            return base + (torch.tensor(self.beliefs[idx]),)
        return base


class Trainer:
    """AlphaZero-style trainer with composite loss.

    Loss = policy_loss + value_loss
           + belief_weight  * belief_loss    (pip-presence BCE)
           + support_weight * support_loss   (end-playability BCE)
    """

    def __init__(self, model, lr=1e-3, weight_decay=1e-4,
                 belief_weight=0.1, support_weight=0.1):
        self.model          = model
        self.device         = next(model.parameters()).device
        self.belief_weight  = belief_weight
        self.support_weight = support_weight
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_epoch(self, dataloader):
        """Train for one epoch.

        Returns:
            total_loss, value_loss, policy_loss, belief_loss, support_loss
        """
        self.model.train()
        total_loss = total_v = total_p = total_b = total_s = 0.0
        n_batches = 0

        for batch in dataloader:
            use_belief  = len(batch) >= 5
            use_support = len(batch) >= 6

            if use_support:
                states, masks, target_pis, target_vs, target_beliefs, target_supports = batch
                target_beliefs  = target_beliefs.to(self.device)
                target_supports = target_supports.to(self.device)
            elif use_belief:
                states, masks, target_pis, target_vs, target_beliefs = batch
                target_beliefs  = target_beliefs.to(self.device)
                target_supports = None
            else:
                states, masks, target_pis, target_vs = batch
                target_beliefs  = None
                target_supports = None

            states     = states.to(self.device)
            masks      = masks.to(self.device)
            target_pis = target_pis.to(self.device)
            target_vs  = target_vs.to(self.device)

            if n_batches == 0:
                _validate_policy_targets_torch(target_pis, masks)

            if use_belief or use_support:
                pred_policy, pred_value, pred_belief_logits, pred_support_logits = self.model(
                    states, valid_actions_mask=masks, return_belief=True
                )
            else:
                pred_policy, pred_value = self.model(states, valid_actions_mask=masks)
                pred_belief_logits  = None
                pred_support_logits = None

            v_loss   = F.mse_loss(pred_value, target_vs)
            log_pred = torch.log(pred_policy + 1e-8)
            p_loss   = -torch.mean(torch.sum(target_pis * log_pred, dim=1))

            b_loss = (
                F.binary_cross_entropy_with_logits(pred_belief_logits, target_beliefs)
                if use_belief else torch.tensor(0.0, device=self.device)
            )
            s_loss = (
                F.binary_cross_entropy_with_logits(pred_support_logits, target_supports)
                if use_support else torch.tensor(0.0, device=self.device)
            )

            loss = (v_loss + p_loss
                    + self.belief_weight  * b_loss
                    + self.support_weight * s_loss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_v    += v_loss.item()
            total_p    += p_loss.item()
            total_b    += b_loss.item()
            total_s    += s_loss.item()
            n_batches  += 1

        if n_batches == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        return (total_loss / n_batches, total_v / n_batches,
                total_p / n_batches,    total_b / n_batches,
                total_s / n_batches)
