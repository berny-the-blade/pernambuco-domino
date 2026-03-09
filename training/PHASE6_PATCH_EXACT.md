# PHASE6_PATCH_EXACT.md
# Exact minimal Phase 6 patch — mapped to actual repo files
# Source: Bernie analysis 2026-03-08
# Based on actual file contents from repo

---

# Patch 1 — training/domino_net.py

## 1A) Add belief head in __init__ (after value head)

After:
```python
        # Value head
        self.value_fc1 = nn.Linear(hidden_dim, 64)
        self.value_bn = nn.BatchNorm1d(64)
        self.value_fc2 = nn.Linear(64, 1)
```

Add:
```python
        # Belief head (training-only auxiliary head)
        # 21 outputs = partner(7) + LHO(7) + RHO(7)
        self.belief_fc1 = nn.Linear(hidden_dim, 128)
        self.belief_bn  = nn.BatchNorm1d(128)
        self.belief_fc2 = nn.Linear(128, 21)
```

## 1B) Replace forward() entirely

```python
    def forward(self, x, valid_actions_mask=None, return_belief=False):
        """
        Args:
            x: (batch, 185) state tensor
            valid_actions_mask: (batch, 57) binary mask of legal actions
            return_belief: if True, also return belief logits (21)

        Returns:
            policy: (batch, 57) probability distribution (masked + softmax)
            value:  (batch, 1)  scalar in [-1, 1]
        or:
            policy, value, belief_logits
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
            p = p + (1 - valid_actions_mask) * (-1e9)
        policy = F.softmax(p, dim=-1)

        # Value head
        v = F.relu(self.value_bn(self.value_fc1(h)))
        v = torch.tanh(self.value_fc2(v))

        if return_belief:
            b = F.relu(self.belief_bn(self.belief_fc1(h)))
            b = self.belief_fc2(b)  # raw logits
            return policy, v, b

        return policy, v
```

## 1C) predict() — no changes needed
Calls `self(state_t, valid_actions_mask=mask_t)` with default `return_belief=False`.
Runtime stays unchanged.

---

# Patch 2 — training/domino_trainer.py

## 2A) Module docstring
Change:
```
Composite loss: MSE(value) + CrossEntropy(policy).
Uses a replay buffer stored as list of (state, mask, pi, value_target) tuples.
```
To:
```
Composite loss: policy + value + optional auxiliary belief loss.
Supports replay tuples:
    (state, mask, pi, value_target)
or
    (state, mask, pi, value_target, belief_target)
```

## 2B) Replace ReplayDataset
```python
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
```

## 2C) Trainer.__init__ — add belief_weight
```python
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, belief_weight=0.2):
        self.model = model
        self.device = next(model.parameters()).device
        self.belief_weight = belief_weight
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
```

## 2D) Replace train_epoch() — returns 4 values
```python
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
```

---

# Patch 3 — training/orchestrator.py

## 3A) Fix import
```python
from domino_env import DominoEnv, TILES
```

## 3B) Add build_belief_target() after imports, before self_play_worker
```python
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
```

## 3C) self_play_worker — add use_belief_head param
```python
def self_play_worker(worker_id, model_state_dict, num_games, use_mcts,
                     mcts_sims, result_queue, use_belief_head=False):
```

## 3D) Worker model load — strict=False
```python
    incompat = model.load_state_dict(model_state_dict, strict=False)
    if incompat.missing_keys:
        print(f"[worker {worker_id}] Missing keys: {incompat.missing_keys}")
    if incompat.unexpected_keys:
        print(f"[worker {worker_id}] Unexpected keys: {incompat.unexpected_keys}")
```

## 3E) Record belief_target in game_history
```python
    record = {
        'state': state_np,
        'mask': valid_mask,
        'pi': target_pi,
        'team': current_team,
    }
    if use_belief_head:
        record['belief_target'] = build_belief_target(env.hands, obs['player'])
    game_history.append(record)
```

## 3F) Include belief_target in replay samples
```python
    if use_belief_head:
        worker_data.append((
            step['state'], step['mask'], step['pi'], v_target,
            step['belief_target'],
        ))
    else:
        worker_data.append((
            step['state'], step['mask'], step['pi'], v_target,
        ))
```

## 3G) Orchestrator.__init__ — add belief params
```python
    def __init__(self, num_workers=4, buffer_size=200000, use_mcts=False,
                 mcts_sims=50, use_belief_head=False, belief_weight=0.2):
        ...
        print(f"Workers: {num_workers}, Buffer: {buffer_size}, "
              f"MCTS: {use_mcts} ({mcts_sims} sims), "
              f"BeliefHead: {use_belief_head}, BeliefWeight: {belief_weight}")
        ...
        self.use_belief_head = use_belief_head
        self.belief_weight   = belief_weight
        self.trainer = Trainer(self.model, lr=1e-3, belief_weight=belief_weight)
```

## 3H) Pass use_belief_head into worker processes
```python
    p = ctx.Process(
        target=self_play_worker,
        args=(w_id, shared_weights, games_per_worker,
              self.use_mcts, self.mcts_sims, result_queue,
              self.use_belief_head)
    )
```

## 3I) Update training printout (4 return values)
```python
    loss, v_loss, p_loss, b_loss = self.trainer.train_epoch(dataloader)
    print(f"  Epoch {epoch+1}/5 | "
          f"Loss: {loss:.4f} "
          f"(V: {v_loss:.4f}, P: {p_loss:.4f}, B: {b_loss:.4f})")
```

## 3J) Champion restore — strict=False + pass belief_weight
```python
    incompat = self.model.load_state_dict(
        {k: v.to(self.device) for k, v in self.champion_weights.items()},
        strict=False
    )
    if incompat.missing_keys:
        print(f"[revert] Missing keys: {incompat.missing_keys}")
    if incompat.unexpected_keys:
        print(f"[revert] Unexpected keys: {incompat.unexpected_keys}")
    self.trainer = Trainer(self.model, lr=1e-3, belief_weight=self.belief_weight)
```

## 3K) load_checkpoint — strict=False
```python
    incompat = self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if incompat.missing_keys:
        print(f"[load_checkpoint] Missing keys: {incompat.missing_keys}")
    if incompat.unexpected_keys:
        print(f"[load_checkpoint] Unexpected keys: {incompat.unexpected_keys}")
```

## 3L) Add CLI flags in main()
```python
    parser.add_argument('--belief-head', action='store_true',
                        help='Enable 21-output auxiliary pip-belief head')
    parser.add_argument('--belief-weight', type=float, default=0.2,
                        help='Auxiliary belief loss weight')
```

And update Orchestrator creation:
```python
    orch = Orchestrator(
        num_workers=args.workers,
        buffer_size=args.buffer_size,
        use_mcts=args.mcts,
        mcts_sims=args.mcts_sims,
        use_belief_head=args.belief_head,
        belief_weight=args.belief_weight,
    )
```

---

# Patch 4 — training/export_model.py

## 4A) Add filtered_inference_state_dict() after imports
```python
def filtered_inference_state_dict(model):
    """Strip training-only belief head weights — browser format stays unchanged."""
    sd = model.state_dict()
    return {
        k: v for k, v in sd.items()
        if not k.startswith("belief_fc1")
        and not k.startswith("belief_bn")
        and not k.startswith("belief_fc2")
    }
```

## 4B) All checkpoint loads — strict=False
Replace every `model.load_state_dict(ckpt['model_state_dict'])` with:
```python
    incompat = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if incompat.missing_keys:    print(f"[export] Missing keys: {incompat.missing_keys}")
    if incompat.unexpected_keys: print(f"[export] Unexpected keys: {incompat.unexpected_keys}")
```
Apply in: export_onnx, export_raw_weights, export_binary_weights.

## 4C) All state_dict() iteration — use filtered version
Replace:
```python
    for name, param in model.state_dict().items():
```
With:
```python
    for name, param in filtered_inference_state_dict(model).items():
```
Apply in: export_raw_weights, export_binary_weights.

---

# Validation sequence

```bash
# 1. Old checkpoint still loads (missing belief keys expected)
python training/orchestrator.py --resume checkpoints/domino_gen_0050.pt --generations 1

# 2. Phase 6 probe run
python training/orchestrator.py \
  --resume checkpoints/domino_gen_0084.pt \
  --generations 5 \
  --workers 10 \
  --games-per-worker 10 \
  --value-target me \
  --policy-target visits \
  --belief-head \
  --belief-weight 0.2

# Expected training output:
#   Epoch 1/5 | Loss: 0.XXXX (V: 0.XXXX, P: 0.XXXX, B: 0.XXXX)

# 3. Export still browser-safe
python training/export_model.py checkpoints/domino_gen_XXXX.pt --format binary
```

---

# What NOT to change
- Do NOT feed belief outputs into search
- Do NOT change JS inference format  
- Do NOT add browser-side belief API
- Do NOT switch to 28x4 tile-location marginals

Phase 6 = trunk reshaping only. Measure. Then Phase 6.5.
