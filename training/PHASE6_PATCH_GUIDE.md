# PHASE6_PATCH_GUIDE.md
# Minimal Phase 6 Patch — File-by-File Implementation
# Source: Bernie analysis 2026-03-08
#
# Design goals:
#   - Minimal changes
#   - Backward-compatible with old checkpoints (strict=False loading)
#   - Belief head is training-only — not consumed by search/inference
#   - Browser/runtime stays completely unchanged
#   - Zero JS loader changes required

---

## Files touched
1. training/orchestrator.py
2. training/domino_trainer.py
3. training/domino_net.py
4. training/export_model.py

---

## New data shape
Each replay sample becomes a 5-tuple:
  (state_np, mask_np, pi_np, v_target, belief_target_np)

belief_target_np: float32 array of shape (21,)
  Index  0..6  = partner has pip 0..6
  Index  7..13 = LHO     has pip 0..6
  Index 14..20 = RHO     has pip 0..6

---

# 1. training/orchestrator.py

## A. Import TILES from domino_env
Change:
  from domino_env import DominoEnv
To:
  from domino_env import DominoEnv, TILES

## B. Add build_belief_target() near top (after imports, before self_play_worker)

```python
def build_belief_target(hidden_hands_by_player, me):
    """
    21-dim binary belief target.
    hidden_hands_by_player: dict[int, list[tile_idx]]
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

## C. Record belief_target in self_play_worker game_history append
Change:
```python
game_history.append({
    'state': state_np,
    'mask': valid_mask,
    'pi': target_pi,
    'team': current_team,
})
```
To:
```python
belief_target = build_belief_target(env.hands, obs['player'])
game_history.append({
    'state': state_np,
    'mask': valid_mask,
    'pi': target_pi,
    'team': current_team,
    'belief_target': belief_target,
})
```

## D. Include belief_target in replay buffer append
Change:
```python
worker_data.append((
    step['state'], step['mask'], step['pi'], v_target
))
```
To:
```python
worker_data.append((
    step['state'],
    step['mask'],
    step['pi'],
    v_target,
    step['belief_target'],
))
```

## E. Backward-compatible checkpoint loading (ALL load sites)
Change:
```python
model.load_state_dict(model_state_dict)
```
To:
```python
missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
if missing:
    print(f"[load_state_dict] Missing keys: {missing}")
if unexpected:
    print(f"[load_state_dict] Unexpected keys: {unexpected}")
```
Apply to: worker model load, arena challenger/champion load, load_checkpoint().

## F. Update loss printing (4 return values from train_epoch)
Change:
```python
loss, v_loss, p_loss = self.trainer.train_epoch(loader)
```
To:
```python
loss, v_loss, p_loss, b_loss = self.trainer.train_epoch(loader)
```
And print:
```python
print(f"  Epoch {epoch+1}/5 | Loss: {loss:.4f} (V: {v_loss:.4f}, P: {p_loss:.4f}, B: {b_loss:.4f})")
```

## G. Add CLI flags
```python
parser.add_argument('--belief-head', action='store_true',
                    help='Enable 21-output auxiliary pip-belief head')
parser.add_argument('--belief-weight', type=float, default=0.2,
                    help='Auxiliary belief loss weight')
```
Use when creating Trainer:
```python
self.trainer = Trainer(self.model, lr=1e-3, belief_weight=args.belief_weight)
```

---

# 2. training/domino_trainer.py

## A. ReplayDataset — support 4- and 5-tuples
```python
class ReplayDataset(Dataset):
    """
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

## B. Trainer — add belief_weight param
```python
class Trainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, belief_weight=0.2):
        self.model = model
        self.device = next(model.parameters()).device
        self.belief_weight = belief_weight
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
```

## C. train_epoch — handle 5-field batches, return 4 values
```python
def train_epoch(self, dataloader):
    """Returns: total_loss, value_loss, policy_loss, belief_loss"""
    self.model.train()
    total_loss = total_v = total_p = total_b = 0.0
    n = 0

    for batch in dataloader:
        if len(batch) == 5:
            states, masks, target_pis, target_vs, target_beliefs = batch
            target_beliefs = target_beliefs.to(self.device)
            use_belief = True
        else:
            states, masks, target_pis, target_vs = batch
            target_beliefs = None
            use_belief = False

        states      = states.to(self.device)
        masks       = masks.to(self.device)
        target_pis  = target_pis.to(self.device)
        target_vs   = target_vs.to(self.device)

        if use_belief:
            pred_policy, pred_value, pred_belief_logits = self.model(
                states, valid_actions_mask=masks, return_belief=True
            )
        else:
            pred_policy, pred_value = self.model(
                states, valid_actions_mask=masks
            )
            pred_belief_logits = None

        v_loss = F.mse_loss(pred_value, target_vs)
        p_loss = -torch.mean(torch.sum(target_pis * torch.log(pred_policy + 1e-8), dim=1))

        if use_belief:
            b_loss = F.binary_cross_entropy_with_logits(pred_belief_logits, target_beliefs)
        else:
            b_loss = torch.tensor(0.0, device=self.device)

        loss = v_loss + p_loss + self.belief_weight * b_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        total_loss += loss.item(); total_v += v_loss.item()
        total_p += p_loss.item(); total_b += b_loss.item()
        n += 1

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    return total_loss/n, total_v/n, total_p/n, total_b/n
```

---

# 3. training/domino_net.py

## A. Add belief head layers to __init__
After the value head:
```python
# Belief head (training-only auxiliary head)
self.belief_fc1 = nn.Linear(hidden_dim, 128)
self.belief_bn  = nn.BatchNorm1d(128)
self.belief_fc2 = nn.Linear(128, 21)
```

## B. Update forward() — optional return_belief param
```python
def forward(self, x, valid_actions_mask=None, return_belief=False):
    """
    Returns (policy, value) by default.
    Returns (policy, value, belief_logits) if return_belief=True.
    """
    if x.shape[0] == 1:
        self.eval()

    # Shared trunk
    h = F.relu(self.input_bn(self.input_fc(x)))
    for block in self.res_blocks:
        h = block(h)

    # Policy head
    p = F.relu(self.policy_bn(self.policy_fc1(h)))
    p = self.policy_fc2(p)
    if valid_actions_mask is not None:
        p = p + (1 - valid_actions_mask) * (-1e9)
    policy = F.softmax(p, dim=-1)

    # Value head
    v = F.relu(self.value_bn(self.value_fc1(h)))
    v = torch.tanh(self.value_fc2(v))

    if return_belief:
        b = F.relu(self.belief_bn(self.belief_fc1(h)))
        b = self.belief_fc2(b)  # raw logits, no activation
        return policy, v, b

    return policy, v
```

## C. predict() stays unchanged (backward-compatible)
```python
def predict(self, state_np, mask_np, device=None):
    if device is None:
        device = next(self.parameters()).device
    self.eval()
    with torch.no_grad():
        state_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
        mask_t  = torch.tensor(mask_np,  dtype=torch.float32).unsqueeze(0).to(device)
        policy, value = self(state_t, valid_actions_mask=mask_t, return_belief=False)
    return policy.squeeze(0).cpu().numpy(), value.item()
```

---

# 4. training/export_model.py

## A. Backward-compatible checkpoint loading
Change all:
```python
model.load_state_dict(ckpt['model_state_dict'])
```
To:
```python
missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
if missing:    print(f"[export] Missing keys: {missing}")
if unexpected: print(f"[export] Unexpected keys: {unexpected}")
```

## B. Add filtered_inference_state_dict() helper
```python
def filtered_inference_state_dict(model):
    """Strip training-only belief head — browser format stays unchanged."""
    sd = model.state_dict()
    return {
        k: v for k, v in sd.items()
        if not k.startswith("belief_fc1")
        and not k.startswith("belief_bn")
        and not k.startswith("belief_fc2")
    }
```

## C. Use filtered dict in JSON + binary export
Replace:
```python
for name, param in model.state_dict().items():
```
With:
```python
for name, param in filtered_inference_state_dict(model).items():
```
Apply to both JSON and binary export paths.

---

# 5. Validation checklist (run in order)

## A. Old checkpoint compatibility
```
python training/orchestrator.py --resume checkpoints/domino_gen_0050.pt --generations 1
```
Expected: loads with "Missing keys: [belief_fc1.*, ...]" warning, but runs fine.

## B. New loss output format
Training should print:
```
Epoch 1/5 | Loss: 0.XXXX (V: 0.XXXX, P: 0.XXXX, B: 0.XXXX)
```

## C. Export unchanged
```
python training/export_model.py checkpoints/domino_gen_XXXX.pt --format binary
```
Binary should NOT contain belief_fc* weights. Browser still loads normally.

---

# 6. First Phase 6 probe command
```bash
python training/orchestrator.py \
  --resume checkpoints/domino_gen_0084.pt \
  --generations 5 \
  --workers 10 \
  --games-per-worker 10 \
  --value-target me \
  --policy-target visits \
  --belief-head \
  --belief-weight 0.2
```

Measure after:
1. Partnership suite score (target: +0.05 vs baseline)
2. Search scaling at 50/100/200/400 sims
3. Belief loss trend (should decrease)
4. Arena vs current champion

---

# 7. What NOT to change in Phase 6

- Do NOT feed belief outputs into search/MCTS
- Do NOT change JS inference format
- Do NOT add browser-side belief API
- Do NOT switch to full 28×4 tile-location marginals
- Do NOT wire belief into predict()

Phase 6 MVP = trunk reshaping only. Measure first. Then Phase 6.5 wires it deeper.
