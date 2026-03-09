# PHASE6_CLI_PATCH.md
# CLI control patches: --value-target and --policy-target
# Source: Bernie analysis 2026-03-08
# Apply AFTER PHASE6_PATCH_EXACT.md

---

# 1) Add CLI flags in main() argparse

```python
parser.add_argument(
    '--value-target',
    choices=['me', 'points'],
    default='me',
    help='Value target: match equity delta (me) or normalized points'
)

parser.add_argument(
    '--policy-target',
    choices=['heuristic', 'visits'],
    default='heuristic',
    help='Policy target source: heuristic softmax or MCTS visit counts'
)
```

---

# 2) Update Orchestrator.__init__

Change signature:
```python
def __init__(self, num_workers=4, buffer_size=200000, use_mcts=False,
             mcts_sims=50, use_belief_head=False, belief_weight=0.2,
             value_target='me', policy_target='heuristic'):
```

Store and print:
```python
        self.value_target  = value_target
        self.policy_target = policy_target

        print(
            f"Workers: {num_workers}, Buffer: {buffer_size}, "
            f"MCTS: {use_mcts} ({mcts_sims} sims), "
            f"BeliefHead: {use_belief_head}, BeliefWeight: {belief_weight}, "
            f"ValueTarget: {value_target}, PolicyTarget: {policy_target}"
        )
```

---

# 3) Pass flags into worker processes

Change worker args from:
```python
args=(w_id, shared_weights, games_per_worker,
      self.use_mcts, self.mcts_sims, result_queue,
      self.use_belief_head)
```
To:
```python
args=(w_id, shared_weights, games_per_worker,
      self.use_mcts, self.mcts_sims, result_queue,
      self.use_belief_head, self.value_target, self.policy_target)
```

---

# 4) Update self_play_worker signature

```python
def self_play_worker(worker_id, model_state_dict, num_games, use_mcts,
                     mcts_sims, result_queue, use_belief_head=False,
                     value_target='me', policy_target='heuristic'):
```

---

# 5) Guard: visits requires MCTS

At top of self_play_worker after setup:
```python
    if policy_target == 'visits' and not use_mcts:
        raise ValueError("policy_target='visits' requires use_mcts=True")
```

---

# 6) Policy target branch at move time

Replace the target_pi construction block:
```python
        # -------------------------------------------------------
        # Build policy target
        # -------------------------------------------------------
        if policy_target == 'visits':
            # Normalized 57-dim visit distribution from MCTS root
            target_pi = search_result["pi"].astype(np.float32)
        elif policy_target == 'heuristic':
            # Existing heuristic softmax target
            target_pi = heuristic_pi.astype(np.float32)
        else:
            raise ValueError(f"Unknown policy_target: {policy_target}")
```

Note: for `policy_target='visits'`, search_result["pi"] must be a normalized
57-dim visit distribution from rootVisitPolicy(). If not yet implemented,
this is where it plugs in.

---

# 7) Value target branch at game end (backfill loop)

Replace hardcoded v_target logic:
```python
        # -------------------------------------------------------
        # Build value target for each recorded state
        # -------------------------------------------------------
        if value_target == 'me':
            # Use existing ME / delta_me logic
            v_target = delta_me(
                winner_team=winner_team,
                points=points,
                team=step['team'],
                my_score=step.get('my_score_before', 0),
                opp_score=step.get('opp_score_before', 0),
                multiplier=step.get('multiplier_before', 1),
            )
        elif value_target == 'points':
            if winner_team < 0:
                v_target = 0.0
            else:
                raw = points / 4.0
                v_target = raw if winner_team == step['team'] else -raw
        else:
            raise ValueError(f"Unknown value_target: {value_target}")
```

NOTE: If existing delta_me() uses global env state at game end rather than
per-step scores, keep existing code and just branch on value_target.
Only add my_score_before/opp_score_before/multiplier_before to the record
if delta_me() actually needs them.

## Move record (add score fields if needed)
```python
        record = {
            'state': state_np,
            'mask':  valid_mask,
            'pi':    target_pi,
            'team':  current_team,
            'my_score_before':  env.match_score[current_team],
            'opp_score_before': env.match_score[1 - current_team],
            'multiplier_before': env.score_multiplier,
        }
```
Only add these if delta_me() needs per-step scores.

---

# 8) Update Orchestrator creation in main()

```python
    orch = Orchestrator(
        num_workers=args.workers,
        buffer_size=args.buffer_size,
        use_mcts=args.mcts,
        mcts_sims=args.mcts_sims,
        use_belief_head=args.belief_head,
        belief_weight=args.belief_weight,
        value_target=args.value_target,
        policy_target=args.policy_target,
    )
```

---

# 9) Experiment commands

## Baseline (no MCTS, heuristic policy)
```bash
python training/orchestrator.py \
  --resume checkpoints/domino_gen_0084.pt \
  --generations 5 --workers 10 --games-per-worker 10 \
  --value-target me --policy-target heuristic
```

## ExIt / visit-target probe
```bash
python training/orchestrator.py \
  --resume checkpoints/domino_gen_0084.pt \
  --generations 5 --workers 10 --games-per-worker 10 \
  --mcts --mcts-sims 50 \
  --value-target me --policy-target visits
```

## Phase 6 full probe
```bash
python training/orchestrator.py \
  --resume checkpoints/domino_gen_0084.pt \
  --generations 5 --workers 10 --games-per-worker 10 \
  --mcts --mcts-sims 50 \
  --value-target me --policy-target visits \
  --belief-head --belief-weight 0.2
```
