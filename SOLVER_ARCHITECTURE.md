# Domino Pernambucano — Solver-Level AI Architecture

## System Overview

The Pernambuco Domino AI operates on two levels:

1. **Browser Engine** (`simulator.html`, ~12,000 LOC) — Real-time heuristic + search AI with bitmask endgame solver (≤16 tiles), ISMCTS, neural net inference, Bayesian beliefs, partner modeling, and signaling inference. Runs in vanilla JS, zero dependencies. **+43 Elo** over heuristic-only baseline (400-game benchmark).

2. **Neural Training Pipeline** (`training/`, ~1,500 LOC Python) — AlphaZero-style self-play with IS-MCTS, two-headed ResNet, and arena gatekeeper. Trains on GPU, generates data on parallel CPU workers. Produces neural weights that can be exported back to the browser engine.

---

## Implementation Status

### Browser Engine (simulator.html) — IMPLEMENTED

| Component | Status | Commit |
|-----------|--------|--------|
| Bitmask Endgame Solver (minimax + alpha-beta, ≤16 tiles) | Done | `afb779f` |
| Move ordering (`_egMoveScore`) | Done | `afb779f` |
| Zobrist hashing + transposition tables | Done | `16794d5` |
| Enhanced ISMCTS (600 iters/300ms, 10-feature heuristic) | Done | `592994e` |
| Always-on Match Equity (removed useME gate) | Done | `683840a` |
| Continuous Bayesian belief updates | Done | `683840a` |
| Partner modeling (P2 style tracking) | Done | `683840a` |
| Signaling inference (opening suits, sacrifice, avoidance) | Done | `683840a` |
| Probabilistic dead number detection | Done | `16794d5` |
| Gibbs sampling for deal generation | Done | `16794d5` |
| Parameterized AI weights (AI_WEIGHTS object) | Done | `16794d5` |
| CMA-ES weight tuner (cma-es-tuner.html) | Done | `16794d5` |
| In-page CMA-ES optimizer (optimizeWeights) | Done | `d520704` |
| Headless self-play API (window.headlessGame) | Done | `16794d5` |
| Upgraded fastAI rollout (dead numbers, phase pips, 3-tile) | Done | `d520704` |
| MC budget increase (800 sims, 600 ISMCTS iters) | Done | `d520704` |
| Neural net browser inference engine | Done | `592994e` |
| Model export script (export_model.py) | Done | `592994e` |

### Neural Training Pipeline (training/) — IMPLEMENTED

| Component | Status | Commit |
|-----------|--------|--------|
| Python game engine (domino_env.py) | Done | `b31d136` |
| Two-headed ResNet (domino_net.py) | Done | `b31d136` |
| 185-dim state encoder (domino_encoder.py) | Done | `b31d136` |
| Composite loss trainer (domino_trainer.py) | Done | `b31d136` |
| IS-MCTS with PUCT (domino_mcts.py) | Done | `b31d136` |
| Self-play orchestrator (orchestrator.py) | Done | `b31d136` |
| Played-tile belief zeroing | Done | `69c319a` |
| Rejection sampling in determinization | Done | `69c319a` |
| Action symmetry (halved search tree) | Done | `69c319a` |
| Arena gatekeeper (55% threshold) | Done | `69c319a` |

### Not Yet Implemented

| Component | Priority | Notes |
|-----------|----------|-------|
| Strong neural model for browser | High | Gen 84 too weak (97/100 arena rejections); need longer training or MCTS self-play |
| WebWorker for heavy computation | Low | Current async chunking sufficient |

---

## PART I — BROWSER ENGINE

### Section 1: Game Rules

**Pernambuco Domino** (Domino Pernambucano):
- 28 tiles (double-six set), 4 players in 2 teams (P0+P2 vs P1+P3)
- 6 tiles dealt per player, 4 removed to dorme (boneyard, face down, unknown)
- Highest double opens; play continues clockwise
- Must play if able; pass only when no legal move exists
- Match to 6 points

**Scoring:**

| Type | Points | Condition |
|------|--------|-----------|
| Batida | 1 | Normal win (last tile played) |
| Carroça | 2 | Last tile is a double |
| Lá-e-lô | 3 | Last tile could play on both ends |
| Cruzada | 4 | Last tile is a double AND could play both ends |
| Blocked | 1 | All 4 players pass consecutively; individual lowest pip count wins |
| Dobrada | 1 | Blocked game with cross-team tie; opener's team wins |

### Section 2: State Representation

```
s = {
  boardEnds: [leftEnd, rightEnd],     // 0-6 each, -1 if empty
  boardLength: int,                    // tiles on board (0-24)
  played: Set<tileIdx>,               // indices of played tiles
  myHand: int[],                       // tile indices in my hand
  handSizes: [int, int, int, int],    // tiles per player
  cantHave: [Set, Set, Set, Set],     // hard elimination per player per number
  playsBy: [int[], int[], ...],       // tiles played by each player
  teamScores: [int, int],            // match scores (0-5 each)
  multiplier: int,                    // current dobrada multiplier (1, 2, 4, ...)
  currentPlayer: int                  // 0-3
}
```

**Action Space (57 dimensions):**
- Actions 0-27: Play tile[action] on LEFT side of board
- Actions 28-55: Play tile[action-28] on RIGHT side of board
- Action 56: Pass

**Action Symmetry Optimization:** When `leftEnd == rightEnd`, right-side plays (28-55) are masked out since they produce identical board states. This halves the search tree for symmetric positions.

### Section 3: Perfect Endgame Solver

When total tiles remaining across all hands ≤ **16**, the engine switches from heuristic/MCTS to **exact minimax with alpha-beta pruning** using a bitmask representation.

**Bitmask Architecture (`_endgameMinimaxBit`):**
- Each player's hand is a 28-bit integer (one bit per tile)
- Legal move generation: `handBits & numberMask[end]` — O(1) bitwise AND
- Move iteration: extract bits via `x & (-x)` (lowest set bit)
- No array allocation during search — pure integer arithmetic

**Algorithm:**
1. `enumerateValidDeals()` — Pool unknown tiles, distribute among 3 hidden players + dorme respecting `cantHave` constraints. Weight by belief marginals.
2. `_endgameMinimaxBit()` — Partnership alpha-beta: Team 0 maximizes, Team 1 minimizes. Terminal evaluation via `_rolloutToMEReward` (match equity).
3. `endgameSolve()` — Aggregate: `expectedME[move] = Σ weight_i × minimax_result_i`. Returns exact ME for each legal move.

**Move Ordering (`_egMoveScore`):**
Priority scoring for alpha-beta cutoff optimization:
1. **Instant win** (+10000): last tile in hand
2. **TT best move** (+5000): best move from transposition table lookup
3. **Double** (+100): doubles played early
4. **High pip count** (+pips): prefer heavy tiles
5. **Both-ends coverage** (+50): tile covers both board ends

Provides 3-10x speedup from earlier cutoffs in alpha-beta search.

**Performance:**
- 16 tiles threshold — handles ~90% of endgame transitions
- Zobrist hashing with transposition table (EXACT/LOWER/UPPER bounds)
- Per-deal TT shared across all root moves
- Budget: 500ms hard cap
- Verified: `endgameVerify(5000)` — ALL PASS against brute-force reference

**Zobrist Hashing:**
```
hash = XOR of:
  _ZOBRIST.tp[tileIdx][player]  for each tile in each hand
  _ZOBRIST.le[leftEnd]          board left end
  _ZOBRIST.re[rightEnd]         board right end
  _ZOBRIST.np[nextPlayer]       whose turn
  _ZOBRIST.pc[passCount]        consecutive passes
  _ZOBRIST.bl[boardLength]      board length (distinguishes positions)
```

**Transposition Table Bounds:**
- Flag 0 (EXACT): `bestVal` fell within (alpha, beta) — exact value
- Flag 1 (LOWER): `bestVal >= beta` — cutoff, value is a lower bound
- Flag 2 (UPPER): `bestVal <= origAlpha` — all moves failed, value is upper bound

### Section 4: ISMCTS (Information Set Monte Carlo Tree Search)

For positions outside endgame threshold, the engine uses ISMCTS with:

- **Progressive Bias**: Heuristic scores from `_ismctsHeuristic` (10-feature function) injected as `H(move) / (visits + 1)`
- **Budget**: 600 iterations / 300ms (Expert difficulty)
- **UCB1**: C=1.41, progressive widening `children < ceil(sqrt(visits+1))`
- **Max nodes**: 5000
- **Determinization**: Hidden hands randomized from beliefs each simulation
- **Evaluation**: `_rolloutToMEReward` for match-equity-optimal play
- **Rollout policy**: `fastAI` with dead number detection, phase-dependent pips, 3-tile closing checks

### Section 5: Bayesian Belief Model

**Binary Layer (cantHave):**
```
cantHave[p] = set of numbers player p cannot hold
Updated when: player p passes on board ends [le, re] → add le, re to cantHave[p]
```

**Probabilistic Layer (BeliefModel):**
```
belief[tile][zone] = P(tile is in zone | all observations)
Zones: partner(0), LHO(1), RHO(2), dorme(3)
```

**Update Rules:**
- `update_on_pass(zone, le, re)`: Zero belief for all tiles matching le or re in that zone
- `update_on_play(zone, tile)`: Zero entire row `belief[tile, :] = 0.0` (tile is visible on table)
- `_sync_belief(obs)`: Re-apply cantHave constraints, normalize rows to sum to 1.0

**Continuous Updates:** Beliefs update every turn (not just during search), ensuring the MCTS sees current information.

### Section 6: Deal Generation

**Three-tier sampling:**

1. **Full rejection sampling** (`generateConsistentDeal`) — Truly unbiased, respects all constraints
2. **Gibbs sampling** (`gibbsSampleDeal`) — Markov chain over valid deals via local tile swaps between players. 50 iterations for mixing. Respects `cantHave` constraints on every swap.
3. **Randomized backtracking** — Fallback for tight constraint scenarios

**Endgame deal loop:** First 10 deals via full generation, remaining via Gibbs mutation from a base deal. Weighted by `dealWeight()` using belief marginals.

### Section 7: Signaling Inference

**Opening Suit Detection:**
- Track which suit each player leads with in early turns (board.length < 8)
- If partner opens suit X → increase belief they hold more of X

**Sacrifice Play Detection:**
- When a player plays a high-pip tile on a losing board position → flag as potential sacrifice
- Reduces belief they hold tiles of that suit (they're dumping)

**Avoidance Refinement:**
- `avoidanceCount` in BeliefModel tracks passes per player per number
- Weighted by game phase: early avoidance = stronger signal than late
- Decay factor: avoidance from 5+ turns ago is less informative

### Section 8: Partner Modeling

All 3 other players tracked (not just opponents):
```
OpponentStyle: {
  aggression,     // EMA of pip-weighted plays
  blockRate,      // tendency to play blocking moves
  doubleRate,     // speed of double disposal
  feedRate,       // partner support plays
  passRate        // forced pass frequency
}
```

If partner is aggressive → slightly reduce own blocking bias (trust partner to close).
Used in `generateConsistentDeal` to weight deals by partner tendency.

### Section 9: Probabilistic Dead Numbers

`isProbablyDead(n, myIdx)` — Returns true when ALL other players are confirmed void on suit `n`, meaning remaining tiles must be in dorme or caller's hand only.

**Scoring integration in smartAI:**
- If dead number and we hold tiles matching it → `+captiveEndBonus` (we monopolize that end)
- If dead number and we don't → `-probDeadPenalty` (end is frozen, likely lock)

### Section 10: Parameterized AI Weights

13 key scoring constants extracted into mutable `AI_WEIGHTS` object:

```javascript
const AI_WEIGHTS = {
  deadEndPenalty: 35,      lockFavorable: 40,       lockUnfavorable: 60,
  chicoteSelf: 25,         chicotePartner: 15,      chicoteOpponent: 25,
  chicoteDorme: 12,        lockApproachGood: 20,    lockApproachBad: 15,
  monopolyBonus: 20,       boardCountGradient: 6,   captiveEndBonus: 20,
  probDeadPenalty: 25,
};
```

Tunable via CMA-ES optimizer (`cma-es-tuner.html`):
- Loads game engine in hidden iframe
- Diagonal CMA-ES with configurable population/generations
- Evaluates via `headlessGame(seed)` with duplicate seeds for noise reduction
- Outputs optimized weight table

### Section 11: Match Equity

**Always-on ME** — Removed the `useME` gate. All evaluations now optimize match-winning probability, not raw points.

```
ME(a, b) = P(team_a wins match | score a vs b, match to 6)

Base cases:
  ME(≥6, b) = 1.0   for b < 6
  ME(a, ≥6) = 0.0   for a < 6

Recursion:
  ME(a, b) = Σ_k P(pts=k) × [0.5 × ME(a+k, b) + 0.5 × ME(a, b+k)]

Point distribution:
  k=1: 0.70  (batida + blocked)
  k=2: 0.16  (carroça)
  k=3: 0.10  (lá-e-lô)
  k=4: 0.04  (cruzada)
```

---

## PART II — NEURAL TRAINING PIPELINE

### Section 12: Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  ORCHESTRATOR                     │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Worker 0 │  │ Worker 1 │  │ Worker N │  CPU  │
│  │ (games)  │  │ (games)  │  │ (games)  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │              │              │              │
│       └──────────────┼──────────────┘              │
│                      ▼                             │
│              ┌───────────────┐                     │
│              │ Replay Buffer │  (sliding deque)    │
│              │   200K max    │                     │
│              └───────┬───────┘                     │
│                      ▼                             │
│              ┌───────────────┐                     │
│              │   TRAINER     │  GPU                │
│              │  5 epochs     │                     │
│              │  batch 256    │                     │
│              └───────┬───────┘                     │
│                      ▼                             │
│              ┌───────────────┐                     │
│              │    ARENA      │  400 games          │
│              │  ≥55% to      │  (champion vs       │
│              │  promote      │   challenger)        │
│              └───────────────┘                     │
└─────────────────────────────────────────────────┘
```

**Training Loop (per generation):**
1. **Phase 1 — Self-Play:** N workers play M games each using frozen model copy. Data collected as `(state, mask, policy, value_target)` tuples.
2. **Phase 2 — Training:** 5 epochs on replay buffer. Composite loss = MSE(value) + CrossEntropy(policy). Adam optimizer with gradient clipping (max_norm=1.0).
3. **Phase 3 — Arena:** Challenger (new weights) vs Champion (current best) in 400-game deathmatch (200 seeds × 2 side-swaps). No noise, greedy argmax. Challenger must win ≥55% of match points to replace champion. If rejected, model reverts to champion weights.

### Section 13: Neural Network (domino_net.py)

**Two-headed ResNet:**

```
Input: 185-dim state vector
  │
  ├─ Linear(185, 256) + ReLU
  │
  ├─ ResBlock(256) × 4
  │     ├─ BatchNorm → ReLU → Linear(256, 256)
  │     ├─ BatchNorm → ReLU → Linear(256, 256)
  │     └─ Skip connection (add input)
  │
  ├─ Policy Head:
  │     ├─ Linear(256, 128) + ReLU
  │     ├─ Linear(128, 57)
  │     ├─ Mask illegal moves (-1e9)
  │     └─ Softmax → π(a|s)
  │
  └─ Value Head:
        ├─ Linear(256, 64) + ReLU
        ├─ Linear(64, 1)
        └─ Tanh → v(s) ∈ [-1, 1]
```

**Parameter count:** ~540K (lightweight, fast CPU inference for self-play)

### Section 14: State Encoder (domino_encoder.py)

**185-dimensional feature vector:**

| Offset | Size | Feature | Encoding |
|--------|------|---------|----------|
| 0-27 | 28 | My hand | Binary: do I hold tile t? |
| 28-55 | 28 | Played tiles | Binary: has tile t been played? |
| 56-62 | 7 | Left end | One-hot (0-6) |
| 63-69 | 7 | Right end | One-hot (0-6) |
| 70-90 | 21 | CantHave | 3 players × 7 numbers, binary |
| 91-174 | 84 | Belief matrix | 3 players × 28 tiles, probabilities |
| 175-178 | 4 | Hand sizes | 4 players, normalized by /6 |
| 179-180 | 2 | Match scores | my_score/6, opp_score/6 |
| 181 | 1 | Multiplier | Normalized by /4 |
| 182 | 1 | Board length | Normalized by /24 |
| 183 | 1 | Game phase | tiles_played / 24 |
| 184 | 1 | My team | 0 or 1 |

**Belief Matrix (28×4):**
- Columns: partner, LHO, RHO, dorme
- Initialized to uniform (0.25 each)
- Updated incrementally via `update_on_pass()` and `update_on_play()`
- `_sync_belief()` re-applies cantHave constraints and normalizes each `encode()` call

### Section 15: IS-MCTS with PUCT (domino_mcts.py)

**Information-Set MCTS** for imperfect-information domino:

1. **Determinize:** Before each simulation, hallucinate hidden hands from Bayesian beliefs using rejection sampling (up to 100 retries, random fallback).
2. **Select:** PUCT formula: `score = Q(a) + c_puct × P(a) × √N_parent / (1 + N_child)`
   - Q-values stored from Team 0's perspective
   - Flipped for Team 1's turns: `q = -child.q_value`
3. **Expand:** Neural net evaluates leaf: `(policy, value) = model.predict(state, mask)`
4. **Backpropagate:** Normalize value to Team 0's perspective, update all nodes in search path.

**Root Exploration:**
- Dirichlet noise: α=0.3, blended 75% neural policy + 25% noise
- Temperature: 1.0 for first 8 moves (explore), 0.1 thereafter (exploit)

**Belief updates inside tree:** Passes update cantHave beliefs; plays zero out tile beliefs. This prevents the tree from hallucinating impossible moves in deep search.

### Section 16: Self-Play Data Generation (orchestrator.py)

**Worker Process:**
```python
def self_play_worker(worker_id, model_state_dict, num_games, ...):
    # 1. Load frozen model on CPU
    # 2. For each game:
    #    a. Reset env + encoder
    #    b. Play to completion using neural policy + Dirichlet noise
    #    c. Record (state, mask, policy, team) per move
    #    d. Update encoder beliefs after each play/pass
    # 3. Retroactive credit assignment:
    #    v_target = ±(points_won / 4.0) based on team vs winner
    # 4. Push all (state, mask, pi, v_target) tuples to result queue
```

**Retroactive Credit Assignment:**
- Reward scaled by point magnitude: cruzada(4pt) → ±1.0, batida(1pt) → ±0.25
- Applied to ALL moves in the game, not just the final move
- Winner's team gets positive, loser's team gets negative

**Exploration:**
- With MCTS (`--mcts`): IS-MCTS policy with temperature decay
- Without MCTS (default): Neural policy + Dirichlet noise (α=0.3, 75/25 blend)

### Section 17: Arena Gatekeeper

Prevents policy collapse (catastrophic forgetting) by requiring proof of improvement:

```
arena_game(champion_weights, challenger_weights, seed, challenger_team):
    # Greedy argmax (no noise, no temperature)
    # Each seed played twice: challenger as team 0 and team 1
    # Returns (winner_team, points)

ARENA_GAMES = 200 seeds × 2 sides = 400 games
ARENA_THRESHOLD = 55% of match points

If challenger ≥ 55%: PROMOTED → becomes new champion, checkpoint saved
If challenger < 55%: REJECTED → model reverts to champion weights
```

This guarantees monotonically increasing skill. The AI can never get worse.

### Section 18: Determinization Safety

**The Impossible Universe Problem:**
Greedy one-by-one tile assignment can create contradictions. Example: If LHO passed on 6s and the only remaining tile is [6|6], but LHO is the only player with hand space left.

**Solution — Rejection Sampling:**
```python
for attempt in range(100):
    shuffle unknown tiles
    assign each tile to a zone based on belief probabilities
    if contradiction (prob_sum == 0):
        retry from scratch
    if all zones filled correctly:
        accept deal

# Fallback: random deal ignoring beliefs (prevents CPU deadlock)
```

### Section 19: Training Configuration

**Default launch:**
```bash
cd training/
python orchestrator.py --workers 4 --generations 100 --games-per-worker 250
```

**Full options:**
```
--workers N           CPU workers for self-play (default: cpu_count - 1)
--generations N       Training generations (default: 100)
--games-per-worker N  Games per worker per generation (default: 250)
--buffer-size N       Replay buffer capacity (default: 200,000)
--mcts                Enable IS-MCTS during self-play (slower but stronger)
--mcts-sims N         MCTS simulations per move (default: 50)
--resume PATH         Resume from checkpoint file
```

**Expected progression:**
- Gen 1-10: Random play → learns rules, stops illegal passes
- Gen 10-25: Basic matching → learns suit following, double management
- Gen 25-50: Partnership synergy → learns partner support, blocking
- Gen 50-100: Match equity awareness → sacrifices, cruzada prevention, endgame precision

### Section 20: File Map

```
pernambuco-domino-repo/
├── simulator.html              # Browser engine (~12,000 LOC)
├── cma-es-tuner.html           # Standalone CMA-ES weight optimizer (iframe-based)
├── SOLVER_ARCHITECTURE.md      # This document
├── DOCUMENTATION.md            # User-facing documentation
├── training/
│   ├── __init__.py
│   ├── requirements.txt        # torch>=2.0.0, numpy>=1.24.0
│   ├── domino_env.py           # Game environment (Python port)
│   ├── domino_net.py           # Two-headed ResNet
│   ├── domino_encoder.py       # 185-dim state encoder
│   ├── domino_trainer.py       # Composite loss trainer
│   ├── domino_mcts.py          # IS-MCTS with PUCT
│   ├── orchestrator.py         # Self-play + training + arena
│   └── export_model.py         # Export .pt → .bin/.onnx/.json for browser
└── checkpoints/                # Generated during training
    └── domino_gen_NNNN.pt      # Champion checkpoints
```

---

## PART III — THEORETICAL REFERENCE

### Section 21: Value Functions

**Round Win Probability:**
```
V_win(s, team) = (1/N) Σ_{d ∈ ConsistentDeals(s)} I[Rollout(s, d).winner == team]
```

**Expected Round Points (signed):**
```
V_pts(s, team) = (1/N) Σ_d [
  if Rollout(s,d).winner == team: +points
  else: -points
]
```

**Match Equity:**
```
V_match(s, team) = Σ_{outcome} P(outcome | s) × ME(scores_after_outcome)
```

**Move Equity Loss:**
```
EL(m) = V_match(s after best_move) - V_match(s after m)
```

### Section 22: Error Classification

| ME Loss | Grade | Label |
|---------|-------|-------|
| < 0.5% | Perfect | Accurate |
| 0.5-1.5% | Good | Inaccuracy |
| 1.5-4.0% | OK | Mistake |
| 4.0-8.0% | Bad | Blunder |
| > 8.0% | Terrible | Howler |

### Section 23: Composite Training Loss

```
L = L_value + L_policy

L_value = MSE(v_predicted, v_target)
        = (1/N) Σ (tanh(f_v(s)) - v*)²

L_policy = -E[Σ_a π*(a) × log(π_θ(a|s))]
         = -(1/N) Σ Σ_a π*(a) × log(π_θ(a|s) + ε)

where:
  v* = ±(points_won / 4.0)  retroactive credit
  π* = MCTS-improved policy (or neural + Dirichlet noise)
  ε = 1e-8 for numerical stability
```

**Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
**Gradient clipping:** max_norm=1.0
**Batch size:** 256

---

### Section 24: Neural Net Browser Inference

The browser engine includes a complete neural network forward pass, enabling model-guided play without any server dependency:

**Weight Loading (`loadNeuralModel`):**
- Binary format: 4-byte header length (little-endian) + JSON header + concatenated float32 arrays
- Header contains: architecture config, per-layer metadata (name, shape, offset, length)
- Loaded into typed Float32Arrays for efficient computation

**Forward Pass (`_nnForward`):**
```
Input: 185-dim state + 57-dim action mask
  ├─ Linear(185, 256) + BatchNorm + ReLU        (input projection)
  ├─ ResBlock × 4:
  │     Linear(256,256) + BN + ReLU + Linear(256,256) + BN + skip + ReLU
  ├─ Policy Head: Linear(256,128) + BN + ReLU + Linear(128,57) + mask + softmax
  └─ Value Head: Linear(256,64) + BN + ReLU + Linear(64,1) + tanh
```

**State Encoder (`_nnEncodeState`):** Matches Python `DominoEncoder` exactly — 185 dimensions with hand, played tiles, board ends (one-hot), cantHave (21 binary), belief proxies (84 floats), hand sizes, match scores, multiplier, board length, phase, team.

**Integration with Expert AI:**
- When model loaded: 20% NN policy blend with ISMCTS/MC search scores
- `neuralEval()` returns policy-ranked moves with value estimate
- Model export via `training/export_model.py`: `python export_model.py checkpoint.pt -o model.bin`

### Section 25: In-Page CMA-ES Weight Optimizer

Console-callable optimizer for the 13 `AI_WEIGHTS` parameters:

```javascript
optimizeWeights(generations=30, popSize=16, gamesPerEval=150)
```

**Architecture:**
- Diagonal CMA-ES engine with step-size adaptation and covariance learning
- Adversarial evaluation: candidate weights vs champion weights, played from both sides
- Each candidate evaluated over `gamesPerEval × 2` games (side-swap for fairness)
- Fitness = net point advantage over champion
- Convergence: stops on σ < 0.5 or 5 stale generations
- Outputs paste-ready `AI_WEIGHTS` object

**Parameter space (13 dimensions):**
```
deadEndPenalty, lockFavorable, lockUnfavorable, chicoteSelf,
chicotePartner, chicoteOpponent, chicoteDorme, lockApproachGood,
lockApproachBad, monopolyBonus, boardCountGradient,
captiveEndBonus, probDeadPenalty
```

**Note:** CMA-ES optimizes for heuristic-vs-heuristic play. Weights optimized this way may not improve MC-Expert performance since they strengthen both the AI and its rollout evaluations symmetrically.

### Section 26: Benchmark Results

**MC-Expert vs Heuristic-Only (400 games, Feb 2026):**
- MC-Expert: 221 wins (56.1%), +43 Elo
- Key contributors: fastAI rollout improvements, ISMCTS heuristic upgrade, raised sim budgets

**AI Decision Stack (in priority order):**
1. Endgame solver (exact, ≤16 tiles) — bitmask minimax with alpha-beta
2. ISMCTS (600 iters/300ms) — tree search with progressive bias
3. Monte Carlo (800 sims) — rollout evaluation with adaptive early stopping
4. Heuristic (`smartAI`) — 35+ weighted features, always available as fallback

---

*Document for `berny-the-blade/pernambuco-domino`*
*Browser engine: single-file vanilla JS, ≤500ms per evaluation, zero dependencies*
*Training pipeline: Python + PyTorch, GPU training, parallel CPU self-play*
*+43 Elo over heuristic baseline (Feb 2026)*
