# Pernambuco Domino — Solver-Level Engine Specification v2

**Author**: Claude Opus 4.6 + Bernd
**Date**: 2026-02-18
**Source of truth**: `simulator.html` (~2900 LOC single-file HTML/JS/Tailwind)
**Scope**: Transform heuristic trainer into mathematically rigorous "best play" engine with full human coaching.

---

## A) EXECUTIVE PLAN

### A.1 Current Engine Dissection

The engine makes decisions through this pipeline:

1. **`smartAI()`** (line 203): Hand-crafted linear scorer with 12 weighted features:
   - Suit control (`myCount * 15`), blocking (`25` per blocked opp), partner support (`pAff * 8`),
     partner penalty (`-10`), pip dump (`tilePips * 2`), double clearing (`12`), isolated-double setup (`20`),
     scarcity penalty (`-8`), partner-close bonus (`15`), closing bonuses (`+200/+80/-30/+25/-15`).
   - Returns ALL scored options sorted descending. No lookahead, no probability.

2. **`Knowledge` class** (line 146): Binary elimination tracker:
   - `cantHave[p]`: Set of numbers player p cannot hold (from passes).
   - `played`: Set of tile IDs on the board.
   - `inferStrength(p)`: Counts how many times player p played each number.
   - `possibleTilesFor(p)`: Filters tiles consistent with hard constraints.
   - **Gap**: No probabilities, no soft inference, no avoidance tracking.

3. **`generateConsistentDeal()`** (line 576): Sampling via rejection + backtracking:
   - Rejection sampling: shuffle pool, cut into hands, check constraints. 200 attempts.
   - Backtracking fallback: randomized assignment with constraint checking.
   - Greedy last-resort: fills hands ignoring some constraints.
   - **Bias**: Rejection sampling is correct but slow when constraints are tight. Backtracking
     is biased (player order affects distribution). Greedy fallback violates constraints.

4. **`monteCarloEval()`** (line 766): Per-tile rollout evaluation:
   - For each playable tile × side: generate N deals, simulate to terminal via `simulateFromPosition()`.
   - Outputs: win%, E[pts], block rate, outcome distribution.
   - **Gap**: No importance weighting, no variance tracking, no confidence intervals, no caching.

5. **`simulateFromPosition()`** (line 706): Rollout policy using `smartAI()`:
   - Plays out full game using heuristic for all players.
   - Correct scoring: normal/carroca/la-e-lo/cruzada/blocked/tie.
   - **Gap**: No tree search, no belief-aware play, no match equity consideration.

6. **Match Equity Table** (line 894): 2D DP `ME[s1][s2]`:
   - Pre-computed for match to 6 using empirical point distribution.
   - **Gap**: No dobrada dimension, no per-move ME optimization, linear approximation in `roundToMatchEquity()`.

7. **Analysis Tab** (line 2480): Async per-move MC evaluation:
   - Runs `monteCarloEval()` for each P0 move, computes equity loss, grades.
   - Leak finder: categorizes errors by game phase and type.

8. **Explanation Builder** (line 1168): Template-based reasoning:
   - 6 categories: board ends, opponent blocking, partner support, doubles, heavy tiles, scarcity.
   - **Gap**: No probability-backed claims, no deduction history, no counterfactual comparisons.

### A.2 Top 15 Upgrade Items (ranked by ROI × correctness)

| # | Item | Category | Impact | Effort |
|---|------|----------|--------|--------|
| 1 | **Fix sampling bias** (replace greedy fallback, add importance weights) | Correctness | Critical | Medium |
| 2 | **Probabilistic belief model** (tile marginals, not just elimination) | Correctness | Critical | High |
| 3 | **DeductionNotebook** with evidence-backed human-readable deductions | Coaching | High | High |
| 4 | **Variance + confidence intervals** on MC results | Correctness | High | Low |
| 5 | **Seedable PRNG** for deterministic replay | Correctness | Medium | Low |
| 6 | **ME table with dobrada** dimension | Correctness | Medium | Low |
| 7 | **WebWorker** for non-blocking MC | UX | High | Medium |
| 8 | **Flexibility/leave value** metric | Strength | High | Medium |
| 9 | **Live deduction panel** ("What You Should Remember") | Coaching | High | High |
| 10 | **Per-move deduction-based error classification** | Coaching | High | Medium |
| 11 | **Counterfactual move comparison** with probability evidence | Explanation | High | Medium |
| 12 | **State hashing + LRU cache** for MC results | Performance | Medium | Medium |
| 13 | **Adaptive sampling** (stop when CI < threshold) | Performance | Medium | Low |
| 14 | **JSONL export** pipeline | Data | Medium | Low |
| 15 | **Control horizon** metric (expected turns of end control) | Strength | Medium | High |

### A.3 Phased Roadmap

**Phase 1 — CORRECTNESS (~500 LOC, 1-2 sessions)**
- Seedable PRNG (xorshift128+)
- Fix `generateConsistentDeal()`: remove greedy fallback, add importance weights
- Add variance tracking to `monteCarloEval()`
- Confidence intervals + adaptive stopping
- ME table with dobrada dimension
- Determinism tests

**Phase 2 — BELIEF + COACHING (~800 LOC, 2-3 sessions)**
- `BeliefModel` class: tile marginals P_p(tile), avoidance tracking, Bayesian updates
- `DeductionNotebook`: evidence log, certain/likely/possible/unclear labels
- Live "What You Should Remember" panel
- Per-player deduction cards
- "Think This Way" checklist on quiz turns
- Deduction-aware error classification ("You missed: West is void in 5")

**Phase 3 — STRENGTH + METRICS (~600 LOC, 2 sessions)**
- Flexibility/leave value computation
- Control horizon metric
- Lock risk detection
- Partner synergy metrics
- Information gain per move
- Motif tagging (block, feed, sacrifice, probe, etc.)

**Phase 4 — ANALYTICS + DATA (~400 LOC, 1 session)**
- WebWorker for MC offloading
- State hashing + LRU cache
- JSONL export pipeline (game logs + evaluations)
- Import/replay mode
- Self-play engine-vs-engine benchmark
- Elo tracking

---

## B) MATHEMATICAL SPEC

### B.1 STATE

```
s = {
  boardEnds: (leftEnd: 0-6, rightEnd: 0-6),
  chain: [tile_0, ..., tile_n],              // ordered tiles on board
  playedTiles: Set<tileId>,                   // 0-28 tiles played
  myHand: [tile_1, ..., tile_k],             // observer's hand
  handSizes: [h_0, h_1, h_2, h_3],          // tiles remaining per player
  passHistory: [                              // complete event log
    { player, turn, action: 'pass'|'play', ends: (a,b), tile? }
  ],
  turnIndex: 0-3,                            // whose turn
  teamScores: [s_team0, s_team1],            // match scores (0-5 each)
  targetScore: 6,                            // match target
  dobradaMultiplier: 1|2|4|8|...,            // stacking tie multiplier
  rulesetParams: {                           // parameterized rules (Rules TBD items)
    scoringType: 'pernambuco',               // normal=1, carroca=2, laelo=3, cruzada=4
    tieRule: 'dobrada',                      // tied block → next game doubled
    startRule: 'highest_double_first',       // first game of match
    subsequentStart: 'winner_chooses_free',  // winning team picks, any tile
    dobraStartRule: 'highest_double',        // dobrada reverts to highest double
    matchTarget: 6,
    tilesPerPlayer: 6,
    dormeTiles: 4                            // undealt tiles
  }
}
```

**Rules TBD** (parameterized in `rulesetParams`):
- Does la-e-lo require the closing tile to match BOTH existing ends, or that all opponents passed before the closer's turn in the final cycle? → **Current code**: `couldPlayBothEnds()` checks tile matches both board ends. This is the standard Pernambuco definition.
- In a blocked game, if BOTH teams have tied lowest individual pip counts, is it a tie (dobrada) or does first-seated lowest win? → **Current code**: true tie (dobrada). Parameterize as `tieBreaker: 'dobrada' | 'first_seat'`.

### B.2 ACTIONS

Legal moves at state s with player p holding hand H:

```
A(s, p) = {
  (tile, side) : tile ∈ H,
    side ∈ {'left', 'right'},
    canPlayOnSide(tile, boardEnd[side])
} ∪ {
  'pass' : if no tile in H can play on either end
}
```

Where `canPlayOnSide(tile, end)` = `tile.left == end || tile.right == end`.

Special case: if `leftEnd == rightEnd`, both sides are equivalent → deduplicate.

Orientation is determined by the side: if playing on left, the tile half matching leftEnd goes inward. This affects the new board end but NOT the action space (orientation is forced).

### B.3 BELIEF STATE (Imperfect Information)

Player 0 (observer) knows: their own hand, all played tiles, all pass events.

**Evidence set**:
```
E = {
  playedTiles,
  myHand,
  handSizes[4],
  passConstraints[p] = { (turn_t, ends=(a,b)) : p passed at turn t with ends a,b },
  playHistory[p] = [ tile_1, tile_2, ... ]    // tiles p has played (in order)
}
```

**Hard constraints** from evidence:
- If player p passed at ends (a,b): `∀ tile with (left==a or right==a) or (left==b or right==b): P_p(tile) = 0` at that point in time.
- Tiles in `playedTiles` or `myHand`: `P_p(tile) = 0` for all p ≠ me.
- Sum constraint: `Σ P_p(tile) over unplayed tiles = handSizes[p]` for each p.

**Tile marginals**: `P_p(tile)` = probability that player p holds tile, given E.

**Number marginals**: `P_p(has_n)` = probability player p holds at least one tile containing number n.

**Sampled deals**: `D_k ~ P(deal | E)`, k = 1..K, with optional importance weights w_k.

### B.4 OBJECTIVES (selectable)

For a candidate move m from state s:

1. **WinProb(m)**: `P(our team wins this hand | play m, then optimal play)`
   ```
   WinProb(m) = (1/K) Σ_k w_k · 𝟙[team wins in rollout from s+m with deal D_k]
   ```

2. **EV_Points(m)**: Expected signed points for our team this hand
   ```
   EV_Points(m) = (1/K) Σ_k w_k · signedPoints(rollout(s+m, D_k))
   ```
   where `signedPoints = +pts if we win, -pts if they win, 0 if tie`.

3. **MatchEquity(m)**: Expected match equity after this hand
   ```
   MatchEquity(m) = (1/K) Σ_k w_k · ME[ourScore + pts_k, oppScore + pts_k_opp, dob_k]
   ```
   Using the extended ME table with dobrada dimension.

4. **Risk(m)**:
   - Variance: `Var(m) = (1/K) Σ w_k · (pts_k - EV)²`
   - CVaR_5%: Expected value of worst 5% of outcomes
   - Tail risk: `P(lose ≥ X pts | m)`

**Point distribution from terminal states**:
```
Terminal → (winner_team, points, type)
  type=normal → pts=1
  type=carroca → pts=2
  type=laelo → pts=3
  type=cruzada → pts=4
  type=blocked → pts=1 (lowest pips wins)
  type=tie → pts=0, dobrada *= 2
```

---

## C) BELIEF / INFERENCE ENGINE

### C.1 Evidence Extraction

**From PASS** (strongest signal):
```
recordPass(player p, turn t, ends (a, b)):
  // CERTAIN: p has no tile containing a AND no tile containing b
  for each unplayed tile T:
    if T.left == a or T.right == a or T.left == b or T.right == b:
      belief.setImpossible(p, T)

  // Store event for deduction display
  deductionLog.push({
    type: 'pass', player: p, turn: t, ends: [a, b],
    implication: `${pn(p)} has no ${a}s and no ${b}s`,
    certainty: 'CERTAIN'
  })
```

**From PLAY** (moderate signal):
```
recordPlay(player p, turn t, tile T, side, ends (a, b)):
  // CERTAIN: tile T is no longer in any hand
  belief.removeTile(T)

  // SOFT INFERENCE: if p had choice of sides but chose one,
  // infer preference (they might have more of the number they exposed)
  if T could play on both sides:
    exposedNumber = newEnd created by play
    belief.softSignal(p, exposedNumber, 'preference', weight=0.3)

  // AVOIDANCE TRACKING: if p played tile that does NOT contain number n,
  // but n was available on the board, record a "missed opportunity"
  for each number n in {a, b}:
    if T.left != n and T.right != n:
      belief.recordAvoidance(p, n, turn=t)
```

**From partnership dynamics** (optional, toggleable):
```
// If player consistently plays heavy tiles → they might be dumping (losing position)
// If player plays from deep suit → they're establishing control
// These are SOFT READS with certainty label
```

### C.2 Sampling Method

#### LEVEL 1: Importance-Sampled Deal Generator (Fast, Practical)

Replace current `generateConsistentDeal()`:

```javascript
// === NEW: BeliefSampler ===
class BeliefSampler {
  constructor(rng) {
    this.rng = rng; // seedable PRNG
  }

  /**
   * Generate K belief-consistent deals with importance weights.
   * Uses constraint-respecting sequential assignment.
   */
  sampleDeals(myHand, handSizes, belief, myIdx, K = 100) {
    const myIds = new Set(myHand.map(t => t.id));
    const pool = ALL_TILES.filter(t => !belief.played.has(t.id) && !myIds.has(t.id));
    const players = [0, 1, 2, 3].filter(p => p !== myIdx);
    const deals = [];
    const weights = [];

    for (let k = 0; k < K; k++) {
      const result = this._sampleOneDeal(pool, players, handSizes, belief, myHand, myIdx);
      if (result) {
        deals.push(result.hands);
        weights.push(result.weight);
      }
    }

    // Normalize weights
    const wSum = weights.reduce((s, w) => s + w, 0);
    if (wSum > 0) {
      for (let i = 0; i < weights.length; i++) weights[i] /= wSum;
    }

    return { deals, weights };
  }

  _sampleOneDeal(pool, players, handSizes, belief, myHand, myIdx) {
    const hands = [[], [], [], []];
    hands[myIdx] = [...myHand];
    const assigned = new Set();
    let logWeight = 0;

    // Shuffle player assignment order for variance reduction
    const order = this._shuffle([...players]);

    for (const p of order) {
      const need = handSizes[p];
      if (need <= 0) continue;

      // Get eligible tiles for this player (respecting hard constraints)
      const eligible = pool.filter(t =>
        !assigned.has(t.id) && belief.canHold(p, t)
      );

      if (eligible.length < need) return null; // infeasible

      // Sample 'need' tiles from eligible using belief-weighted selection
      const selected = this._weightedSampleWithout(eligible, need, t => {
        return belief.getTileMarginal(p, t.id) || (1.0 / eligible.length);
      });

      if (!selected) return null;

      for (const t of selected.tiles) {
        hands[p].push(t);
        assigned.add(t.id);
      }
      logWeight += selected.logWeight;
    }

    return { hands, weight: Math.exp(logWeight) };
  }

  _weightedSampleWithout(items, k, weightFn) {
    // Weighted sampling without replacement
    // Returns { tiles, logWeight } where logWeight = log(proposal/target) for IS correction
    const remaining = [...items];
    const selected = [];
    let logW = 0;

    for (let i = 0; i < k; i++) {
      if (remaining.length === 0) return null;
      const weights = remaining.map(weightFn);
      const wSum = weights.reduce((s, w) => s + w, 0);
      if (wSum <= 0) return null;

      // Sample proportional to weights
      let r = this.rng.random() * wSum;
      let idx = 0;
      for (; idx < weights.length - 1; idx++) {
        r -= weights[idx];
        if (r <= 0) break;
      }

      selected.push(remaining[idx]);
      logW += Math.log(weights[idx] / wSum); // log proposal probability
      remaining.splice(idx, 1);
    }

    // Target: uniform over valid assignments → log(1/C(n,k))
    // IS weight = target / proposal
    const n = items.length;
    const logUniform = -logCombination(n, k);
    logW = logUniform - logW; // IS correction

    return { tiles: selected, logWeight: logW };
  }

  _shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}
```

#### LEVEL 2: MCMC Swap Sampler (Strong, More Correct)

```javascript
class MCMCSampler {
  /**
   * Gibbs-style swap sampler: start from a valid deal,
   * repeatedly swap tiles between players while maintaining constraints.
   * Converges to uniform over valid deals.
   */
  sample(initialDeal, belief, myIdx, burnIn = 50, thinning = 5, K = 100) {
    let current = initialDeal.map(h => [...h]);
    const players = [0, 1, 2, 3].filter(p => p !== myIdx);
    const deals = [];
    let accepted = 0, total = 0;

    for (let iter = 0; iter < burnIn + K * thinning; iter++) {
      // Pick two random non-observer players
      const p1 = players[Math.floor(this.rng.random() * players.length)];
      const p2 = players[Math.floor(this.rng.random() * players.length)];
      if (p1 === p2 || current[p1].length === 0 || current[p2].length === 0) continue;

      // Pick random tile from each
      const i1 = Math.floor(this.rng.random() * current[p1].length);
      const i2 = Math.floor(this.rng.random() * current[p2].length);
      const t1 = current[p1][i1];
      const t2 = current[p2][i2];

      // Check if swap is valid (both tiles satisfy other player's constraints)
      total++;
      if (belief.canHold(p1, t2) && belief.canHold(p2, t1)) {
        current[p1][i1] = t2;
        current[p2][i2] = t1;
        accepted++;
      }

      // Collect sample after burn-in, with thinning
      if (iter >= burnIn && (iter - burnIn) % thinning === 0) {
        deals.push(current.map(h => [...h]));
      }
    }

    return {
      deals,
      weights: new Array(deals.length).fill(1.0 / deals.length),
      diagnostics: {
        acceptanceRate: accepted / total,
        totalSwapAttempts: total,
        effectiveSampleSize: deals.length // approximate; could compute via autocorrelation
      }
    };
  }
}
```

### C.3 BeliefModel Class

```javascript
class BeliefModel {
  constructor() {
    // Hard constraints (certain)
    this.impossible = [new Set(), new Set(), new Set(), new Set()]; // P_p(tile) = 0
    this.played = new Set();

    // Tile marginals (computed from sampling or inference)
    this.marginals = [{}, {}, {}, {}]; // marginals[p][tileId] = probability

    // Avoidance evidence (soft inference)
    this.avoidanceCount = [
      [0,0,0,0,0,0,0], [0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]
    ]; // avoidanceCount[p][n] = times p could have played n but didn't

    // Evidence log
    this.events = []; // { type, player, turn, data, certainty }
  }

  canHold(p, tile) {
    if (this.played.has(tile.id)) return false;
    if (this.impossible[p].has(tile.id)) return false;
    // Check number constraints
    if (tile.left === tile.right) {
      return !this._numberImpossible(p, tile.left);
    }
    return !this._numberImpossible(p, tile.left) || !this._numberImpossible(p, tile.right);
  }

  _numberImpossible(p, n) {
    // Player p cannot hold any tile with number n
    // True if every unplayed tile containing n is in impossible[p]
    for (const t of ALL_TILES) {
      if (this.played.has(t.id)) continue;
      if (t.left !== n && t.right !== n) continue;
      if (!this.impossible[p].has(t.id)) return false;
    }
    return true;
  }

  recordPass(p, turn, leftEnd, rightEnd) {
    // Hard constraint: p has no tile containing leftEnd or rightEnd
    for (const t of ALL_TILES) {
      if (this.played.has(t.id)) continue;
      if (t.left === leftEnd || t.right === leftEnd ||
          t.left === rightEnd || t.right === rightEnd) {
        this.impossible[p].add(t.id);
      }
    }
    this.events.push({
      type: 'pass', player: p, turn,
      ends: [leftEnd, rightEnd],
      certainty: 'CERTAIN',
      implication: `Has no ${leftEnd}s and no ${rightEnd}s`
    });
  }

  recordPlay(p, turn, tile, side, leftEnd, rightEnd) {
    this.played.add(tile.id);
    for (let q = 0; q < 4; q++) this.impossible[q].add(tile.id);

    // Avoidance: if board had ends (a,b) and tile doesn't contain one of them
    if (leftEnd !== undefined) {
      for (const n of [leftEnd, rightEnd]) {
        if (tile.left !== n && tile.right !== n) {
          // p chose not to play a tile with number n (if they had other options)
          // Only count if p had playable tiles with n (we don't know for sure, so count conservatively)
          this.avoidanceCount[p][n]++;
        }
      }
    }

    this.events.push({
      type: 'play', player: p, turn, tile,
      certainty: 'CERTAIN',
      implication: `Played [${tile.left}|${tile.right}]`
    });
  }

  getTileMarginal(p, tileId) {
    return this.marginals[p][tileId] || 0;
  }

  getNumberStrength(p, n) {
    // P(player p has at least one tile with number n)
    let probNone = 1.0;
    for (const t of ALL_TILES) {
      if (this.played.has(t.id)) continue;
      if (t.left !== n && t.right !== n) continue;
      const m = this.marginals[p][t.id] || 0;
      probNone *= (1 - m);
    }
    return 1 - probNone;
  }

  getEntropy(p) {
    // Shannon entropy of player p's hand distribution
    let H = 0;
    for (const t of ALL_TILES) {
      if (this.played.has(t.id)) continue;
      const m = this.marginals[p][t.id] || 0;
      if (m > 0 && m < 1) {
        H -= m * Math.log2(m) + (1 - m) * Math.log2(1 - m);
      }
    }
    return H;
  }

  updateMarginals(deals, weights, myIdx) {
    // Recompute marginals from sampled deals
    for (let p = 0; p < 4; p++) {
      if (p === myIdx) continue;
      this.marginals[p] = {};
    }
    for (let k = 0; k < deals.length; k++) {
      const w = weights[k];
      for (let p = 0; p < 4; p++) {
        if (p === myIdx) continue;
        for (const t of deals[k][p]) {
          this.marginals[p][t.id] = (this.marginals[p][t.id] || 0) + w;
        }
      }
    }
  }

  clone() {
    const b = new BeliefModel();
    b.impossible = this.impossible.map(s => new Set(s));
    b.played = new Set(this.played);
    b.marginals = this.marginals.map(m => ({...m}));
    b.avoidanceCount = this.avoidanceCount.map(a => [...a]);
    b.events = [...this.events];
    return b;
  }
}
```

### C.4 Output Belief Metrics

| Metric | Formula | Display |
|--------|---------|---------|
| Tile marginal P_p(tile) | From sampling | 7×7 heatmap per player |
| Number strength P_p(has_n) | 1 - Π(1 - P_p(tile)) for tiles with n | Bar chart 0-6 per player |
| Player entropy H_p | -Σ p log p over marginals | Sparkline per player |
| Total entropy | Σ H_p | Single number |
| Info gain of move m | H_before - E[H_after \| play m] | Per-move metric |

---

## D) EVALUATION ENGINE

### D.1 Evaluation Pipeline

For each legal move m:
1. Generate K belief-consistent deals using `BeliefSampler` or `MCMCSampler`.
2. For each deal D_k with weight w_k:
   a. Apply move m to state s, producing s'.
   b. Roll out from s' to terminal using rollout policy π.
   c. Record outcome: (winnerTeam, points, type).
3. Compute metrics with importance weights.

```javascript
class Evaluator {
  constructor(sampler, rolloutPolicy, rng) {
    this.sampler = sampler;
    this.policy = rolloutPolicy;
    this.rng = rng;
    this.cache = new LRUCache(500);
  }

  /**
   * Quick eval: fewer samples, heuristic rollout
   * Target: <200ms
   */
  quickEval(snap, player, K = 60) {
    return this._evaluate(snap, player, K, 'heuristic');
  }

  /**
   * Deep eval: more samples, stronger rollout
   * Target: <2s (or via WebWorker)
   */
  deepEval(snap, player, K = 200) {
    return this._evaluate(snap, player, K, 'strong');
  }

  _evaluate(snap, player, K, policyLevel) {
    const hand = snap.hands[player];
    const lE = snap.leftEnd, rE = snap.rightEnd;
    const playable = hand.filter(t => canPlay(t, lE, rE, snap.board.length));
    if (playable.length === 0) return [];
    if (playable.length === 1) return [this._trivialResult(playable[0])];

    // Check cache
    const key = this._stateHash(snap, player);
    if (this.cache.has(key)) return this.cache.get(key);

    // Sample deals
    const handSizes = snap.hands.map(h => h.length);
    const { deals, weights } = this.sampler.sampleDeals(
      hand, handSizes, snap.belief, player, K
    );

    const results = [];

    for (const tile of playable) {
      const sides = this._getLegalSides(tile, lE, rE, snap.board.length);

      let bestResult = null;

      for (const side of sides) {
        const stats = this._rolloutStats(
          snap, player, tile, side, deals, weights, policyLevel
        );
        if (!bestResult || stats.expectedPoints > bestResult.expectedPoints) {
          bestResult = { tile, side, ...stats };
        }
      }

      if (bestResult) results.push(bestResult);
    }

    results.sort((a, b) => b.expectedPoints - a.expectedPoints || b.winRate - a.winRate);
    this.cache.set(key, results);
    return results;
  }

  _rolloutStats(snap, player, tile, side, deals, weights, policyLevel) {
    const myTeam = player % 2;
    let wins = 0, totalPts = 0, sumPtsSq = 0;
    let blocks = 0, ties = 0;
    const outcomes = { normal: 0, carroca: 0, laelo: 0, cruzada: 0, blocked: 0, tie: 0 };
    const ptsArray = []; // for CVaR computation

    // Compute new state after move
    const { newLE, newRE, newBLen } = this._applyMove(snap, tile, side);

    // Check instant win
    if (snap.hands[player].length === 1) {
      // This tile empties hand → instant win
      return this._instantWinResult(tile, side, snap, player);
    }

    const newHand = snap.hands[player].filter(t => t.id !== tile.id);
    const newHandSizes = snap.hands.map(h => h.length);
    newHandSizes[player]--;
    const newKnowledge = snap.knowledge.clone();
    newKnowledge.recordPlay(player, tile);

    let wSum = 0;
    for (let k = 0; k < deals.length; k++) {
      const w = weights[k];
      wSum += w;

      // Build hands for this deal
      const simHands = deals[k].map(h => [...h]);
      simHands[player] = [...newHand];

      const result = simulateFromPosition(
        simHands, null, newLE, newRE,
        (player + 1) % 4, newKnowledge, newBLen
      );

      const signedPts = result.winnerTeam === myTeam ? result.points :
                         result.winnerTeam >= 0 ? -result.points : 0;

      wins += w * (result.winnerTeam === myTeam ? 1 : 0);
      totalPts += w * signedPts;
      sumPtsSq += w * signedPts * signedPts;
      ptsArray.push({ pts: signedPts, w });

      if (result.type === 'blocked' || result.type === 'tie') blocks += w;
      if (result.type === 'tie') ties += w;
      if (outcomes[result.type] !== undefined) outcomes[result.type] += w;
    }

    const ep = wSum > 0 ? totalPts / wSum : 0;
    const wr = wSum > 0 ? wins / wSum : 0.5;
    const variance = wSum > 0 ? (sumPtsSq / wSum) - ep * ep : 0;
    const stdDev = Math.sqrt(Math.max(0, variance));

    // CVaR 5%: expected value of worst 5% of outcomes
    ptsArray.sort((a, b) => a.pts - b.pts);
    let cvarSum = 0, cvarW = 0;
    const cvarThreshold = 0.05 * wSum;
    for (const { pts, w } of ptsArray) {
      if (cvarW >= cvarThreshold) break;
      const take = Math.min(w, cvarThreshold - cvarW);
      cvarSum += take * pts;
      cvarW += take;
    }
    const cvar5 = cvarW > 0 ? cvarSum / cvarW : ep;

    // Confidence interval (normal approximation)
    const n_eff = deals.length; // effective sample size (approximate)
    const ci95 = 1.96 * stdDev / Math.sqrt(Math.max(1, n_eff));

    return {
      winRate: wr,
      expectedPoints: ep,
      variance,
      stdDev,
      cvar5,
      ci95,
      blockRate: wSum > 0 ? blocks / wSum : 0,
      tieRate: wSum > 0 ? ties / wSum : 0,
      outcomes,
      sims: deals.length
    };
  }

  _stateHash(snap, player) {
    // Fast hash of game state for caching
    const parts = [
      snap.leftEnd, snap.rightEnd,
      snap.hands[player].map(t => t.id).sort().join(','),
      snap.hands.map(h => h.length).join(','),
      [...snap.knowledge.played].sort().join(','),
      snap.knowledge.cantHave.map(s => [...s].sort().join(',')).join('|')
    ];
    return parts.join('::');
  }

  _getLegalSides(tile, lE, rE, bLen) {
    if (bLen === 0) return [null];
    const sides = [];
    if (tile.left === lE || tile.right === lE) sides.push('left');
    if ((tile.left === rE || tile.right === rE) && lE !== rE) sides.push('right');
    if (sides.length === 0 && (tile.left === rE || tile.right === rE)) sides.push('right');
    return sides;
  }
}
```

### D.2 Adaptive Stopping

```javascript
// In the evaluation loop, check CI after every batch of 20 samples:
if (k > 0 && k % 20 === 0) {
  const currentCI = 1.96 * currentStdDev / Math.sqrt(k);
  if (currentCI < 0.05) break; // CI tight enough, stop early
}
```

### D.3 Rollout Policy Levels

| Level | Method | Speed |
|-------|--------|-------|
| `heuristic` | Current `smartAI()` — fast, no lookahead | ~0.1ms/game |
| `strong` | `smartAI()` + 1-ply MC tiebreaker for close decisions | ~1ms/game |
| `deep` | 2-ply minimax with alpha-beta over top-3 moves | ~10ms/game |

---

## E) MATCH EQUITY

### E.1 Extended ME Table with Dobrada

```javascript
// ME[s1][s2][dob] = P(team1 wins match | scores s1 vs s2, dobrada multiplier dob)
// dob ∈ {1, 2, 4, 8} (cap at 8 for practical purposes)

const DOB_VALUES = [1, 2, 4, 8];
const ME3D = (() => {
  const T = MATCH_TARGET;
  // me[s1][s2][dobIdx] where dobIdx indexes into DOB_VALUES
  const me = Array.from({length: T + 5}, () =>
    Array.from({length: T + 5}, () => new Float64Array(DOB_VALUES.length))
  );

  // Base cases
  for (let d = 0; d < DOB_VALUES.length; d++) {
    for (let s2 = 0; s2 <= T + 4; s2++) {
      for (let s1 = T; s1 <= T + 4; s1++) {
        me[s1][s2][d] = s2 >= T ? 0.5 : 1.0;
      }
    }
    for (let s1 = 0; s1 < T; s1++) {
      for (let s2 = T; s2 <= T + 4; s2++) {
        me[s1][s2][d] = 0.0;
      }
    }
  }

  // Fill bottom-up
  for (let s1 = T - 1; s1 >= 0; s1--) {
    for (let s2 = T - 1; s2 >= 0; s2--) {
      for (let d = 0; d < DOB_VALUES.length; d++) {
        const dob = DOB_VALUES[d];
        let val = 0;
        for (const { pts: basePts, prob } of POINT_DIST) {
          const pts = basePts * dob;
          const s1w = Math.min(s1 + pts, T + 4);
          const s2w = Math.min(s2 + pts, T + 4);
          // After a decisive round, dobrada resets to 1 (index 0)
          val += 0.5 * prob * me[s1w][s2][0] + 0.5 * prob * me[s1][s2w][0];
        }
        // Tied/blocked rounds: no score change, dobrada doubles
        const tieProb = 0.03; // ~3% of rounds are ties
        const nextDobIdx = Math.min(d + 1, DOB_VALUES.length - 1);
        val = val * (1 - tieProb) + tieProb * me[s1][s2][nextDobIdx];

        me[s1][s2][d] = val;
      }
    }
  }
  return me;
})();

function getMatchEquity3D(s1, s2, dobMultiplier) {
  const dIdx = DOB_VALUES.indexOf(dobMultiplier);
  const d = dIdx >= 0 ? dIdx : 0;
  return ME3D[Math.min(s1, MATCH_TARGET + 4)][Math.min(s2, MATCH_TARGET + 4)][d];
}
```

### E.2 Per-Move ME Computation

```javascript
function moveMatchEquity(snap, player, tile, side, evalResult) {
  const myTeam = player % 2;
  const myScore = matchScore[myTeam];
  const oppScore = matchScore[1 - myTeam];
  const dob = scoreMultiplier;

  // Current ME before this hand
  const meBefore = getMatchEquity3D(myScore, oppScore, dob);

  // Expected ME after this hand, given rollout outcomes
  let meAfter = 0;
  const { outcomes, sims } = evalResult;
  // Weight by actual outcome distribution from MC
  // ... (integrate over outcome distribution)

  return { meBefore, meAfter, meDelta: meAfter - meBefore };
}
```

---

## F) COMPLETE METRICS LIBRARY

### F.1 Equity Metrics
```javascript
const EquityMetrics = {
  winPct: (r) => r.winRate * 100,
  evPoints: (r) => r.expectedPoints,
  matchEquity: (r, snap) => moveMatchEquity(snap, 0, r.tile, r.side, r),
  equityLoss: (best, chosen) => best.expectedPoints - chosen.expectedPoints,
  meLoss: (best, chosen, snap) => {
    const meBest = moveMatchEquity(snap, 0, best.tile, best.side, best);
    const meChosen = moveMatchEquity(snap, 0, chosen.tile, chosen.side, chosen);
    return meBest.meAfter - meChosen.meAfter;
  }
};
```

### F.2 Risk Metrics
```javascript
const RiskMetrics = {
  variance: (r) => r.variance,
  stdDev: (r) => r.stdDev,
  cvar5: (r) => r.cvar5,
  tailRisk: (r, X) => {
    // P(lose >= X points) — from outcome distribution
    // Requires raw pts array stored in result
  },
  lockProbability: (r) => r.blockRate + r.tieRate
};
```

### F.3 Tempo / Initiative Metrics
```javascript
function computeTempoMetrics(snap, player, tile, side) {
  // Expected turns to empty for each team
  const myTeam = player % 2;
  const myTurns = snap.hands.filter((h, i) => i % 2 === myTeam)
    .reduce((s, h) => s + h.length, 0);
  const oppTurns = snap.hands.filter((h, i) => i % 2 !== myTeam)
    .reduce((s, h) => s + h.length, 0);

  // Prob next player can play (from belief)
  const nextPlayer = (player + 1) % 4;
  const { newLE, newRE } = applyMove(snap, tile, side);
  const nextCanPlay = snap.belief.getNumberStrength(nextPlayer, newLE) +
                       snap.belief.getNumberStrength(nextPlayer, newRE);
  // Crude: if nextPlayer is opponent, lower is better
  const nextIsOpp = nextPlayer % 2 !== myTeam;

  return {
    teamTurnsLeft: myTurns,
    oppTurnsLeft: oppTurns,
    nextPlayerCanPlay: Math.min(1, nextCanPlay),
    controlRetention: nextIsOpp ? (1 - nextCanPlay) : nextCanPlay
  };
}
```

### F.4 Control & Blocking Metrics
```javascript
function computeControlMetrics(snap, player, tile, side, belief) {
  const { newLE, newRE } = applyMove(snap, tile, side);

  // End control: P(we still control these numbers after one cycle)
  // Approximate: count our remaining tiles matching new ends
  const hand = snap.hands[player].filter(t => t.id !== tile.id);
  const myEndTiles = hand.filter(t =>
    t.left === newLE || t.right === newLE || t.left === newRE || t.right === newRE
  ).length;

  // Suit dominance: for each end number, P(we own majority)
  const suitDom = {};
  for (const n of [newLE, newRE]) {
    const remaining = ALL_TILES.filter(t =>
      !belief.played.has(t.id) && (t.left === n || t.right === n)
    ).length;
    const weHave = hand.filter(t => t.left === n || t.right === n).length;
    suitDom[n] = remaining > 0 ? weHave / remaining : 0;
  }

  // Expected forced passes (sum over opponents)
  let expectedForcedPasses = 0;
  for (const opp of [1, 3]) {
    const canPlayLE = belief.getNumberStrength(opp, newLE);
    const canPlayRE = belief.getNumberStrength(opp, newRE);
    const canPlay = 1 - (1 - canPlayLE) * (1 - canPlayRE);
    expectedForcedPasses += (1 - canPlay);
  }

  return {
    myEndTiles,
    suitDominance: suitDom,
    expectedForcedPasses,
    squeezeValue: expectedForcedPasses > 0.5 ? 'high' : expectedForcedPasses > 0.2 ? 'medium' : 'low'
  };
}
```

### F.5 Flexibility / Leave Value
```javascript
function computeFlexibility(hand, tile) {
  // Hand after playing this tile
  const leave = hand.filter(t => t.id !== tile.id);
  if (leave.length === 0) return { flexibility: 1.0, leaveEntropy: 0 };

  // Number distribution in remaining hand
  const numCount = new Array(7).fill(0);
  for (const t of leave) {
    numCount[t.left]++;
    if (t.left !== t.right) numCount[t.right]++;
  }

  // Flexibility: expected playable tiles across all possible future ends
  // Average over all 7×7 end combinations (weighted by plausibility)
  let totalPlayable = 0, combos = 0;
  for (let a = 0; a <= 6; a++) {
    for (let b = a; b <= 6; b++) {
      const playable = leave.filter(t =>
        t.left === a || t.right === a || t.left === b || t.right === b
      ).length;
      totalPlayable += playable;
      combos++;
    }
  }
  const flexibility = totalPlayable / combos / Math.max(1, leave.length);

  // Leave entropy: entropy of number distribution
  const total = numCount.reduce((s, c) => s + c, 0);
  let entropy = 0;
  for (const c of numCount) {
    if (c > 0 && total > 0) {
      const p = c / total;
      entropy -= p * Math.log2(p);
    }
  }

  return { flexibility, leaveEntropy: entropy, numDistribution: numCount };
}
```

### F.6 Information Metrics
```javascript
function computeInfoGain(snap, player, tile, side, belief) {
  // Entropy before move
  const H_before = [1, 2, 3].reduce((s, p) => s + belief.getEntropy(p), 0);

  // After our move, the next player must respond.
  // If they pass → big information gain (eliminates numbers)
  // If they play → moderate info gain (reveals one tile)
  const { newLE, newRE } = applyMove(snap, tile, side);

  // Expected entropy after, weighted by P(next player passes/plays)
  const nextP = (player + 1) % 4;
  const passProb = 1 - Math.min(1,
    belief.getNumberStrength(nextP, newLE) +
    belief.getNumberStrength(nextP, newRE)
  );

  // If pass: eliminates 2 numbers → high info gain
  const H_after_pass = H_before * 0.7; // rough: 30% entropy reduction from pass
  // If play: reveals 1 tile → moderate info gain
  const H_after_play = H_before * 0.9; // rough: 10% reduction from play

  const H_after = passProb * H_after_pass + (1 - passProb) * H_after_play;
  const infoGain = H_before - H_after;

  return {
    infoGain,
    passProb_nextPlayer: passProb,
    diagnosticValue: infoGain > 0.5 ? 'high' : infoGain > 0.2 ? 'medium' : 'low'
  };
}
```

### F.7 Partner Synergy Metrics
```javascript
function computePartnerSynergy(snap, player, tile, side, belief) {
  const partner = (player + 2) % 4;
  const { newLE, newRE } = applyMove(snap, tile, side);

  // Partner playability next turn
  const partnerCanPlayLE = belief.getNumberStrength(partner, newLE);
  const partnerCanPlayRE = belief.getNumberStrength(partner, newRE);
  const partnerPlayProb = 1 - (1 - partnerCanPlayLE) * (1 - partnerCanPlayRE);

  // Partner rescue value: if partner was locked, does this unlock them?
  const partnerWasLocked = snap.knowledge.cantHave[partner].has(snap.leftEnd) &&
                            snap.knowledge.cantHave[partner].has(snap.rightEnd);
  const partnerNowUnlocked = partnerPlayProb > 0.3;
  const rescueValue = partnerWasLocked && partnerNowUnlocked ? 1.0 : 0.0;

  return {
    partnerPlayProb,
    partnerRescueValue: rescueValue,
    feedsPartnerSuit: partnerCanPlayLE > 0.5 || partnerCanPlayRE > 0.5
  };
}
```

### F.8 Error / Training Metrics
```javascript
const ErrorClassifier = {
  thresholds: {
    perfect: 0.02,    // < 0.02 E[pts] loss
    inaccuracy: 0.15, // < 0.15
    mistake: 0.40,    // < 0.40
    blunder: Infinity  // >= 0.40
  },

  classify(equityLoss) {
    if (equityLoss < this.thresholds.perfect) return 'perfect';
    if (equityLoss < this.thresholds.inaccuracy) return 'inaccuracy';
    if (equityLoss < this.thresholds.mistake) return 'mistake';
    return 'blunder';
  },

  // Detailed error type from deduction analysis
  classifyWithReason(userMove, bestMove, snap, belief, deductions) {
    const el = bestMove.expectedPoints - (userMove.expectedPoints || 0);
    const grade = this.classify(el);
    if (grade === 'perfect') return { grade, reasons: [] };

    const reasons = [];

    // Check if user ignored a known void
    for (const opp of [1, 3]) {
      const deadNums = [...snap.knowledge.cantHave[opp]];
      const bestEnds = applyMove(snap, bestMove.tile, bestMove.side);
      const userEnds = applyMove(snap, userMove.tile, userMove.side);

      if (deadNums.includes(bestEnds.newLE) && deadNums.includes(bestEnds.newRE) &&
          !(deadNums.includes(userEnds.newLE) && deadNums.includes(userEnds.newRE))) {
        reasons.push({
          type: 'inference_error',
          text: `${pn(opp)} is void in {${deadNums.join(',')}} — best move blocks them, yours doesn't`,
          deduction: `Pass evidence from turn ${deductions.getPassTurn(opp)}`
        });
      }
    }

    // Check counting error
    // Check tempo error
    // Check partner coordination error
    // Check risk error (match context)

    return { grade, reasons, equityLoss: el };
  }
};
```

---

## G) EXPLANATION ENGINE

### G.1 Three-Layer Explanations

```javascript
class ExplanationBuilder {
  /**
   * Generate explanation for why bestMove is best.
   * Returns { tldr, keyDrivers, deepDive, motifs, counterfactuals }
   */
  explain(bestMove, allMoves, snap, belief, deductions, metrics) {
    return {
      tldr: this._buildTLDR(bestMove, allMoves, snap, belief),
      keyDrivers: this._buildKeyDrivers(bestMove, snap, belief, metrics),
      deepDive: this._buildDeepDive(bestMove, allMoves, snap, belief),
      motifs: this._tagMotifs(bestMove, snap, belief, metrics),
      counterfactuals: this._buildCounterfactuals(bestMove, allMoves, snap, metrics)
    };
  }

  _buildTLDR(best, all, snap, belief) {
    // One sentence: "Play [3|5] on the left because it blocks Opp Left
    // while keeping 3 open for your partner."
    const { newLE, newRE } = applyMove(snap, best.tile, best.side);
    const reasons = [];

    // Check blocking
    for (const opp of [1, 3]) {
      const blocked = belief.events.filter(e =>
        e.type === 'pass' && e.player === opp
      );
      if (blocked.length > 0) {
        const deadNums = [...snap.knowledge.cantHave[opp]];
        if (deadNums.includes(newLE) && deadNums.includes(newRE)) {
          reasons.push(`blocks ${pn(opp)} (void in ${deadNums.join(',')})`);
        }
      }
    }

    // Check partner feed
    const partnerStr = snap.knowledge.inferStrength(2);
    if (partnerStr[newLE] >= 2 || partnerStr[newRE] >= 2) {
      reasons.push(`feeds partner's strong suit`);
    }

    // Check flexibility
    const flex = computeFlexibility(snap.hands[0], best.tile);
    if (flex.flexibility > 0.5) {
      reasons.push(`keeps hand flexible`);
    }

    if (reasons.length === 0) reasons.push(`highest overall equity`);

    return `Play [${best.tile.left}|${best.tile.right}] on ${best.side} — ${reasons.join(', ')}.`;
  }

  _buildKeyDrivers(best, snap, belief, metrics) {
    const drivers = [];

    drivers.push({
      metric: 'E[pts]', value: best.expectedPoints.toFixed(2),
      explanation: `Expected ${best.expectedPoints >= 0 ? 'gain' : 'loss'} of ${Math.abs(best.expectedPoints).toFixed(2)} points`
    });
    drivers.push({
      metric: 'Win%', value: (best.winRate * 100).toFixed(1) + '%',
      explanation: `Team wins this hand ${(best.winRate * 100).toFixed(1)}% of the time`
    });

    if (metrics.control) {
      drivers.push({
        metric: 'Forced passes', value: metrics.control.expectedForcedPasses.toFixed(1),
        explanation: `Expected to force ${metrics.control.expectedForcedPasses.toFixed(1)} opponent passes`
      });
    }

    if (metrics.partner) {
      drivers.push({
        metric: 'Partner play%', value: (metrics.partner.partnerPlayProb * 100).toFixed(0) + '%',
        explanation: `Partner can play next turn ${(metrics.partner.partnerPlayProb * 100).toFixed(0)}% of the time`
      });
    }

    return drivers;
  }

  _tagMotifs(best, snap, belief, metrics) {
    const motifs = [];
    const { newLE, newRE } = applyMove(snap, best.tile, best.side);

    // Block attempt
    for (const opp of [1, 3]) {
      const dead = [...snap.knowledge.cantHave[opp]];
      if (dead.includes(newLE) && dead.includes(newRE)) {
        motifs.push({
          name: 'BLOCK',
          definition: 'Leave board ends that an opponent cannot play on, forcing a pass.',
          evidence: `${pn(opp)} passed → void in {${dead.join(',')}}. New ends ${newLE}/${newRE} both blocked.`
        });
      }
    }

    // Partner feed
    const pStr = snap.knowledge.inferStrength(2);
    if (pStr[newLE] >= 2 || pStr[newRE] >= 2) {
      const suit = pStr[newLE] >= 2 ? newLE : newRE;
      motifs.push({
        name: 'PARTNER FEED',
        definition: 'Leave an end number your partner has demonstrated strength in.',
        evidence: `Partner played ${pStr[suit]}x tiles with ${suit}. Leaving ${suit} open for them.`
      });
    }

    // Double clearing
    if (best.tile.left === best.tile.right) {
      motifs.push({
        name: 'DOUBLE CLEAR',
        definition: 'Play a double while opportunity exists — doubles are inflexible.',
        evidence: `[${best.tile.left}|${best.tile.right}] only connects to ${best.tile.left}s.`
      });
    }

    // Information probe
    if (metrics.info && metrics.info.infoGain > 0.4) {
      motifs.push({
        name: 'INFO PROBE',
        definition: 'Play that maximizes information gain about opponent hands.',
        evidence: `Expected info gain: ${metrics.info.infoGain.toFixed(2)}. Next player ${(metrics.info.passProb_nextPlayer * 100).toFixed(0)}% to pass → reveals void.`
      });
    }

    // Tempo push
    if (metrics.tempo && metrics.tempo.controlRetention > 0.6) {
      motifs.push({
        name: 'TEMPO',
        definition: 'Maintain initiative by playing tiles that keep opponents struggling.',
        evidence: `Control retention: ${(metrics.tempo.controlRetention * 100).toFixed(0)}%.`
      });
    }

    return motifs;
  }

  _buildCounterfactuals(best, allMoves, snap, metrics) {
    // Compare top 3 moves
    const top3 = allMoves.slice(0, 3);
    return top3.map((m, i) => {
      if (i === 0) return { move: m, comparison: 'BEST', delta: 0 };
      const delta = best.expectedPoints - m.expectedPoints;
      const { newLE: bLE, newRE: bRE } = applyMove(snap, best.tile, best.side);
      const { newLE: mLE, newRE: mRE } = applyMove(snap, m.tile, m.side);

      let reason = '';
      if (delta > 0.3) {
        reason = `Loses ${delta.toFixed(2)} E[pts] mainly because `;
        // Identify the biggest difference
        if (m.winRate < best.winRate - 0.05) {
          reason += `win% drops from ${(best.winRate*100).toFixed(0)}% to ${(m.winRate*100).toFixed(0)}%.`;
        } else {
          reason += `worse board position (ends ${mLE}/${mRE} vs ${bLE}/${bRE}).`;
        }
      } else {
        reason = `Close alternative (${delta.toFixed(2)} E[pts] worse). Viable in some deals.`;
      }

      return { move: m, comparison: `#${i+1}`, delta, reason };
    });
  }
}
```

### G.2 Uncertainty Communication

```javascript
function explainUncertainty(results) {
  const best = results[0];
  const second = results.length > 1 ? results[1] : null;
  if (!second) return null;

  const gap = best.expectedPoints - second.expectedPoints;
  const ciOverlap = gap < (best.ci95 + second.ci95);

  if (ciOverlap) {
    return {
      message: `Close call: [${best.tile.left}|${best.tile.right}] and [${second.tile.left}|${second.tile.right}] are within confidence intervals (gap: ${gap.toFixed(2)}, CI overlap).`,
      recommendation: best.variance < second.variance
        ? `[${best.tile.left}|${best.tile.right}] is also lower variance — safer choice.`
        : `[${second.tile.left}|${second.tile.right}] has lower variance if you prefer safety.`
    };
  }
  return null;
}
```

---

## G2) HUMAN DEDUCTION COACH

### G2.1 DeductionNotebook Data Structure

```javascript
class DeductionNotebook {
  constructor() {
    // Per-player deductions
    this.players = [0, 1, 2, 3].map(() => ({
      cannotHaveNumbers: new Set(),      // hard: from passes
      voidSince: {},                      // voidSince[n] = turn number when void confirmed
      avoidanceEvidence: {},              // avoidanceEvidence[n] = { count, turns: [] }
      passEvents: [],                     // { turn, ends: [a,b] }
      playEvents: [],                     // { turn, tile, side }
      tileMarginals: {},                  // tileId → probability
    }));

    // Deduction bullets (human-readable, ordered by recency)
    this.bullets = [];

    // Global state
    this.remainingByNumber = [7, 7, 7, 7, 7, 7, 7]; // each number on 7 tiles
    this.playedTiles = new Set();
    this.turn = 0;

    // Thresholds for certainty labels
    this.thresholds = {
      certain: 1.0,    // P = 1.0 (hard logic)
      likely: 0.70,    // P >= 0.70
      possible: 0.40,  // P >= 0.40
      unclear: 0.0     // P < 0.40
    };
  }

  /**
   * Update after a PASS event.
   * This is the strongest signal — produces CERTAIN deductions.
   */
  recordPass(player, turn, leftEnd, rightEnd) {
    this.turn = turn;
    const p = this.players[player];

    // Hard deduction: player has no tiles with leftEnd or rightEnd
    const newVoids = [];
    for (const n of [leftEnd, rightEnd]) {
      if (!p.cannotHaveNumbers.has(n)) {
        p.cannotHaveNumbers.add(n);
        p.voidSince[n] = turn;
        newVoids.push(n);
      }
    }

    p.passEvents.push({ turn, ends: [leftEnd, rightEnd] });

    // Generate human-readable bullet
    if (newVoids.length > 0) {
      this.bullets.push({
        turn,
        player,
        certainty: 'CERTAIN',
        evidence: `${pn(player)} PASSED when ends were ${leftEnd}/${rightEnd}`,
        implication: `${pn(player)} has no ${newVoids.join('s and no ')}s`,
        strategy: this._passStrategy(player, newVoids, leftEnd, rightEnd),
        icon: '🚫'
      });
    }
  }

  /**
   * Update after a PLAY event.
   */
  recordPlay(player, turn, tile, side, leftEnd, rightEnd) {
    this.turn = turn;
    const p = this.players[player];

    // Remove tile from all possibilities
    this.playedTiles.add(tile.id);
    this.remainingByNumber[tile.left]--;
    if (tile.left !== tile.right) this.remainingByNumber[tile.right]--;

    p.playEvents.push({ turn, tile, side });

    // Track avoidance: did player choose NOT to play a number that was on the board?
    for (const n of [leftEnd, rightEnd]) {
      if (n !== undefined && tile.left !== n && tile.right !== n) {
        if (!p.avoidanceEvidence[n]) {
          p.avoidanceEvidence[n] = { count: 0, turns: [] };
        }
        p.avoidanceEvidence[n].count++;
        p.avoidanceEvidence[n].turns.push(turn);
      }
    }

    // Generate bullets for notable plays
    if (tile.left === tile.right) {
      this.bullets.push({
        turn, player, certainty: 'CERTAIN',
        evidence: `${pn(player)} played double [${tile.left}|${tile.left}]`,
        implication: `Remaining ${tile.left}s: ${this.remainingByNumber[tile.left]}`,
        strategy: this.remainingByNumber[tile.left] <= 2
          ? `${tile.left}s are now SCARCE — controlling the ${tile.left}-end is valuable`
          : `Double cleared early — ${pn(player)} likely has more ${tile.left}s`,
        icon: '🎲'
      });
    }

    // Check for avoidance patterns (STRONG INFERENCE)
    for (const n of [leftEnd, rightEnd]) {
      if (n === undefined) continue;
      const avoid = p.avoidanceEvidence[n];
      if (avoid && avoid.count >= 2 && !p.cannotHaveNumbers.has(n)) {
        this.bullets.push({
          turn, player, certainty: 'LIKELY',
          evidence: `${pn(player)} avoided playing ${n}s ${avoid.count} times (turns ${avoid.turns.join(', ')})`,
          implication: `${pn(player)} is likely LOW on ${n}s (not void, but weak)`,
          strategy: player % 2 === 0
            ? `Partner may need help with ${n}s — consider keeping ${n}-end open`
            : `Opponent weak in ${n} — keeping ${n} on board pressures them`,
          icon: '🔍'
        });
      }
    }

    // Scarcity alerts
    for (let n = 0; n <= 6; n++) {
      if (this.remainingByNumber[n] === 1 && this._prevRemaining(n) === 2) {
        this.bullets.push({
          turn, player: -1, certainty: 'CERTAIN',
          evidence: `Only 1 tile left with number ${n}`,
          implication: `${n} is nearly EXHAUSTED — opening a ${n}-end risks stalling the board`,
          strategy: `Avoid placing ${n} as an end unless you hold the last ${n}-tile`,
          icon: '⚠️'
        });
      }
    }
  }

  _prevRemaining(n) {
    // Count remaining before this turn's play (approximate)
    return this.remainingByNumber[n] + 1;
  }

  _passStrategy(player, voidNumbers, leftEnd, rightEnd) {
    const isOpp = player % 2 === 1;
    const isPartner = player === 2;

    if (isOpp) {
      return `KEEP ${voidNumbers.join(' and ')} on the board to force ${pn(player)} to pass again. ` +
             `Stack blocked numbers to lock them out.`;
    } else if (isPartner) {
      return `AVOID leaving ${voidNumbers.join(' or ')} as the only playable end for partner. ` +
             `Open other numbers to help them.`;
    }
    return '';
  }

  /**
   * Get the full deduction card for a player.
   */
  getPlayerCard(player) {
    const p = this.players[player];
    return {
      name: pn(player),
      team: player % 2 === 0 ? 'Your team' : 'Opponents',
      hardConstraints: [...p.cannotHaveNumbers].sort(),
      voidSince: { ...p.voidSince },
      avoidance: Object.entries(p.avoidanceEvidence)
        .filter(([n, e]) => e.count >= 2)
        .map(([n, e]) => ({ number: parseInt(n), count: e.count, turns: e.turns })),
      tilesPlayed: p.playEvents.length,
      estimatedHand: Math.max(0, 6 - p.playEvents.length),
      riskFlags: this._getRiskFlags(player)
    };
  }

  _getRiskFlags(player) {
    const flags = [];
    const p = this.players[player];
    const isOpp = player % 2 === 1;

    // Check for numbers where opponent is strong (we should NOT open them)
    if (isOpp) {
      for (let n = 0; n <= 6; n++) {
        if (p.cannotHaveNumbers.has(n)) continue;
        const avoid = p.avoidanceEvidence[n];
        const played = p.playEvents.filter(e => e.tile.left === n || e.tile.right === n).length;
        if (played >= 2 && this.remainingByNumber[n] >= 2) {
          flags.push({
            type: 'danger',
            text: `Do NOT open ${n}: ${pn(player)} played ${played}x ${n}s — likely strong`,
            certainty: 'LIKELY'
          });
        }
      }
    }

    return flags;
  }

  /**
   * Generate the "Think This Way" checklist for the current turn.
   */
  getThinkChecklist(snap, playerIdx) {
    const lE = snap.leftEnd, rE = snap.rightEnd;
    const items = [];

    // 1. Current ends
    items.push({
      category: 'Board State',
      text: `Ends now: ${lE} / ${rE}`,
      detail: null
    });

    // 2. Who is void on each end?
    for (const n of [lE, rE]) {
      const voidPlayers = [];
      for (let p = 0; p < 4; p++) {
        if (p === playerIdx) continue;
        if (this.players[p].cannotHaveNumbers.has(n)) {
          voidPlayers.push(pn(p));
        }
      }
      if (voidPlayers.length > 0) {
        items.push({
          category: 'Void Info',
          text: `${n}-end: ${voidPlayers.join(', ')} void`,
          detail: 'CERTAIN from pass evidence',
          certainty: 'CERTAIN'
        });
      }
    }

    // 3. Which end is safer to open?
    const safetyL = this._endSafety(lE, snap, playerIdx);
    const safetyR = this._endSafety(rE, snap, playerIdx);
    if (safetyL !== safetyR) {
      const safer = safetyL > safetyR ? lE : rE;
      const riskier = safetyL > safetyR ? rE : lE;
      items.push({
        category: 'Safety',
        text: `${safer}-end is SAFER to change; ${riskier}-end better to keep`,
        detail: `Safety scores: ${lE}=${safetyL.toFixed(1)}, ${rE}=${safetyR.toFixed(1)}`,
        certainty: 'LIKELY'
      });
    }

    // 4. Suit exhaustion warnings
    for (const n of [lE, rE]) {
      if (this.remainingByNumber[n] <= 2) {
        items.push({
          category: 'Counting',
          text: `Only ${this.remainingByNumber[n]} tile(s) left with ${n}`,
          detail: `${n} is ${this.remainingByNumber[n] === 0 ? 'EXHAUSTED' : 'scarce'}`,
          certainty: 'CERTAIN'
        });
      }
    }

    // 5. Partner status
    const partnerHand = Math.max(0, 6 - this.players[2].playEvents.length);
    if (partnerHand <= 2) {
      items.push({
        category: 'Partner',
        text: `Partner has ~${partnerHand} tile(s) left — close to winning!`,
        detail: 'Prioritize feeding partner over your own suit control',
        certainty: 'CERTAIN'
      });
    }

    return items;
  }

  _endSafety(endNum, snap, playerIdx) {
    // Higher = safer to open a new number by playing on this end
    let safety = 0;

    // Good: opponent is void on this end (they pass regardless)
    for (const opp of [1, 3]) {
      if (this.players[opp].cannotHaveNumbers.has(endNum)) safety += 2;
    }

    // Good: we have many tiles with this number (we can follow up)
    const myTiles = snap.hands[playerIdx].filter(t =>
      t.left === endNum || t.right === endNum
    ).length;
    safety += myTiles;

    // Bad: partner is void on this end (changing it might help them)
    if (this.players[2].cannotHaveNumbers.has(endNum)) safety -= 1;

    return safety;
  }

  /**
   * Get recent bullets (last N), prioritized by importance.
   */
  getRecentBullets(maxCount = 10) {
    // Sort by: CERTAIN first, then by recency
    const sorted = [...this.bullets].sort((a, b) => {
      const certOrder = { CERTAIN: 0, LIKELY: 1, POSSIBLE: 2, UNCLEAR: 3 };
      const ca = certOrder[a.certainty] || 3;
      const cb = certOrder[b.certainty] || 3;
      if (ca !== cb) return ca - cb;
      return b.turn - a.turn; // more recent first within same certainty
    });
    return sorted.slice(0, maxCount);
  }

  /**
   * When user makes a mistake, identify which deductions they missed.
   */
  getMissedDeductions(userMove, bestMove, snap) {
    const missed = [];
    const { newLE: uLE, newRE: uRE } = applyMove(snap, userMove.tile, userMove.side);
    const { newLE: bLE, newRE: bRE } = applyMove(snap, bestMove.tile, bestMove.side);

    // Check: did user fail to block a void opponent?
    for (const opp of [1, 3]) {
      const dead = this.players[opp].cannotHaveNumbers;
      const bestBlocks = dead.has(bLE) && dead.has(bRE);
      const userBlocks = dead.has(uLE) && dead.has(uRE);
      if (bestBlocks && !userBlocks) {
        missed.push({
          type: 'inference_error',
          deduction: `${pn(opp)} is void in {${[...dead].join(',')}}`,
          impact: `Best move blocks ${pn(opp)} completely; yours doesn't`,
          evidence: this.players[opp].passEvents.map(e =>
            `Turn ${e.turn}: passed at ${e.ends[0]}/${e.ends[1]}`
          )
        });
      }
    }

    // Check: did user block their own partner?
    const partnerDead = this.players[2].cannotHaveNumbers;
    if (partnerDead.has(uLE) && partnerDead.has(uRE) &&
        !(partnerDead.has(bLE) && partnerDead.has(bRE))) {
      missed.push({
        type: 'partner_error',
        deduction: `Partner is void in {${[...partnerDead].join(',')}}`,
        impact: 'Your move blocks your own partner!',
        evidence: this.players[2].passEvents.map(e =>
          `Turn ${e.turn}: partner passed at ${e.ends[0]}/${e.ends[1]}`
        )
      });
    }

    // Check: counting error (opened a scarce number)
    for (const n of [uLE, uRE]) {
      if (this.remainingByNumber[n] <= 1 && n !== bLE && n !== bRE) {
        missed.push({
          type: 'counting_error',
          deduction: `Only ${this.remainingByNumber[n]} tile(s) left with ${n}`,
          impact: `Opening ${n} risks stalling the board`,
          evidence: [`${7 - this.remainingByNumber[n]} of 7 tiles with ${n} already played`]
        });
      }
    }

    return missed;
  }

  clone() {
    const nb = new DeductionNotebook();
    nb.players = this.players.map(p => ({
      cannotHaveNumbers: new Set(p.cannotHaveNumbers),
      voidSince: { ...p.voidSince },
      avoidanceEvidence: JSON.parse(JSON.stringify(p.avoidanceEvidence)),
      passEvents: [...p.passEvents],
      playEvents: [...p.playEvents],
      tileMarginals: { ...p.tileMarginals }
    }));
    nb.bullets = [...this.bullets];
    nb.remainingByNumber = [...this.remainingByNumber];
    nb.playedTiles = new Set(this.playedTiles);
    nb.turn = this.turn;
    return nb;
  }
}
```

### G2.2 UI Specification

#### Panel 1: "What You Should Remember" (always visible)

```html
<!-- New tab in right panel: "Coach" -->
<div class="bg-slate-900 rounded-lg p-2 space-y-1.5 overflow-y-auto flex-1">
  <div class="text-xs font-bold text-emerald-400 mb-1">
    🧠 What You Should Remember
  </div>

  <!-- Bullet list: most important deductions -->
  ${notebook.getRecentBullets(8).map(b => `
    <div class="flex gap-2 items-start text-xs
      ${b.certainty === 'CERTAIN' ? 'bg-red-950/30 border-l-2 border-red-500' :
        b.certainty === 'LIKELY' ? 'bg-amber-950/30 border-l-2 border-amber-500' :
        'bg-slate-800 border-l-2 border-slate-600'} p-1.5 rounded-r">
      <span>${b.icon}</span>
      <div>
        <div class="text-slate-300">
          <span class="font-semibold">[Turn ${b.turn}]</span> ${b.evidence}
        </div>
        <div class="text-slate-400 mt-0.5">→ ${b.implication}</div>
        <div class="text-emerald-400 mt-0.5 italic">💡 ${b.strategy}</div>
        <span class="text-xs px-1 rounded
          ${b.certainty === 'CERTAIN' ? 'bg-red-900 text-red-300' :
            b.certainty === 'LIKELY' ? 'bg-amber-900 text-amber-300' :
            'bg-slate-700 text-slate-400'}">${b.certainty}</span>
      </div>
    </div>
  `).join('')}
</div>
```

#### Panel 2: Per-Player Deduction Cards

```html
${[1, 2, 3].map(p => {
  const card = notebook.getPlayerCard(p);
  return `
  <div class="bg-slate-900 rounded-lg p-2 mb-2">
    <div class="text-xs font-bold mb-1"
         style="color:${teamColors[p%2]}">${card.name}
      <span class="text-slate-500">(~${card.estimatedHand} tiles)</span>
    </div>
    <!-- Hard constraints -->
    ${card.hardConstraints.length > 0 ?
      `<div class="text-xs text-red-400">
        🚫 Void: {${card.hardConstraints.join(', ')}}
      </div>` : ''}
    <!-- Avoidance -->
    ${card.avoidance.length > 0 ?
      card.avoidance.map(a =>
        `<div class="text-xs text-amber-400">
          🔍 Weak in ${a.number}s (avoided ${a.count}x)
        </div>`
      ).join('') : ''}
    <!-- Risk flags -->
    ${card.riskFlags.map(f =>
      `<div class="text-xs text-red-300 font-semibold">⚠️ ${f.text}</div>`
    ).join('')}
  </div>`;
}).join('')}
```

#### Panel 3: "Think This Way" Checklist (on quiz turns)

```html
<div class="bg-emerald-950/30 border border-emerald-800 rounded-lg p-2 mb-2">
  <div class="text-xs font-bold text-emerald-400 mb-1">
    🤔 Before You Pick — Think About:
  </div>
  ${notebook.getThinkChecklist(snap, 0).map(item => `
    <div class="flex items-start gap-1.5 text-xs mb-1">
      <span class="text-slate-500 w-16 flex-shrink-0">${item.category}</span>
      <span class="text-slate-200">${item.text}</span>
      ${item.certainty ?
        `<span class="px-1 rounded text-xs
          ${item.certainty === 'CERTAIN' ? 'bg-red-900 text-red-300' :
            'bg-amber-900 text-amber-300'}">${item.certainty}</span>` : ''}
    </div>
  `).join('')}
</div>
```

### G2.3 Integration Points in simulator.html

| Current function | Change needed |
|---|---|
| `Knowledge.recordPass()` | Also call `notebook.recordPass()` |
| `Knowledge.recordPlay()` | Also call `notebook.recordPlay()` |
| `render()` | Add "Coach" tab in right panel, render notebook |
| `finishQuizPick()` | Call `notebook.getMissedDeductions()` for error coaching |
| `buildDetailedExplanation()` | Incorporate deduction evidence into explanations |
| `renderAnalysisPanel()` | Show per-move deduction-awareness in analysis report |

### G2.4 Safety Against Overclaiming

Every bullet and card entry MUST carry a `certainty` label:

| Label | Condition | Display |
|---|---|---|
| CERTAIN | From pass constraint or tile counting (100%) | Red badge |
| LIKELY | P ≥ 70% from avoidance evidence or belief model | Amber badge |
| POSSIBLE | 40% ≤ P < 70% | Gray badge |
| UNCLEAR | P < 40% | Not displayed (filtered out) |

The coach NEVER says "West has no 5s" without CERTAIN evidence. It says "West is LIKELY weak in 5s" when based on avoidance only.

### G2.5 Example Deduction Sequence

**Turn 3**: Opp Left passes at ends 4/5
```
🚫 [Turn 3] Opp Left PASSED when ends were 4/5
→ Opp Left has no 4s and no 5s
💡 KEEP 4 and 5 on the board to force Opp Left to pass again.
[CERTAIN]
```

**Turn 7**: Opp Right plays [6|2] when ends are 6/3 (avoids 3)
```
🔍 [Turn 7] Opp Right avoided playing 3s 2 times (turns 5, 7)
→ Opp Right is likely LOW on 3s (not void, but weak)
💡 Opponent weak in 3 — keeping 3 on board pressures them
[LIKELY]
```

**Turn 9**: Partner plays [3|3] (double-3)
```
🎲 [Turn 9] Partner played double [3|3]
→ Remaining 3s: 2
💡 3s are now SCARCE — controlling the 3-end is valuable
[CERTAIN]
```

---

## H) DATA & REPLAY PIPELINE

### H.1 JSONL Export Format

```javascript
class GameExporter {
  export(game, evaluations, notebook) {
    const lines = [];

    // Header
    lines.push(JSON.stringify({
      type: 'header',
      version: '2.0',
      timestamp: new Date().toISOString(),
      matchScore: [...matchScore],
      dobrada: scoreMultiplier,
      engineConfig: {
        mcSims: MC_SIMS_QUIZ,
        objectiveType: 'EV_Points',
        seed: currentSeed || null
      }
    }));

    // Initial state
    lines.push(JSON.stringify({
      type: 'deal',
      hands: game.originalHands,
      autoPlayedBy: game.autoPlayedBy,
      autoTile: game.autoTile
    }));

    // Decision points
    for (const snap of game.snapshots) {
      if (snap.action !== 'play' || snap.player < 0) continue;
      const eval = evaluations[snap.move] || null;
      lines.push(JSON.stringify({
        type: 'decision',
        move: snap.move,
        player: snap.player,
        tile: { left: snap.tile.left, right: snap.tile.right },
        side: snap.side,
        boardEnds: [snap.leftEnd, snap.rightEnd],
        handSize: snap.hands[snap.player].length,
        allOptions: snap.allOptions.map(o => ({
          tile: { left: o.tile.left, right: o.tile.right },
          side: o.side,
          heuristicScore: o.score
        })),
        evaluation: eval ? {
          expectedPoints: eval.expectedPoints,
          winRate: eval.winRate,
          variance: eval.variance,
          sims: eval.sims
        } : null,
        deductions: notebook ? notebook.getRecentBullets(5) : null
      }));
    }

    // Outcome
    lines.push(JSON.stringify({
      type: 'outcome',
      ...game.outcome
    }));

    return lines.join('\n');
  }
}

// UI: "Export Game" button → downloads .jsonl file
function exportGame() {
  const exporter = new GameExporter();
  const jsonl = exporter.export(currentGame, currentGame._analysisCache || {}, notebook);
  const blob = new Blob([jsonl], { type: 'application/jsonl' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `domino-game-${Date.now()}.jsonl`;
  a.click();
  URL.revokeObjectURL(url);
}
```

### H.2 Import/Replay

```javascript
function importGame(jsonlText) {
  const lines = jsonlText.trim().split('\n').map(JSON.parse);
  const header = lines.find(l => l.type === 'header');
  const deal = lines.find(l => l.type === 'deal');
  const decisions = lines.filter(l => l.type === 'decision');
  const outcome = lines.find(l => l.type === 'outcome');

  // Reconstruct game from deal + decisions
  // ... (rebuild snapshots from decision log)
  // Load into currentGame and render
}
```

### H.3 Self-Play Dataset Builder

```javascript
function selfPlayBatch(numGames = 100) {
  const results = [];
  for (let i = 0; i < numGames; i++) {
    const game = generateGame();
    // Optionally run MC analysis on each P0 decision
    results.push({
      outcome: game.outcome,
      moves: game.snapshots.filter(s => s.action === 'play').length
    });
  }
  return results;
}
```

---

## I) ENGINEERING: EXACT CODE CHANGES

### I.1 Functions to Replace

| Current | Replace with | Lines |
|---------|-------------|-------|
| `Knowledge` class | `BeliefModel` + keep `Knowledge` as lightweight wrapper | 146-200 |
| `generateConsistentDeal()` | `BeliefSampler.sampleDeals()` | 576-678 |
| `monteCarloEval()` | `Evaluator.quickEval()` / `Evaluator.deepEval()` | 766-868 |
| `computeTileProbs()` | `BeliefModel.updateMarginals()` + `getTileMarginal()` | 682-701 |
| `buildDetailedExplanation()` | `ExplanationBuilder.explain()` | 1168-1336 |
| `ME` table (2D) | `ME3D` table with dobrada | 894-924 |

### I.2 New Modules (in-file sections)

```
// ========== SEEDABLE PRNG ==========
class SplitMix64 { ... }

// ========== BELIEF MODEL ==========
class BeliefModel { ... }

// ========== DEDUCTION NOTEBOOK ==========
class DeductionNotebook { ... }

// ========== DEAL SAMPLER ==========
class BeliefSampler { ... }   // Level 1: importance-sampled
class MCMCSampler { ... }     // Level 2: Gibbs swap

// ========== EVALUATOR ==========
class Evaluator { ... }       // quick + deep eval
class LRUCache { ... }        // bounded cache

// ========== METRICS LIBRARY ==========
const EquityMetrics = { ... }
const RiskMetrics = { ... }
function computeTempoMetrics() { ... }
function computeControlMetrics() { ... }
function computeFlexibility() { ... }
function computeInfoGain() { ... }
function computePartnerSynergy() { ... }

// ========== EXPLANATION ENGINE ==========
class ExplanationBuilder { ... }
const ErrorClassifier = { ... }

// ========== MATCH EQUITY 3D ==========
const ME3D = (() => { ... })();

// ========== EXPORT/REPLAY ==========
class GameExporter { ... }

// ========== WEBWORKER BRIDGE ==========
// (inline worker using Blob URL)
```

### I.3 Integration Order

1. **Add `SplitMix64`** PRNG → replace `Math.random()` calls in sampling/shuffle
2. **Add `BeliefModel`** → instantiate alongside `Knowledge` in `simulateFullRound()`
3. **Add `DeductionNotebook`** → instantiate alongside `Knowledge`, update in pass/play handlers
4. **Add `BeliefSampler`** → replace `generateConsistentDeal()` calls in `monteCarloEval()`
5. **Add `Evaluator`** → replace `monteCarloEval()` with `Evaluator.quickEval()`
6. **Add `ME3D`** → replace `ME` and `getMatchEquity()`
7. **Add `ExplanationBuilder`** → replace `buildDetailedExplanation()`
8. **Add Coach UI tab** → new right panel tab "Coach" rendering notebook
9. **Add Export button** → "Export Game" in controls
10. **Add WebWorker** → offload `deepEval()` to worker

---

## J) PERFORMANCE + PARALLELISM

### J.1 Seedable PRNG

```javascript
class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) || 0n;
  }

  next() {
    this.state += 0x9e3779b97f4a7c15n;
    let z = this.state;
    z = (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n;
    z = (z ^ (z >> 27n)) * 0x94d049bb133111ebn;
    z = z ^ (z >> 31n);
    return z;
  }

  random() {
    // Returns float in [0, 1)
    return Number(this.next() & 0xFFFFFFFFn) / 0x100000000;
  }

  randomInt(max) {
    return Math.floor(this.random() * max);
  }
}
```

### J.2 WebWorker (Inline)

```javascript
function createEvalWorker() {
  const workerCode = `
    // Include: SplitMix64, BeliefModel, BeliefSampler, simulateFromPosition, smartAI, etc.
    // (serialized as string and included in worker)

    self.onmessage = function(e) {
      const { snap, player, K, taskId } = e.data;
      // Run evaluation
      const results = evaluate(snap, player, K);
      self.postMessage({ taskId, results });
    };
  `;
  const blob = new Blob([workerCode], { type: 'application/javascript' });
  return new Worker(URL.createObjectURL(blob));
}
```

### J.3 LRU Cache

```javascript
class LRUCache {
  constructor(maxSize = 500) {
    this.max = maxSize;
    this.map = new Map();
  }

  get(key) {
    if (!this.map.has(key)) return null;
    const val = this.map.get(key);
    // Move to end (most recent)
    this.map.delete(key);
    this.map.set(key, val);
    return val;
  }

  set(key, val) {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, val);
    if (this.map.size > this.max) {
      // Delete oldest
      const first = this.map.keys().next().value;
      this.map.delete(first);
    }
  }

  has(key) { return this.map.has(key); }
}
```

---

## K) TEST PLAN

### K.1 Rule Correctness Tests

```javascript
// Test: scoring types
assert(scoreOutcome({left:3, right:3}, {left:3, right:5}) === {pts:2, type:'carroca'});
// [3|3] closing, not both ends → carroca

assert(scoreOutcome({left:3, right:5}, {left:3, right:5}) === {pts:3, type:'laelo'});
// [3|5] closing when ends are 3,5 → la-e-lo (matches both)

assert(scoreOutcome({left:4, right:4}, {left:4, right:4}) === {pts:4, type:'cruzada'});
// [4|4] closing when ends are 4,4 → cruzada (double + both ends)

// Test: pass constraint
const k = new Knowledge();
k.recordPass(1, 3, 5); // P1 passes at ends 3,5
assert(k.cantHave[1].has(3) && k.cantHave[1].has(5));
assert(!k.canPlayerHold(1, {left:3, right:6})); // has 3 → impossible
assert(k.canPlayerHold(1, {left:1, right:2}));   // no 3 or 5 → ok
```

### K.2 Sampling Tests

```javascript
// Distribution sanity: with no constraints, each player should get ~uniform tiles
function testUniformDistribution() {
  const sampler = new BeliefSampler(new SplitMix64(42));
  const belief = new BeliefModel();
  const counts = {};
  const K = 10000;
  const myHand = ALL_TILES.slice(0, 6);
  const handSizes = [6, 6, 6, 6];

  const { deals } = sampler.sampleDeals(myHand, handSizes, belief, 0, K);

  for (const deal of deals) {
    for (const t of deal[1]) {
      counts[t.id] = (counts[t.id] || 0) + 1;
    }
  }

  // Each of the 22 remaining tiles should appear ~6/22 * K times in P1's hand
  const expected = 6 / 22 * K;
  for (const [id, count] of Object.entries(counts)) {
    const ratio = count / expected;
    assert(ratio > 0.8 && ratio < 1.2, `Tile ${id}: ratio ${ratio} outside [0.8, 1.2]`);
  }
}

// Constraint satisfaction: with tight constraints, no invalid deals
function testConstraintSatisfaction() {
  const belief = new BeliefModel();
  belief.recordPass(1, 1, 3, 5); // P1 void in 3,5
  belief.recordPass(3, 2, 2, 4); // P3 void in 2,4

  const sampler = new BeliefSampler(new SplitMix64(123));
  const myHand = ALL_TILES.slice(0, 6);
  const { deals } = sampler.sampleDeals(myHand, [6,6,6,6], belief, 0, 1000);

  for (const deal of deals) {
    for (const t of deal[1]) {
      assert(t.left !== 3 && t.right !== 3 && t.left !== 5 && t.right !== 5,
        `P1 got tile with 3 or 5: ${t.id}`);
    }
    for (const t of deal[3]) {
      assert(t.left !== 2 && t.right !== 2 && t.left !== 4 && t.right !== 4,
        `P3 got tile with 2 or 4: ${t.id}`);
    }
  }
}
```

### K.3 Determinism Tests

```javascript
function testDeterminism() {
  const rng1 = new SplitMix64(42);
  const rng2 = new SplitMix64(42);
  const sampler1 = new BeliefSampler(rng1);
  const sampler2 = new BeliefSampler(rng2);

  const belief = new BeliefModel();
  const myHand = ALL_TILES.slice(0, 6);

  const r1 = sampler1.sampleDeals(myHand, [6,6,6,6], belief, 0, 50);
  const r2 = sampler2.sampleDeals(myHand, [6,6,6,6], belief, 0, 50);

  assert(r1.deals.length === r2.deals.length);
  for (let i = 0; i < r1.deals.length; i++) {
    for (let p = 0; p < 4; p++) {
      const ids1 = r1.deals[i][p].map(t => t.id).join(',');
      const ids2 = r2.deals[i][p].map(t => t.id).join(',');
      assert(ids1 === ids2, `Deal ${i} player ${p} mismatch`);
    }
  }
}
```

### K.4 Regression Tests (Known Positions)

```javascript
// Position: P0 has [5|6], board ends 5/3, P1 void in {3,5}, Partner played 2x 6s
// Expected: play [5|6] on left (opens 6 for partner, blocks P1 on 5)
function testKnownPosition1() {
  const snap = createTestPosition({
    hand: [{left:5, right:6}],
    leftEnd: 5, rightEnd: 3,
    p1Void: [3, 5],
    partnerPlayed: [{left:6, right:2}, {left:6, right:1}]
  });
  const results = evaluator.quickEval(snap, 0, 200);
  assert(results[0].side === 'left', 'Should play on left to expose 6');
}
```

### K.5 Self-Play Elo Tracking

```javascript
function selfPlayElo(engineA, engineB, numGames = 200) {
  let winsA = 0, winsB = 0, draws = 0;

  for (let i = 0; i < numGames; i++) {
    // Alternate which team each engine plays
    const teamA = i % 2;
    const game = runGame(engineA, engineB, teamA);

    if (game.outcome.team === teamA) winsA++;
    else if (game.outcome.type === 'tie') draws++;
    else winsB++;
  }

  const winRate = winsA / (winsA + winsB + draws);
  const elo = -400 * Math.log10(1 / winRate - 1);
  return { winsA, winsB, draws, winRate, eloDiff: elo };
}
```

### K.6 Stress Tests

```javascript
// Worst case: all 4 players have passed multiple times → very tight constraints
function stressTestTightConstraints() {
  const belief = new BeliefModel();
  // Each opponent void in 3+ numbers
  belief.recordPass(1, 1, 0, 1);
  belief.recordPass(1, 5, 2, 3);
  belief.recordPass(3, 2, 4, 5);
  belief.recordPass(3, 6, 6, 0);

  const sampler = new BeliefSampler(new SplitMix64(999));
  const myHand = [{left:1,right:2,id:'1-2'},{left:3,right:4,id:'3-4'}];
  const start = performance.now();
  const { deals } = sampler.sampleDeals(myHand, [2,4,6,4], belief, 0, 100);
  const elapsed = performance.now() - start;

  console.log(`Tight constraints: ${deals.length}/100 valid deals in ${elapsed.toFixed(0)}ms`);
  assert(deals.length >= 50, 'Should generate at least 50 valid deals');
  assert(elapsed < 500, 'Should complete in <500ms');
}
```

---

## IMPLEMENTATION CHECKLIST

### Phase 1 — Correctness
- [ ] Implement `SplitMix64` PRNG
- [ ] Replace `Math.random()` in `generateConsistentDeal()` and `shuffleDeck()`
- [ ] Implement `BeliefSampler` with importance weighting
- [ ] Remove greedy fallback from deal generation
- [ ] Add variance, stdDev, CI to `monteCarloEval()` results
- [ ] Add adaptive stopping (CI < threshold)
- [ ] Implement `ME3D` with dobrada dimension
- [ ] Replace `getMatchEquity()` and `roundToMatchEquity()`
- [ ] Write rule correctness tests
- [ ] Write sampling distribution tests
- [ ] Write determinism tests

### Phase 2 — Belief + Coaching
- [ ] Implement `BeliefModel` class
- [ ] Implement `DeductionNotebook` class
- [ ] Hook `recordPass` / `recordPlay` into simulation engine
- [ ] Add "Coach" tab in right panel
- [ ] Render "What You Should Remember" bullets
- [ ] Render per-player deduction cards
- [ ] Render "Think This Way" checklist on quiz turns
- [ ] Add `getMissedDeductions()` to quiz error feedback
- [ ] Integrate deduction evidence into `buildDetailedExplanation()`
- [ ] Add certainty labels (CERTAIN/LIKELY/POSSIBLE/UNCLEAR)
- [ ] Write deduction correctness tests (scripted sequences)

### Phase 3 — Strength + Metrics
- [ ] Implement `computeFlexibility()`
- [ ] Implement `computeControlMetrics()`
- [ ] Implement `computeTempoMetrics()`
- [ ] Implement `computeInfoGain()`
- [ ] Implement `computePartnerSynergy()`
- [ ] Implement `ExplanationBuilder` with motif tagging
- [ ] Implement counterfactual comparison (top 3 moves)
- [ ] Implement `ErrorClassifier.classifyWithReason()`
- [ ] Add `MCMCSampler` (Level 2 sampling)
- [ ] Add metrics display in MC panel

### Phase 4 — Analytics + Data
- [ ] Implement `LRUCache` for eval results
- [ ] Implement `GameExporter` (JSONL)
- [ ] Add Export Game button in UI
- [ ] Implement import/replay functionality
- [ ] Implement WebWorker for deep eval
- [ ] Implement self-play benchmark
- [ ] Add Elo tracking
- [ ] Stress test full pipeline

---

*End of specification. Total estimated LOC: ~2300 across all phases.*
*Priority: Phase 1 (correctness) → Phase 2 (coaching) → Phase 3 (strength) → Phase 4 (data)*
