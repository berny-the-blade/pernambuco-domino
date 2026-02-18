# Domino Pernambucano — Solver-Level Analytical Engine Architecture

## Current State Assessment

| Component | Current | Target |
|-----------|---------|--------|
| Belief model | Binary elimination (cantHave sets) | Bayesian probability distributions |
| Sampling | Rejection + backtracking (unbiased) | Gibbs/MCMC with convergence diagnostics |
| Evaluation | Heuristic weights + MC E[pts] | Full rollout equity with variance |
| Match equity | Pre-computed DP table (symmetric) | Asymmetric ME with dobrada stack |
| Error classification | E[pts] thresholds | Match equity loss (like XG) |
| Flexibility | Follow-up count (implicit) | Explicit leave value + connectivity |
| Information | cantHave sets + pass tracking | Entropy tracking + info gain per move |
| Performance | Synchronous main thread | Chunked async (WebWorker deferred) |

---

## SECTION 1 — State Formalization

### 1.1 Game State

```
s = {
  boardEnds: [leftEnd, rightEnd],     // 0-6 each, null if empty
  boardLength: int,                    // tiles on board
  played: Set<tileId>,                 // 28 tiles total
  myHand: Tile[],                      // known tiles
  handSizes: [int, int, int, int],     // tiles per player
  passHistory: PassEvent[],            // {player, leftEnd, rightEnd}
  cantHave: [Set, Set, Set, Set],      // hard elimination
  playsBy: [Tile[], Tile[], ...],      // tiles played by each player
  teamScores: [int, int],             // match scores (0-5 each)
  dobrada: int,                        // current multiplier (1, 2, 4, ...)
  currentPlayer: int                   // 0-3
}
```

### 1.2 Value Functions

**Round Win Probability:**
```
V_win(s, team) = (1/N) Σ_{d ∈ ConsistentDeals(s)} I[Rollout(s, d).winner == team]
```

**Expected Round Points (signed, from team's perspective):**
```
V_pts(s, team) = (1/N) Σ_d [
  if Rollout(s,d).winner == team: +Rollout(s,d).points
  else if Rollout(s,d).winner == opp: -Rollout(s,d).points
  else: 0  (tie)
]
```

**Match Equity:**
```
V_match(s, team) = Σ_{outcome ∈ Outcomes} P(outcome | s) × ME(scores_after_outcome)
```

Where:
```
ME(a, b) = P(team_a wins match | score a vs b, match to 6)

Base cases:
  ME(≥6, b) = 1.0   for b < 6
  ME(a, ≥6) = 0.0   for a < 6
  ME(≥6, ≥6) = 0.5  (shouldn't happen)

Recursion:
  ME(a, b) = Σ_k P(pts=k) × [0.5 × ME(a+k*d, b) + 0.5 × ME(a, b+k*d)]

  where d = dobrada multiplier, P(pts=k) from empirical distribution:
    k=1: 0.70  (normal + blocked wins)
    k=2: 0.16  (carroca)
    k=3: 0.10  (la-e-lo)
    k=4: 0.04  (cruzada)
```

**Move Equity Loss (for error classification):**
```
EL(m) = V_match(s after best_move) - V_match(s after m)
```

### 1.3 Move Evaluation

For each legal move m from state s:
```
Eval(m) = {
  winRate:        E[I(win) | m],
  expectedPoints: E[signed_pts | m],
  matchEquity:    E[ME_after | m],
  variance:       Var[signed_pts | m],
  downsideRisk:   P(lose ≥ 2 pts | m),
  equityLoss:     V_match(best) - V_match(m),
  flexibility:    E[playable_tiles_next_turn | m],
  infoGain:       H(beliefs_before) - E[H(beliefs_after) | m]
}
```

---

## SECTION 2 — Bayesian Belief Model

### 2.1 Current Model (Binary Elimination)

```
For player p, tile t:
  P(p holds t) ∈ {0, "possible"}

cantHave[p] = set of numbers → if p passed on end n, n ∈ cantHave[p]
A tile t is "impossible" for p iff:
  t is double [n|n] and n ∈ cantHave[p], OR
  t.left ∈ cantHave[p] AND t.right ∈ cantHave[p]
```

### 2.2 Target Model (Bayesian Probabilities)

**Definition:**
```
B[p][t] = P(player p holds tile t | all observations)
```

**Constraints:**
```
1. Tile conservation:  Σ_p B[p][t] = 1  for unplayed tiles not in my hand
2. Hand size:          Σ_t B[p][t] = handSizes[p]  for each player p
3. Hard elimination:   B[p][t] = 0  if t.left ∈ cantHave[p] AND t.right ∈ cantHave[p]
4. Known tiles:        B[me][t] = 1 if t in myHand, 0 otherwise
5. Played tiles:       B[p][t] = 0  for all p if t ∈ played
6. Dorme tiles:        B[dorme][t] = 1 - Σ_p B[p][t]  (implicit)
```

### 2.3 Computing Beliefs — Weighted Monte Carlo

**Algorithm: ComputeBeliefs(knowledge, handSizes, myHand, myIdx, N=500)**

```python
counts = {p: {t: 0 for t in unplayed_not_mine} for p in other_players}
valid = 0

for i in range(N):
    deal = generateConsistentDeal(myHand, handSizes, knowledge, myIdx)
    valid += 1
    for p in other_players:
        for t in deal[p]:
            counts[p][t.id] += 1

beliefs = {}
for p in other_players:
    beliefs[p] = {}
    for t_id in counts[p]:
        beliefs[p][t_id] = counts[p][t_id] / valid

return beliefs
```

This is what `computeTileProbs` already does. To make it a proper Bayesian model:

### 2.4 Incorporating Play Inference (Bridge-style)

Beyond pass constraints, incorporate **play patterns** as soft evidence:

```
P(p holds t | p played tiles with number n frequently)
  ∝ P(p plays n-tiles | p holds t) × P(p holds t | prior)
```

**Implementation: Soft Constraints via Importance Weighting**

When generating consistent deals, weight each deal by its plausibility:

```python
def deal_weight(deal, knowledge):
    w = 1.0
    for p in range(4):
        # Players who played many n-tiles likely hold more n-tiles
        for n in range(7):
            played_count = knowledge.inferStrength(p)[n]
            held_count = sum(1 for t in deal[p] if t.left == n or t.right == n)
            # Bayesian: P(hold k | played c) ∝ P(played c | hold k)
            # Approximate: players play from their strong suits
            if played_count >= 2 and held_count > 0:
                w *= 1.0 + 0.3 * held_count  # soft boost
    return w
```

Use importance sampling:
```
B[p][t] = Σ_i (w_i × I[t ∈ deal_i[p]]) / Σ_i w_i
```

### 2.5 Entropy Tracking

**Per-player entropy:**
```
H(p) = - Σ_t [B[p][t] × log2(B[p][t]) + (1 - B[p][t]) × log2(1 - B[p][t])]
```

More practically, entropy over the possible hand compositions:
```
H_composition(p) = log2(|consistent_deals_for_p|)
                  ≈ log2(C(eligible_tiles, hand_size))
```

**Information gain per move:**
```
IG(move) = H(beliefs_before) - H(beliefs_after_move)
```

A pass reveals more info (2 dead numbers) than a play (1 tile revealed).

### 2.6 Data Structure

```javascript
class BeliefState {
  constructor(knowledge, handSizes, myHand, myIdx) {
    this.probs = {};      // {playerId: {tileId: probability}}
    this.entropy = {};    // {playerId: float}
    this.infoGain = 0;    // bits gained this move
    this._compute(knowledge, handSizes, myHand, myIdx);
  }

  _compute(knowledge, handSizes, myHand, myIdx, nSamples = 300) {
    // Monte Carlo belief estimation with importance weighting
    const weights = [];
    const samples = [];
    for (let i = 0; i < nSamples; i++) {
      const deal = generateConsistentDeal(myHand, handSizes, knowledge, myIdx);
      const w = this._dealWeight(deal, knowledge);
      weights.push(w);
      samples.push(deal);
    }
    const totalW = weights.reduce((s, w) => s + w, 0);

    for (const p of [0, 1, 2, 3]) {
      if (p === myIdx) continue;
      this.probs[p] = {};
      for (const t of ALL_TILES) {
        if (knowledge.played.has(t.id) || myHand.some(m => m.id === t.id)) continue;
        let wSum = 0;
        for (let i = 0; i < samples.length; i++) {
          if (samples[i][p].some(st => st.id === t.id)) wSum += weights[i];
        }
        this.probs[p][t.id] = wSum / totalW;
      }
      // Compute entropy
      this.entropy[p] = 0;
      for (const tid of Object.keys(this.probs[p])) {
        const pr = this.probs[p][tid];
        if (pr > 0 && pr < 1) {
          this.entropy[p] -= pr * Math.log2(pr) + (1 - pr) * Math.log2(1 - pr);
        }
      }
    }
  }

  _dealWeight(deal, knowledge) {
    let w = 1.0;
    for (let p = 0; p < 4; p++) {
      const str = knowledge.inferStrength(p);
      for (let n = 0; n < 7; n++) {
        if (str[n] >= 2) {
          const held = deal[p].filter(t => t.left === n || t.right === n).length;
          w *= 1.0 + 0.2 * held;
        }
      }
    }
    return w;
  }

  probHolds(player, tileId) {
    return this.probs[player]?.[tileId] || 0;
  }

  playerEntropy(player) {
    return this.entropy[player] || 0;
  }
}
```

---

## SECTION 3 — Correct Sampling Engine

### 3.1 Current Implementation (Already Upgraded)

Three-tier approach already implemented:
1. **Rejection sampling** (200 attempts) — truly unbiased
2. **Randomized backtracking** — handles tight constraints
3. **Greedy fallback** — last resort

### 3.2 Gibbs Sampling Enhancement

For tighter constraint scenarios, add Gibbs sampling as intermediate tier:

```python
def gibbs_sample(initial_deal, knowledge, hand_sizes, n_iters=50):
    """
    Start from any valid deal, repeatedly:
    1. Pick random pair of players (p1, p2)
    2. Pool their tiles
    3. Redistribute respecting constraints
    """
    deal = deep_copy(initial_deal)

    for _ in range(n_iters):
        # Pick two random non-self players
        p1, p2 = random.sample(other_players, 2)

        # Pool their tiles
        pool = deal[p1] + deal[p2]
        random.shuffle(pool)

        # Redistribute respecting constraints
        new_p1, new_p2 = [], []
        remaining = list(pool)

        for t in remaining:
            if len(new_p1) >= hand_sizes[p1]:
                new_p2.append(t)
            elif len(new_p2) >= hand_sizes[p2]:
                new_p1.append(t)
            elif canPlayerHold(p1, t, knowledge) and not canPlayerHold(p2, t, knowledge):
                new_p1.append(t)
            elif canPlayerHold(p2, t, knowledge) and not canPlayerHold(p1, t, knowledge):
                new_p2.append(t)
            else:
                # Both can hold — random assignment
                if random.random() < hand_sizes[p1] / (hand_sizes[p1] + hand_sizes[p2]):
                    new_p1.append(t)
                else:
                    new_p2.append(t)

        if len(new_p1) == hand_sizes[p1] and len(new_p2) == hand_sizes[p2]:
            deal[p1] = new_p1
            deal[p2] = new_p2

    return deal
```

### 3.3 Sampling Validation

```javascript
function validateSamplingUniformity(knowledge, handSizes, myHand, myIdx, nTrials = 5000) {
  const tileCounts = {};  // {tileId: {playerId: count}}

  for (let i = 0; i < nTrials; i++) {
    const deal = generateConsistentDeal(myHand, handSizes, knowledge, myIdx);
    for (let p = 0; p < 4; p++) {
      for (const t of deal[p]) {
        if (!tileCounts[t.id]) tileCounts[t.id] = {};
        tileCounts[t.id][p] = (tileCounts[t.id][p] || 0) + 1;
      }
    }
  }

  // Chi-squared test for uniformity across eligible players
  let totalChiSq = 0, df = 0;
  for (const tid of Object.keys(tileCounts)) {
    const eligible = [0,1,2,3].filter(p => p !== myIdx && canPlayerHold(p, {left: +tid[0], right: +tid[2]}, knowledge));
    if (eligible.length <= 1) continue;
    const total = eligible.reduce((s, p) => s + (tileCounts[tid][p] || 0), 0);
    const expected = total / eligible.length;
    for (const p of eligible) {
      const obs = tileCounts[tid][p] || 0;
      totalChiSq += (obs - expected) ** 2 / expected;
      df++;
    }
  }

  return { chiSquared: totalChiSq, degreesOfFreedom: df, pValue: 1 - chiSquaredCDF(totalChiSq, df) };
}
```

---

## SECTION 4 — Move Evaluation Pipeline

### 4.1 Full Evaluation Pipeline

```javascript
function evaluateMove(snap, player, move, numSims = 120) {
  const { tile, side } = move;
  const hand = snap.hands[player];
  const myTeam = player % 2;

  // Apply move to get new state
  const newState = applyMove(snap, player, tile, side);

  let wins = 0, totalPts = 0, totalPtsSq = 0;
  let blocks = 0, bigLosses = 0;
  const outcomes = { normal: 0, carroca: 0, laelo: 0, cruzada: 0, blocked: 0, tie: 0 };

  // Check instant win
  if (newState.handSizes[player] === 0) {
    const pts = computeWinPoints(tile, snap.leftEnd, snap.rightEnd, snap.board.length);
    return {
      winRate: 1.0,
      expectedPoints: pts.points,
      variance: 0,
      downsideRisk: 0,
      matchEquity: getMatchEquity(
        matchScore[myTeam] + pts.points * dobrada,
        matchScore[1 - myTeam]
      ),
      outcomes: { [pts.type]: numSims },
      flexibility: 0,
      instantWin: true
    };
  }

  for (let sim = 0; sim < numSims; sim++) {
    const deal = generateConsistentDeal(
      newState.hands[player], newState.handSizes, newState.knowledge, player
    );
    const result = simulateFromPosition(
      deal, null, newState.leftEnd, newState.rightEnd,
      (player + 1) % 4, newState.knowledge, newState.boardLength
    );

    const signedPts = result.winnerTeam === myTeam ? result.points
                    : result.winnerTeam >= 0 ? -result.points : 0;
    totalPts += signedPts;
    totalPtsSq += signedPts * signedPts;
    if (result.winnerTeam === myTeam) wins++;
    if (signedPts <= -2) bigLosses++;
    if (result.type === 'blocked' || result.type === 'tie') blocks++;
    outcomes[result.type]++;
  }

  const ep = totalPts / numSims;
  const variance = totalPtsSq / numSims - ep * ep;

  // Compute match equity after expected outcome
  const meAfterWin = getMatchEquity(matchScore[myTeam] + 1 * dobrada, matchScore[1 - myTeam]);
  const meAfterLoss = getMatchEquity(matchScore[myTeam], matchScore[1 - myTeam] + 1 * dobrada);
  const matchEq = (wins / numSims) * meAfterWin + (1 - wins / numSims) * meAfterLoss;

  // Flexibility: how many tiles can player play next turn on new ends?
  const remaining = hand.filter(t => t.id !== tile.id);
  const flexibility = remaining.filter(t =>
    t.left === newState.leftEnd || t.right === newState.leftEnd ||
    t.left === newState.rightEnd || t.right === newState.rightEnd
  ).length;

  return {
    winRate: wins / numSims,
    expectedPoints: ep,
    variance,
    downsideRisk: bigLosses / numSims,
    matchEquity: matchEq,
    blockRate: blocks / numSims,
    outcomes,
    flexibility,
    instantWin: false,
    sims: numSims
  };
}
```

### 4.2 Output Table Format

```
Move       | Win%  | E[pts] | ME     | σ²   | Risk↓ | Flex | EL    | Grade
[0|5] R    | 72%   | +0.84  | 0.812  | 1.2  | 5%    | 1    | 0.000 | BEST
[0|5] L    | 48%   | +0.22  | 0.743  | 2.1  | 18%   | 0    | 0.069 | MISTAKE
```

### 4.3 Adaptive Monte Carlo (Stop Early)

```javascript
function adaptiveMC(snap, player, maxSims = 200, ciThreshold = 0.05) {
  const results = [];
  // ... setup per tile ...

  for (let sim = 0; sim < maxSims; sim++) {
    // Run one sim for each tile
    // After every 20 sims, check confidence intervals
    if (sim > 0 && sim % 20 === 0) {
      const sorted = results.sort((a, b) => b.ep - a.ep);
      const best = sorted[0], second = sorted[1];
      if (!second) break; // only one option

      // 95% CI for difference
      const diffMean = best.ep - second.ep;
      const diffVar = best.variance / sim + second.variance / sim;
      const diffCI = 1.96 * Math.sqrt(diffVar);

      // If best is clearly better than second, stop
      if (diffMean - diffCI > 0) break;
    }
  }
  return results;
}
```

---

## SECTION 5 — Match Equity Model

### 5.1 Full ME Table with Dobrada

The current ME table assumes dobrada=1. For full accuracy:

```
ME[a][b][d] = P(team_a wins match | score a vs b, dobrada multiplier d)

Base cases:
  ME[≥6][b][d] = 1.0 for b < 6
  ME[a][≥6][d] = 0.0 for a < 6

Recursion:
  ME[a][b][d] = Σ_k P(pts=k) × [
    0.5 × ME[min(a + k*d, 10)][b][1] +    // we win, dobrada resets
    0.5 × ME[a][min(b + k*d, 10)][1]       // they win, dobrada resets
  ]

Tie case (adds to dobrada, no points):
  P(tie) ≈ 0.08
  Tie contribution: P(tie) × ME[a][b][d*2]   // dobrada stacks

Full recursion:
  ME[a][b][d] = P(tie) × ME[a][b][min(d*2, 8)] +
                (1 - P(tie)) × Σ_k P(pts=k | not_tie) × [
                  0.5 × ME[min(a + k*d, 10)][b][1] +
                  0.5 × ME[a][min(b + k*d, 10)][1]
                ]
```

### 5.2 Implementation

```javascript
function buildMETable() {
  const MAX_SCORE = 10; // buffer above 6
  const MAX_DOB = 4;    // dobrada indices: 0=x1, 1=x2, 2=x4, 3=x8

  // ME[a][b][d] where d is dobrada index (0-3)
  const ME = Array.from({length: MAX_SCORE + 1}, () =>
    Array.from({length: MAX_SCORE + 1}, () =>
      new Float64Array(MAX_DOB)
    )
  );

  // Base cases
  for (let b = 0; b <= MAX_SCORE; b++)
    for (let d = 0; d < MAX_DOB; d++) {
      for (let a = 6; a <= MAX_SCORE; a++) ME[a][b][d] = b >= 6 ? 0.5 : 1.0;
      for (let a = 0; a < 6; a++) if (b >= 6) ME[a][b][d] = 0.0;
    }

  const P_TIE = 0.08;
  const POINT_DIST = [{pts:1, prob:0.70}, {pts:2, prob:0.16}, {pts:3, prob:0.10}, {pts:4, prob:0.04}];

  // Iterate until convergence (for tie → dobrada recursion)
  for (let iter = 0; iter < 20; iter++) {
    for (let a = 5; a >= 0; a--) {
      for (let b = 5; b >= 0; b--) {
        for (let d = MAX_DOB - 1; d >= 0; d--) {
          const mult = 1 << d; // 1, 2, 4, 8
          let val = 0;

          // Tie case
          const nextDob = Math.min(d + 1, MAX_DOB - 1);
          val += P_TIE * ME[a][b][nextDob];

          // Win/loss cases
          const nonTieProb = 1 - P_TIE;
          for (const {pts, prob} of POINT_DIST) {
            const scored = pts * mult;
            const aWin = Math.min(a + scored, MAX_SCORE);
            const bWin = Math.min(b + scored, MAX_SCORE);
            val += nonTieProb * prob * 0.5 * ME[aWin][b][0]; // we win, dobrada resets
            val += nonTieProb * prob * 0.5 * ME[a][bWin][0]; // they win, dobrada resets
          }

          ME[a][b][d] = val;
        }
      }
    }
  }

  return ME;
}
```

### 5.3 Risk-Aware Move Selection

At certain match scores, risk tolerance changes:

```
If behind 1-5: prefer high-variance moves (swing for carroca/cruzada)
If ahead 5-1: prefer safe moves (avoid giving opponent 4-pt cruzada)

riskAdjustedEV(m) = E[pts](m) + λ × σ(m)
  where λ = sign(our_score - opp_score) × risk_factor
```

---

## SECTION 6 — Flexibility & Leave Value

### 6.1 Flexibility Score

```javascript
function computeFlexibility(hand, tile, newLE, newRE) {
  const remaining = hand.filter(t => t.id !== tile.id);
  if (remaining.length === 0) return { flex: Infinity, connectivity: 1.0 }; // going out

  // Immediate playability on new ends
  const playableNext = remaining.filter(t =>
    t.left === newLE || t.right === newLE ||
    t.left === newRE || t.right === newRE
  ).length;

  // Suit entropy: how many different numbers can we play?
  const numbers = new Set();
  for (const t of remaining) { numbers.add(t.left); numbers.add(t.right); }

  // Connectivity: fraction of possible end values we can respond to
  const connectivity = numbers.size / 7;

  // Leave value: expected playability over random future board ends
  // P(can play) = 1 - P(can't play on left) × P(can't play on right)
  // P(can play on end e) = 1 if any remaining tile has number e
  let leaveValue = 0;
  for (let e1 = 0; e1 <= 6; e1++) {
    for (let e2 = 0; e2 <= 6; e2++) {
      const canPlayAny = remaining.some(t =>
        t.left === e1 || t.right === e1 || t.left === e2 || t.right === e2
      );
      leaveValue += canPlayAny ? 1 : 0;
    }
  }
  leaveValue /= 49; // normalize to [0, 1]

  return {
    flex: playableNext,
    connectivity,
    leaveValue,
    suitEntropy: -[...numbers].length * Math.log2(1 / numbers.size) // rough
  };
}
```

### 6.2 Two-Turn Lookahead Flexibility

```javascript
function flexibility2Turn(hand, tile, side, lE, rE, knowledge) {
  const remaining = hand.filter(t => t.id !== tile.id);
  const newEnds = computeNewEnds(tile, side, lE, rE);

  // Turn 1: what can I play?
  const t1Playable = remaining.filter(t =>
    canPlay(t, newEnds.newLeft, newEnds.newRight, 1)
  );

  // Turn 2: for each t1 option, what remains playable?
  let t2PlayableSum = 0;
  for (const t1 of t1Playable) {
    for (const t1Side of ['left', 'right']) {
      if (!canPlayOnSide(t1, t1Side === 'left' ? newEnds.newLeft : newEnds.newRight)) continue;
      const t1Ends = computeNewEnds(t1, t1Side, newEnds.newLeft, newEnds.newRight);
      const t2Hand = remaining.filter(t => t.id !== t1.id);
      const t2Playable = t2Hand.filter(t =>
        canPlay(t, t1Ends.newLeft, t1Ends.newRight, 1)
      );
      t2PlayableSum += t2Playable.length;
    }
  }

  return {
    turn1Options: t1Playable.length,
    avgTurn2Options: t1Playable.length > 0 ? t2PlayableSum / t1Playable.length : 0
  };
}
```

---

## SECTION 7 — Control Horizon

### 7.1 Number Control Probability

```javascript
function computeControlMap(hand, knowledge, handSizes) {
  // For each number 0-6, estimate P(we control it as board end)
  const control = new Float64Array(7);

  for (let n = 0; n < 7; n++) {
    // How many of our tiles have this number?
    const ourCount = hand.filter(t => t.left === n || t.right === n).length;

    // How many tiles with this number remain (not played, not ours)?
    const remaining = knowledge.remainingWithNumber(n) - ourCount;

    // How many opponents can play this number?
    // If both opponents lack it (passed), we have full control
    const opp1Has = !knowledge.cantHave[1].has(n);
    const opp3Has = !knowledge.cantHave[3].has(n);
    const oppCanPlay = (opp1Has ? 1 : 0) + (opp3Has ? 1 : 0);

    // Control ≈ our_share / (our_share + opp_share)
    if (ourCount + remaining === 0) {
      control[n] = 0; // number exhausted
    } else {
      const estOppHeld = remaining * (oppCanPlay / 3); // rough
      control[n] = ourCount / (ourCount + estOppHeld + 0.01);
    }
  }

  return control;
}
```

### 7.2 Control Horizon (Expected Turns of Control)

```javascript
function controlHorizon(hand, tile, side, lE, rE, knowledge) {
  const newEnds = computeNewEnds(tile, side, lE, rE);
  const controlMap = computeControlMap(hand.filter(t => t.id !== tile.id), knowledge, null);

  // Expected turns before opponent can flip both ends
  const leftControl = controlMap[newEnds.newLeft] || 0;
  const rightControl = controlMap[newEnds.newRight] || 0;

  // P(opponent flips end) ≈ 1 - control
  // Expected turns = 1 / P(flip) approximately
  const horizon = 1 / (1 - Math.max(leftControl, rightControl) + 0.01);

  return {
    leftControl: controlMap[newEnds.newLeft],
    rightControl: controlMap[newEnds.newRight],
    horizon: Math.min(horizon, 10), // cap
    controlMap
  };
}
```

---

## SECTION 8 — Tile Exhaustion & Lock Risk

### 8.1 Lockout Probability

```javascript
function computeLockRisk(hand, tile, side, lE, rE, knowledge, beliefs) {
  const newEnds = computeNewEnds(tile, side, lE, rE);
  const remaining = hand.filter(t => t.id !== tile.id);

  // P(we pass next turn) = P(no remaining tile matches either end)
  const canPlayNext = remaining.some(t =>
    canPlay(t, newEnds.newLeft, newEnds.newRight, 1)
  );
  const immediateLock = canPlayNext ? 0 : 1;

  // P(locked within 2 cycles) — need opponent play simulation
  // Quick estimate: if we can play next, what if opponent changes ends?
  let lockIn2 = 0;
  if (canPlayNext) {
    // Estimate P(after our play + 3 opponent plays, we can't play)
    // Conservative: check if we have coverage for likely end values
    const ourNumbers = new Set();
    for (const t of remaining) { ourNumbers.add(t.left); ourNumbers.add(t.right); }
    // Each opponent play could change 1 end to any of 7 values
    // P(both ends become numbers we lack) ≈ P(end1 bad) × P(end2 bad)
    const badEndProb = 1 - ourNumbers.size / 7;
    lockIn2 = badEndProb * badEndProb;
  } else {
    lockIn2 = 1; // already locked
  }

  // Tempo: expected passes forced in next cycle
  const expectedPasses = immediateLock * 1 + lockIn2 * (1 - immediateLock) * 0.5;

  return {
    immediateLock,
    lockIn2Cycles: lockIn2,
    expectedPassesForced: expectedPasses,
    tempoGain: canPlayNext ? remaining.filter(t =>
      canPlay(t, newEnds.newLeft, newEnds.newRight, 1)
    ).length / Math.max(remaining.length, 1) : -1
  };
}
```

### 8.2 Suit Exhaustion Warning

```javascript
function suitExhaustionWarnings(knowledge) {
  const warnings = [];
  for (let n = 0; n < 7; n++) {
    const rem = knowledge.remainingWithNumber(n);
    if (rem === 0) warnings.push({ number: n, level: 'dead', text: `${n}s exhausted — will lock if both ends` });
    else if (rem === 1) warnings.push({ number: n, level: 'critical', text: `Only 1 tile with ${n} remaining` });
    else if (rem === 2) warnings.push({ number: n, level: 'warning', text: `${n}s scarce (${rem} left)` });
  }
  return warnings;
}
```

---

## SECTION 9 — Blunder Classification (Match Equity Based)

### 9.1 Classification System

Convert from E[pts] equity loss to Match Equity loss for proper classification:

```javascript
function classifyError(equityLoss, matchEquityLoss) {
  // Primary: use match equity loss (percentage points)
  const meLoss = matchEquityLoss * 100; // convert to percentage

  if (meLoss < 0.5) return { grade: 'perfect', label: 'Accurate', color: '#22c55e' };
  if (meLoss < 1.5) return { grade: 'good', label: 'Inaccuracy', color: '#86efac' };
  if (meLoss < 4.0) return { grade: 'ok', label: 'Mistake', color: '#eab308' };
  if (meLoss < 8.0) return { grade: 'bad', label: 'Blunder', color: '#ef4444' };
  return { grade: 'terrible', label: 'Howler', color: '#dc2626' };

  // Fallback when ME not available — use E[pts] equity loss
  // These thresholds calibrated so that:
  // 0.02 E[pts] ≈ 0.5% ME at score 0-0
  // 0.06 E[pts] ≈ 1.5% ME
  // 0.15 E[pts] ≈ 4% ME
  // 0.30 E[pts] ≈ 8% ME
}
```

### 9.2 Session Error Rate

```javascript
const sessionStats = {
  moves: 0,
  totalMELoss: 0,         // sum of match equity losses
  totalEPLoss: 0,          // sum of E[pts] losses
  grades: { perfect: 0, good: 0, ok: 0, bad: 0, terrible: 0 },
  byPhase: {
    opening: { moves: 0, loss: 0 },   // moves 1-3
    midgame: { moves: 0, loss: 0 },   // moves 4-8
    endgame: { moves: 0, loss: 0 }    // moves 9+
  },
  byNumber: [0,0,0,0,0,0,0],          // EL by number played
  doublesErrors: 0,
  closingErrors: 0
};

function updateSessionStats(move, equityLoss, meLoss, grade, phase, tile) {
  sessionStats.moves++;
  sessionStats.totalMELoss += meLoss;
  sessionStats.totalEPLoss += equityLoss;
  sessionStats.grades[grade]++;
  sessionStats.byPhase[phase].moves++;
  sessionStats.byPhase[phase].loss += meLoss;
  sessionStats.byNumber[tile.left] += equityLoss;
  sessionStats.byNumber[tile.right] += equityLoss;
  if (tile.left === tile.right && grade !== 'perfect') sessionStats.doublesErrors++;
}
```

### 9.3 Leak Finder Rules

```javascript
function findLeaks(stats) {
  const leaks = [];

  // Average error rate
  const avgME = stats.totalMELoss / Math.max(stats.moves, 1) * 100;

  // Phase leaks
  for (const [phase, data] of Object.entries(stats.byPhase)) {
    if (data.moves < 3) continue;
    const phaseAvg = data.loss / data.moves * 100;
    if (phaseAvg > avgME * 1.5) {
      leaks.push({
        type: phase.toUpperCase(),
        severity: phaseAvg > 5 ? 'critical' : 'moderate',
        text: `${phase} avg ME loss ${phaseAvg.toFixed(1)}% — ${(phaseAvg/avgME).toFixed(1)}x your overall rate`,
        fix: phase === 'opening' ? 'Focus on suit control and double management' :
             phase === 'endgame' ? 'Enable MC analysis for closing situations' :
             'Review blocking and partner support plays'
      });
    }
  }

  // Number-specific leaks
  const maxNumLoss = Math.max(...stats.byNumber);
  const worstNum = stats.byNumber.indexOf(maxNumLoss);
  if (maxNumLoss > stats.totalEPLoss * 0.3) {
    leaks.push({
      type: `NUMBER ${worstNum}`,
      severity: 'moderate',
      text: `${(maxNumLoss / stats.totalEPLoss * 100).toFixed(0)}% of your errors involve ${worstNum}s`,
      fix: `Review plays involving number ${worstNum} — possible suit reading gap`
    });
  }

  // Doubles leak
  if (stats.doublesErrors > stats.moves * 0.3) {
    leaks.push({
      type: 'DOUBLES',
      severity: 'critical',
      text: `Doubles in ${(stats.doublesErrors / stats.moves * 100).toFixed(0)}% of errors`,
      fix: 'Play doubles early unless they set up isolated double combo'
    });
  }

  return leaks;
}
```

---

## SECTION 10 — Exploitability & Style Modeling

### 10.1 Opponent Style Parameters

```javascript
class OpponentModel {
  constructor(playerId) {
    this.id = playerId;
    this.profile = {
      aggressiveness: 0.5,    // 0 = conservative, 1 = aggressive
      doubleSpeed: 0.5,       // tendency to play doubles early
      heavyDump: 0.5,         // tendency to dump high-pip tiles
      blockFocus: 0.5,        // tendency to block vs support partner
      suitLoyalty: 0.5        // tendency to keep strong suit connected
    };
    this.history = [];         // recent play decisions for updating
  }

  updateFromPlay(tile, boardEnds, handSize, knowledge) {
    const pips = tile.left + tile.right;
    const isDouble = tile.left === tile.right;

    // Update heavy dump tendency
    if (pips >= 9) this.profile.heavyDump = 0.8 * this.profile.heavyDump + 0.2 * 0.8;
    else if (pips <= 3) this.profile.heavyDump = 0.8 * this.profile.heavyDump + 0.2 * 0.2;

    // Update double speed
    if (isDouble && handSize > 3) this.profile.doubleSpeed = 0.8 * this.profile.doubleSpeed + 0.2 * 0.8;

    this.history.push({ tile, boardEnds, handSize });
  }
}
```

### 10.2 Adjusted Beliefs Using Style

```javascript
function styleAdjustedDealWeight(deal, knowledge, opponentModels) {
  let w = 1.0;
  for (let p = 0; p < 4; p++) {
    if (!opponentModels[p]) continue;
    const model = opponentModels[p];

    // Players with high heavyDump tendency less likely to hold heavy tiles
    for (const t of deal[p]) {
      const pips = t.left + t.right;
      if (model.profile.heavyDump > 0.6 && pips >= 8) w *= 0.7;
      if (model.profile.suitLoyalty > 0.6) {
        const str = knowledge.inferStrength(p);
        if (str[t.left] >= 2 || str[t.right] >= 2) w *= 1.3;
      }
    }
  }
  return w;
}
```

---

## SECTION 11 — UI Additions

### 11.1 New Panels

**Move Analysis Table** (in quiz result and watch mode):
```
┌────────┬──────┬────────┬───────┬──────┬──────┬──────┬────────┐
│ Move   │ Win% │ E[pts] │  ME   │  σ²  │ Risk │ Flex │  EL    │
├────────┼──────┼────────┼───────┼──────┼──────┼──────┼────────┤
│[0|5] R │ 72%  │ +0.84  │ 81.2% │ 1.2  │  5%  │  1   │ BEST   │
│[0|5] L │ 48%  │ +0.22  │ 74.3% │ 2.1  │ 18%  │  0   │ -0.069 │
└────────┴──────┴────────┴───────┴──────┴──────┴──────┴────────┘
```

**Belief Heatmap** (upgrade probability grid):
- 7×7 grid per opponent with gradient colors
- Show P(holds tile) as percentage
- Already partially implemented — enhance with importance-weighted beliefs

**Control Heatmap** (new):
```
Numbers:  0    1    2    3    4    5    6
Control: [██] [░░] [██] [██] [░░] [  ] [░░]
          92%  31%  78%  85%  45%   0%  38%
```

**Entropy Over Time** (new sparkline):
- Show H(opp1), H(opp2), H(partner) over game moves
- Steep drops = high-info moves (passes)

**Error Report** (upgrade Analysis tab):
- Per-move ME loss graph
- Phase breakdown (opening/mid/endgame)
- Leak finder rules

### 11.2 Tab Structure

```
[Reasoning] [Tracker] [Analysis] [Leaks] [Stats] [Guide]
```

New "Leaks" tab:
- Leak finder dashboard
- Number-specific error rates
- Phase error rates
- Doubles error rate
- Trend over sessions

---

## SECTION 12 — Performance

### 12.1 State Hashing

```javascript
function stateHash(snap) {
  // Deterministic hash from game state for MC result caching
  const parts = [
    snap.leftEnd, snap.rightEnd,
    snap.hands[0].map(t => t.id).sort().join(','),
    [...snap.knowledge.played].sort().join(','),
    snap.knowledge.cantHave.map(s => [...s].sort().join(',')).join('|')
  ];
  return parts.join('::');
}
```

### 12.2 MC Result Caching

```javascript
const mcCache = new Map();
const MC_CACHE_MAX = 200;

function cachedMC(snap, player, numSims) {
  const hash = stateHash(snap) + '::' + player;
  if (mcCache.has(hash)) return mcCache.get(hash);

  const result = monteCarloEval(snap, player, numSims);

  if (mcCache.size >= MC_CACHE_MAX) {
    // Evict oldest entry
    const firstKey = mcCache.keys().next().value;
    mcCache.delete(firstKey);
  }
  mcCache.set(hash, result);
  return result;
}
```

### 12.3 Chunked Async Evaluation

Already implemented for Analysis tab. Extend to all heavy computations:

```javascript
async function evaluateAllMovesAsync(snap, player, numSims, onProgress) {
  const playable = snap.hands[player].filter(t =>
    canPlay(t, snap.leftEnd, snap.rightEnd, snap.board.length)
  );
  const results = [];

  for (let i = 0; i < playable.length; i++) {
    const tile = playable[i];
    const sides = getSides(tile, snap.leftEnd, snap.rightEnd, snap.board.length);

    for (const side of sides) {
      const eval = evaluateMove(snap, player, { tile, side }, numSims);
      results.push({ tile, side, ...eval });
    }

    onProgress((i + 1) / playable.length);
    await new Promise(r => setTimeout(r, 5)); // yield to UI
  }

  results.sort((a, b) => b.expectedPoints - a.expectedPoints);
  return results;
}
```

### 12.4 WebWorker (Future Phase)

When Blob-based WebWorker becomes necessary (>500 sims):

```javascript
// Create worker from inline code
function createMCWorker() {
  const workerCode = `
    // Include: ALL_TILES, Knowledge, canPlay, smartAI, generateConsistentDeal,
    //          simulateFromPosition, monteCarloEval
    // (extracted as string literals)

    self.onmessage = function(e) {
      const { snap, player, numSims, taskId } = e.data;
      // Reconstruct Knowledge from serialized data
      const result = monteCarloEval(snap, player, numSims);
      self.postMessage({ taskId, result });
    };
  `;
  const blob = new Blob([workerCode], { type: 'application/javascript' });
  return new Worker(URL.createObjectURL(blob));
}
```

Deferred to Phase 4 — current async chunking is sufficient for ≤200 sims.

---

## SECTION 13 — Testing Plan

### 13.1 Deterministic Regression

```javascript
function seedRandom(seed) {
  // Mulberry32 PRNG for reproducible tests
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// Known position tests
const TEST_POSITIONS = [
  {
    name: 'Obvious close: 1 tile left, play to win',
    hand: [{left:0, right:5, id:'0-5'}],
    lE: 5, rE: 3, bLen: 20,
    expected: { tile: '0-5', side: 'left', reason: 'instant win' }
  },
  {
    name: 'Setup close: 2 tiles, one enables going out',
    hand: [{left:0, right:5, id:'0-5'}, {left:0, right:2, id:'0-2'}],
    lE: 0, rE: 5, bLen: 18,
    expected: { tile: '0-5', bestSide: 'right' } // leaves 0,0 for 0-2
  },
  // ... more known positions
];
```

### 13.2 Sampling Uniformity

```javascript
function testSamplingBias(nTrials = 10000) {
  // Create scenario with known uniform distribution
  // Empty knowledge → all deals equally likely
  const knowledge = new Knowledge();
  const myHand = ALL_TILES.slice(0, 6);
  const handSizes = [6, 6, 6, 6];

  const tileDist = {};
  for (let i = 0; i < nTrials; i++) {
    const deal = generateConsistentDeal(myHand, handSizes, knowledge, 0);
    for (let p = 1; p <= 3; p++) {
      for (const t of deal[p]) {
        const key = `${p}:${t.id}`;
        tileDist[key] = (tileDist[key] || 0) + 1;
      }
    }
  }

  // Each unplayed tile should appear equally often for each eligible player
  // Expected: 6/18 × nTrials = nTrials/3 for 3 players with equal hand sizes
  const expected = nTrials / 3;
  let maxDeviation = 0;
  for (const [key, count] of Object.entries(tileDist)) {
    const dev = Math.abs(count - expected) / expected;
    maxDeviation = Math.max(maxDeviation, dev);
  }

  console.assert(maxDeviation < 0.05, `Sampling bias detected: ${(maxDeviation*100).toFixed(1)}% max deviation`);
}
```

### 13.3 Self-Play Elo Benchmark

```javascript
function selfPlayBenchmark(agentA, agentB, nGames = 1000) {
  let winsA = 0, winsB = 0, ties = 0;
  let ptsA = 0, ptsB = 0;

  for (let g = 0; g < nGames; g++) {
    // Teams: A = players 0,2; B = players 1,3
    const agents = [agentA, agentB, agentA, agentB];
    const result = playGame(agents);
    if (result.winnerTeam === 0) { winsA++; ptsA += result.points; }
    else if (result.winnerTeam === 1) { winsB++; ptsB += result.points; }
    else ties++;
  }

  const winRate = winsA / (winsA + winsB);
  const elo = -400 * Math.log10(1 / winRate - 1);

  return { winsA, winsB, ties, ptsA, ptsB, winRate, eloDiff: elo.toFixed(0) };
}

// Compare: heuristicOnly vs heuristic+closingBonus vs heuristic+MC
```

---

## PHASED ROADMAP

### Phase 1 — Immediate (Done + This Session)
- [x] Rejection sampling (unbiased)
- [x] E[pts] evaluation with full scoring
- [x] Match equity table (basic)
- [x] Equity loss grading
- [x] Closing bonus in heuristic
- [x] MC-assisted endgame generation
- [x] Analysis tab with leak finder
- [x] Gradient probability tracker

### Phase 2 — Belief Model + Enhanced ME (Next)
- [ ] BeliefState class with importance-weighted MC
- [ ] Full ME table with dobrada dimension ME[a][b][d]
- [ ] Flexibility & leave value computation
- [ ] Match-equity-based error classification
- [ ] Entropy tracking per move
- [ ] Enhanced leak finder (by number, by phase)
- [ ] State hashing + MC result caching
- **Estimated complexity**: ~400 LOC
- **Performance impact**: +50ms per belief computation

### Phase 3 — Control + Exhaustion + UI
- [ ] Control horizon computation
- [ ] Control heatmap in UI
- [ ] Lock risk computation
- [ ] Suit exhaustion warnings
- [ ] Move analysis table in quiz results
- [ ] Entropy sparkline
- [ ] "Leaks" tab with full dashboard
- [ ] Adaptive MC (early stopping)
- **Estimated complexity**: ~500 LOC
- **Performance impact**: minimal (lightweight computations)

### Phase 4 — Opponent Modeling + Advanced
- [ ] OpponentModel class
- [ ] Style-adjusted beliefs
- [ ] Risk-aware move selection at extreme match scores
- [ ] Gibbs sampling as intermediate sampling tier
- [ ] Sampling uniformity validation (debug tool)
- [ ] Self-play Elo benchmark
- [ ] WebWorker extraction (if needed for >500 sims)
- [ ] Deterministic regression test suite
- **Estimated complexity**: ~600 LOC
- **Performance impact**: varies; opponent modeling is lightweight, WebWorker is structural

### Priority Order (Impact / Effort)

| Upgrade | Impact | Effort | Priority |
|---------|--------|--------|----------|
| ME with dobrada | High | Low | 1 |
| BeliefState class | High | Medium | 2 |
| Flexibility/leave value | High | Low | 3 |
| ME-based error classification | Medium | Low | 4 |
| MC caching | Medium | Low | 5 |
| Control horizon | Medium | Medium | 6 |
| Lock risk | Medium | Low | 7 |
| Entropy tracking | Low | Medium | 8 |
| Opponent modeling | Low | High | 9 |
| WebWorker | Low | High | 10 |
| Gibbs sampling | Low | High | 11 |

---

*Document generated for `berny-the-blade/pernambuco-domino` simulator.html*
*Architecture targets: single-file browser JS, ≤200ms per evaluation, no external dependencies*
