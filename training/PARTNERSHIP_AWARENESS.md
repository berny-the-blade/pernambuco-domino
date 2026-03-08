# PARTNERSHIP_AWARENESS.md
# Partner-communication / signaling awareness improvements
# Source: Bernie analysis 2026-03-08

## Why it matters
Most domino AI mistakes at this stage are not "search too shallow" — they are
"engine doesn't understand what the move is communicating."

Example:
- opener: 55, opponent: 51, partner to move
- partner should continue the 1-side, NOT reconnect to 5
- this is: preserving pressure + not stealing partner's information channel

Estimated value: +20–30 Elo + big jump in "looks like a strong dupla"

---

## Phase 1 — Fast win: new-end preservation heuristic (Commit B)

### Rule
If:
- opponent's last move opened a new end x
- current ends are (old, x)
- a legal move exists that keeps x alive

Then: apply soft penalty to moves that remove x and reconnect to old side,
UNLESS immediate tactical gain exists:
- guaranteed block
- immediate go-out
- clear pip swing threshold
- endgame solver says it's better

### Suggested weight
Start: -8 to -15 equivalent smartAI score penalty (not -30+, let suite calibrate)

---

## Phase 2 — Partnership tactical suite (Commit A)

File: training/tests/partnership_suite.json
Script: training/tests/test_partnership_suite.py

### Categories (30-50 positions)
1. Don't steal partner's suit
2. Preserve newly opened end  ← your example goes here
3. Keep pressure on opponent weak end
4. Probe for information
5. Don't collapse signaling channel unless tactical gain
6. Damage-control when partner misplays

### Metrics
- % positions: engine chooses preferred move
- % positions: engine chooses discouraged move

---

## Phase 3 — Auxiliary NN head: void prediction (Commit D)

### Option A: void prediction (recommended first)
Predict for each other player: P(void in pip 0..6)
Output: 3 players × 7 pips = 21-dim auxiliary head
Loss: BCE per output
Labels: from self-play logs (full hidden state known during generation)

### Option B: partner-compatibility target
Predict: P(partner can continue each current end)
More custom, more directly aligned to game conventions.

### Why it helps
Teaches trunk: hidden-info inference + communication through actions.
Improves both policy and value.

---

## Phase 4 — Other-Play robustness (longer-term)
- Randomize seat identities, equivalent tile encodings, suit-label symmetries
- Prevents brittleness in ad-hoc human partnership play

---

## Commit Plan

| Commit | Change | Pass gate |
|--------|--------|-----------|
| A | partnership_suite.json + test script | suite runs, baseline recorded |
| B | new-end preservation heuristic in smartAI | suite improves, no >10 Elo regression |
| C | partnership diagnostics counter | behavior change confirmed in logs |
| D | auxiliary void-prediction head (21 outputs) | aux loss decreases, no policy/value regression |
| E | benchmark vs heuristic partner + ad-hoc partner | validates beyond self-play |

---

## Concrete feature pseudocode (new-end preservation)

```
For each candidate move m:
  if lastMoveOpenedNewEnd(x) and currentEnds = (old, x):
    if moveRemovesEnd(m, x) and moveReconnectsTo(m, old):
      if hasAlternativeThatKeepsXAlive(legalMoves):
        if not immeditateTacticalGain(m):
          score[m] -= NEW_END_PRESERVATION_PENALTY  // start: 8-15
```

---

## Benchmark additions (Commit E)
Test vs:
- heuristic-only partner
- heuristic-only opponents  
- self-play clone
- "ad-hoc" partner (slight randomness)

Tells you: improvement is real-world, not just self-play artifact.
