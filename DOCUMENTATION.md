# Domino Pernambucano — Strategy Trainer & Simulator

## Overview

A comprehensive single-file web application for learning and mastering **Domino Pernambucano** (Brazilian partnership domino). Built as a training tool that combines AI game simulation, Monte Carlo evaluation, interactive quizzes, and deep strategic analysis to help players improve from beginner to expert level.

**Live file**: `simulator.html` (~7200 LOC) — open directly in any modern browser, no build step or server required.

---

## Table of Contents

1. [Game Rules](#game-rules)
2. [Game Modes](#game-modes)
3. [Analysis Tabs](#analysis-tabs)
4. [AI Engine](#ai-engine)
5. [Keyboard Shortcuts](#keyboard-shortcuts)
6. [Data & Persistence](#data--persistence)
7. [Technical Architecture](#technical-architecture)

---

## Game Rules

### Setup
- **28 tiles** (standard double-six set: [0|0] through [6|6])
- **4 players** in fixed partnerships: P0+P2 (your team) vs P1+P3 (opponents)
- **24 tiles dealt** (6 per player), **4 removed** face-down (unknown to all)
- Players sit: You (bottom), Opp Left, Partner (top), Opp Right

### Play
- First game: player with the **highest double** opens (auto-played)
- Subsequent games: **winning team chooses** who opens with free tile choice
- Each turn: play a tile matching either board end, or **pass** if you can't
- Board forms a snake — two open ends tracked at all times

### Scoring (per round)

| Finish Type | Points | Condition |
|------------|--------|-----------|
| **Normal** | 1 | Go out with a regular tile |
| **Carroca** | 2 | Go out with a **double** |
| **La-e-lo** | 3 | Go out with a tile that matches **both** board ends |
| **Cruzada** | 4 | Go out with a **double** that matches **both** ends |
| **Blocked** | 1 | All 4 players pass consecutively — lowest pip total wins |
| **Tie** | 0 | Blocked game with tied pip totals — triggers **dobrada** |

### Match System
- **First to 6 points** wins the match
- **Dobrada** (doubling): after a tied block, the next game's points are multiplied (×2, then ×4, ×8)
- **Match Equity Table**: 3D lookup `ME3D[score1][score2][dobradaLevel]` computing win probability from any score state

---

## Game Modes

### 1. Watch AI Play (green button)
Step through complete AI-vs-AI games with full transparency.

- All 4 hands visible (face-up)
- AI reasoning shown for every move in the Reasoning tab
- Manual navigation: arrow keys or Next/Back buttons
- **Auto-play** with speed control (Slow 2s / Med 1s / Fast 0.5s)
- **MC toggle** (MC: ON/OFF button): enables Monte Carlo deep analysis overlay showing expected points per option
- Best mode for learning — watch optimal play and understand why

### 2. Quiz Mode (purple button)
Test your skills against the AI's optimal play.

- AI plays P1, P2, P3 automatically
- When it's **your turn (P0)**, playable tiles glow yellow — click to play
- If a tile can go on both ends, a **side picker** overlay appears
- After each decision, you're **graded**:
  - **Perfect** (green): Matched AI's best choice or within 0.02 EP
  - **Inaccuracy** (light green): Within 0.15 EP of best
  - **Mistake** (yellow): Within 0.40 EP of best
  - **Blunder** (red): More than 0.40 EP loss
- Equity loss shown for each decision
- Elo rating tracked across games

### 3. Play vs AI (yellow button)
Play real games with configurable AI difficulty.

- **Difficulty picker** screen with separate controls for:
  - **Opponents** (P1, P3): Beginner (0.25) / Intermediate (0.50) / Advanced (0.75) / Expert (1.00)
  - **Partner** (P2): Same scale
- **AI delay**: ~600ms per AI move for natural pacing
- Full match scoring (first to 6 points, dobrada on ties)
- **Hint system**: click the hint button to see AI's recommendation
- **Post-game review**: after game ends, use arrow keys to step through all moves
- **Match winner overlay** when a team reaches 6 points
- Game automatically saved to History for later review

### 4. Strategy Puzzles (amber button)
12 hand-crafted positions testing specific skills.

**6 categories, 2 puzzles each:**

| Category | Puzzles | Skills Tested |
|----------|---------|---------------|
| Blocking | Seal the Exit (★★★), Double Lock (★★★★) | Using void info to trap opponents |
| Feeding | Partner's Suit (★★), The Sacrifice (★★★) | Supporting partner's strong suits |
| Doubles | Isolated Double Emergency (★★), Double Timing (★★★) | Managing dangerous doubles |
| Endgame | La-e-Lo Setup (★★★★), Pip Count (★★★) | Maximizing points, blocked game prep |
| Reading | Count the Tiles (★★★), Pressure the Weakness (★★★★) | Board counting, opponent inference |
| Tempo | Suit Dominance (★★★), Board Width (★★★★) | Controlling the pace of play |

- Hint button reveals the best move
- Graded with MC evaluation (500 sims per tile)
- Completion tracked persistently

### 5. Opening Quiz (cyan button)
Practice the opening decision: should you or your partner open?

- Shows your 6-tile hand
- Choose: "I open" or "Partner opens"
- System runs **40 MC rollouts** per scenario to determine optimal choice
- **Hand analysis** based on 1M-game empirical study:
  - **Suit depth** (#1 predictor): depth 5 = +0.20 EP, depth 4 = +0.05 EP
  - **Doubles count** (#2): 1 double = +0.04 EP, 0 doubles = −0.09 EP
  - **Suit coverage** (#3): 4 suits only = −0.15 EP
- Streak tracking and stats in localStorage

### 6. Deep Study (indigo button)
The most comprehensive analysis mode — MC evaluation + tile probabilities + strategic commentary for **every player's every move**.

- Generates a watch-mode game with **all hands visible**
- Step through with arrow keys — each move auto-analyzed:
  - **MC Evaluation** (1000 rollouts per option): all legal moves ranked by Expected Points
  - **Play Grade**: OPTIMAL / GOOD / MISTAKE / BLUNDER with EP loss
  - **Tile Probability Matrix**: Bayesian inference showing probability each unknown tile is held by P1, P2, P3, or removed
  - **Deduction Bullets**: what P0 can logically infer from this move (voids, avoidance patterns, scarcity)
  - **Strategic Notes**: board state, suit scarcity, partner coordination, critical decision analysis
  - **Heuristic Reasoning**: every AI reason is clickable, expanding to show:
    - **What**: plain English explanation of the concept
    - **Math**: exact scoring formula with values from this position
    - **Why it matters**: strategic reasoning for improving your game
- MC option list shows heuristic score `(H:XX)` per tile — click any option to see its reason breakdown

---

## Analysis Tabs

The right panel has 8 tabs, available during any game:

### Reasoning
Current move's AI reasoning: which tile was chosen, why, and all alternatives ranked by heuristic score. When MC is enabled, options are re-ranked by expected points with equity loss shown.

### Tracker
Visual probability grid showing which tiles each opponent could hold. Based on void detection (passes) and Monte Carlo sampling. Shows a 7×7 number matrix with color-coded probability for each player.

### Coach
Deduction notebook with educational commentary:
- **"Before You Pick" checklist** during quiz turns
- **Certainty-tagged bullets**: CERTAIN (from passes), LIKELY (from avoidance patterns)
- **Per-player cards**: void numbers, avoidance evidence, risk flags
- **Play history** timeline

### Analysis
Post-game performance report (auto-computed with MC):
- **Average Equity Loss** per move with game rating
- **Move-by-move list**: grade, tile played vs best tile, EP loss
- **Leak finder**: detects patterns (opening weakness, endgame errors, double management issues)
- Click any move to navigate to that position

### Study
Deep Study analysis — see [Deep Study mode](#6-deep-study-indigo-button) above.

### Stats
Lifetime statistics:
- Perfect play rate and accuracy breakdown
- Average equity loss with skill level rating
- Match equity display
- Recent decision history

### History
Game database (last 100 games in localStorage):
- **Performance overview**: avg EL, win/loss record, trend (last 5 vs previous 5)
- **Weakness areas**: opening EL, endgame EL, doubles error rate, blunder count
- **EL trend chart**: sparkline showing improvement over time
- **Game list**: click any game to see move-by-move detail or replay it
- **Clear history** button

### Guide
Reference card for Pernambuco rules, scoring, and strategy principles.

---

## AI Engine

### Heuristic Evaluation (`smartAI`)

Every legal move is scored by combining these weighted factors:

| Factor | Points | Description |
|--------|--------|-------------|
| **Suit control** | +15 per tile | Count of remaining hand tiles matching new board ends |
| **Blocking** | +25 per opponent | Both ends = opponent's confirmed void numbers |
| **Partner support** | +8 per play count | Partner previously played tiles with this number |
| **Partner void** | −10 | Opening a number partner is confirmed void in |
| **Pip weight** | +2 per pip | Heavier tiles score more (tiebreaker for blocked games) |
| **Double clearance** | +12 | Playing any double (they're inflexible, risky to hold) |
| **Isolated double rescue** | +20 | Creates board end matching a double with no support tiles |
| **Board counting** | (2−N)×6 | N = opponent tiles with that number. Low N = soft block |
| **Go out** | +200 + pts×30 | Instant win, bonus scales with point type |
| **Near-close setup** | +80 | Last tile is playable on resulting board ends |
| **Stranding** | −30 | Last tile can't play on resulting ends |
| **2-tile coverage** | +25 / −15 | Both or neither remaining tile matches new ends |

### Monte Carlo Simulation

When enabled, MC simulation plays out hundreds/thousands of random complete games from the current position:

1. Fix the known hand; randomly deal remaining tiles to other players (respecting known voids)
2. Play game to completion using `smartAI` for all players
3. Record outcome (win/loss, points, type)
4. Repeat N times, compute statistics

**Simulation counts by mode:**

| Mode | Sims per tile | Purpose |
|------|--------------|---------|
| Watch (MC:ON) | 120 | Real-time analysis |
| Quiz | 100 | Grading player moves |
| Puzzles | 500 | High confidence for curated positions |
| Analysis tab | 60 | Many moves, moderate quality |
| Deep Study | 1000 | Maximum accuracy for learning |
| Game generation | 30 | Endgame MC override (≤3 tiles) |

**MC output per option:**
- Expected Points (EP): average signed points across all sims
- Win Rate: fraction of sims where your team wins
- Block Rate: fraction ending in blocked/tied games
- Outcome distribution: normal / carroca / la-e-lo / cruzada / blocked / tie counts
- Variance, StdDev, CI95, CVaR5% (tail risk)

### Difficulty System

For Play vs AI mode, the `selectMoveByDifficulty()` function uses **softmax temperature** selection:

```
temperature = max(0.05, (1 - difficulty) × 2.5)
weight[i] = exp((score[i] - maxScore) / (temperature × 25 + 1))
```

- **Expert (1.0)**: Always picks best move (temperature ≈ 0)
- **Advanced (0.75)**: Strong play, occasional 2nd-best picks
- **Intermediate (0.50)**: Weighted random, makes real mistakes
- **Beginner (0.25)**: Nearly uniform random from legal moves

### Knowledge Tracking

The `Knowledge` class tracks hard constraints learned during play:

- **cantHave[player]**: Set of numbers a player definitely doesn't have (from passes)
- **played**: Set of all tile IDs that have been played
- **playsBy[player]**: Array of tiles each player has shown
- **passedOn[player]**: Record of which ends a player passed on
- **inferStrength(player)**: Suit frequency array showing how many tiles with each number a player has played
- **remainingWithNumber(n)**: Count of unplayed tiles containing number n
- **possibleTilesFor(player)**: All tiles a player could potentially hold

### Deduction Notebook

The `DeductionNotebook` class provides narrative analysis:

- Records plays and passes with context
- Detects **avoidance patterns** (player avoids playing a number 2+ times)
- Tracks **suit scarcity** (numbers becoming exhausted)
- Generates **certainty-tagged bullets** (CERTAIN from passes, LIKELY from avoidance)
- Produces **"Think About" checklists** for quiz mode

---

## Keyboard Shortcuts

| Key | Action | Available In |
|-----|--------|-------------|
| → or Space | Next step | Watch, Quiz (when not waiting) |
| ← | Previous step | Watch, Quiz, Play (post-game) |
| N | New game | All modes |
| P | Toggle auto-play | Watch mode |

---

## Data & Persistence

All data is stored in browser `localStorage` (survives page refresh, cleared with browser data).

| Key | Contents | Max Size |
|-----|----------|----------|
| `domino-trainer-stats` | Quiz accuracy stats: total moves, perfect/good/ok/bad counts, last 30 decisions | ~5KB |
| `domino_elo` | Elo rating for quiz mode | ~1KB |
| `domino-game-history` | Last 100 Play vs AI games with outcomes, moves, and MC analysis | ~200KB |
| `domino-puzzle-completion` | Puzzle grades per puzzle ID | ~1KB |
| `domino-opening-quiz-stats` | Opening quiz correct/total/streak/history | ~5KB |

### Export / Import
- **Export**: Downloads current game as `.jsonl` file (all snapshots, reasoning, MC results)
- **Import**: Upload a `.jsonl` file to replay any exported game in watch mode

---

## Technical Architecture

### Stack
- **Vanilla JavaScript** — no framework, no dependencies, no build step
- **Tailwind CSS** via CDN — utility-first styling
- **Single HTML file** — entire application in `simulator.html`
- **SplitMix64 PRNG** — seedable random number generator for reproducible games

### File Structure
```
pernambuco-domino-repo/
├── simulator.html          # Main application (~7200 LOC)
├── opening_study.js        # Standalone Node.js 1M-game empirical study
├── SOLVER_ARCHITECTURE.md  # Design doc for future solver (not implemented)
├── SOLVER_SPEC_V2.md       # Solver specification v2
├── DOCUMENTATION.md        # This file
└── .git/
```

### Key Code Sections (simulator.html)

| Section | Lines | Description |
|---------|-------|-------------|
| CSS & Styles | 1-78 | Tile rendering, animations, UI classes |
| Dot Patterns & Tile HTML | 84-130 | Domino visual rendering |
| Deck & Utility Functions | 130-440 | Tile creation, shuffling, board logic |
| DeductionNotebook | 442-700 | Narrative deduction tracking |
| BeliefModel & Sampler | 700-1720 | Probabilistic hand inference |
| Knowledge Class | 1728-1782 | Hard constraint tracking |
| smartAI Heuristic | 1785-1941 | Core AI evaluation (all 12 heuristic factors) |
| Difficulty System | 1943-1967 | Softmax temperature selection |
| Hand Strength | 1969-2015 | Hand quality evaluation meter |
| simulateFullRound | 2015-2174 | Complete game simulation with snapshots |
| generateGame | 2174-2208 | Game creation (deal + simulate) |
| Live Play Engine | 2210-2700 | Incremental play mode (human + AI) |
| generateConsistentDeal | 2705-2826 | MC sampling with constraint satisfaction |
| computeAllTileProbs | 2828-2860 | Bayesian tile probability matrix |
| Study Engine | 2861-3085 | Deep Study: deductions, strategy, MC per step |
| simulateFromPosition | 3110-3170 | Fast game simulation for MC |
| monteCarloEval | 3180-3310 | Full MC evaluation with caching |
| State Variables | 3310-3430 | All application state |
| render() | 4010-4180 | Main render function |
| Player Areas | 4313-4660 | Hand display for all 4 players |
| Board Rendering | 4660-4790 | Snake board layout |
| Reasoning Panel | 4790-4860 | Heuristic reasoning display |
| Coach Panel | 5300-5400 | Deduction notebook UI |
| Analysis Panel | 5580-5710 | Post-game MC analysis |
| getReasonExplanation | 5712-5925 | Deep explanations for all 17 heuristics |
| Study Panel | 5925-6120 | Deep Study tab UI |
| Stats Panel | 6120-6280 | Lifetime statistics |
| History Panel | 6280-6430 | Game database UI |
| Guide Panel | 6430-6500 | Rules reference |
| Opening Quiz | 6500-6600 | Opening decision quiz |
| Start Screen | 6600-6660 | Mode selection with feature cards |
| Controls & Navigation | 6660-6920 | Step/play/mode switching |
| Keyboard Handler | 6920-6955 | Shortcut bindings |

### Performance
- Snapshot-based architecture: each move stores a complete game state
- MC cache (`mcCache`): avoids re-computing identical positions
- Study cache (`studyCache`): stores per-step analysis results
- Async computation with `setTimeout` yielding to keep UI responsive
- Typical game: ~24 snapshots, renders instantly
- Deep Study: ~3-8 seconds per step (1000 MC sims)

### opening_study.js
Standalone Node.js script that extracts the core game engine and runs 1,000,000 simulated games to empirically determine which hand features predict opening advantage. Key findings:
- **Suit depth** is the #1 predictor (depth 5: +0.20 EP)
- **1 double** is optimal (+0.04 EP), 0 doubles is bad (−0.09 EP)
- **4 suit coverage** is bad (−0.15 EP)
- **Connectivity and pip count** are NOT significant
- Overall opening advantage is negligible (−0.003 EP) — it's a coin flip on average

---

## Version History

| Commit | Feature |
|--------|---------|
| Match winner overlay | Celebration screen when team reaches 6 points |
| Game history database | localStorage storage for 100 games + aggregate learning |
| Opening Decision Quiz | MC-validated quiz for open/pass decisions |
| 1M-game empirical study | Data-backed hand analysis replacing heuristic guesses |
| Bold categories + EP formatting | Improved hand analysis display |
| Deep Study mode | MC eval + tile probabilities + strategic analysis for every move |
| Hands visible in Study | All tiles face-up when Study tab active |
| 1000 rollouts | Increased MC accuracy for Deep Study |
| Deep heuristic explanations | Click-to-expand What/Math/Why for all 17 AI reasons |
| Pro Analytics tab | OPP/PPP flow, trap radar, go-out threats |
| BeliefModel & EntropyTracker | Probabilistic inference engine |
| OpponentStyle profiling | EMA-based behavioral labels |
| ISMCTS search | Information Set MCTS with UCB1 and progressive widening |
| ME3D match equity table | 3D DP table with dobrada dimension |
| Match equity in search | `_rolloutToMEReward` integrating ME into MC/ISMCTS |
| 40 curated puzzles | 12 original + 28 audit positions across 8 categories |
| LeakAggregator | Persistent weakness tracking across sessions |
| Coach overlay | Real-time strategic narration during play |
| Internationalization | Full EN/PT bilingual support (~221 keys) |

---

## Advanced AI Reference

### `smartAI` — Complete Scoring Table

**Signature**: `smartAI(hand, lE, rE, bLen, player, knowledge, returnAll = false)`

Phase multiplier (`deadMul`): ≤4 tiles → 0.4, 5–14 → 1.0, >14 → 1.4.

Opening (bLen=0): `sc[left]*10 + sc[right]*10 + (double?15:0) + pips*2`

**All ~35 mid-game factors:**

| Factor | Points | Condition |
|--------|--------|-----------|
| Suit control | `+myCount × 15` | Tiles matching new/other end |
| Block opp1 (next) | `+35` | Both ends in opp1's void set |
| Block opp2 | `+25` | Both ends in opp2's void set |
| Opp1 blocker style | `+8` | opp1 blockingFocus > 0.6, void on newEnd |
| Opp2 blocker style | `+5` | Same for opp2 |
| Partner affinity | `+pAff × 8` | Partner played that suit |
| Partner void newEnd | `−10` | Partner confirmed void |
| Partner blocked both | `−20` | Partner void on both ends |
| Partner good coverage | `+8` | Both ends have 3+ partner tiles |
| Few partner tiles | `−8` | pTilesNew=0, partner not void |
| Pip weight | `+round(pips × 2 × deadMul)` | Always |
| Play double early | `+12` | Tile is a double |
| Isolated double setup | `+20` | Creates end matching isolated double |
| Board count gradient | `+(2−oppCouldHave) × 6` | +12 to −12 range |
| Dead end penalty | `−round(35 × deadMul)` | Creates single dead end |
| Deliberate lock win | `+round(40 × deadMul)` | Both dead, our pips ≤ opp−2 |
| Both dead, bad pips | `−round(60 × deadMul)` | Both dead, pips unfavorable |
| Chicote: we hold | `+round(25 × deadMul)` | Self holds last tile of number |
| Chicote self+opp void | `+12` | Lock threat bonus |
| Chicote: partner | `+round(15 × deadMul)` | Partner holds chicote |
| Chicote: opponent | `−round(25 × deadMul)` | Opponent holds chicote |
| Chicote: dorme | `−12` | Chicote in removed tiles |
| Chicote: unknown | `−round(15 × oppProb)` | Proportional to opp probability |
| Near-dead, not ours | `−12` | 1 remaining, we don't hold chicote |
| Lock favors us | `+20` | Few tiles, bLen≥8, our pips < opp−3 |
| Lock hurts us | `−15` | Same, our pips > opp+5 |
| Suit exhaustion | `+8 + (3−rem)×5` | 1–3 remaining, we hold ≥ rem |
| True monopoly | `+round(20 × deadMul)` | Hold ALL remaining of a number |
| Near-monopoly | `+round((weHold/rem) × 12 × deadMul)` | Hold 2+ of last 3–4 |
| Info hiding: exposed | `−8` | bLen 1–6, strongest suit exposed |
| Info hiding: diversify | `+5` | Low suit count, different end |
| Partner close sacrifice | `+25` | Partner 1 tile, their suit on newEnd |
| Partner close suit | `+10/+15` | Partner 1–2 tiles, keep their suit |
| Match point pips | `+round(pips × 1.5)` | Our score ≥ 5 |
| Match point options | `+8` | Our score ≥ 5, suit count ≥ 2 |
| Behind, play double | `+8` | Double, opp≥5, our<4 |
| Point denial (normal) | `−8` | Feeds opp with 1 tile left |
| Point denial (cruzada) | `−15` | Same + tile is double |
| Go out | `+200 + pts×30` | Last tile played |
| Near-close plays | `+80` | 1 tile left, can play |
| Strands last tile | `−30` | 1 tile left, can't play |
| 2 tiles: both play | `+25` | Both remaining match ends |
| 2 tiles: neither | `−15` | Neither matches |
| 2-ply: opp goes out | `−25` | Opp 1 tile can go out (≤3 hand, bLen≥10) |
| 2-ply: opp 2 both play | `−10` | Opp 2 tiles both playable |

### `fastAI` — Lightweight Rollout Policy

Used by MC/ISMCTS rollouts. No chicote/lock/match awareness.

| Factor | Points | Condition |
|--------|--------|-----------|
| Suit control | `+mc × 12` | Matching tiles |
| Block opp1 | `+25` | Both ends voided |
| Block opp2 | `+18` | Both ends voided |
| Partner void | `−15` | Partner void both ends |
| Pip weight | `+(left+right) × 2` | Always |
| Double clearing | `+10` | Is double |
| Near-close plays | `+60` | 2 tiles, last can play |
| Near-close stranded | `−20` | 2 tiles, last can't |

### `monteCarloEval` Parameters

| Parameter | Value |
|-----------|-------|
| Default sims | 80 |
| Batch size | 20 |
| Min before early stop | 40 |
| Early stopping | 90% CI: `diffMean > 1.65 × StdErr && > 0.05` |
| Cache | LRUCache(300), BitSet hex keys |

### `ismctsEval` Parameters

| Parameter | Value |
|-----------|-------|
| Iterations | 300 |
| Time limit | 150ms |
| Max nodes | 5000 |
| UCB1 C | 1.41 |
| Widening | `children < ceil(sqrt(visits+1))` |

### MC Simulation Counts by Context

| Context | Sims |
|---------|------|
| Expert AI (Play) | 500 |
| Coach overlay | 500 |
| Deep Study | 1000 |
| Pro Analytics | 50 |
| Default MC | 80 |
| Puzzle grading | 500 |
| Watch MC:ON | 120 |
| Quiz grading | 100 |
| Analysis tab | 60 |
| Endgame override | 30 |
| Opening quiz | 40 |

---

## Knowledge & Belief Systems

### `Knowledge` Class
- `cantHave[p]`: Set of voided numbers (from passes)
- `played`: Set of played tile IDs
- `playsBy[p]`: tiles played by each player
- `_remainingCount[7]`: O(1) tiles-per-number lookup
- Methods: `recordPlay`, `recordPass`, `inferStrength(p)` (7-element array), `remainingWithNumber(n)`, `possibleTilesFor(p)`, `chicoteFor(n, hand)` → `{tile, holder, confidence}`

### `BeliefModel` Class
- `marginals[p]`: tileId → probability
- `avoidanceCount[4][7]`: avoidance tracking
- Methods: `recordPass`, `recordPlay`, `getTileMarginal`, `getNumberStrength`, `getEntropy` (Shannon), `updateMarginals(deals, weights)`, `syncFromKnowledge`

### `OpponentStyle` Class (EMA α=0.3, P1+P3 only)
5 metrics: `heavyDump`, `doubleSpeed`, `blockingFocus`, `suitLoyalty`, `aggressiveness`
Labels: Heavy Dumper (>0.65), Blocker (>0.65), Aggressive (>0.65+doubles>0.55), Suit Player (>0.7)

### `EntropyTracker`
Records Shannon entropy snapshots. Methods: `infoGain(move)`, `totalInfoGain()`, `getTimeline()`

### `LeakAggregator` (6 types)
`BLOCKING_MISS`, `FEED_ERROR`, `DOUBLE_TIMING`, `DEAD_END`, `PIP_WASTE`, `PARTNER_IGNORE`
Persisted in localStorage `domino-leak-stats`.

---

## Match Equity System

### ME3D Table
- Dimensions: 11×11×4 (score1 × score2 × dobrada level)
- `POINT_DIST`: 1pt=70%, 2pt=16%, 3pt=10%, 4pt=4%
- `TIE_PROB`: 3%
- `DOB_VALUES`: [1, 2, 4, 8]
- Fill: bottom-up DP, ties at max dobrada solved analytically

### `_rolloutToMEReward(winnerTeam, points, myTeam)`
Returns ME delta: `newME − currentME`. Positive = good for us.
`useME` gate: active when `matchScore[0]>0 || matchScore[1]>0 || scoreMultiplier>1`.

---

## Data Structures

- **`BitSet`**: Dual 32-bit unsigned ints. Methods: has/set/clear/or/and/xor/not/popcount/forEach/toString(hex)
- **`ISMCTSNode`**: Tree node with UCB1(C=1.41), selectChild, expand, update
- **`LRUCache`**: MC cache, 300 entries, evicts LRU on overflow
- **`DeductionNotebook`**: Narrative analysis, avoidance patterns, certainty-tagged bullets

---

## Puzzle System Details

### Format
```
{ id, title, difficulty, category, myHand, board,
  otherHands: [P1,P2,P3], voids, passHistory,
  correctTile, correctSide, explanation }
```

**Invariant**: myHand + board + otherHands + 4(dorme) = 28, no duplicates.

### 40 Puzzles by Category
| Category | Count | IDs |
|----------|-------|-----|
| blocking | 10 | block-01/02, audit-block-01–08 |
| feeding | 7 | feed-01/02, audit-partner-01–05 |
| doubles | 2 | double-01/02 |
| endgame | 5 | end-01/02, audit-end-01–03 |
| reading | 2 | read-01/02 |
| tempo | 6 | tempo-01/02, audit-tempo-01–04 |
| pip | 4 | audit-pip-01–04 |
| suit-control | 4 | audit-suit-01–04 |

### Auto-Generation
- `extractPuzzleFromAnalysis`: creates from game position
- `validatePuzzle`: tile accounting + eqLoss ≥ 0.15
- `scanGameForPuzzles`: finds positions with eqLoss ≥ 0.30
- localStorage max 20 FIFO, category `'generated'`

---

## Internationalization

- Languages: Portuguese (pt, default) and English (en)
- `TR` object: ~221 keys, each `{en, pt}`
- `_t(key)` lookup, `_tr(reason)` for AI reason translation
- Persisted in localStorage `pernambuco_lang`

---

## All localStorage Keys

| Key | Contents |
|-----|----------|
| `pernambuco_lang` | UI language |
| `domino-trainer-stats` | Quiz accuracy |
| `domino_elo` | Elo rating |
| `domino-game-history` | Last 100 games |
| `domino-puzzle-completion` | Puzzle grades |
| `domino-generated-puzzles` | Auto puzzles (max 20) |
| `domino-opening-quiz-stats` | Opening quiz stats |
| `domino-leak-stats` | Leak aggregator |
| `pernambuco_coach` | Coach enabled |
| `pernambuco_coach_collapsed` | Coach collapsed |

---

## Self-Play Benchmark

### `window.runAIBenchmark(numGames)`
Console function. Plays numGames×2 games: heuristic vs MC-expert (500 sims/move).
Reports: win rates, Pts/Game, Elo diff = `400 × log10(mcWR / heuristicWR)`.

### `selfPlayBatch(numGames)`
Pure heuristic self-play. Returns: games, wins, avgPts, blockedPct.

---

## Elo System

| Parameter | Value |
|-----------|-------|
| Starting | 1200 |
| K-factor | 32 |
| Perfect move opponent | 1600 |
| Blunder opponent | 800 |

---

## Development Guide

### Adding a Puzzle
1. Add to `PUZZLES[]` before `];`
2. Verify: myHand + board + otherHands + 4 = 28, no dupes
3. Voids must match passHistory
4. correctTile should have ≥ 0.15 EP gap

### Tuning AI Weights
- Search `ss +=` / `ss -=` in `smartAI`
- Test: `runAIBenchmark(200)`, check 40 puzzles still pass

### Adding a Leak Type
1. Add to `LEAK_TYPES`
2. Add detection in Analysis grading loop
3. Call `leakAggregator.addLeak(type, severity)`

### Key Gotchas
- `tileHTML` takes **7 params** — missing any silently breaks
- `lE`/`rE` not `leftEnd`/`rightEnd`
- CSS `!important` needed for board tile borders
- Play mode uses `liveLeftEnd`/`liveRightEnd`
- `'3-1'` and `'1-3'` are the same tile — compare by ID
