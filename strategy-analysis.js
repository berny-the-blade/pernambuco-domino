// Domino Pernambucano - Starter Strategy Analysis
// For each deal, simulate with BOTH possible starters and compare outcomes
// Then correlate winning starter's hand characteristics

// ========== CORE GAME LOGIC ==========

function createDeck() {
  const deck = [];
  for (let i = 0; i <= 6; i++) {
    for (let j = i; j <= 6; j++) {
      deck.push({ left: i, right: j, id: i + '-' + j });
    }
  }
  return deck;
}

function shuffleDeck(deck) {
  const shuffled = [...deck];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

function findBestBotMove(hand, leftEnd, rightEnd, boardLen) {
  const playable = hand.filter(t => {
    if (boardLen === 0) return true;
    return t.left === leftEnd || t.right === leftEnd || t.left === rightEnd || t.right === rightEnd;
  });
  if (playable.length === 0) return null;

  let bestTile = playable[0];
  let bestScore = -1;
  for (let tile of playable) {
    let score = tile.left + tile.right;
    if (tile.left === tile.right) score += 10;
    if (score > bestScore) {
      bestScore = score;
      bestTile = tile;
    }
  }
  return bestTile;
}

function determineSide(tile, leftEnd, rightEnd) {
  const canLeft = tile.left === leftEnd || tile.right === leftEnd;
  const canRight = tile.left === rightEnd || tile.right === rightEnd;
  if (canLeft && !canRight) return 'left';
  if (canRight && !canLeft) return 'right';
  if (canLeft && canRight) return Math.random() > 0.5 ? 'left' : 'right';
  return null;
}

function couldPlayOnBothEnds(tile, left, right) {
  if (left === null || right === null) return false;
  if (left === right) return false;
  const canLeft = tile.left === left || tile.right === left;
  const canRight = tile.left === right || tile.right === right;
  return canLeft && canRight;
}

// ========== HAND ANALYSIS ==========

function analyzeHand(hand) {
  const doubles = hand.filter(t => t.left === t.right);
  const pipCount = hand.reduce((sum, t) => sum + t.left + t.right, 0);

  // Count suit frequency (how many tiles contain each number 0-6)
  const suitCount = [0, 0, 0, 0, 0, 0, 0];
  for (const t of hand) {
    suitCount[t.left]++;
    if (t.left !== t.right) suitCount[t.right]++;
  }
  const maxSuit = Math.max(...suitCount);
  const dominantSuit = suitCount.indexOf(maxSuit);

  // Count unique numbers covered
  const numbersSet = new Set();
  for (const t of hand) {
    numbersSet.add(t.left);
    numbersSet.add(t.right);
  }

  // High tiles (5 or 6 on either side)
  const highTiles = hand.filter(t => t.left >= 5 || t.right >= 5).length;

  // Low tiles (both sides <= 2)
  const lowTiles = hand.filter(t => t.left <= 2 && t.right <= 2).length;

  return {
    doubleCount: doubles.length,
    pipCount: pipCount,
    maxSuitCount: maxSuit,
    dominantSuit: dominantSuit,
    uniqueNumbers: numbersSet.size,
    highTiles: highTiles,
    lowTiles: lowTiles,
    tiles: hand.map(t => `${t.left}-${t.right}`).join(', ')
  };
}

// ========== ROUND SIMULATION ==========

function simulateRound(hands, startPlayer) {
  // Deep copy hands
  const h = hands.map(hand => hand.map(t => ({ ...t })));
  let board = [];
  let leftEnd = null;
  let rightEnd = null;
  let currentPlayer = startPlayer;
  let passCount = 0;
  let prevLeftEnd = null;
  let prevRightEnd = null;
  let moveCount = 0;

  while (moveCount < 200) {
    moveCount++;
    const tile = findBestBotMove(h[currentPlayer], leftEnd, rightEnd, board.length);

    if (tile) {
      passCount = 0;
      h[currentPlayer] = h[currentPlayer].filter(t => t.id !== tile.id);

      if (board.length === 0) {
        board = [tile];
        leftEnd = tile.left;
        rightEnd = tile.right;
      } else {
        const side = determineSide(tile, leftEnd, rightEnd);
        let placedTile = { ...tile };
        if (side === 'left') {
          if (tile.left === leftEnd) placedTile = { ...tile, left: tile.right, right: tile.left };
          board.unshift(placedTile);
          leftEnd = placedTile.left;
        } else if (side === 'right') {
          if (tile.right === rightEnd) placedTile = { ...tile, left: tile.right, right: tile.left };
          board.push(placedTile);
          rightEnd = placedTile.right;
        }
      }

      if (h[currentPlayer].length === 0) {
        const isDouble = tile.left === tile.right;
        const wasOnBothEnds = couldPlayOnBothEnds(tile, prevLeftEnd, prevRightEnd);
        let basePoints;
        if (isDouble && wasOnBothEnds) basePoints = 4;
        else if (isDouble) basePoints = 2;
        else if (wasOnBothEnds) basePoints = 3;
        else basePoints = 1;

        return { type: 'win', winner: currentPlayer, team: currentPlayer % 2, points: basePoints };
      }

      prevLeftEnd = leftEnd;
      prevRightEnd = rightEnd;
      currentPlayer = (currentPlayer + 1) % 4;
    } else {
      passCount++;
      if (passCount >= 4) {
        const handValues = h.map((hand, idx) => ({
          player: idx,
          points: hand.reduce((sum, t) => sum + t.left + t.right, 0)
        }));
        const minValue = Math.min(...handValues.map(v => v.points));
        const winners = handValues.filter(v => v.points === minValue);

        if (winners.length > 1 && winners.some(w => w.player % 2 === 0) && winners.some(w => w.player % 2 === 1)) {
          return { type: 'tie', team: -1, points: 0 };
        }
        return { type: 'blocked', winner: winners[0].player, team: winners[0].player % 2, points: 1 };
      }
      currentPlayer = (currentPlayer + 1) % 4;
    }
  }
  return { type: 'abort' };
}

// ========== STRATEGY COMPARISON ==========

const NUM_TRIALS = 50000;
const WINNING_TEAM = 0; // Team 0 (players 0, 2) always picks

// Track: for each characteristic comparison, how often the player with MORE of that trait wins
const comparisons = {
  moreDoubles: { chosen: 0, won: 0 },
  fewerDoubles: { chosen: 0, won: 0 },
  higherPips: { chosen: 0, won: 0 },
  lowerPips: { chosen: 0, won: 0 },
  higherMaxSuit: { chosen: 0, won: 0 },
  lowerMaxSuit: { chosen: 0, won: 0 },
  moreHighTiles: { chosen: 0, won: 0 },
  fewerHighTiles: { chosen: 0, won: 0 },
  moreUniqueNums: { chosen: 0, won: 0 },
  fewerUniqueNums: { chosen: 0, won: 0 },
  random: { chosen: 0, won: 0 },
};

// Detailed tracking: for each deal, try both starters
let totalDeals = 0;
let player0Better = 0;
let player2Better = 0;
let tiedOutcome = 0;

// Characteristic buckets for detailed analysis
const bucketStats = {
  doubles: { 0: { starts: 0, wins: 0 }, 1: { starts: 0, wins: 0 }, 2: { starts: 0, wins: 0 }, 3: { starts: 0, wins: 0 }, 4: { starts: 0, wins: 0 }, 5: { starts: 0, wins: 0 }, 6: { starts: 0, wins: 0 } },
  pipRange: { 'low(0-15)': { starts: 0, wins: 0 }, 'mid(16-25)': { starts: 0, wins: 0 }, 'high(26+)': { starts: 0, wins: 0 } },
  maxSuit: { 1: { starts: 0, wins: 0 }, 2: { starts: 0, wins: 0 }, 3: { starts: 0, wins: 0 }, 4: { starts: 0, wins: 0 }, 5: { starts: 0, wins: 0 }, 6: { starts: 0, wins: 0 } },
  uniqueNums: { 2: { starts: 0, wins: 0 }, 3: { starts: 0, wins: 0 }, 4: { starts: 0, wins: 0 }, 5: { starts: 0, wins: 0 }, 6: { starts: 0, wins: 0 }, 7: { starts: 0, wins: 0 } },
};

function getPipBucket(pips) {
  if (pips <= 15) return 'low(0-15)';
  if (pips <= 25) return 'mid(16-25)';
  return 'high(26+)';
}

console.log(`=== Domino Pernambucano - Starter Strategy Analysis ===`);
console.log(`Running ${NUM_TRIALS.toLocaleString()} deals, simulating each with both possible starters...\n`);

for (let trial = 0; trial < NUM_TRIALS; trial++) {
  const deck = shuffleDeck(createDeck());
  const hands = [[], [], [], []];
  for (let i = 0; i < 24; i++) {
    hands[i % 4].push(deck[i]);
  }

  // Team 0 = players 0 and 2. Simulate with each starting.
  // Use same random seed concept: run deterministically by using same hands
  const handsA = hands.map(h => h.map(t => ({ ...t })));
  const handsB = hands.map(h => h.map(t => ({ ...t })));

  const resultA = simulateRound(handsA, 0); // Player 0 starts
  const resultB = simulateRound(handsB, 2); // Player 2 starts

  if (resultA.type === 'abort' || resultB.type === 'abort') continue;

  totalDeals++;

  const p0analysis = analyzeHand(hands[0]);
  const p2analysis = analyzeHand(hands[2]);

  // Track bucket stats for ALL starts
  // Player 0 starts
  const p0won = resultA.team === WINNING_TEAM;
  bucketStats.doubles[p0analysis.doubleCount].starts++;
  if (p0won) bucketStats.doubles[p0analysis.doubleCount].wins++;
  bucketStats.pipRange[getPipBucket(p0analysis.pipCount)].starts++;
  if (p0won) bucketStats.pipRange[getPipBucket(p0analysis.pipCount)].wins++;
  bucketStats.maxSuit[p0analysis.maxSuitCount].starts++;
  if (p0won) bucketStats.maxSuit[p0analysis.maxSuitCount].wins++;
  if (bucketStats.uniqueNums[p0analysis.uniqueNumbers]) {
    bucketStats.uniqueNums[p0analysis.uniqueNumbers].starts++;
    if (p0won) bucketStats.uniqueNums[p0analysis.uniqueNumbers].wins++;
  }

  // Player 2 starts
  const p2won = resultB.team === WINNING_TEAM;
  bucketStats.doubles[p2analysis.doubleCount].starts++;
  if (p2won) bucketStats.doubles[p2analysis.doubleCount].wins++;
  bucketStats.pipRange[getPipBucket(p2analysis.pipCount)].starts++;
  if (p2won) bucketStats.pipRange[getPipBucket(p2analysis.pipCount)].wins++;
  bucketStats.maxSuit[p2analysis.maxSuitCount].starts++;
  if (p2won) bucketStats.maxSuit[p2analysis.maxSuitCount].wins++;
  if (bucketStats.uniqueNums[p2analysis.uniqueNumbers]) {
    bucketStats.uniqueNums[p2analysis.uniqueNumbers].starts++;
    if (p2won) bucketStats.uniqueNums[p2analysis.uniqueNumbers].wins++;
  }

  // Head-to-head: which starter gave team 0 a better outcome?
  const aTeamWin = resultA.team === WINNING_TEAM;
  const bTeamWin = resultB.team === WINNING_TEAM;

  if (aTeamWin && !bTeamWin) player0Better++;
  else if (bTeamWin && !aTeamWin) player2Better++;
  else tiedOutcome++;

  // Strategy comparisons (only when characteristics differ)
  function trackStrategy(stratName, chooseA) {
    comparisons[stratName].chosen++;
    if (chooseA && aTeamWin) comparisons[stratName].won++;
    else if (!chooseA && bTeamWin) comparisons[stratName].won++;
  }

  // More doubles
  if (p0analysis.doubleCount !== p2analysis.doubleCount) {
    trackStrategy('moreDoubles', p0analysis.doubleCount > p2analysis.doubleCount);
    trackStrategy('fewerDoubles', p0analysis.doubleCount < p2analysis.doubleCount);
  }

  // Higher/lower pips
  if (p0analysis.pipCount !== p2analysis.pipCount) {
    trackStrategy('higherPips', p0analysis.pipCount > p2analysis.pipCount);
    trackStrategy('lowerPips', p0analysis.pipCount < p2analysis.pipCount);
  }

  // Higher/lower max suit concentration
  if (p0analysis.maxSuitCount !== p2analysis.maxSuitCount) {
    trackStrategy('higherMaxSuit', p0analysis.maxSuitCount > p2analysis.maxSuitCount);
    trackStrategy('lowerMaxSuit', p0analysis.maxSuitCount < p2analysis.maxSuitCount);
  }

  // More/fewer high tiles
  if (p0analysis.highTiles !== p2analysis.highTiles) {
    trackStrategy('moreHighTiles', p0analysis.highTiles > p2analysis.highTiles);
    trackStrategy('fewerHighTiles', p0analysis.highTiles < p2analysis.highTiles);
  }

  // More/fewer unique numbers
  if (p0analysis.uniqueNumbers !== p2analysis.uniqueNumbers) {
    trackStrategy('moreUniqueNums', p0analysis.uniqueNumbers > p2analysis.uniqueNumbers);
    trackStrategy('fewerUniqueNums', p0analysis.uniqueNumbers < p2analysis.uniqueNumbers);
  }

  // Random baseline
  comparisons.random.chosen++;
  if (Math.random() > 0.5) { if (aTeamWin) comparisons.random.won++; }
  else { if (bTeamWin) comparisons.random.won++; }
}

// ========== REPORT ==========

console.log(`Total deals analyzed: ${totalDeals.toLocaleString()}\n`);

console.log('=== HEAD-TO-HEAD: Does Starter Choice Matter? ===');
console.log(`Starter A better:  ${player0Better} (${(player0Better/totalDeals*100).toFixed(1)}%)`);
console.log(`Starter B better:  ${player2Better} (${(player2Better/totalDeals*100).toFixed(1)}%)`);
console.log(`Same outcome:      ${tiedOutcome} (${(tiedOutcome/totalDeals*100).toFixed(1)}%)`);
console.log(`=> Starter choice matters in ${((player0Better + player2Better)/totalDeals*100).toFixed(1)}% of deals\n`);

console.log('=== STRATEGY COMPARISON (win rate when using strategy) ===');
console.log('Strategy               | Applicable | Win Rate');
console.log('-----------------------|------------|--------');
const stratOrder = ['moreDoubles', 'fewerDoubles', 'higherPips', 'lowerPips', 'higherMaxSuit', 'lowerMaxSuit', 'moreHighTiles', 'fewerHighTiles', 'moreUniqueNums', 'fewerUniqueNums', 'random'];
const labels = {
  moreDoubles: 'More doubles',
  fewerDoubles: 'Fewer doubles',
  higherPips: 'Higher pip count',
  lowerPips: 'Lower pip count',
  higherMaxSuit: 'Stronger suit',
  lowerMaxSuit: 'Weaker suit',
  moreHighTiles: 'More high tiles',
  fewerHighTiles: 'Fewer high tiles',
  moreUniqueNums: 'More variety',
  fewerUniqueNums: 'Less variety',
  random: 'Random (baseline)',
};

for (const strat of stratOrder) {
  const s = comparisons[strat];
  const winRate = s.chosen > 0 ? (s.won / s.chosen * 100).toFixed(1) : 'N/A';
  console.log(`${labels[strat].padEnd(23)}| ${String(s.chosen).padStart(10)} | ${winRate}%`);
}

console.log('\n=== DETAILED: WIN RATE BY STARTER HAND CHARACTERISTICS ===\n');

console.log('--- By Number of Doubles in Starting Hand ---');
console.log('Doubles | Starts    | Wins      | Win Rate');
console.log('--------|-----------|-----------|--------');
for (let d = 0; d <= 6; d++) {
  const b = bucketStats.doubles[d];
  if (b.starts > 0) {
    console.log(`${String(d).padStart(7)} | ${String(b.starts).padStart(9)} | ${String(b.wins).padStart(9)} | ${(b.wins/b.starts*100).toFixed(1)}%`);
  }
}

console.log('\n--- By Pip Count Range ---');
console.log('Pips       | Starts    | Wins      | Win Rate');
console.log('-----------|-----------|-----------|--------');
for (const range of ['low(0-15)', 'mid(16-25)', 'high(26+)']) {
  const b = bucketStats.pipRange[range];
  if (b.starts > 0) {
    console.log(`${range.padEnd(11)}| ${String(b.starts).padStart(9)} | ${String(b.wins).padStart(9)} | ${(b.wins/b.starts*100).toFixed(1)}%`);
  }
}

console.log('\n--- By Strongest Suit (max tiles sharing a number) ---');
console.log('Max Suit | Starts    | Wins      | Win Rate');
console.log('---------|-----------|-----------|--------');
for (let s = 1; s <= 6; s++) {
  const b = bucketStats.maxSuit[s];
  if (b.starts > 0) {
    console.log(`${String(s).padStart(8)} | ${String(b.starts).padStart(9)} | ${String(b.wins).padStart(9)} | ${(b.wins/b.starts*100).toFixed(1)}%`);
  }
}

console.log('\n--- By Number Variety (unique numbers 0-6 in hand) ---');
console.log('Unique # | Starts    | Wins      | Win Rate');
console.log('---------|-----------|-----------|--------');
for (let n = 2; n <= 7; n++) {
  const b = bucketStats.uniqueNums[n];
  if (b && b.starts > 0) {
    console.log(`${String(n).padStart(8)} | ${String(b.starts).padStart(9)} | ${String(b.wins).padStart(9)} | ${(b.wins/b.starts*100).toFixed(1)}%`);
  }
}

console.log('\n=== SUMMARY & RECOMMENDATIONS ===\n');

// Find best strategy
let bestStrat = 'random';
let bestRate = comparisons.random.chosen > 0 ? comparisons.random.won / comparisons.random.chosen : 0;
for (const strat of stratOrder) {
  const s = comparisons[strat];
  if (s.chosen > 1000) { // need enough samples
    const rate = s.won / s.chosen;
    if (rate > bestRate) {
      bestRate = rate;
      bestStrat = strat;
    }
  }
}
console.log(`Best starter strategy: ${labels[bestStrat]} (${(bestRate * 100).toFixed(1)}% win rate)`);
console.log(`Random baseline:       ${(comparisons.random.won / comparisons.random.chosen * 100).toFixed(1)}% win rate`);
console.log(`Advantage:             +${((bestRate - comparisons.random.won / comparisons.random.chosen) * 100).toFixed(1)} percentage points\n`);

console.log('=== ANALYSIS COMPLETE ===');
