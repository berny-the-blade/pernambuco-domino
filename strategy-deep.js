// Domino Pernambucano - Deep Strategy Analysis
// Factors: isolated doubles, protected doubles, hand connectivity, gaps,
// double of dominant suit, pair depth, heavy doubles, blocking potential

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
    if (score > bestScore) { bestScore = score; bestTile = tile; }
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
  return (tile.left === left || tile.right === left) && (tile.left === right || tile.right === right);
}

function simulateRound(hands, startPlayer) {
  const h = hands.map(hand => hand.map(t => ({ ...t })));
  let board = [], leftEnd = null, rightEnd = null;
  let currentPlayer = startPlayer, passCount = 0;
  let prevLeftEnd = null, prevRightEnd = null, moveCount = 0;

  while (moveCount < 200) {
    moveCount++;
    const tile = findBestBotMove(h[currentPlayer], leftEnd, rightEnd, board.length);
    if (tile) {
      passCount = 0;
      h[currentPlayer] = h[currentPlayer].filter(t => t.id !== tile.id);
      if (board.length === 0) {
        board = [tile]; leftEnd = tile.left; rightEnd = tile.right;
      } else {
        const side = determineSide(tile, leftEnd, rightEnd);
        let p = { ...tile };
        if (side === 'left') {
          if (tile.left === leftEnd) p = { ...tile, left: tile.right, right: tile.left };
          board.unshift(p); leftEnd = p.left;
        } else if (side === 'right') {
          if (tile.right === rightEnd) p = { ...tile, left: tile.right, right: tile.left };
          board.push(p); rightEnd = p.right;
        }
      }
      if (h[currentPlayer].length === 0) {
        const isD = tile.left === tile.right;
        const both = couldPlayOnBothEnds(tile, prevLeftEnd, prevRightEnd);
        let pts;
        if (isD && both) pts = 4;
        else if (isD) pts = 2;
        else if (both) pts = 3;
        else pts = 1;
        return { type: 'win', winner: currentPlayer, team: currentPlayer % 2, points: pts };
      }
      prevLeftEnd = leftEnd; prevRightEnd = rightEnd;
      currentPlayer = (currentPlayer + 1) % 4;
    } else {
      passCount++;
      if (passCount >= 4) {
        const hv = h.map((hand, idx) => ({ player: idx, points: hand.reduce((s, t) => s + t.left + t.right, 0) }));
        const min = Math.min(...hv.map(v => v.points));
        const winners = hv.filter(v => v.points === min);
        if (winners.length > 1 && winners.some(w => w.player % 2 === 0) && winners.some(w => w.player % 2 === 1))
          return { type: 'tie', team: -1, points: 0 };
        return { type: 'blocked', winner: winners[0].player, team: winners[0].player % 2, points: 1 };
      }
      currentPlayer = (currentPlayer + 1) % 4;
    }
  }
  return { type: 'abort' };
}

// ========== DEEP HAND ANALYSIS ==========

function deepAnalyze(hand) {
  const suitCount = [0, 0, 0, 0, 0, 0, 0]; // how many tiles contain each number
  const doubles = [];
  const nonDoubles = [];

  for (const t of hand) {
    suitCount[t.left]++;
    if (t.left !== t.right) {
      suitCount[t.right]++;
      nonDoubles.push(t);
    } else {
      doubles.push(t);
    }
  }

  const maxSuitCount = Math.max(...suitCount);
  const dominantSuit = suitCount.indexOf(maxSuitCount);

  // Isolated doubles: double N-N where suitCount[N] === 1 (only the double itself)
  const isolatedDoubles = doubles.filter(d => suitCount[d.left] === 1).length;

  // Protected doubles: double N-N where suitCount[N] >= 3 (double + 2 or more tiles with N)
  const protectedDoubles = doubles.filter(d => suitCount[d.left] >= 3).length;

  // Semi-protected doubles: double N-N where suitCount[N] === 2 (double + 1 tile with N)
  const semiProtectedDoubles = doubles.filter(d => suitCount[d.left] === 2).length;

  // Has double of dominant suit?
  const hasDominantDouble = doubles.some(d => d.left === dominantSuit);

  // Gaps: numbers 0-6 you have ZERO tiles for
  const gaps = suitCount.filter(c => c === 0).length;

  // Hand connectivity: longest chain you can form
  // Build adjacency: from each tile, what numbers connect
  const connectivity = computeConnectivity(hand);

  // Heavy doubles (5-5 or 6-6)
  const heavyDoubles = doubles.filter(d => d.left >= 5).length;

  // Light doubles (0-0, 1-1)
  const lightDoubles = doubles.filter(d => d.left <= 1).length;

  // Average pip per tile
  const totalPips = hand.reduce((s, t) => s + t.left + t.right, 0);
  const avgPip = totalPips / hand.length;

  // Dominant suit depth (how many tiles have the dominant suit number)
  const dominantDepth = suitCount[dominantSuit];

  // Second strongest suit
  const sorted = [...suitCount].sort((a, b) => b - a);
  const secondSuit = sorted[1];

  // Balance: difference between strongest and second strongest
  const suitBalance = maxSuitCount - secondSuit;

  return {
    doubleCount: doubles.length,
    isolatedDoubles,
    protectedDoubles,
    semiProtectedDoubles,
    hasDominantDouble,
    gaps,
    connectivity,
    heavyDoubles,
    lightDoubles,
    totalPips,
    avgPip,
    dominantSuit,
    dominantDepth,
    secondSuit,
    suitBalance,
    maxSuitCount,
  };
}

function computeConnectivity(hand) {
  // Find longest sequence of tiles that chain together
  // Each tile connects number A to number B
  // Find longest path in this graph
  if (hand.length === 0) return 0;

  let maxChain = 1;

  function dfs(currentEnd, used, length) {
    if (length > maxChain) maxChain = length;
    for (let i = 0; i < hand.length; i++) {
      if (used[i]) continue;
      const t = hand[i];
      if (t.left === currentEnd) {
        used[i] = true;
        dfs(t.right, used, length + 1);
        used[i] = false;
      } else if (t.right === currentEnd) {
        used[i] = true;
        dfs(t.left, used, length + 1);
        used[i] = false;
      }
    }
  }

  const used = new Array(hand.length).fill(false);
  for (let i = 0; i < hand.length; i++) {
    used[i] = true;
    dfs(hand[i].left, used, 1);
    dfs(hand[i].right, used, 1);
    used[i] = false;
  }

  return maxChain;
}

// ========== SIMULATION ==========

const NUM_TRIALS = 50000;

// Bucket trackers
function makeBuckets(keys) {
  const b = {};
  for (const k of keys) b[k] = { starts: 0, wins: 0 };
  return b;
}

const buckets = {
  isolatedDoubles: makeBuckets([0, 1, 2, 3, 4, 5]),
  protectedDoubles: makeBuckets([0, 1, 2, 3]),
  semiProtectedDoubles: makeBuckets([0, 1, 2, 3]),
  hasDominantDouble: makeBuckets(['yes', 'no']),
  gaps: makeBuckets([0, 1, 2, 3, 4, 5]),
  connectivity: makeBuckets([1, 2, 3, 4, 5, 6]),
  heavyDoubles: makeBuckets([0, 1, 2]),
  lightDoubles: makeBuckets([0, 1, 2]),
  dominantDepth: makeBuckets([1, 2, 3, 4, 5, 6]),
  suitBalance: makeBuckets([0, 1, 2, 3, 4]),
  avgPipBucket: makeBuckets(['low(0-3)', 'mid(3-5)', 'high(5+)']),
};

// Strategy head-to-head
const strategies = {};
const stratDefs = [
  ['fewerIsolatedDoubles', (a, b) => a.isolatedDoubles < b.isolatedDoubles],
  ['moreProtectedDoubles', (a, b) => a.protectedDoubles > b.protectedDoubles],
  ['hasDominantDouble', (a, b) => a.hasDominantDouble && !b.hasDominantDouble],
  ['fewerGaps', (a, b) => a.gaps < b.gaps],
  ['higherConnectivity', (a, b) => a.connectivity > b.connectivity],
  ['noHeavyDoubles', (a, b) => a.heavyDoubles < b.heavyDoubles],
  ['deeperDominant', (a, b) => a.dominantDepth > b.dominantDepth],
  ['moreSuitBalance', (a, b) => a.suitBalance < b.suitBalance],
  ['lowerAvgPip', (a, b) => a.avgPip < b.avgPip],
  ['higherAvgPip', (a, b) => a.avgPip > b.avgPip],
  // Combos
  ['fewIso+highConn', (a, b) => a.isolatedDoubles <= b.isolatedDoubles && a.connectivity > b.connectivity],
  ['domDouble+deepSuit', (a, b) => a.hasDominantDouble && a.dominantDepth >= 4 && !(b.hasDominantDouble && b.dominantDepth >= 4)],
  ['1double+highVar', (a, b) => a.doubleCount === 1 && a.connectivity >= 4 && !(b.doubleCount === 1 && b.connectivity >= 4)],
  ['0isoDoubles+fewGaps', (a, b) => a.isolatedDoubles === 0 && a.gaps <= 2 && !(b.isolatedDoubles === 0 && b.gaps <= 2)],
];

for (const [name] of stratDefs) {
  strategies[name] = { applicable: 0, won: 0 };
}
strategies['random'] = { applicable: 0, won: 0 };

console.log('=== Domino Pernambucano - Deep Strategy Analysis ===');
console.log(`Running ${NUM_TRIALS.toLocaleString()} deals...\n`);

function record(bucket, key, won) {
  if (bucket[key]) {
    bucket[key].starts++;
    if (won) bucket[key].wins++;
  }
}

for (let trial = 0; trial < NUM_TRIALS; trial++) {
  const deck = shuffleDeck(createDeck());
  const hands = [[], [], [], []];
  for (let i = 0; i < 24; i++) hands[i % 4].push(deck[i]);

  const handsA = hands.map(h => h.map(t => ({ ...t })));
  const handsB = hands.map(h => h.map(t => ({ ...t })));

  const resultA = simulateRound(handsA, 0);
  const resultB = simulateRound(handsB, 2);
  if (resultA.type === 'abort' || resultB.type === 'abort') continue;

  const aWin = resultA.team === 0;
  const bWin = resultB.team === 0;

  const analA = deepAnalyze(hands[0]);
  const analB = deepAnalyze(hands[2]);

  // Record bucket stats for BOTH starters
  for (const [player, anal, won] of [[0, analA, aWin], [2, analB, bWin]]) {
    record(buckets.isolatedDoubles, anal.isolatedDoubles, won);
    record(buckets.protectedDoubles, anal.protectedDoubles, won);
    record(buckets.semiProtectedDoubles, anal.semiProtectedDoubles, won);
    record(buckets.hasDominantDouble, anal.hasDominantDouble ? 'yes' : 'no', won);
    record(buckets.gaps, anal.gaps, won);
    record(buckets.connectivity, Math.min(anal.connectivity, 6), won);
    record(buckets.heavyDoubles, Math.min(anal.heavyDoubles, 2), won);
    record(buckets.lightDoubles, Math.min(anal.lightDoubles, 2), won);
    record(buckets.dominantDepth, Math.min(anal.dominantDepth, 6), won);
    record(buckets.suitBalance, Math.min(anal.suitBalance, 4), won);
    const apb = anal.avgPip < 3 ? 'low(0-3)' : anal.avgPip < 5 ? 'mid(3-5)' : 'high(5+)';
    record(buckets.avgPipBucket, apb, won);
  }

  // Strategy head-to-head
  for (const [name, pred] of stratDefs) {
    const aPreferred = pred(analA, analB);
    const bPreferred = pred(analB, analA);
    if (aPreferred && !bPreferred) {
      strategies[name].applicable++;
      if (aWin) strategies[name].won++;
    } else if (bPreferred && !aPreferred) {
      strategies[name].applicable++;
      if (bWin) strategies[name].won++;
    }
  }

  strategies['random'].applicable++;
  if (Math.random() > 0.5) { if (aWin) strategies['random'].won++; }
  else { if (bWin) strategies['random'].won++; }
}

// ========== REPORT ==========

function printBucket(title, bucket) {
  console.log(`\n--- ${title} ---`);
  console.log('Value    | Starts    | Wins      | Win Rate');
  console.log('---------|-----------|-----------|--------');
  const keys = Object.keys(bucket).sort((a, b) => {
    const na = parseFloat(a), nb = parseFloat(b);
    if (!isNaN(na) && !isNaN(nb)) return na - nb;
    return a.localeCompare(b);
  });
  for (const k of keys) {
    const b = bucket[k];
    if (b.starts > 0) {
      console.log(`${String(k).padEnd(9)}| ${String(b.starts).padStart(9)} | ${String(b.wins).padStart(9)} | ${(b.wins/b.starts*100).toFixed(1)}%`);
    }
  }
}

console.log('\n========================================');
console.log('  DETAILED WIN RATES BY HAND FACTOR');
console.log('========================================');

printBucket('Isolated Doubles (double with no suit support)', buckets.isolatedDoubles);
printBucket('Protected Doubles (double with 2+ suit tiles)', buckets.protectedDoubles);
printBucket('Semi-Protected Doubles (double with 1 suit tile)', buckets.semiProtectedDoubles);
printBucket('Has Double of Dominant Suit', buckets.hasDominantDouble);
printBucket('Gaps (numbers 0-6 not in hand at all)', buckets.gaps);
printBucket('Hand Connectivity (longest chain length)', buckets.connectivity);
printBucket('Heavy Doubles (5-5 or 6-6)', buckets.heavyDoubles);
printBucket('Light Doubles (0-0 or 1-1)', buckets.lightDoubles);
printBucket('Dominant Suit Depth (tiles with strongest number)', buckets.dominantDepth);
printBucket('Suit Balance (gap between 1st and 2nd strongest)', buckets.suitBalance);
printBucket('Average Pip Value Per Tile', buckets.avgPipBucket);

console.log('\n\n========================================');
console.log('  STRATEGY HEAD-TO-HEAD COMPARISON');
console.log('========================================\n');
console.log('Strategy                          | Applicable | Win Rate | vs Random');
console.log('----------------------------------|------------|----------|----------');

const randomRate = strategies['random'].applicable > 0 ? strategies['random'].won / strategies['random'].applicable : 0;

const sorted = [...stratDefs.map(s => s[0]), 'random'].sort((a, b) => {
  const ra = strategies[a].applicable > 0 ? strategies[a].won / strategies[a].applicable : 0;
  const rb = strategies[b].applicable > 0 ? strategies[b].won / strategies[b].applicable : 0;
  return rb - ra;
});

for (const name of sorted) {
  const s = strategies[name];
  if (s.applicable < 100) continue;
  const rate = s.won / s.applicable;
  const diff = rate - randomRate;
  const diffStr = diff >= 0 ? `+${(diff * 100).toFixed(1)}%` : `${(diff * 100).toFixed(1)}%`;
  console.log(`${name.padEnd(34)}| ${String(s.applicable).padStart(10)} | ${(rate * 100).toFixed(1)}%    | ${diffStr}`);
}

console.log('\n\n========================================');
console.log('  TOP INSIGHTS & RECOMMENDATIONS');
console.log('========================================\n');

// Find top 3 strategies
const ranked = stratDefs.map(([name]) => {
  const s = strategies[name];
  return { name, rate: s.applicable > 100 ? s.won / s.applicable : 0, samples: s.applicable };
}).filter(s => s.samples > 500).sort((a, b) => b.rate - a.rate);

console.log('TOP STARTER SELECTION RULES (by win rate):\n');
for (let i = 0; i < Math.min(5, ranked.length); i++) {
  const r = ranked[i];
  const diff = ((r.rate - randomRate) * 100).toFixed(1);
  console.log(`  ${i + 1}. ${r.name} — ${(r.rate * 100).toFixed(1)}% win rate (+${diff}% vs random, ${r.samples.toLocaleString()} samples)`);
}

// Find worst factors
console.log('\nDANGER SIGNS (hands that lose more when starting):\n');
const dangerFactors = [];
for (const [factorName, bucket] of Object.entries(buckets)) {
  for (const [key, val] of Object.entries(bucket)) {
    if (val.starts > 500) {
      const rate = val.wins / val.starts;
      if (rate < 0.55) {
        dangerFactors.push({ factor: factorName, value: key, rate, samples: val.starts });
      }
    }
  }
}
dangerFactors.sort((a, b) => a.rate - b.rate);
for (const d of dangerFactors.slice(0, 5)) {
  console.log(`  - ${d.factor}=${d.value}: ${(d.rate * 100).toFixed(1)}% win rate (${d.samples.toLocaleString()} samples)`);
}

console.log('\n=== ANALYSIS COMPLETE ===');
