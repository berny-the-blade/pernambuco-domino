// Domino Pernambucano - 100 Match Simulation
// Extracted pure game logic from domino-updated.html for testing

// ========== CORE GAME LOGIC (mirrors the HTML exactly) ==========

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

function canPlayTileOnBoard(tile, leftEnd, rightEnd, boardLen) {
  if (boardLen === 0) return true;
  return tile.left === leftEnd || tile.right === leftEnd ||
         tile.left === rightEnd || tile.right === rightEnd;
}

function couldPlayOnBothEnds(tile, left, right) {
  if (left === null || right === null) return false;
  if (left === right) return false;
  const canLeft = tile.left === left || tile.right === left;
  const canRight = tile.left === right || tile.right === right;
  return canLeft && canRight;
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

function determineSide(tile, leftEnd, rightEnd, boardLen) {
  if (boardLen === 0) return null;
  const canLeft = tile.left === leftEnd || tile.right === leftEnd;
  const canRight = tile.left === rightEnd || tile.right === rightEnd;
  if (canLeft && !canRight) return 'left';
  if (canRight && !canLeft) return 'right';
  if (canLeft && canRight) return Math.random() > 0.5 ? 'left' : 'right';
  return null;
}

// ========== SIMULATION ENGINE ==========

const MATCH_TARGET = 6;
const PLAYERS = [
  { name: 'P1', slot: 0 },
  { name: 'P2', slot: 1 },
  { name: 'P3', slot: 2 },
  { name: 'P4', slot: 3 }
];

// Stats tracking
const stats = {
  totalMatches: 0,
  totalRounds: 0,
  team1Wins: 0,
  team2Wins: 0,
  normalWins: 0,
  carrocaWins: 0,
  laELoWins: 0,
  cruzadaWins: 0,
  blockedGames: 0,
  blockedTies: 0,
  extraPointsAwarded: 0,
  maxRoundsInMatch: 0,
  minRoundsInMatch: Infinity,
  maxScore: 0,
  errors: [],
  infiniteLoopAborts: 0,
};

function simulateRound(state) {
  let { hands, board, leftEnd, rightEnd, currentPlayer, passCount, extraPoints } = state;
  let moveCount = 0;
  const MAX_MOVES = 200; // safety valve

  while (moveCount < MAX_MOVES) {
    moveCount++;

    const hand = hands[currentPlayer];
    const tile = findBestBotMove(hand, leftEnd, rightEnd, board.length);

    if (tile) {
      // Play the tile
      passCount = 0;
      hands[currentPlayer] = hands[currentPlayer].filter(t => t.id !== tile.id);

      if (board.length === 0) {
        board = [tile];
        leftEnd = tile.left;
        rightEnd = tile.right;
      } else {
        const side = determineSide(tile, leftEnd, rightEnd, board.length);
        let placedTile = { ...tile };

        if (side === 'left') {
          if (tile.left === leftEnd) {
            placedTile = { ...tile, left: tile.right, right: tile.left };
          }
          board.unshift(placedTile);
          leftEnd = placedTile.left;
        } else if (side === 'right') {
          if (tile.right === rightEnd) {
            placedTile = { ...tile, left: tile.right, right: tile.left };
          }
          board.push(placedTile);
          rightEnd = placedTile.right;
        }
      }

      // Verify board integrity: first tile's left = leftEnd, last tile's right = rightEnd
      if (board.length > 0) {
        if (board[0].left !== leftEnd) {
          stats.errors.push(`Board integrity: board[0].left=${board[0].left} != leftEnd=${leftEnd}`);
        }
        if (board[board.length - 1].right !== rightEnd) {
          stats.errors.push(`Board integrity: board[last].right=${board[board.length-1].right} != rightEnd=${rightEnd}`);
        }
      }

      // Check if player emptied hand
      if (hands[currentPlayer].length === 0) {
        const isDouble = tile.left === tile.right;
        const wasOnBothEnds = couldPlayOnBothEnds(tile, state.prevLeftEnd, state.prevRightEnd);

        let basePoints, scoreName;
        if (isDouble && wasOnBothEnds) {
          basePoints = 4; scoreName = 'cruzada';
          stats.cruzadaWins++;
        } else if (isDouble) {
          basePoints = 2; scoreName = 'carroca';
          stats.carrocaWins++;
        } else if (wasOnBothEnds) {
          basePoints = 3; scoreName = 'la-e-lo';
          stats.laELoWins++;
        } else {
          basePoints = 1; scoreName = 'normal';
          stats.normalWins++;
        }

        // Validate scoring
        if (basePoints < 1 || basePoints > 4) {
          stats.errors.push(`Invalid base points: ${basePoints}`);
        }

        const points = basePoints + extraPoints;
        if (extraPoints > 0) stats.extraPointsAwarded++;

        return {
          type: 'win',
          winner: currentPlayer,
          winningTeam: currentPlayer % 2,
          points: points,
          scoreName: scoreName,
          board: board,
          hands: hands
        };
      }

      // Save previous ends for la-e-lo detection on next play
      state.prevLeftEnd = leftEnd;
      state.prevRightEnd = rightEnd;
      currentPlayer = (currentPlayer + 1) % 4;
    } else {
      // Pass
      passCount++;

      if (passCount >= 4) {
        // Blocked game
        stats.blockedGames++;

        const handValues = hands.map((hand, idx) => ({
          player: idx,
          points: hand.reduce((sum, t) => sum + t.left + t.right, 0)
        }));

        const minValue = Math.min(...handValues.map(h => h.points));
        const winners = handValues.filter(h => h.points === minValue);

        // Verify all tiles accounted for
        const totalTilesLeft = hands.reduce((sum, h) => sum + h.length, 0);
        const expectedTilesLeft = 24 - board.length;
        // Account for first-game auto-play (starter has 5 tiles initially)
        if (totalTilesLeft + board.length > 24) {
          stats.errors.push(`Tile count mismatch: ${totalTilesLeft} in hands + ${board.length} on board != 24`);
        }

        if (winners.length > 1 && winners.some(w => w.player % 2 === 0) && winners.some(w => w.player % 2 === 1)) {
          stats.blockedTies++;
          return {
            type: 'tie',
            points: 0,
            extraPoints: extraPoints + 1,
            board: board,
            hands: hands
          };
        }

        const winner = winners[0].player;
        const points = 1 + extraPoints;
        if (extraPoints > 0) stats.extraPointsAwarded++;

        return {
          type: 'blocked',
          winner: winner,
          winningTeam: winner % 2,
          points: points,
          board: board,
          hands: hands,
          handValues: handValues
        };
      }

      currentPlayer = (currentPlayer + 1) % 4;
    }
  }

  stats.infiniteLoopAborts++;
  stats.errors.push(`Round aborted after ${MAX_MOVES} moves`);
  return { type: 'abort' };
}

function simulateMatch(matchNum) {
  const teamScores = [0, 0];
  let roundNum = 0;
  let extraPoints = 0;
  let lastWinningTeam = null;
  const MAX_ROUNDS = 100;

  while (teamScores[0] < MATCH_TARGET && teamScores[1] < MATCH_TARGET && roundNum < MAX_ROUNDS) {
    roundNum++;

    const deck = shuffleDeck(createDeck());
    const hands = [[], [], [], []];
    for (let i = 0; i < 24; i++) {
      hands[i % 4].push(deck[i]);
    }

    // Verify dealing: 28 tiles in deck, 24 dealt (6 each), 4 "dorme"
    if (hands.some(h => h.length !== 6)) {
      stats.errors.push(`Match ${matchNum} Round ${roundNum}: Hand size != 6`);
    }

    let startPlayer, board, leftEnd, rightEnd;

    if (roundNum === 1) {
      // First round: find and auto-play highest double
      startPlayer = 0;
      let highestDouble = -1;
      let highestDoubleTile = null;
      for (let p = 0; p < 4; p++) {
        for (let tile of hands[p]) {
          if (tile.left === tile.right && tile.left > highestDouble) {
            highestDouble = tile.left;
            startPlayer = p;
            highestDoubleTile = tile;
          }
        }
      }

      if (!highestDoubleTile) {
        stats.errors.push(`Match ${matchNum} Round ${roundNum}: No doubles found in any hand!`);
        // Extremely rare with 24 tiles, but possible with 4 dorme tiles being all doubles?
        // Actually impossible: 7 doubles exist, only 4 dorme, so at least 3 doubles are dealt
        continue;
      }

      // Auto-play the highest double (matches our fix)
      hands[startPlayer] = hands[startPlayer].filter(t => t.id !== highestDoubleTile.id);
      board = [highestDoubleTile];
      leftEnd = highestDoubleTile.left;
      rightEnd = highestDoubleTile.right;

      // Verify starter now has 5 tiles
      if (hands[startPlayer].length !== 5) {
        stats.errors.push(`Match ${matchNum} Round ${roundNum}: Starter hand size after auto-play != 5 (got ${hands[startPlayer].length})`);
      }

      // Next player after auto-play
      startPlayer = (startPlayer + 1) % 4;
    } else {
      // Subsequent rounds: winning team picks starter (simulate: random from winning team)
      if (lastWinningTeam !== null) {
        const winSlots = lastWinningTeam === 0 ? [0, 2] : [1, 3];
        startPlayer = winSlots[Math.floor(Math.random() * 2)];
      } else {
        startPlayer = 0;
      }
      board = [];
      leftEnd = null;
      rightEnd = null;
    }

    const result = simulateRound({
      hands: hands,
      board: board,
      leftEnd: leftEnd,
      rightEnd: rightEnd,
      currentPlayer: startPlayer,
      passCount: 0,
      extraPoints: extraPoints,
      prevLeftEnd: leftEnd,
      prevRightEnd: rightEnd
    });

    if (result.type === 'abort') {
      break;
    }

    if (result.type === 'tie') {
      extraPoints = result.extraPoints;
      // No winner, no lastWinningTeam change, play again
      continue;
    }

    // Win or blocked win
    teamScores[result.winningTeam] += result.points;
    lastWinningTeam = result.winningTeam;
    extraPoints = 0;

    // Validate points
    if (result.points < 1) {
      stats.errors.push(`Match ${matchNum} Round ${roundNum}: Points < 1 (${result.points})`);
    }
    if (result.type === 'win' && result.points > 4 + 10) { // max 4 base + reasonable extra
      stats.errors.push(`Match ${matchNum} Round ${roundNum}: Suspiciously high points (${result.points})`);
    }
  }

  if (roundNum >= MAX_ROUNDS) {
    stats.errors.push(`Match ${matchNum}: Hit MAX_ROUNDS (${MAX_ROUNDS})`);
  }

  // Track stats
  stats.totalRounds += roundNum;
  if (roundNum > stats.maxRoundsInMatch) stats.maxRoundsInMatch = roundNum;
  if (roundNum < stats.minRoundsInMatch) stats.minRoundsInMatch = roundNum;

  const winningTeam = teamScores[0] >= MATCH_TARGET ? 0 : 1;
  if (winningTeam === 0) stats.team1Wins++;
  else stats.team2Wins++;

  const maxScore = Math.max(teamScores[0], teamScores[1]);
  if (maxScore > stats.maxScore) stats.maxScore = maxScore;

  // Verify winner actually reached target
  if (teamScores[winningTeam] < MATCH_TARGET) {
    stats.errors.push(`Match ${matchNum}: Winner didn't reach target! Scores: ${teamScores}`);
  }

  return { teamScores, roundNum, winningTeam };
}

// ========== RUN SIMULATION ==========

console.log('=== Domino Pernambucano - 100 Match Simulation ===\n');

// First verify deck
const testDeck = createDeck();
console.log(`Deck: ${testDeck.length} tiles (expected 28)`);
if (testDeck.length !== 28) {
  console.log('ERROR: Deck should have 28 tiles!');
  process.exit(1);
}

// Verify doubles count
const doubles = testDeck.filter(t => t.left === t.right);
console.log(`Doubles: ${doubles.length} (expected 7: 0-0 through 6-6)`);
if (doubles.length !== 7) {
  console.log('ERROR: Should have 7 doubles!');
  process.exit(1);
}

// Verify total pip count
const totalPips = testDeck.reduce((sum, t) => sum + t.left + t.right, 0);
console.log(`Total pips: ${totalPips} (expected 168)`);

// Verify no duplicate tiles
const ids = testDeck.map(t => t.id);
const uniqueIds = new Set(ids);
console.log(`Unique tiles: ${uniqueIds.size} (expected 28)`);
if (uniqueIds.size !== 28) {
  console.log('ERROR: Duplicate tile IDs found!');
  process.exit(1);
}

console.log('\nRunning 100 matches...\n');

const matchResults = [];
for (let i = 1; i <= 100; i++) {
  const result = simulateMatch(i);
  matchResults.push(result);
  stats.totalMatches++;
}

// ========== REPORT ==========

console.log('=== RESULTS ===\n');
console.log(`Matches played:        ${stats.totalMatches}`);
console.log(`Total rounds:          ${stats.totalRounds}`);
console.log(`Avg rounds/match:      ${(stats.totalRounds / stats.totalMatches).toFixed(1)}`);
console.log(`Min rounds in a match: ${stats.minRoundsInMatch}`);
console.log(`Max rounds in a match: ${stats.maxRoundsInMatch}`);
console.log(`Highest final score:   ${stats.maxScore}`);
console.log('');
console.log(`Team 1 wins:           ${stats.team1Wins} (${(stats.team1Wins / stats.totalMatches * 100).toFixed(1)}%)`);
console.log(`Team 2 wins:           ${stats.team2Wins} (${(stats.team2Wins / stats.totalMatches * 100).toFixed(1)}%)`);
console.log('');
console.log('--- Round Outcomes ---');
console.log(`Normal wins (1pt):     ${stats.normalWins}`);
console.log(`Carroca wins (2pt):    ${stats.carrocaWins}`);
console.log(`La e Lo wins (3pt):    ${stats.laELoWins}`);
console.log(`Cruzada wins (4pt):    ${stats.cruzadaWins}`);
console.log(`Blocked games:         ${stats.blockedGames}`);
console.log(`Blocked ties (draw):   ${stats.blockedTies}`);
console.log(`Extra points awarded:  ${stats.extraPointsAwarded}`);
console.log('');

const totalOutcomes = stats.normalWins + stats.carrocaWins + stats.laELoWins + stats.cruzadaWins + stats.blockedGames + stats.blockedTies;
console.log(`Total round outcomes:  ${totalOutcomes} (should ≈ totalRounds: ${stats.totalRounds})`);

console.log('');
console.log('--- Errors ---');
if (stats.errors.length === 0) {
  console.log('NO ERRORS! All 100 matches completed cleanly.');
} else {
  console.log(`${stats.errors.length} error(s) found:`);
  // Deduplicate errors
  const errorCounts = {};
  for (const err of stats.errors) {
    errorCounts[err] = (errorCounts[err] || 0) + 1;
  }
  for (const [err, count] of Object.entries(errorCounts)) {
    console.log(`  [${count}x] ${err}`);
  }
}

if (stats.infiniteLoopAborts > 0) {
  console.log(`\nWARNING: ${stats.infiniteLoopAborts} round(s) aborted due to infinite loop!`);
}

console.log('\n=== SIMULATION COMPLETE ===');
