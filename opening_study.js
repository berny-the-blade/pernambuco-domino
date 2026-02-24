#!/usr/bin/env node
// opening_study.js — Large-scale MC study: "Should I open or let partner open?"
// Runs N simulations, extracts hand features, determines which features predict opening advantage.
// Uses heuristic-only AI (no MC lookahead) for speed.

const NUM_SIMS = 1000000;
const REPORT_EVERY = 50000;

// ========== CORE GAME ENGINE (extracted from simulator.html) ==========

function createDeck() {
  const d = [];
  for (let i = 0; i <= 6; i++)
    for (let j = i; j <= 6; j++)
      d.push({ left: i, right: j, id: `${i}-${j}` });
  return d;
}

const ALL_TILES = createDeck();
const PLAYER_NAMES = ['You (P0)', 'Opp Left (P1)', 'Partner (P2)', 'Opp Right (P3)'];
function pn(i) { return PLAYER_NAMES[i] || `P${i}`; }

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function canPlay(t, lE, rE, bLen) {
  if (bLen === 0) return true;
  return t.left === lE || t.right === lE || t.left === rE || t.right === rE;
}

function couldPlayBothEnds(t, l, r) {
  if (l === null || r === null) return false;
  return (t.left === l || t.right === l) && (t.left === r || t.right === r);
}

// ========== KNOWLEDGE CLASS ==========

class Knowledge {
  constructor() {
    this.cantHave = [new Set(), new Set(), new Set(), new Set()];
    this.played = new Set();
    this.playsBy = [[], [], [], []];
    this.passedOn = [[], [], [], []];
    this._strengthCache = [null, null, null, null];
  }
  clone() {
    const k = new Knowledge();
    k.cantHave = this.cantHave.map(s => new Set(s));
    k.played = new Set(this.played);
    k.playsBy = this.playsBy.map(a => [...a]);
    k.passedOn = this.passedOn.map(a => [...a]);
    k._strengthCache = [null, null, null, null];
    return k;
  }
  recordPlay(p, t) {
    this.played.add(t.id);
    if (p >= 0 && p <= 3) {
      this.playsBy[p].push(t);
      this._strengthCache[p] = null;
    }
  }
  recordPass(p, lE, rE) {
    this.cantHave[p].add(lE);
    this.cantHave[p].add(rE);
    this.passedOn[p].push({ lE, rE });
  }
  inferStrength(p) {
    if (this._strengthCache[p]) return this._strengthCache[p];
    const s = [0,0,0,0,0,0,0];
    for (const t of this.playsBy[p]) { s[t.left]++; if (t.left !== t.right) s[t.right]++; }
    this._strengthCache[p] = s;
    return s;
  }
  deadNumbers(p) { return [...this.cantHave[p]]; }
  remainingWithNumber(n) {
    let c = 0;
    for (const t of ALL_TILES) { if (!this.played.has(t.id) && (t.left === n || t.right === n)) c++; }
    return c;
  }
}

// ========== STUB NOTEBOOK (minimal, just enough for simulateFullRound) ==========

class DeductionNotebook {
  constructor() { this.playedTiles = new Set(); this.remainingByNumber = [7,7,7,7,7,7,7]; this.players = [0,1,2,3].map(() => ({ cannotHaveNumbers: new Set(), voidSince: {}, avoidanceEvidence: {}, passEvents: [], playEvents: [] })); this.bullets = []; this.turn = 0; }
  recordPass(player, turn, lE, rE) {
    this.turn = turn;
    const p = this.players[player];
    for (const n of [lE, rE]) { p.cannotHaveNumbers.add(n); if (!p.voidSince[n]) p.voidSince[n] = turn; }
    p.passEvents.push({ turn, ends: [lE, rE] });
  }
  recordPlay(player, turn, tile, side, lE, rE) {
    this.turn = turn;
    if (player >= 0 && player <= 3) this.players[player].playEvents.push({ turn, tile, side });
    this.playedTiles.add(tile.id);
    this.remainingByNumber[tile.left]--;
    if (tile.left !== tile.right) this.remainingByNumber[tile.right]--;
  }
  clone() {
    const nb = new DeductionNotebook();
    nb.players = this.players.map(p => ({
      cannotHaveNumbers: new Set(p.cannotHaveNumbers), voidSince: {...p.voidSince},
      avoidanceEvidence: JSON.parse(JSON.stringify(p.avoidanceEvidence)),
      passEvents: [...p.passEvents], playEvents: p.playEvents.map(e => ({...e, tile: {...e.tile}}))
    }));
    nb.remainingByNumber = [...this.remainingByNumber];
    nb.playedTiles = new Set(this.playedTiles);
    nb.turn = this.turn;
    nb.bullets = [];
    return nb;
  }
}

// ========== SMART AI (heuristic only, no MC) ==========

function smartAI(hand, lE, rE, bLen, player, knowledge, returnAll = false) {
  const playable = hand.filter(t => canPlay(t, lE, rE, bLen));
  if (playable.length === 0) return returnAll ? [] : null;

  if (bLen === 0) {
    const sc = [0,0,0,0,0,0,0];
    for (const t of hand) { sc[t.left]++; if (t.left !== t.right) sc[t.right]++; }
    const scored = playable.map(t => {
      let s = sc[t.left]*10 + sc[t.right]*10 + (t.left===t.right?15:0) + (t.left+t.right)*2;
      return { tile: t, score: s, side: null, reasons: [] };
    });
    scored.sort((a, b) => b.score - a.score);
    return returnAll ? scored : scored[0];
  }

  const partner = (player + 2) % 4;
  const opp1 = (player + 1) % 4, opp2 = (player + 3) % 4;

  const scored = playable.flatMap(tile => {
    const options = [];
    for (const side of ['left', 'right']) {
      const cS = side === 'left' ? (tile.left === lE || tile.right === lE) : (tile.left === rE || tile.right === rE);
      if (!cS) continue;

      let newEnd;
      if (side === 'left') newEnd = tile.left === lE ? tile.right : tile.left;
      else newEnd = tile.right === rE ? tile.left : tile.right;
      const otherEnd = side === 'left' ? rE : lE;
      let ss = 0;
      const sr = [];

      // Suit control
      const myCount = hand.filter(x => x.id !== tile.id && (x.left === newEnd || x.right === newEnd || x.left === otherEnd || x.right === otherEnd)).length;
      ss += myCount * 15;

      // Blocking
      if (knowledge.cantHave[opp1].has(newEnd) && knowledge.cantHave[opp1].has(otherEnd)) ss += 25;
      if (knowledge.cantHave[opp2].has(newEnd) && knowledge.cantHave[opp2].has(otherEnd)) ss += 25;

      // Partner support
      const pStr = knowledge.inferStrength(partner);
      let pAff = 0;
      for (const endNum of (newEnd === otherEnd ? [newEnd] : [newEnd, otherEnd])) {
        const played = pStr[endNum] || 0;
        if (played > 0) {
          const remaining = knowledge.remainingWithNumber(endNum);
          const weHold = hand.filter(x => x.id !== tile.id && (x.left === endNum || x.right === endNum)).length;
          const othersCouldHave = remaining - weHold;
          if (othersCouldHave > 0 && !knowledge.cantHave[partner].has(endNum)) pAff += played;
        }
      }
      if (pAff > 0) ss += pAff * 8;
      if (knowledge.cantHave[partner].has(newEnd)) ss -= 10;

      // Pip weight
      ss += (tile.left + tile.right) * 2;

      // Doubles
      if (tile.left === tile.right) ss += 12;

      // Isolated double setup
      for (const h of hand) {
        if (h.id === tile.id || h.left !== h.right) continue;
        const sup = hand.filter(x => x.id !== h.id && x.id !== tile.id && (x.left === h.left || x.right === h.left)).length;
        if (sup === 0 && (newEnd === h.left || otherEnd === h.left)) ss += 20;
      }

      // Board counting
      const remNew = knowledge.remainingWithNumber(newEnd);
      const weHoldNew = hand.filter(x => x.id !== tile.id && (x.left === newEnd || x.right === newEnd)).length;
      const oppCouldHaveNew = Math.max(0, remNew - weHoldNew);
      ss += (2 - oppCouldHaveNew) * 6;

      // Partner close
      const estPHand = Math.max(0, 6 - knowledge.playsBy[partner].length);
      if (estPHand <= 2 && estPHand > 0) ss += 15;

      // Closing bonus
      const remainAfter = hand.length - 1;
      if (remainAfter === 0) {
        const isD = tile.left === tile.right;
        const wasBoth = bLen > 0 && couldPlayBothEnds(tile, lE, rE);
        let closePts = 1;
        if (isD && wasBoth) closePts = 4;
        else if (isD) closePts = 2;
        else if (wasBoth) closePts = 3;
        ss += 200 + closePts * 30;
      } else if (remainAfter === 1) {
        const lastTile = hand.find(t => t.id !== tile.id);
        if (lastTile) {
          const canPlayLast = (lastTile.left === newEnd || lastTile.right === newEnd ||
                               lastTile.left === otherEnd || lastTile.right === otherEnd);
          if (canPlayLast) {
            ss += 80;
          } else {
            const allStrand = hand.filter(t => t.id !== tile.id).length === 1 && (() => {
              const ot = lastTile;
              const otherPlayable = (ot.left === lE || ot.right === lE || ot.left === rE || ot.right === rE);
              if (!otherPlayable) return false;
              for (const oSide of ['left', 'right']) {
                const oEnd = oSide === 'left' ? lE : rE;
                if (ot.left !== oEnd && ot.right !== oEnd) continue;
                const oNewEnd = (ot.left === oEnd) ? ot.right : ot.left;
                const oOtherEnd = oSide === 'left' ? rE : lE;
                if (tile.left === oNewEnd || tile.right === oNewEnd || tile.left === oOtherEnd || tile.right === oOtherEnd) return false;
              }
              return true;
            })();
            ss -= 30;
          }
        }
      } else if (remainAfter === 2) {
        const others = hand.filter(t => t.id !== tile.id);
        const coverCount = others.filter(t =>
          t.left === newEnd || t.right === newEnd || t.left === otherEnd || t.right === otherEnd
        ).length;
        if (coverCount === 2) ss += 25;
        else if (coverCount === 0) ss -= 15;
      }

      options.push({ tile, score: ss, reasons: sr, side });
    }
    return options;
  });

  scored.sort((a, b) => b.score - a.score);
  return returnAll ? scored : (scored[0] || null);
}

// ========== SIMULATE FULL ROUND (no MC, heuristic only) ==========

function simulateFullRound(initHands, startPlayer, initBoard, initLeft, initRight, autoPlayer) {
  const hands = initHands.map(h => h.map(t => ({...t})));
  let board = [...initBoard], lE = initLeft, rE = initRight;
  let cur = startPlayer, passCount = 0;
  const knowledge = new Knowledge();
  const nb = new DeductionNotebook();
  for (const t of initBoard) {
    knowledge.recordPlay(autoPlayer != null ? autoPlayer : -1, t);
    nb.recordPlay(autoPlayer != null ? autoPlayer : -1, 0, t, null, undefined, undefined);
  }

  let moveNum = 0;
  while (moveNum < 200) {
    moveNum++;
    const allOptions = smartAI(hands[cur], lE, rE, board.length, cur, knowledge, true);
    let decision = allOptions.length > 0 ? allOptions[0] : null;

    if (decision) {
      passCount = 0;
      const { tile, side: prefSide } = decision;
      let side = prefSide;
      if (board.length > 0 && !side) side = (tile.left === lE || tile.right === lE) ? 'left' : 'right';

      hands[cur] = hands[cur].filter(t => t.id !== tile.id);
      knowledge.recordPlay(cur, tile);
      nb.recordPlay(cur, moveNum, tile, side, lE, rE);

      const pL = lE, pR = rE;

      if (board.length === 0) {
        board = [{...tile}]; lE = tile.left; rE = tile.right;
      } else {
        let placed = {...tile};
        if (side === 'left') {
          if (tile.left === lE) placed = {left: tile.right, right: tile.left, id: tile.id};
          board.unshift(placed); lE = placed.left;
        } else {
          if (tile.right === rE) placed = {left: tile.right, right: tile.left, id: tile.id};
          board.push(placed); rE = placed.right;
        }
      }

      if (hands[cur].length === 0) {
        const isD = tile.left === tile.right;
        const wasBoth = couldPlayBothEnds(tile, pL, pR);
        let pts;
        if (isD && wasBoth) pts = 4;
        else if (isD) pts = 2;
        else if (wasBoth) pts = 3;
        else pts = 1;
        return { winner: cur, team: cur % 2, points: pts, type: 'win' };
      }
      cur = (cur + 1) % 4;
    } else {
      passCount++;
      knowledge.recordPass(cur, lE, rE);
      nb.recordPass(cur, moveNum, lE, rE);

      if (passCount >= 4) {
        const hv = hands.map((h, i) => ({ player: i, team: i%2, points: h.reduce((s,t)=>s+t.left+t.right,0) }));
        const min = Math.min(...hv.map(v => v.points));
        const winners = hv.filter(v => v.points === min);
        if (winners.length > 1 && winners.some(w => w.team===0) && winners.some(w => w.team===1)) {
          return { type: 'tie', points: 0, team: -1, winner: -1 };
        }
        return { type: 'blocked', winner: winners[0].player, team: winners[0].team, points: 1 };
      }
      cur = (cur + 1) % 4;
    }
  }
  return { type: 'abort', team: -1, winner: -1, points: 0 };
}

// ========== PICK OPENING TILE ==========

function pickOpeningTile(hand) {
  const suitCount = {};
  for (let n = 0; n <= 6; n++) suitCount[n] = hand.filter(t => t.left === n || t.right === n).length;
  return hand.slice().sort((a, b) => {
    const aIsD = a.left === a.right ? 1 : 0;
    const bIsD = b.left === b.right ? 1 : 0;
    if (bIsD !== aIsD) return bIsD - aIsD;
    const aDepth = Math.max(suitCount[a.left], suitCount[a.right]);
    const bDepth = Math.max(suitCount[b.left], suitCount[b.right]);
    if (bDepth !== aDepth) return bDepth - aDepth;
    return (b.left + b.right) - (a.left + a.right);
  })[0];
}

// ========== GENERATE GAME (simplified) ==========

function generateGame(forcedStarter, hands) {
  const hp = forcedStarter;
  const ht = pickOpeningTile(hands[hp]);
  hands[hp] = hands[hp].filter(t => t.id !== ht.id);
  const starter = (hp + 1) % 4;
  return simulateFullRound(hands, starter, [{...ht}], ht.left, ht.right, hp);
}

// ========== HAND FEATURE EXTRACTION ==========

function extractFeatures(hand) {
  const doubles = hand.filter(t => t.left === t.right);
  const totalPips = hand.reduce((s, t) => s + t.left + t.right, 0);

  const suitDepth = {};
  for (let n = 0; n <= 6; n++) suitDepth[n] = hand.filter(t => t.left === n || t.right === n).length;
  const maxSuitDepth = Math.max(...Object.values(suitDepth));
  const deepSuit = parseInt(Object.entries(suitDepth).sort((a,b) => b[1] - a[1])[0][0]);

  const uniqueSuits = new Set();
  hand.forEach(t => { uniqueSuits.add(t.left); uniqueSuits.add(t.right); });

  let chains = 0;
  for (let i = 0; i < hand.length; i++)
    for (let j = i+1; j < hand.length; j++) {
      const a = hand[i], b = hand[j];
      if (a.left === b.left || a.left === b.right || a.right === b.left || a.right === b.right) chains++;
    }

  const heavyTiles = hand.filter(t => t.left + t.right >= 9).length;

  // Has specific doubles
  const hasD6 = hand.some(t => t.left === 6 && t.right === 6);
  const hasD5 = hand.some(t => t.left === 5 && t.right === 5);
  const hasD0 = hand.some(t => t.left === 0 && t.right === 0);

  // Highest double value
  const highestDouble = doubles.length > 0 ? Math.max(...doubles.map(t => t.left)) : -1;

  // Average pip per tile
  const avgPip = totalPips / 6;

  // "Longest run" — max consecutive suit depth
  const suitArr = Object.values(suitDepth).sort((a,b) => b - a);

  return {
    doubles: doubles.length, totalPips, maxSuitDepth, deepSuit,
    suitCoverage: uniqueSuits.size, connectivity: chains,
    heavyTiles, hasD6, hasD5, hasD0, highestDouble, avgPip,
    topSuitDepths: suitArr.slice(0, 3)
  };
}

// ========== MAIN SIMULATION ==========

console.log(`\n=== PERNAMBUCO DOMINO OPENING STUDY ===`);
console.log(`Running ${(NUM_SIMS/1000000).toFixed(1)}M simulations...\n`);

const startTime = Date.now();

// Accumulators for feature buckets
const buckets = {
  // Each bucket: { p0Sum, p2Sum, count }
  doubles: {},      // 0, 1, 2, 3
  totalPips: {},    // bucketed
  maxSuitDepth: {}, // 1-6
  suitCoverage: {}, // 3-7
  connectivity: {}, // bucketed
  heavyTiles: {},   // 0, 1, 2, 3+
  hasD6: {},        // true/false
  hasD5: {},
  hasD0: {},
  highestDouble: {},
  pipBucket: {},
  // Combo features
  doublesXdepth: {},
  doublesXpips: {},
};

function addToBucket(cat, key, p0EP, p2EP) {
  if (!buckets[cat][key]) buckets[cat][key] = { p0Sum: 0, p2Sum: 0, count: 0, p0Wins: 0, p2Wins: 0, ties: 0 };
  const b = buckets[cat][key];
  b.p0Sum += p0EP;
  b.p2Sum += p2EP;
  b.count++;
  if (p0EP > p2EP) b.p0Wins++;
  else if (p2EP > p0EP) b.p2Wins++;
  else b.ties++;
}

let totalP0EP = 0, totalP2EP = 0;
let p0BetterCount = 0, p2BetterCount = 0, tiedCount = 0;

for (let sim = 0; sim < NUM_SIMS; sim++) {
  // Deal
  const deck = shuffle(ALL_TILES.map(t => ({...t})));
  const hands = [[], [], [], []];
  for (let i = 0; i < 24; i++) hands[i%4].push(deck[i]);

  const p0Hand = hands[0].map(t => ({...t}));
  const features = extractFeatures(p0Hand);

  // P0 opens
  const hands0 = hands.map(h => h.map(t => ({...t})));
  const out0 = generateGame(0, hands0);
  const ep0 = out0.team === 0 ? (out0.points || 0) : -(out0.points || 0);

  // P2 opens
  const hands2 = hands.map(h => h.map(t => ({...t})));
  const out2 = generateGame(2, hands2);
  const ep2 = out2.team === 0 ? (out2.points || 0) : -(out2.points || 0);

  totalP0EP += ep0;
  totalP2EP += ep2;
  if (ep0 > ep2) p0BetterCount++;
  else if (ep2 > ep0) p2BetterCount++;
  else tiedCount++;

  // Bucket by features
  addToBucket('doubles', features.doubles, ep0, ep2);

  const pipBucket = features.totalPips < 20 ? '<20' : features.totalPips < 25 ? '20-24' : features.totalPips < 30 ? '25-29' : features.totalPips < 35 ? '30-34' : '35+';
  addToBucket('pipBucket', pipBucket, ep0, ep2);
  addToBucket('totalPips', features.totalPips, ep0, ep2);
  addToBucket('maxSuitDepth', features.maxSuitDepth, ep0, ep2);
  addToBucket('suitCoverage', features.suitCoverage, ep0, ep2);

  const connBucket = features.connectivity <= 5 ? 'low(0-5)' : features.connectivity <= 9 ? 'med(6-9)' : 'high(10+)';
  addToBucket('connectivity', connBucket, ep0, ep2);
  addToBucket('heavyTiles', Math.min(features.heavyTiles, 3), ep0, ep2);
  addToBucket('hasD6', features.hasD6, ep0, ep2);
  addToBucket('hasD5', features.hasD5, ep0, ep2);
  addToBucket('hasD0', features.hasD0, ep0, ep2);
  addToBucket('highestDouble', features.highestDouble, ep0, ep2);

  // Combos
  addToBucket('doublesXdepth', `${features.doubles}d_depth${features.maxSuitDepth}`, ep0, ep2);
  addToBucket('doublesXpips', `${features.doubles}d_${pipBucket}pip`, ep0, ep2);

  if ((sim + 1) % REPORT_EVERY === 0) {
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = (sim + 1) / elapsed;
    const eta = (NUM_SIMS - sim - 1) / rate;
    process.stdout.write(`  ${((sim+1)/1000)}K done (${rate.toFixed(0)}/s, ETA ${eta.toFixed(0)}s)\r`);
  }
}

const elapsed = (Date.now() - startTime) / 1000;
console.log(`\nDone in ${elapsed.toFixed(1)}s (${(NUM_SIMS/elapsed).toFixed(0)} games/sec)\n`);

// ========== ANALYSIS ==========

console.log('='.repeat(70));
console.log('OVERALL RESULTS');
console.log('='.repeat(70));
console.log(`Total simulations: ${NUM_SIMS.toLocaleString()}`);
console.log(`Avg EP when YOU open:     ${(totalP0EP / NUM_SIMS).toFixed(4)}`);
console.log(`Avg EP when PARTNER opens: ${(totalP2EP / NUM_SIMS).toFixed(4)}`);
console.log(`Difference (you - partner): ${((totalP0EP - totalP2EP) / NUM_SIMS).toFixed(4)}`);
console.log(`You better: ${p0BetterCount} (${(p0BetterCount/NUM_SIMS*100).toFixed(1)}%)`);
console.log(`Partner better: ${p2BetterCount} (${(p2BetterCount/NUM_SIMS*100).toFixed(1)}%)`);
console.log(`Tied: ${tiedCount} (${(tiedCount/NUM_SIMS*100).toFixed(1)}%)`);

function printBucket(name, bucket, sortByKey = false) {
  console.log(`\n${'─'.repeat(70)}`);
  console.log(`FEATURE: ${name}`);
  console.log(`${'─'.repeat(70)}`);
  const entries = Object.entries(bucket);
  if (sortByKey) entries.sort((a, b) => {
    const na = parseFloat(a[0]), nb = parseFloat(b[0]);
    if (!isNaN(na) && !isNaN(nb)) return na - nb;
    return a[0] < b[0] ? -1 : 1;
  });

  console.log(`${'Value'.padEnd(18)} ${'Count'.padStart(8)} ${'%'.padStart(6)} ${'P0 EP'.padStart(8)} ${'P2 EP'.padStart(8)} ${'Δ(P0-P2)'.padStart(9)} ${'P0 better'.padStart(10)} ${'Verdict'.padStart(12)}`);
  for (const [key, b] of entries) {
    if (b.count < 100) continue; // skip tiny buckets
    const p0Avg = b.p0Sum / b.count;
    const p2Avg = b.p2Sum / b.count;
    const delta = p0Avg - p2Avg;
    const pct = (b.count / NUM_SIMS * 100).toFixed(1);
    const p0BetterPct = (b.p0Wins / b.count * 100).toFixed(0);
    const verdict = Math.abs(delta) < 0.02 ? 'TOSS-UP' : delta > 0 ? 'YOU OPEN' : 'PARTNER';
    console.log(`${String(key).padEnd(18)} ${String(b.count).padStart(8)} ${pct.padStart(5)}% ${p0Avg.toFixed(4).padStart(8)} ${p2Avg.toFixed(4).padStart(8)} ${(delta >= 0 ? '+' : '') + delta.toFixed(4).padStart(8)} ${(p0BetterPct + '%').padStart(9)} ${verdict.padStart(12)}`);
  }
}

printBucket('Number of Doubles', buckets.doubles, true);
printBucket('Total Pips (exact)', buckets.totalPips, true);
printBucket('Total Pips (bucketed)', buckets.pipBucket, true);
printBucket('Max Suit Depth', buckets.maxSuitDepth, true);
printBucket('Suit Coverage (unique numbers)', buckets.suitCoverage, true);
printBucket('Connectivity (tile links)', buckets.connectivity, true);
printBucket('Heavy Tiles (9+ pips)', buckets.heavyTiles, true);
printBucket('Has Double-Six', buckets.hasD6, true);
printBucket('Has Double-Five', buckets.hasD5, true);
printBucket('Has Double-Blank', buckets.hasD0, true);
printBucket('Highest Double Value', buckets.highestDouble, true);
printBucket('Doubles x Suit Depth', buckets.doublesXdepth, true);
printBucket('Doubles x Pip Range', buckets.doublesXpips, true);

// ========== KEY FINDINGS SUMMARY ==========

console.log(`\n${'='.repeat(70)}`);
console.log('KEY FINDINGS — Features that most predict opening advantage');
console.log('='.repeat(70));

// Find the features with biggest delta range
const featureSummaries = [];
for (const [name, bucket] of Object.entries(buckets)) {
  const entries = Object.entries(bucket).filter(([_, b]) => b.count >= 1000);
  if (entries.length < 2) continue;
  const deltas = entries.map(([key, b]) => ({ key, delta: b.p0Sum / b.count - b.p2Sum / b.count, count: b.count }));
  const minD = Math.min(...deltas.map(d => d.delta));
  const maxD = Math.max(...deltas.map(d => d.delta));
  const range = maxD - minD;
  featureSummaries.push({ name, range, minD, maxD, deltas });
}
featureSummaries.sort((a, b) => b.range - a.range);

for (const f of featureSummaries.slice(0, 10)) {
  console.log(`\n${f.name}: range ${f.range.toFixed(4)} (${f.minD.toFixed(4)} to ${f.maxD.toFixed(4)})`);
  const sorted = f.deltas.sort((a, b) => b.delta - a.delta);
  for (const d of sorted.slice(0, 5)) {
    console.log(`  ${String(d.key).padEnd(20)} Δ=${(d.delta >= 0 ? '+' : '') + d.delta.toFixed(4)}  (n=${d.count})`);
  }
}

console.log('\n\nDone!');
