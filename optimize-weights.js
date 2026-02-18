#!/usr/bin/env node
'use strict';

// ============================================================
// Pernambuco Domino AI — Weight Optimizer v2
// Simulated Annealing + Head-to-Head Validation
// ============================================================

// --- GAME ENGINE (matches simulator.html exactly) ---

function createDeck() {
  const d = [];
  for (let i = 0; i <= 6; i++)
    for (let j = i; j <= 6; j++)
      d.push({ left: i, right: j, id: `${i}-${j}` });
  return d;
}

const ALL_TILES = createDeck();

function shuffleDeck(deck) {
  const s = [...deck];
  for (let i = s.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [s[i], s[j]] = [s[j], s[i]];
  }
  return s;
}

function canPlay(t, lE, rE, bLen) {
  if (bLen === 0) return true;
  return t.left === lE || t.right === lE || t.left === rE || t.right === rE;
}

function couldPlayBothEnds(t, l, r) {
  if (l === null || r === null) return false;
  return (t.left === l || t.right === l) && (t.left === r || t.right === r);
}

class Knowledge {
  constructor() {
    this.cantHave = [new Set(), new Set(), new Set(), new Set()];
    this.played = new Set();
    this.playsBy = [[], [], [], []];
  }
  clone() {
    const k = new Knowledge();
    k.cantHave = this.cantHave.map(s => new Set(s));
    k.played = new Set(this.played);
    k.playsBy = this.playsBy.map(a => a.map(t => ({ ...t })));
    return k;
  }
  recordPlay(p, t) {
    this.played.add(t.id);
    if (p >= 0 && p <= 3) this.playsBy[p].push(t);
  }
  recordPass(p, lE, rE) {
    this.cantHave[p].add(lE);
    this.cantHave[p].add(rE);
  }
  inferStrength(p) {
    const s = [0, 0, 0, 0, 0, 0, 0];
    for (const t of this.playsBy[p]) { s[t.left]++; if (t.left !== t.right) s[t.right]++; }
    return s;
  }
  remainingWithNumber(n) {
    let c = 0;
    for (const t of ALL_TILES) { if (!this.played.has(t.id) && (t.left === n || t.right === n)) c++; }
    return c;
  }
}

// --- SMART AI WITH CONFIGURABLE WEIGHTS ---
// Weight indices:
// 0: suit control, 1: block, 2: partner support, 3: partner can't have,
// 4: dump pip mult, 5: clear double, 6: isolated double setup, 7: scarcity,
// 8: partner close, 9: open suit, 10: open double, 11: open pip

const WEIGHT_NAMES = [
  'suit_ctrl', 'block', 'partner_sup', 'partner_cant', 'dump_pip',
  'clr_double', 'iso_double', 'scarcity', 'partner_close',
  'open_suit', 'open_double', 'open_pip'
];

const CURRENT_WEIGHTS = [15, 25, 8, -10, 2, 12, 20, -8, 15, 10, 15, 2];

function smartAI(w, hand, lE, rE, bLen, player, knowledge) {
  const playable = hand.filter(t => canPlay(t, lE, rE, bLen));
  if (playable.length === 0) return null;

  if (bLen === 0) {
    const sc = [0, 0, 0, 0, 0, 0, 0];
    for (const t of hand) { sc[t.left]++; if (t.left !== t.right) sc[t.right]++; }
    let best = null, bestS = -Infinity;
    for (const t of playable) {
      let s = sc[t.left] * w[9] + sc[t.right] * w[9] + (t.left === t.right ? w[10] : 0) + (t.left + t.right) * w[11];
      if (s > bestS) { bestS = s; best = { tile: t, side: null, score: s }; }
    }
    return best;
  }

  const partner = (player + 2) % 4;
  const opp1 = (player + 1) % 4, opp2 = (player + 3) % 4;
  let best = null, bestS = -Infinity;

  for (const tile of playable) {
    for (const side of ['left', 'right']) {
      const cS = side === 'left' ? (tile.left === lE || tile.right === lE) : (tile.left === rE || tile.right === rE);
      if (!cS) continue;

      let newEnd;
      if (side === 'left') newEnd = tile.left === lE ? tile.right : tile.left;
      else newEnd = tile.right === rE ? tile.left : tile.right;

      const otherEnd = side === 'left' ? rE : lE;
      let ss = 0;

      const myCount = hand.filter(x => x.id !== tile.id && (x.left === newEnd || x.right === newEnd || x.left === otherEnd || x.right === otherEnd)).length;
      ss += myCount * w[0];

      if (knowledge.cantHave[opp1].has(newEnd) && knowledge.cantHave[opp1].has(otherEnd)) ss += w[1];
      if (knowledge.cantHave[opp2].has(newEnd) && knowledge.cantHave[opp2].has(otherEnd)) ss += w[1];

      const pStr = knowledge.inferStrength(partner);
      let pAff = 0;
      for (const endNum of [newEnd, otherEnd]) {
        const played = pStr[endNum] || 0;
        if (played > 0) {
          const remaining = knowledge.remainingWithNumber(endNum);
          const weHold = hand.filter(x => x.id !== tile.id && (x.left === endNum || x.right === endNum)).length;
          const othersCouldHave = remaining - weHold;
          if (othersCouldHave > 0 && !knowledge.cantHave[partner].has(endNum)) pAff += played;
        }
      }
      if (pAff > 0) ss += pAff * w[2];
      if (knowledge.cantHave[partner].has(newEnd)) ss += w[3];

      if (tile.left + tile.right >= 9) ss += (tile.left + tile.right) * w[4];
      else ss += (tile.left + tile.right) * w[4];

      if (tile.left === tile.right) ss += w[5];

      for (const h of hand) {
        if (h.id === tile.id || h.left !== h.right) continue;
        const sup = hand.filter(x => x.id !== h.id && x.id !== tile.id && (x.left === h.left || x.right === h.left)).length;
        if (sup === 0 && (newEnd === h.left || otherEnd === h.left)) ss += w[6];
      }

      if (knowledge.remainingWithNumber(newEnd) <= 1) ss += w[7];

      const estPHand = Math.max(0, 6 - knowledge.playsBy[partner].length);
      if (estPHand <= 2 && estPHand > 0) ss += w[8];

      if (ss > bestS) { bestS = ss; best = { tile, side, score: ss }; }
    }
  }

  return best;
}

// --- GAME SIMULATION ---

function simulateGame(weightsTeam0, weightsTeam1) {
  const deck = shuffleDeck(ALL_TILES);
  const hands = [[], [], [], []];
  for (let i = 0; i < 24; i++) hands[i % 4].push({ ...deck[i] });

  let hd = -1, hp = 0, ht = null;
  for (let p = 0; p < 4; p++)
    for (const t of hands[p])
      if (t.left === t.right && t.left > hd) { hd = t.left; hp = p; ht = t; }

  const board = [{ ...ht }];
  hands[hp] = hands[hp].filter(t => t.id !== ht.id);
  let lE = ht.left, rE = ht.right;
  let cur = (hp + 1) % 4;

  const knowledge = new Knowledge();
  knowledge.recordPlay(hp, ht);

  let passCount = 0;

  for (let moveNum = 0; moveNum < 200; moveNum++) {
    const w = cur % 2 === 0 ? weightsTeam0 : weightsTeam1;
    const decision = smartAI(w, hands[cur], lE, rE, board.length, cur, knowledge);

    if (decision) {
      passCount = 0;
      const { tile, side: prefSide } = decision;
      let side = prefSide || ((tile.left === lE || tile.right === lE) ? 'left' : 'right');

      hands[cur] = hands[cur].filter(t => t.id !== tile.id);
      knowledge.recordPlay(cur, tile);

      const pL = lE, pR = rE;

      let placed = { ...tile };
      if (side === 'left') {
        if (tile.left === lE) placed = { left: tile.right, right: tile.left, id: tile.id };
        board.unshift(placed); lE = placed.left;
      } else {
        if (tile.right === rE) placed = { left: tile.right, right: tile.left, id: tile.id };
        board.push(placed); rE = placed.right;
      }

      if (hands[cur].length === 0) {
        const isD = tile.left === tile.right;
        const wasBoth = couldPlayBothEnds(tile, pL, pR);
        let pts;
        if (isD && wasBoth) pts = 4;
        else if (isD) pts = 2;
        else if (wasBoth) pts = 3;
        else pts = 1;
        return { team: cur % 2, points: pts };
      }
      cur = (cur + 1) % 4;
    } else {
      passCount++;
      knowledge.recordPass(cur, lE, rE);
      if (passCount >= 4) {
        const hv = hands.map((h, i) => ({
          team: i % 2, points: h.reduce((s, t) => s + t.left + t.right, 0)
        }));
        const t0 = hv.filter(v => v.team === 0).reduce((s, v) => s + v.points, 0);
        const t1 = hv.filter(v => v.team === 1).reduce((s, v) => s + v.points, 0);
        if (t0 < t1) return { team: 0, points: 1 };
        if (t1 < t0) return { team: 1, points: 1 };
        return { team: -1, points: 0 };
      }
      cur = (cur + 1) % 4;
    }
  }
  return { team: -1, points: 0 };
}

// --- EVALUATION: head-to-head ---

function evaluate(weightsA, weightsB, numGames) {
  let winsA = 0, winsB = 0, ties = 0;
  const half = Math.floor(numGames / 2);

  for (let i = 0; i < half; i++) {
    const r = simulateGame(weightsA, weightsB);
    if (r.team === 0) winsA++; else if (r.team === 1) winsB++; else ties++;
  }
  for (let i = 0; i < half; i++) {
    const r = simulateGame(weightsB, weightsA);
    if (r.team === 1) winsA++; else if (r.team === 0) winsB++; else ties++;
  }

  const total = winsA + winsB + ties;
  return { winRate: total > 0 ? winsA / total : 0.5, winsA, winsB, ties, total };
}

// --- SIMULATED ANNEALING ---

const WEIGHT_BOUNDS = [
  [3, 40],    // suit control
  [5, 60],    // block
  [1, 30],    // partner support
  [-30, -1],  // partner can't have
  [0.1, 5],   // dump pip mult
  [3, 50],    // clear double
  [3, 50],    // isolated double setup
  [-25, -1],  // scarcity
  [3, 40],    // partner close
  [2, 25],    // open suit
  [3, 35],    // open double
  [0.1, 5],   // open pip mult
];

function clamp(w) {
  return w.map((v, i) => Math.max(WEIGHT_BOUNDS[i][0], Math.min(WEIGHT_BOUNDS[i][1], v)));
}

function perturb(w, scale) {
  const nw = [...w];
  // Perturb 2-4 random weights
  const numChanges = 2 + Math.floor(Math.random() * 3);
  for (let c = 0; c < numChanges; c++) {
    const i = Math.floor(Math.random() * nw.length);
    const range = WEIGHT_BOUNDS[i][1] - WEIGHT_BOUNDS[i][0];
    nw[i] += (Math.random() * 2 - 1) * range * scale;
  }
  return clamp(nw);
}

function optimize() {
  console.log('='.repeat(60));
  console.log('  Pernambuco Domino AI — Weight Optimizer v2');
  console.log('  Method: Simulated Annealing + Multi-restart');
  console.log('='.repeat(60));
  console.log();

  const GAMES_PER_EVAL = 600;
  const SA_STEPS = 150;
  const RESTARTS = 5;
  const CONFIRM_GAMES = 2000;

  // First: baseline
  console.log('Baseline evaluation (2000 games each)...');
  const baseVsRandom = evaluate(CURRENT_WEIGHTS, [0,0,0,0,0,0,0,0,0,0,0,0], 2000); // dummy weights = very bad AI
  console.log(`Current weights: [${CURRENT_WEIGHTS.join(', ')}]`);
  console.log();

  const startTime = Date.now();
  let globalBest = [...CURRENT_WEIGHTS];
  let globalBestRate = 0.5;

  for (let restart = 0; restart < RESTARTS; restart++) {
    console.log(`\n--- Restart ${restart + 1}/${RESTARTS} ---`);

    let current = restart === 0 ? [...CURRENT_WEIGHTS] : perturb(CURRENT_WEIGHTS, 0.4);
    let currentRate = 0.5;

    // Evaluate starting point
    const startEval = evaluate(current, CURRENT_WEIGHTS, GAMES_PER_EVAL);
    currentRate = startEval.winRate;
    console.log(`Start: ${(currentRate * 100).toFixed(1)}% vs baseline`);

    let best = [...current];
    let bestRate = currentRate;
    let accepted = 0, improved = 0;

    for (let step = 0; step < SA_STEPS; step++) {
      const temp = 1.0 - step / SA_STEPS; // temperature: 1.0 → 0.0
      const scale = 0.15 * (0.3 + 0.7 * temp); // perturbation scale decreases

      const candidate = perturb(current, scale);
      const result = evaluate(candidate, CURRENT_WEIGHTS, GAMES_PER_EVAL);
      const candidateRate = result.winRate;

      const delta = candidateRate - currentRate;

      // Accept if better, or with probability based on temperature
      if (delta > 0 || Math.random() < Math.exp(delta * 20 / (temp + 0.01))) {
        current = candidate;
        currentRate = candidateRate;
        accepted++;
        if (delta > 0) improved++;

        if (candidateRate > bestRate) {
          best = [...candidate];
          bestRate = candidateRate;
        }
      }

      if ((step + 1) % 15 === 0) {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(
          `  Step ${String(step + 1).padStart(3)}/${SA_STEPS} | ` +
          `Current: ${(currentRate * 100).toFixed(1)}% | ` +
          `Best: ${(bestRate * 100).toFixed(1)}% | ` +
          `Accepted: ${accepted}/${step + 1} | ` +
          `Improved: ${improved} | ` +
          `${elapsed}s`
        );
      }
    }

    console.log(`Restart ${restart + 1} best: ${(bestRate * 100).toFixed(1)}%`);

    // Confirm with more games
    const confirm = evaluate(best, CURRENT_WEIGHTS, CONFIRM_GAMES);
    console.log(`Confirmed: ${(confirm.winRate * 100).toFixed(1)}% (${confirm.winsA}W/${confirm.winsB}L/${confirm.ties}T over ${confirm.total} games)`);

    if (confirm.winRate > globalBestRate) {
      globalBest = [...best];
      globalBestRate = confirm.winRate;
      console.log('  >>> New global best!');
    }
  }

  // === FINAL VALIDATION ===
  console.log('\n' + '='.repeat(60));
  console.log('  FINAL VALIDATION (5000 games)');
  console.log('='.repeat(60));

  const finalResult = evaluate(globalBest, CURRENT_WEIGHTS, 5000);
  console.log(`\nOptimized vs Current: ${(finalResult.winRate * 100).toFixed(1)}% (${finalResult.winsA}W/${finalResult.winsB}L/${finalResult.ties}T)`);

  // Also test vs a random baseline (very weak AI)
  const vsWeak = evaluate(globalBest, perturb(CURRENT_WEIGHTS, 0.8), 2000);
  const curVsWeak = evaluate(CURRENT_WEIGHTS, perturb(CURRENT_WEIGHTS, 0.8), 2000);
  console.log(`Optimized vs Weak AI: ${(vsWeak.winRate * 100).toFixed(1)}%`);
  console.log(`Current vs Weak AI:   ${(curVsWeak.winRate * 100).toFixed(1)}%`);

  console.log('\n' + '='.repeat(60));
  console.log('  OPTIMIZED WEIGHTS');
  console.log('='.repeat(60));

  const rounded = globalBest.map(v => Math.round(v * 10) / 10);
  console.log('\nWeight comparison:');
  console.log('Name'.padEnd(16) + 'Current'.padStart(10) + 'Optimized'.padStart(12) + 'Change'.padStart(10));
  console.log('-'.repeat(48));
  for (let i = 0; i < WEIGHT_NAMES.length; i++) {
    const pct = Math.abs(CURRENT_WEIGHTS[i]) > 0
      ? ((rounded[i] - CURRENT_WEIGHTS[i]) / Math.abs(CURRENT_WEIGHTS[i]) * 100).toFixed(0)
      : 'N/A';
    console.log(
      WEIGHT_NAMES[i].padEnd(16) +
      String(CURRENT_WEIGHTS[i]).padStart(10) +
      String(rounded[i]).padStart(12) +
      `${pct > 0 ? '+' : ''}${pct}%`.padStart(10)
    );
  }

  console.log('\n// Copy this into simulator.html:');
  console.log(`const OPTIMIZED_WEIGHTS = [${rounded.join(', ')}];`);
  console.log(`// Names: ${WEIGHT_NAMES.join(', ')}`);
  console.log(`// Win rate vs original: ${(finalResult.winRate * 100).toFixed(1)}%`);

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\nTotal time: ${totalTime}s`);
}

optimize();
