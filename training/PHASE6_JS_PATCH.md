# PHASE6_JS_PATCH.md
# JS patch: rootVisitPolicy() + ismctsEval() policy export
# Source: Bernie analysis 2026-03-08
# Purpose: export 57-dim visit-count policy from search root for Python trainer

---

# Patch 1 — Add helpers near _nnActionMask() / action utilities

```js
// Convert symbolic move -> canonical 57-action index
// 0..27 = tile-left, 28..55 = tile-right, 56 = pass
function _moveToActionIdx(move) {
    if (!move) return -1;
    if (move.pass) return 56;
    if (!move.tile || move.tile.id == null) return -1;

    const tileIdx = TILE_INDEX[move.tile.id];
    if (tileIdx == null) return -1;

    if (move.side === 'right') return 28 + tileIdx;
    // "left" and "open" both map to left-side convention
    return tileIdx;
}

// Extract 57-dim visit-count policy from root children (NO dedup)
// legalMask is optional but recommended.
// If root has no visits, returns uniform over legal actions.
function rootVisitPolicy(root, legalMask, tau = 1.0) {
    const pi = new Float32Array(NN_NUM_ACTIONS);
    let total = 0.0;

    for (const child of root.children) {
        if (!child || child.visits <= 0) continue;

        const a = (child.actionIdx != null) ? child.actionIdx : _moveToActionIdx(child.move);
        if (a < 0 || a >= NN_NUM_ACTIONS) continue;
        if (legalMask && !legalMask[a]) continue;

        const w = (tau === 1.0) ? child.visits : Math.pow(child.visits, 1.0 / tau);
        pi[a] += w;
        total += w;
    }

    if (total > 0) {
        for (let a = 0; a < NN_NUM_ACTIONS; a++) pi[a] /= total;
        return pi;
    }

    // Fallback: uniform over legal
    if (legalMask) {
        let nLegal = 0;
        for (let a = 0; a < NN_NUM_ACTIONS; a++) if (legalMask[a]) nLegal++;
        if (nLegal > 0) {
            const u = 1.0 / nLegal;
            for (let a = 0; a < NN_NUM_ACTIONS; a++) if (legalMask[a]) pi[a] = u;
        }
    }
    return pi;
}
```

---

# Patch 2 — Extend ismctsEval() signature

Change:
```js
function ismctsEval(hand, lE, rE, bLen, player, knowledge, beliefModel) {
```
To:
```js
function ismctsEval(hand, lE, rE, bLen, player, knowledge, beliefModel, returnPolicy = false, policyTau = 1.0) {
```
All existing callers are unaffected (new params default to false/1.0).

---

# Patch 3 — Replace root extraction block in ismctsEval()

Find the existing block that starts with:
```js
    // Extract results from root's children
    if (root.children.length === 0) return [];
```

Replace the entire block with:

```js
    // Extract results from root's children
    if (root.children.length === 0) {
        return returnPolicy ? { results: [], pi: new Float32Array(NN_NUM_ACTIONS) } : [];
    }

    const legalMask = _nnActionMask(hand, lE, rE, bLen);
    const pi57 = rootVisitPolicy(root, legalMask, policyTau);

    // Keep current UI/display behavior: dedup by tile for ranked move display
    const tileResults = new Map();
    for (const child of root.children) {
        if (!child || child.visits === 0) continue;

        const avgReward = child.totalReward / child.visits;
        const actionIdx = (child.actionIdx != null) ? child.actionIdx : _moveToActionIdx(child.move);

        // Note: expectedPoints naming kept for backward compatibility/UI,
        // but avgReward is now effectively ΔME / reward-space, not literal points.
        const r = {
            tile: child.move.tile,
            side: child.move.side,
            actionIdx,
            winRate: (avgReward + 1) / 2,       // legacy display only
            expectedPoints: avgReward * 4,       // legacy display only
            avgReward: avgReward,
            visits: child.visits,
            blockRate: 0,
            sims: child.visits,
            outcomes: { normal: 0, carroca: 0, laelo: 0, cruzada: 0, blocked: 0, tie: 0 },
            variance: 0, stdDev: 0, ci95: 0, cvar5: 0,
            ismcts: true
        };

        const tileKey = child.move.tile.id;
        if (!tileResults.has(tileKey) || r.avgReward > tileResults.get(tileKey).avgReward) {
            tileResults.set(tileKey, r);
        }
    }

    const finalResults = [...tileResults.values()];
    finalResults.sort((a, b) => b.avgReward - a.avgReward || b.visits - a.visits);

    if (returnPolicy) {
        return { results: finalResults, pi: pi57 };
    }
    return finalResults;
```

### Why this is correct
- `pi57` is computed BEFORE dedup, from raw child visits — no information lost
- Display/UI behavior is preserved exactly
- Old callers still get the same array of ranked moves
- New training/export callers ask for `{ results, pi }`

---

# Patch 4 — (optional) Store actionIdx at node expand time

If ISMCTSNode.expand(move) creates a child, add:
```js
child.actionIdx = _moveToActionIdx(move);
```
Not strictly required (the helper fallback derives it on demand), but cleaner.

---

# Patch 5 — Export path usage

Wherever training export calls ismctsEval(), change to:
```js
const out = ismctsEval(hand, lE, rE, bLen, player, knowledge, beliefModel, true, 1.0);
const rankedMoves = out.results;
const policy57  = Array.from(out.pi);

// Then write training record:
const record = {
    snapshot: ...,
    x:    Array.from(_nnEncodeState(hand, lE, rE, bLen, player, knowledge)),
    mask: Array.from(_nnActionMask(hand, lE, rE, bLen)),
    pi:   policy57,
    // v: backfilled later as ΔME
};
```

---

# Patch 6 — Temperature

For training always start with:
```js
policyTau = 1.0
```
Can experiment with 0.8 / 0.5 later. 1.0 is the correct default.

---

# Quick validation (browser console)

```js
const out = ismctsEval(hand, lE, rE, bLen, player, knowledge, null, true, 1.0);
out.pi.length                                              // → 57
out.results.length                                         // → ranked display moves
out.pi.reduce((a,b)=>a+b,0)                               // → ~1.0

// No probability mass on illegal actions:
const mask = _nnActionMask(hand, lE, rE, bLen);
out.pi.filter((p,i)=>mask[i]===0 && p>1e-9).length       // → 0
```

---

# Files to patch
- `index.html` (simulator used for training data export)
- `simulator.html` if it has its own copy of ismctsEval()
