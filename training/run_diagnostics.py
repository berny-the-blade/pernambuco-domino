"""
Diagnostics 1 + 2 — Value Calibration & Policy/Value/Search Agreement
=======================================================================

Run against any Phase 6.5-compatible checkpoint.

Diagnostic 1: Value Calibration Curve
  Plots predicted value vs actual outcome across bins.
  Reveals whether the value head is over/underconfident.

Diagnostic 2: Policy / Value-only / Search Agreement Audit
  For sampled positions, compares top-1 move from:
    • Raw policy head (greedy)
    • Value-only lookahead (evaluate each legal next-state)
    • Search at live budget (default 100 sims)
    • Reference oracle (high-budget search, default 800 sims)
  Reports accuracy vs oracle and agreement between heads.

Usage:
    python training/run_diagnostics.py --checkpoint checkpoints/best_100sims.pt
    python training/run_diagnostics.py \\
        --checkpoint checkpoints/best_100sims.pt \\
        --ref-checkpoint training/checkpoints/domino_gen_0050.pt \\
        --positions 500 --live-sims 100 --ref-sims 800 --no-plot
"""

import os, sys, copy, json, argparse, time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless — saves PNG instead of opening window
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domino_env     import DominoEnv, DominoMatch, TILES
from domino_net     import DominoNet
from domino_encoder import DominoEncoder
from domino_mcts    import DominoMCTS
from match_equity   import delta_me


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path, device):
    net = DominoNet()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    net.load_state_dict(state, strict=False)
    net.to(device).eval()
    return net


def value_only_move(env, encoder, model, device,
                    my_score, opp_score, multiplier):
    """
    Enumerate legal moves, evaluate each next-state with the value head,
    return the action index with the best predicted value.

    The value head returns from the perspective of the player who just moved
    (i.e., the state fed in is encoded from the next-to-move player's POV,
    so we negate for mid-game continuations where teams alternate).
    """
    mask = env.get_legal_moves_mask()
    legal = np.where(mask)[0]
    if len(legal) == 1:
        return int(legal[0])

    best_action, best_v = None, -np.inf

    for action in legal:
        env_copy    = copy.deepcopy(env)
        enc_copy    = copy.deepcopy(encoder)

        obs_next, _, done, info = env_copy.step(action)

        if done:
            # Terminal: use the game outcome directly
            # Outcome is from Team 0 perspective; flip if current mover is Team 1
            outcome = info.get("delta_me", 0.0)
            v = outcome
        else:
            # Encode from next player's perspective
            next_team    = env_copy.current_team
            ns           = 1 - next_team          # naive flip for 2-team game
            next_my_sc   = my_score  if next_team == 0 else opp_score
            next_opp_sc  = opp_score if next_team == 0 else my_score
            state_next   = enc_copy.encode(obs_next,
                                           my_score=next_my_sc,
                                           opp_score=next_opp_sc,
                                           multiplier=multiplier)
            mask_next    = env_copy.get_legal_moves_mask()
            _, v         = model.predict(state_next, mask_next, device)
            # Negate: value is from next-mover's POV; we want current-mover's POV
            if env_copy.current_team != env.current_team:
                v = -v

        if v > best_v:
            best_v, best_action = v, int(action)

    return best_action


def mcts_top1(mcts_obj, env, encoder):
    """Return the argmax of MCTS visit-count probabilities (temperature=0)."""
    probs = mcts_obj.get_action_probs(env, encoder, temperature=0.01)
    return int(np.argmax(probs))


def policy_top1(model, state_np, mask_np, device):
    """Return argmax of raw policy head output."""
    probs, _ = model.predict(state_np, mask_np, device)
    return int(np.argmax(probs))


# ─────────────────────────────────────────────────────────────────────────────
# Data collection — self-play with diagnostic logging
# ─────────────────────────────────────────────────────────────────────────────

def collect_positions(model, device, num_positions=500,
                      sample_every=3, seed=42):
    """
    Run self-play until we have `num_positions` diagnostic records.

    Each record:
        state_np   : encoded state
        mask_np    : legal moves mask
        v_pred     : value head prediction at this state
        outcome    : final delta_me outcome (from current player's POV)
        env        : deep-copied env snapshot for move evaluation
        encoder    : deep-copied encoder snapshot
        my_score   : match score (for context encoding)
        opp_score  : match score
        multiplier : match multiplier
    """
    np.random.seed(seed)
    records = []
    games_played = 0

    while len(records) < num_positions:
        match   = DominoMatch(target_points=6)
        encoder = DominoEncoder()
        obs     = match.new_game()
        encoder.reset()

        scores_before  = list(match.scores)
        mult_before    = match.multiplier
        game_positions = []   # buffer — flush when game ends

        step = 0
        while not match.env.is_over() and step < 200:
            env  = match.env
            mask = env.get_legal_moves_mask()

            if mask.sum() == 0:
                break

            my_team   = env.current_team
            my_sc     = match.scores[my_team]
            opp_sc    = match.scores[1 - my_team]
            state_np  = encoder.encode(obs,
                                       my_score=my_sc,
                                       opp_score=opp_sc,
                                       multiplier=match.multiplier)

            # Sample every Nth position to reduce correlation
            if step % sample_every == 0:
                _, v_pred = model.predict(state_np, mask, device)
                game_positions.append(dict(
                    state_np  = state_np.copy(),
                    mask_np   = mask.copy(),
                    v_pred    = v_pred,
                    env       = copy.deepcopy(env),
                    encoder   = copy.deepcopy(encoder),
                    my_score  = my_sc,
                    opp_score = opp_sc,
                    multiplier= match.multiplier,
                ))

            # Play out with policy (no MCTS to keep collection fast)
            probs, _ = model.predict(state_np, mask, device)
            action   = int(np.random.choice(len(probs), p=probs))
            obs, _, done, _ = match.env.step(action)
            encoder.update(obs)
            step += 1

            if done:
                break

        # Compute outcome for this game
        # Use match equity delta from team 0's perspective as ground truth
        outcome_t0 = delta_me(scores_before, list(match.scores),
                               mult_before, match.target_points)

        # Assign outcome to each saved position (from that player's POV)
        for rec in game_positions:
            team    = rec["env"].current_team
            outcome = outcome_t0 if team == 0 else -outcome_t0
            rec["outcome"] = outcome
            records.append(rec)

        games_played += 1
        if games_played % 10 == 0:
            print(f"  Collected {len(records)}/{num_positions} positions "
                  f"({games_played} games)...", flush=True)

        # Advance match if not over
        if not match.match_over:
            try:
                obs = match.new_game()
                encoder.reset()
                scores_before = list(match.scores)
                mult_before   = match.multiplier
            except Exception:
                pass

    return records[:num_positions]


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 1 — Value Calibration
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic_calibration(records, n_bins=10, out_path="diag1_calibration.png"):
    preds   = np.array([r["v_pred"]   for r in records])
    outcomes= np.array([r["outcome"]  for r in records])

    bins    = np.linspace(-1, 1, n_bins + 1)
    dig     = np.digitize(preds, bins)

    xs, ys, counts = [], [], []
    for i in range(1, len(bins)):
        mask = dig == i
        if mask.sum() < 10:
            continue
        xs.append(preds[mask].mean())
        ys.append(outcomes[mask].mean())
        counts.append(mask.sum())

    # ECE (expected calibration error)
    ece = sum(c * abs(y - x) for x, y, c in zip(xs, ys, counts)) / max(sum(counts), 1)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([-1, 1], [-1, 1], "k--", label="Perfect calibration", linewidth=1.5)
    sc = ax.scatter(xs, ys, c=counts, cmap="viridis", s=80, zorder=5)
    ax.plot(xs, ys, "o-", color="steelblue", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Sample count")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Actual outcome (mean)")
    ax.set_title(f"Diagnostic 1 — Value Calibration  (ECE={ece:.4f})")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  [Diag 1] Saved: {out_path}")

    return dict(ece=ece, xs=xs, ys=ys, counts=counts)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 2 — Policy / Value-only / Search Agreement
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic_agreement(records, model, device,
                          live_sims=100, ref_sims=800,
                          max_positions=300,
                          out_path="diag2_agreement.json"):
    """
    For each position (up to max_positions), compute:
        policy_move    : argmax(raw policy)
        value_move     : best value-only lookahead
        search_move    : MCTS @ live_sims
        ref_move       : MCTS @ ref_sims  (oracle)

    Then report accuracy vs oracle and pairwise agreement.
    """
    mcts_live = DominoMCTS(model, num_simulations=live_sims)
    mcts_ref  = DominoMCTS(model, num_simulations=ref_sims)

    subset = records[:max_positions]
    n      = len(subset)

    pol_correct = val_correct = srch_correct = 0
    pol_vs_srch = val_vs_srch = pol_vs_val  = 0

    print(f"  [Diag 2] Evaluating {n} positions "
          f"(live={live_sims} sims, ref={ref_sims} sims)...")

    for i, rec in enumerate(subset):
        env     = rec["env"]
        encoder = rec["encoder"]
        mask    = rec["mask_np"]
        state   = rec["state_np"]

        # Skip forced moves (no information content)
        if mask.sum() <= 1:
            n -= 1
            continue

        pol_m  = policy_top1(model, state, mask, device)
        val_m  = value_only_move(env, encoder, model, device,
                                 rec["my_score"], rec["opp_score"],
                                 rec["multiplier"])
        srch_m = mcts_top1(mcts_live, env, encoder)
        ref_m  = mcts_top1(mcts_ref,  env, encoder)

        pol_correct  += int(pol_m  == ref_m)
        val_correct  += int(val_m  == ref_m)
        srch_correct += int(srch_m == ref_m)
        pol_vs_srch  += int(pol_m  == srch_m)
        val_vs_srch  += int(val_m  == srch_m)
        pol_vs_val   += int(pol_m  == val_m)

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(subset)} done...", flush=True)

    if n == 0:
        print("  [Diag 2] No non-forced positions found.")
        return {}

    results = dict(
        n                  = n,
        live_sims          = live_sims,
        ref_sims           = ref_sims,
        policy_accuracy    = round(pol_correct  / n, 4),
        value_accuracy     = round(val_correct  / n, 4),
        search_accuracy    = round(srch_correct / n, 4),
        policy_vs_search   = round(pol_vs_srch  / n, 4),
        value_vs_search    = round(val_vs_srch  / n, 4),
        policy_vs_value    = round(pol_vs_val   / n, 4),
    )

    # Verdict
    pol_a  = results["policy_accuracy"]
    val_a  = results["value_accuracy"]
    srch_a = results["search_accuracy"]

    if   pol_a > val_a + 0.10 and srch_a < pol_a - 0.05:
        verdict = "VALUE_HURTING_SEARCH — value head is dragging down search"
    elif val_a > pol_a + 0.10:
        verdict = "POLICY_WEAK — policy prior is the main bottleneck"
    elif pol_a < 0.45 and val_a < 0.45:
        verdict = "BOTH_WEAK — representation may be the bottleneck"
    elif srch_a < pol_a - 0.05 and srch_a < val_a - 0.05:
        verdict = "SEARCH_INTEGRATION_ISSUE — search worse than both heads alone"
    else:
        verdict = "BALANCED — no single clear bottleneck"

    results["verdict"] = verdict

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [Diag 2] Saved: {out_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(label, cal, agr):
    print(f"\n{'='*60}")
    print(f"  RESULTS — {label}")
    print(f"{'='*60}")
    print(f"\n  Diagnostic 1 — Value Calibration")
    print(f"    ECE (lower=better):  {cal.get('ece', '?'):.4f}")
    if cal.get("xs"):
        mid = len(cal["xs"]) // 2
        print(f"    Sample bin (mid):    pred={cal['xs'][mid]:.2f}  "
              f"actual={cal['ys'][mid]:.2f}")

    print(f"\n  Diagnostic 2 — Move Agreement")
    if agr:
        print(f"    N positions:         {agr['n']}")
        print(f"    Policy accuracy:     {agr['policy_accuracy']:.1%}")
        print(f"    Value-only accuracy: {agr['value_accuracy']:.1%}")
        print(f"    Search accuracy:     {agr['search_accuracy']:.1%}")
        print(f"    Policy vs search:    {agr['policy_vs_search']:.1%}")
        print(f"    Value  vs search:    {agr['value_vs_search']:.1%}")
        print(f"    Policy vs value:     {agr['policy_vs_value']:.1%}")
        print(f"\n  >> VERDICT: {agr['verdict']}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Value calibration + policy/value/search audit")
    ap.add_argument("--checkpoint",  required=True,
                    help="Path to .pt checkpoint to evaluate")
    ap.add_argument("--ref-checkpoint", default=None,
                    help="Optional second checkpoint for side-by-side comparison "
                         "(e.g., production champion)")
    ap.add_argument("--positions",   type=int, default=500,
                    help="Number of diagnostic positions to collect (default 500)")
    ap.add_argument("--live-sims",   type=int, default=100,
                    help="MCTS sims for 'search' move (default 100, = live budget)")
    ap.add_argument("--ref-sims",    type=int, default=800,
                    help="MCTS sims for oracle reference move (default 800)")
    ap.add_argument("--diag2-positions", type=int, default=200,
                    help="Subset of positions for slow Diag 2 (default 200)")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--no-plot",     action="store_true",
                    help="Skip matplotlib output (just print + JSON)")
    ap.add_argument("--outdir",      default="logs",
                    help="Directory for output files (default: logs/)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.outdir, exist_ok=True)
    tag = Path(args.checkpoint).stem

    print(f"\n[1/4] Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    print(f"\n[2/4] Collecting {args.positions} positions via self-play...")
    t0      = time.time()
    records = collect_positions(model, device,
                                num_positions=args.positions,
                                seed=args.seed)
    print(f"  Done in {time.time()-t0:.1f}s  ({len(records)} records)")

    # ── Diagnostic 1 ──────────────────────────────────────────────
    print(f"\n[3/4] Running Diagnostic 1 (value calibration)...")
    cal_path = os.path.join(args.outdir, f"diag1_{tag}.png")
    cal = diagnostic_calibration(records,
                                 out_path=cal_path if not args.no_plot else "/dev/null")

    # ── Diagnostic 2 ──────────────────────────────────────────────
    print(f"\n[4/4] Running Diagnostic 2 (agreement audit, "
          f"{args.diag2_positions} positions)...")
    agr_path = os.path.join(args.outdir, f"diag2_{tag}.json")
    agr = diagnostic_agreement(records, model, device,
                                live_sims=args.live_sims,
                                ref_sims=args.ref_sims,
                                max_positions=args.diag2_positions,
                                out_path=agr_path)

    print_summary(tag, cal, agr)

    # ── Optional side-by-side comparison ──────────────────────────
    if args.ref_checkpoint:
        ref_tag = Path(args.ref_checkpoint).stem
        print(f"\n[+] Running comparison checkpoint: {args.ref_checkpoint}")
        ref_model   = load_model(args.ref_checkpoint, device)
        ref_records = collect_positions(ref_model, device,
                                        num_positions=args.positions,
                                        seed=args.seed + 1)
        ref_cal_path = os.path.join(args.outdir, f"diag1_{ref_tag}.png")
        ref_cal = diagnostic_calibration(ref_records,
                                         out_path=ref_cal_path if not args.no_plot else "/dev/null")
        ref_agr_path = os.path.join(args.outdir, f"diag2_{ref_tag}.json")
        ref_agr = diagnostic_agreement(ref_records, ref_model, device,
                                        live_sims=args.live_sims,
                                        ref_sims=args.ref_sims,
                                        max_positions=args.diag2_positions,
                                        out_path=ref_agr_path)
        print_summary(ref_tag, ref_cal, ref_agr)

        # Combined comparison table
        print(f"\n  {'Metric':<28} {'Challenger':>14} {'Champion':>14}")
        print(f"  {'-'*56}")
        for key in ("policy_accuracy", "value_accuracy", "search_accuracy",
                    "policy_vs_search", "value_vs_search"):
            if agr and ref_agr:
                label = key.replace("_", " ").title()
                print(f"  {label:<28} {agr[key]:>13.1%} {ref_agr[key]:>13.1%}")
        print(f"\n  Challenger verdict: {agr.get('verdict','?')}")
        print(f"  Champion verdict:   {ref_agr.get('verdict','?')}")
        print()


if __name__ == "__main__":
    main()
