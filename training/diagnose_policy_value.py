"""
diagnose_policy_value.py — Policy / Value / Search Diagnostics
==============================================================

Loads a pre-collected positions file (from collect_positions.py) and runs:

  Diagnostic A — Value Calibration Curve
    Bins value predictions vs actual outcomes.
    Outputs: value_calibration.csv, value_calibration.png

  Diagnostic B — Policy / Value-only / Search Agreement Audit
    Compares top-1 move from policy / value-lookahead / live search / ref search.
    Outputs: agreement_audit.csv, policy_value_search_bar.png

  summary.json — Combined metrics and auto-verdict

Usage:
    # First collect positions:
    python training/collect_positions.py \\
        --checkpoint checkpoints/best_100sims.pt \\
        --positions 1000 --out diagnostics/positions.pkl

    # Then diagnose:
    python training/diagnose_policy_value.py \\
        --checkpoint checkpoints/best_100sims.pt \\
        --positions diagnostics/positions.pkl \\
        --live-sims 100 --ref-sims 800

    # Side-by-side vs champion:
    python training/diagnose_policy_value.py \\
        --checkpoint checkpoints/best_100sims.pt \\
        --positions diagnostics/positions.pkl \\
        --ref-checkpoint training/checkpoints/domino_gen_0050.pt
"""

import os, sys, copy, json, csv, pickle, argparse, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domino_net  import DominoNet
from domino_mcts import DominoMCTS


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path, device):
    """Load any Phase 5 / 6 / 6.5 checkpoint with strict=False."""
    net  = DominoNet()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    incomp = net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    if incomp.missing_keys:
        print(f"  [load] Missing keys (random-init): {incomp.missing_keys[:5]}...")
    return net.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Move selectors
# ─────────────────────────────────────────────────────────────────────────────

def policy_top1(model, rec, device):
    """Argmax of raw masked policy head."""
    probs, _ = model.predict(rec["state_np"], rec["mask_np"], device)
    return int(np.argmax(probs))


def value_only_top1(model, rec, device):
    """
    Evaluate each legal next-state with the value head.
    Returns the action with the highest predicted value from current player's POV.
    """
    mask    = rec["mask_np"]
    legal   = np.where(mask)[0]
    if len(legal) == 1:
        return int(legal[0])

    env     = rec["env"]
    encoder = rec["encoder"]
    best_a, best_v = None, -np.inf

    for action in legal:
        env_c = copy.deepcopy(env)
        enc_c = copy.deepcopy(encoder)

        obs_next, _, done, info = env_c.step(action)

        if done:
            v = float(info.get("delta_me", 0.0))
        else:
            team_next   = env_c.current_team
            next_my_sc  = rec["my_score"]  if team_next == 0 else rec["opp_score"]
            next_opp_sc = rec["opp_score"] if team_next == 0 else rec["my_score"]
            state_next  = enc_c.encode(obs_next, next_my_sc, next_opp_sc,
                                       rec["multiplier"])
            mask_next   = env_c.get_legal_moves_mask()
            _, v        = model.predict(state_next, mask_next, device)
            # Negate if opponent is now to move
            if env_c.current_team != env.current_team:
                v = -v

        if v > best_v:
            best_v, best_a = v, int(action)

    return best_a


def search_top1(mcts_obj, rec):
    """Top-1 from MCTS visit-count distribution (temperature→0)."""
    probs = mcts_obj.get_action_probs(rec["env"], rec["encoder"], temperature=0.01)
    return int(np.argmax(probs))


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic A — Value Calibration
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(records, model, device, n_bins=10):
    """
    Returns per-record rows and bin summary.
    """
    rows = []
    for i, rec in enumerate(records):
        _, v_pred = model.predict(rec["state_np"], rec["mask_np"], device)
        rows.append(dict(idx=i, v_pred=float(v_pred), outcome=float(rec["outcome"])))

    # Bin summary
    preds    = np.array([r["v_pred"]  for r in rows])
    outcomes = np.array([r["outcome"] for r in rows])
    edges    = np.linspace(-1, 1, n_bins + 1)
    dig      = np.digitize(preds, edges)

    bins = []
    for i in range(1, len(edges)):
        m = dig == i
        if m.sum() < 5:
            continue
        bins.append(dict(
            bin_mid         = round(float(preds[m].mean()), 4),
            mean_predicted  = round(float(preds[m].mean()), 4),
            mean_actual     = round(float(outcomes[m].mean()), 4),
            count           = int(m.sum()),
        ))

    # Summary metrics
    mse        = float(np.mean((preds - outcomes) ** 2))
    brier      = float(np.mean(((preds + 1) / 2 - (outcomes + 1) / 2) ** 2))
    sign_acc   = float(np.mean(np.sign(preds) == np.sign(outcomes)))
    # Calibration slope via linear regression
    if len(preds) > 2:
        slope, intercept = float(np.polyfit(preds, outcomes, 1))
    else:
        slope, intercept = 1.0, 0.0
    ece = sum(b["count"] * abs(b["mean_actual"] - b["mean_predicted"]) for b in bins) \
          / max(sum(b["count"] for b in bins), 1)

    metrics = dict(mse=round(mse, 5), brier=round(brier, 5),
                   sign_accuracy=round(sign_acc, 4),
                   calibration_slope=round(slope, 4),
                   calibration_intercept=round(intercept, 4),
                   ece=round(ece, 4))

    return rows, bins, metrics


def plot_calibration(bins, metrics, out_path, label=""):
    xs = [b["mean_predicted"] for b in bins]
    ys = [b["mean_actual"]    for b in bins]
    cs = [b["count"]          for b in bins]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([-1, 1], [-1, 1], "k--", lw=1.5, label="Perfect calibration")
    # Regression line
    slope = metrics["calibration_slope"]
    intercept = metrics["calibration_intercept"]
    xs_fit = np.array([-1, 1])
    ax.plot(xs_fit, slope * xs_fit + intercept, "r-", lw=1, alpha=0.6,
            label=f"Fitted  (slope={slope:.2f})")
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=80, zorder=5)
    ax.plot(xs, ys, "o-", color="steelblue", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Sample count")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Actual outcome (mean)")
    title = f"Value Calibration{' — ' + label if label else ''}"
    ax.set_title(f"{title}\nECE={metrics['ece']:.4f}  slope={slope:.3f}  "
                 f"sign_acc={metrics['sign_accuracy']:.1%}")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic B — Agreement Audit
# ─────────────────────────────────────────────────────────────────────────────

def run_agreement(records, model, device, live_sims=100, ref_sims=800,
                  max_positions=300):
    mcts_live = DominoMCTS(model, num_simulations=live_sims)
    mcts_ref  = DominoMCTS(model, num_simulations=ref_sims)

    subset = [r for r in records[:max_positions] if r["mask_np"].sum() > 1]
    n = len(subset)
    print(f"  Agreement audit: {n} non-forced positions  "
          f"(live={live_sims}, ref={ref_sims} sims)...")

    rows = []
    for i, rec in enumerate(subset):
        pol_m  = policy_top1   (model, rec, device)
        val_m  = value_only_top1(model, rec, device)
        srch_m = search_top1   (mcts_live, rec)
        ref_m  = search_top1   (mcts_ref,  rec)

        rows.append(dict(
            idx                 = i,
            policy_top1         = pol_m,
            value_top1          = val_m,
            search_top1         = srch_m,
            ref_top1            = ref_m,
            policy_correct      = int(pol_m  == ref_m),
            value_correct       = int(val_m  == ref_m),
            search_correct      = int(srch_m == ref_m),
            policy_search_agree = int(pol_m  == srch_m),
            value_search_agree  = int(val_m  == srch_m),
            policy_value_agree  = int(pol_m  == val_m),
        ))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n}...", flush=True)

    # Summary
    def avg(key): return round(sum(r[key] for r in rows) / max(n, 1), 4)

    summary = dict(
        n                       = n,
        live_sims               = live_sims,
        ref_sims                = ref_sims,
        policy_accuracy         = avg("policy_correct"),
        value_accuracy          = avg("value_correct"),
        search_accuracy         = avg("search_correct"),
        policy_search_agreement = avg("policy_search_agree"),
        value_search_agreement  = avg("value_search_agree"),
        policy_value_agreement  = avg("policy_value_agree"),
    )

    # Auto-verdict
    pa = summary["policy_accuracy"]
    va = summary["value_accuracy"]
    sa = summary["search_accuracy"]

    if   pa > va + 0.10 and sa < pa - 0.05:
        verdict = "VALUE_HURTING_SEARCH"
        detail  = "Policy knows the move; value mis-ranks leaf states → search degraded"
    elif va > pa + 0.10:
        verdict = "POLICY_WEAK"
        detail  = "Value head carries search; policy prior is the bottleneck"
    elif pa < 0.45 and va < 0.45:
        verdict = "BOTH_WEAK"
        detail  = "Representation / training quality is still the main bottleneck"
    elif sa < pa - 0.05 and sa < va - 0.05:
        verdict = "SEARCH_INTEGRATION_ISSUE"
        detail  = "Search is worse than both heads alone — value calibration or PUCT weighting"
    else:
        verdict = "BALANCED"
        detail  = "No single clear bottleneck identified"

    summary["verdict"] = verdict
    summary["verdict_detail"] = detail
    return rows, summary


def plot_agreement_bar(summary, out_path, label=""):
    metrics = ["policy_accuracy", "value_accuracy", "search_accuracy",
               "policy_search_agreement", "value_search_agreement",
               "policy_value_agreement"]
    short   = ["Policy\naccuracy", "Value\naccuracy", "Search\naccuracy",
               "Policy\nvs search", "Value\nvs search", "Policy\nvs value"]
    vals    = [summary[m] for m in metrics]
    colors  = ["#4c78a8"] * 3 + ["#72b7b2"] * 3

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(short, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0.5, color="gray", lw=1, ls="--")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    title = f"Policy / Value / Search Agreement{' — ' + label if label else ''}"
    ax.set_title(f"{title}\nVerdict: {summary['verdict']}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Run one checkpoint end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def run_all(records, checkpoint_path, device, live_sims, ref_sims,
            max_diag_b, out_dir):
    label = Path(checkpoint_path).stem
    os.makedirs(out_dir, exist_ok=True)

    model = load_model(checkpoint_path, device)

    # ── Diagnostic A ─────────────────────────────────────────────────
    print(f"\n[A] Value calibration ({len(records)} positions)...")
    cal_rows, cal_bins, cal_metrics = run_calibration(records, model, device)
    write_csv(cal_rows, os.path.join(out_dir, "value_calibration.csv"))
    plot_calibration(cal_bins, cal_metrics,
                     os.path.join(out_dir, "value_calibration.png"), label=label)

    # ── Diagnostic B ─────────────────────────────────────────────────
    print(f"\n[B] Agreement audit (up to {max_diag_b} positions)...")
    agr_rows, agr_summary = run_agreement(records, model, device,
                                          live_sims=live_sims, ref_sims=ref_sims,
                                          max_positions=max_diag_b)
    write_csv(agr_rows, os.path.join(out_dir, "agreement_audit.csv"))
    plot_agreement_bar(agr_summary,
                       os.path.join(out_dir, "policy_value_search_bar.png"),
                       label=label)

    # ── Summary JSON ─────────────────────────────────────────────────
    summary = dict(
        checkpoint  = str(checkpoint_path),
        label       = label,
        n_positions = len(records),
        calibration = cal_metrics,
        agreement   = agr_summary,
    )
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Console report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(s):
    c = s["calibration"]
    a = s["agreement"]
    print(f"\n{'='*60}")
    print(f"  {s['label']}  ({s['n_positions']} positions)")
    print(f"{'='*60}")
    print(f"\n  ── Calibration ──")
    print(f"  ECE:               {c['ece']:.4f}  (0 = perfect)")
    print(f"  Slope:             {c['calibration_slope']:.3f}  (1.0 = perfect)")
    print(f"  Sign accuracy:     {c['sign_accuracy']:.1%}")
    print(f"  MSE:               {c['mse']:.5f}")
    print(f"\n  ── Agreement  (n={a['n']}) ──")
    print(f"  Policy accuracy:   {a['policy_accuracy']:.1%}  (vs {a['ref_sims']}-sim oracle)")
    print(f"  Value accuracy:    {a['value_accuracy']:.1%}")
    print(f"  Search accuracy:   {a['search_accuracy']:.1%}  (@{a['live_sims']} sims)")
    print(f"  Policy vs search:  {a['policy_search_agreement']:.1%}")
    print(f"  Value  vs search:  {a['value_search_agreement']:.1%}")
    print(f"  Policy vs value:   {a['policy_value_agreement']:.1%}")
    print(f"\n  >> {a['verdict']}: {a['verdict_detail']}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Policy / value / search diagnostics for Pernambuco Domino AI")
    ap.add_argument("--checkpoint",         required=True,
                    help="Checkpoint to evaluate (.pt)")
    ap.add_argument("--positions",          required=True,
                    help="Path to positions pickle (from collect_positions.py)")
    ap.add_argument("--live-sims",          type=int, default=100)
    ap.add_argument("--ref-sims",           type=int, default=800)
    ap.add_argument("--max-positions",      type=int, default=1000,
                    help="Cap on positions used for Diag A calibration")
    ap.add_argument("--max-diag-b",         type=int, default=200,
                    help="Cap on positions used for Diag B agreement (slow)")
    ap.add_argument("--device",             default=None,
                    help="cuda|cpu (auto-detect if omitted)")
    ap.add_argument("--out-dir",            default=None,
                    help="Output directory (default: diagnostics/run_YYYYMMDD_HHMM/)")
    ap.add_argument("--ref-checkpoint",     default=None,
                    help="Optional second checkpoint for side-by-side comparison")
    ap.add_argument("--ref-positions",      default=None,
                    help="Separate positions file for ref-checkpoint (optional; "
                         "reuses --positions if omitted)")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Default output dir
    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.out_dir = os.path.join("diagnostics", f"run_{stamp}")

    # Load positions
    print(f"\nLoading positions: {args.positions}")
    with open(args.positions, "rb") as f:
        records = pickle.load(f)
    records = records[:args.max_positions]
    print(f"  {len(records)} records loaded")

    # Run challenger
    tag      = Path(args.checkpoint).stem
    out_dir  = os.path.join(args.out_dir, tag)
    summary  = run_all(records, args.checkpoint, device,
                       args.live_sims, args.ref_sims, args.max_diag_b, out_dir)
    print_report(summary)

    # Optional comparison checkpoint
    if args.ref_checkpoint:
        ref_records = records   # reuse same positions by default
        if args.ref_positions:
            print(f"\nLoading ref positions: {args.ref_positions}")
            with open(args.ref_positions, "rb") as f:
                ref_records = pickle.load(f)[:args.max_positions]

        ref_tag     = Path(args.ref_checkpoint).stem
        ref_out_dir = os.path.join(args.out_dir, ref_tag)
        ref_summary = run_all(ref_records, args.ref_checkpoint, device,
                              args.live_sims, args.ref_sims,
                              args.max_diag_b, ref_out_dir)
        print_report(ref_summary)

        # Comparison table
        print(f"{'='*60}")
        print(f"  COMPARISON — {tag}  vs  {ref_tag}")
        print(f"{'='*60}")
        ca, ra = summary["agreement"], ref_summary["agreement"]
        cc, rc = summary["calibration"], ref_summary["calibration"]
        rows = [
            ("ECE",              cc["ece"],               rc["ece"],               True),
            ("Calib slope",      cc["calibration_slope"], rc["calibration_slope"], False),
            ("Sign accuracy",    cc["sign_accuracy"],     rc["sign_accuracy"],     False),
            ("Policy accuracy",  ca["policy_accuracy"],   ra["policy_accuracy"],   False),
            ("Value accuracy",   ca["value_accuracy"],    ra["value_accuracy"],    False),
            ("Search accuracy",  ca["search_accuracy"],   ra["search_accuracy"],   False),
            ("Policy vs search", ca["policy_search_agreement"], ra["policy_search_agreement"], False),
            ("Value  vs search", ca["value_search_agreement"],  ra["value_search_agreement"],  False),
        ]
        print(f"\n  {'Metric':<24} {'Challenger':>14} {'Reference':>14}")
        print(f"  {'-'*52}")
        for name, chall, ref, lower_is_better in rows:
            better = (chall < ref) if lower_is_better else (chall > ref)
            marker = " ✓" if better else "  "
            print(f"  {name:<24} {chall:>13.4f} {ref:>13.4f}{marker}")
        print(f"\n  Challenger: {summary['agreement']['verdict']}")
        print(f"  Reference:  {ref_summary['agreement']['verdict']}")
        print()


if __name__ == "__main__":
    main()
