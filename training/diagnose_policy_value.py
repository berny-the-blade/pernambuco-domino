"""
diagnose_policy_value.py — Policy / Value / Search Bottleneck Diagnostics
==========================================================================

Diagnostic 1: Value calibration curve
  Bins value predictions vs actual outcomes.
  → value_calibration.csv

Diagnostic 2: Policy / value-only / search agreement audit
  Compares top-1 move from:
    - Raw policy head (challenger)
    - Value-only lookahead (challenger)
    - Live search @ live_sims (challenger)
    - Reference oracle @ ref_sims (reference/production champion model)
  → agreement_audit.csv

→ summary.json

Usage:
    # Step 1 — collect positions:
    python training/collect_positions.py \\
        --checkpoint checkpoints/best_100sims.pt \\
        --positions 500 --out diagnostics/positions.pkl

    # Step 2 — diagnose:
    python training/diagnose_policy_value.py \\
        --positions diagnostics/positions.pkl \\
        --checkpoint checkpoints/best_100sims.pt \\
        --reference-checkpoint training/checkpoints/domino_gen_0050.pt \\
        --live-sims 100 --ref-sims 800 --max-positions 500 \\
        --device cpu --out-dir diagnostics/pv_audit_run1
"""

import argparse
import copy
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domino_net  import DominoNet
from domino_mcts import DominoMCTS


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Diagnose policy/value/search bottlenecks.")
    p.add_argument("--positions",             type=str, required=True,
                   help="Pickle file from collect_positions.py")
    p.add_argument("--checkpoint",            type=str, required=True,
                   help="Challenger checkpoint")
    p.add_argument("--reference-checkpoint",  type=str, required=True,
                   help="Reference / production champion checkpoint")
    p.add_argument("--live-sims",             type=int, default=100,
                   help="Search budget for live-search move (default 100)")
    p.add_argument("--ref-sims",              type=int, default=800,
                   help="Search budget for reference oracle move (default 800)")
    p.add_argument("--max-positions",         type=int, default=500,
                   help="Max positions to evaluate (default 500)")
    p.add_argument("--device",                type=str, default="cpu",
                   help="cpu or cuda")
    p.add_argument("--out-dir",               type=str, default="diagnostics/latest",
                   help="Output directory")
    return p.parse_args()


def load_pickle_positions(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of position records, got {type(data)}")
    return data


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DominoNet()
    incompat = model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    if getattr(incompat, "missing_keys", None):
        print(f"[load_model] Missing keys for {checkpoint_path}: {incompat.missing_keys}")
    if getattr(incompat, "unexpected_keys", None):
        print(f"[load_model] Unexpected keys for {checkpoint_path}: {incompat.unexpected_keys}")
    model.to(device).eval()
    return model


def legal_actions_from_mask(mask_np):
    return np.where(mask_np > 0.5)[0]


def policy_move(model, pos, device):
    policy_np, _ = model.predict(pos["state_np"], pos["mask_np"], device)
    return int(np.argmax(policy_np)), policy_np


def value_only_move(model, pos, device):
    """
    For each legal move:
      - step env copy
      - update encoder
      - encode next state
      - evaluate with value head
    Pick move with best value from current player's team POV.
    """
    env        = pos["env"]
    encoder    = pos["encoder"]
    my_score   = pos["my_score"]
    opp_score  = pos["opp_score"]
    multiplier = pos["multiplier"]

    legal = legal_actions_from_mask(pos["mask_np"])
    if len(legal) == 0:
        raise ValueError("No legal actions in position")

    cur_team = env.current_team if hasattr(env, "current_team") else (pos["player"] % 2)

    best_action = None
    best_value  = -1e18

    for action_idx in legal:
        env_copy = copy.deepcopy(env)
        enc_copy = copy.deepcopy(encoder)

        obs_next, reward, done, info = env_copy.step(int(action_idx))
        enc_copy.update(obs_next)

        state_next = enc_copy.encode(obs_next, my_score, opp_score, multiplier)
        mask_next  = env_copy.get_legal_moves_mask()

        _, value_next = model.predict(state_next, mask_next, device)

        next_team = env_copy.current_team if hasattr(env_copy, "current_team") else cur_team

        # Convert back to current team's perspective if needed
        if next_team != cur_team:
            value_next = -value_next

        if value_next > best_value:
            best_value  = value_next
            best_action = int(action_idx)

    return best_action, float(best_value)


def search_move(model, pos, sims):
    mcts  = DominoMCTS(model, num_simulations=sims)
    probs = mcts.get_action_probs(pos["env"], pos["encoder"], temperature=0.01)
    return int(np.argmax(probs)), probs


# ─────────────────────────────────────────────────────────────────────────────

def summarize_value_calibration(rows, out_dir: Path):
    preds    = np.array([r["v_pred"]  for r in rows], dtype=np.float32)
    outcomes = np.array([r["outcome"] for r in rows], dtype=np.float32)

    mse      = float(np.mean((preds - outcomes) ** 2))
    sign_acc = float(np.mean(np.sign(preds) == np.sign(outcomes)))

    bins      = np.linspace(-1.0, 1.0, 11)
    digitized = np.digitize(preds, bins)

    calib_rows = []
    for i in range(1, len(bins)):
        mask = digitized == i
        n    = int(mask.sum())
        if n == 0:
            continue
        calib_rows.append({
            "bin_left":     float(bins[i - 1]),
            "bin_right":    float(bins[i]),
            "count":        n,
            "pred_mean":    float(preds[mask].mean()),
            "outcome_mean": float(outcomes[mask].mean()),
        })

    with open(out_dir / "value_calibration.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bin_left", "bin_right", "count",
                                               "pred_mean", "outcome_mean"])
        writer.writeheader()
        writer.writerows(calib_rows)

    return {
        "mse":          mse,
        "sign_accuracy": sign_acc,
        "n_rows":       len(rows),
    }


def summarize_agreement(rows, out_dir: Path):
    def mean_key(k):
        return float(np.mean([r[k] for r in rows])) if rows else 0.0

    summary = {
        "policy_correct":       mean_key("policy_correct"),
        "value_correct":        mean_key("value_correct"),
        "search_correct":       mean_key("search_correct"),
        "policy_search_agree":  mean_key("policy_search_agree"),
        "value_search_agree":   mean_key("value_search_agree"),
        "policy_value_agree":   mean_key("policy_value_agree"),
        "n_rows":               len(rows),
    }

    with open(out_dir / "agreement_audit.csv", "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Auto-verdict
    pa = summary["policy_correct"]
    va = summary["value_correct"]
    sa = summary["search_correct"]

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

    summary["verdict"]        = verdict
    summary["verdict_detail"] = detail
    return summary


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading positions from {args.positions}")
    positions = load_pickle_positions(args.positions)[:args.max_positions]
    print(f"Loaded {len(positions)} positions")

    print(f"Loading challenger:  {args.checkpoint}")
    challenger = load_model(args.checkpoint, args.device)

    print(f"Loading reference:   {args.reference_checkpoint}")
    reference  = load_model(args.reference_checkpoint, args.device)

    calib_rows = []
    audit_rows = []

    for i, pos in enumerate(positions):
        if i % 50 == 0:
            print(f"  ... {i}/{len(positions)}", flush=True)

        # Skip forced moves — no information content
        if legal_actions_from_mask(pos["mask_np"]).size <= 1:
            continue

        _, v_pred = challenger.predict(pos["state_np"], pos["mask_np"], args.device)
        calib_rows.append({
            "idx":     i,
            "v_pred":  float(v_pred),
            "outcome": float(pos["outcome"]),
        })

        policy_top1, policy_probs   = policy_move(challenger, pos, args.device)
        value_top1,  value_score    = value_only_move(challenger, pos, args.device)
        search_top1, search_probs   = search_move(challenger, pos, args.live_sims)
        ref_top1,    ref_probs      = search_move(reference,  pos, args.ref_sims)

        audit_rows.append({
            "idx":                i,
            "policy_top1":        int(policy_top1),
            "value_top1":         int(value_top1),
            "search_top1":        int(search_top1),
            "ref_top1":           int(ref_top1),
            "policy_correct":     int(policy_top1  == ref_top1),
            "value_correct":      int(value_top1   == ref_top1),
            "search_correct":     int(search_top1  == ref_top1),
            "policy_search_agree": int(policy_top1 == search_top1),
            "value_search_agree":  int(value_top1  == search_top1),
            "policy_value_agree":  int(policy_top1 == value_top1),
            "policy_top1_prob":   float(policy_probs[policy_top1]),
            "search_top1_prob":   float(search_probs[search_top1]),
            "ref_top1_prob":      float(ref_probs[ref_top1]),
            "value_top1_score":   float(value_score),
            "outcome":            float(pos["outcome"]),
            "v_pred":             float(v_pred),
            "board_len":          int(pos["board_len"]),
            "player":             int(pos["player"]),
            "move_idx":           int(pos["move_idx"]),
        })

    calib_summary     = summarize_value_calibration(calib_rows, out_dir)
    agreement_summary = summarize_agreement(audit_rows, out_dir)

    summary = {
        "checkpoint":           args.checkpoint,
        "reference_checkpoint": args.reference_checkpoint,
        "positions":            len(positions),
        "live_sims":            args.live_sims,
        "ref_sims":             args.ref_sims,
        "value_calibration":    calib_summary,
        "agreement":            agreement_summary,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nVerdict: {agreement_summary['verdict']}")
    print(f"Detail:  {agreement_summary['verdict_detail']}")


if __name__ == "__main__":
    main()
