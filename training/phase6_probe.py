"""
phase6_probe.py — Phase 6 belief-head λ sweep experiment.

Runs 3 short 5-gen training probes starting from the same base checkpoint,
each with a different belief auxiliary weight:
    Probe A: λ_belief = 0.1
    Probe B: λ_belief = 0.2
    Probe C: λ_belief = 0.3

After each probe, evaluates the resulting model on:
  1. Partnership suite  (if available)
  2. Search scaling     (50/100/200/400 sims, 50 duplicate pairs)
  3. Anchor eval        (vs gen 50 at 100 sims, 200 games)

Results written to: training/logs/phase6_probe_results.json

Usage:
    # From repo root:
    python training/phase6_probe.py

    # Specify base checkpoint explicitly:
    python training/phase6_probe.py --base-checkpoint training/checkpoints/domino_gen_0100.pt

    # Dry run (no training, just evaluate base checkpoint 3 times):
    python training/phase6_probe.py --dry-run

    # Fewer generations per probe (faster, noisier):
    python training/phase6_probe.py --gens-per-probe 3

    # Control workers and games:
    python training/phase6_probe.py --workers 4 --games-per-worker 100

Notes:
    - Each probe is INDEPENDENT: all start from the same base checkpoint.
    - Probes run sequentially (not parallel) to avoid GPU contention.
    - Training is CPU self-play + GPU training (same as orchestrator.py).
    - Checkpoints saved as: checkpoints/phase6_probe_{lambda}/domino_gen_{N:04d}.pt
    - If partnership suite (tests/test_partnership_suite.py) is missing, that
      metric is skipped gracefully.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

# Allow running from repo root or from training/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from domino_env import DominoEnv, DominoMatch, TILES
from domino_net import DominoNet
from domino_encoder import DominoEncoder
from domino_mcts import DominoMCTS
from domino_trainer import Trainer, ReplayDataset
from match_equity import get_match_equity, delta_me
from orchestrator import self_play_worker, build_belief_target

CHECKPOINTS_DIR = os.path.join(_THIS_DIR, "checkpoints")
LOGS_DIR        = os.path.join(_THIS_DIR, "logs")
RESULTS_PATH    = os.path.join(LOGS_DIR, "phase6_probe_results.json")

LAMBDA_VALUES   = [0.1, 0.2, 0.3]
DEFAULT_GENS    = 5
DEFAULT_WORKERS = max(2, (os.cpu_count() or 4) - 1)
DEFAULT_GAMES   = 150   # games per worker per gen (small for a probe)
EVAL_SIMS_LIST  = [50, 100, 200, 400]
EVAL_PAIRS      = 50    # duplicate pairs per sim level for search scaling
ANCHOR_PAIRS    = 100   # duplicate pairs for anchor eval (200 games)
ANCHOR_GEN      = 50    # reference generation for anchor eval


# ─── helpers ─────────────────────────────────────────────────────────────────

def find_best_base_checkpoint():
    """Find best available base checkpoint.
    Priority: best_100sims.pt > best_200sims.pt > latest domino_gen_????.pt
    """
    for candidate in [
        os.path.join(CHECKPOINTS_DIR, "best_100sims.pt"),
        os.path.join(CHECKPOINTS_DIR, "best_200sims.pt"),
        os.path.join(CHECKPOINTS_DIR, "best_50sims.pt"),
    ]:
        if os.path.exists(candidate):
            return candidate, os.path.basename(candidate)

    pattern = os.path.join(CHECKPOINTS_DIR, "domino_gen_????.pt")
    files = [f for f in glob.glob(pattern) if "BACKUP" not in f and "phase6" not in f]
    if not files:
        raise FileNotFoundError("No base checkpoint found. Run training first.")
    latest = max(files, key=lambda p: int(
        os.path.basename(p).replace("domino_gen_", "").replace(".pt", "")
    ))
    return latest, os.path.basename(latest)


def load_model(path: str) -> DominoNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    input_dim = sd["input_fc.weight"].shape[1]
    m = DominoNet(input_dim=input_dim, hidden_dim=256, num_actions=57, num_blocks=4)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [load_model] Missing keys (belief head expected on non-belief ckpt): {missing[:4]}")
    m.eval()
    return m


def play_duplicate_pair_simple(model_a, model_b, seed, num_sims):
    """Play one duplicate pair (two games, swap sides). Returns a_wins (0, 1, or 2)."""
    device = torch.device("cpu")
    mcts_a = DominoMCTS(model_a, num_simulations=num_sims)
    mcts_b = DominoMCTS(model_b, num_simulations=num_sims)
    a_wins = 0
    for a_side in [0, 1]:
        env = DominoEnv()
        enc_a = DominoEncoder(); enc_a.reset()
        enc_b = DominoEncoder(); enc_b.reset()
        obs = env.reset(seed=seed)
        step = 0
        while not env.is_over() and step < 200:
            mask = env.get_legal_moves_mask()
            if mask.sum() == 0:
                break
            team = env.current_player % 2
            if team == a_side:
                pi = mcts_a.get_action_probs(env, enc_a, temperature=0.1)
            else:
                pi = mcts_b.get_action_probs(env, enc_b, temperature=0.1)
            action = int(np.argmax(pi * mask))
            obs, _, done, _ = env.step(action)
            step += 1
        if env.game_over and env.winner_team == a_side:
            a_wins += 1
    return a_wins


def eval_search_scaling(model_probe, model_ref, sim_list=None, num_pairs=EVAL_PAIRS, seed_base=70000):
    """Evaluate model_probe vs model_ref at multiple sim budgets. Returns list of result dicts."""
    if sim_list is None:
        sim_list = EVAL_SIMS_LIST
    results = []
    for num_sims in sim_list:
        wins_probe = 0
        total = 0
        for i in range(num_pairs):
            seed = seed_base + i
            a_wins = play_duplicate_pair_simple(model_probe, model_ref, seed, num_sims)
            wins_probe += a_wins
            total += 2  # 2 games per pair
        wr = wins_probe / total if total > 0 else 0.5
        print(f"    [{num_sims:>4} sims] win%={wr*100:.1f}% ({wins_probe}/{total})")
        results.append({"sims": num_sims, "win_pct": round(wr * 100, 1),
                         "wins": wins_probe, "total": total})
    return results


def eval_anchor(model_probe, anchor_path, num_sims=100, num_pairs=ANCHOR_PAIRS, seed_base=80000):
    """Evaluate model_probe vs anchor. Returns win_pct and elo_delta."""
    if not os.path.exists(anchor_path):
        print(f"    [anchor eval] Anchor not found: {anchor_path} — skipping")
        return None
    model_ref = load_model(anchor_path)
    wins_probe = 0
    total = 0
    for i in range(num_pairs):
        seed = seed_base + i
        a_wins = play_duplicate_pair_simple(model_probe, model_ref, seed, num_sims)
        wins_probe += a_wins
        total += 2
    wr = wins_probe / total if total > 0 else 0.5
    # Logistic ELO delta: Δ = 400 * log10(wr / (1-wr))
    elo = round(400 * (np.log10(wr + 1e-9) - np.log10(1 - wr + 1e-9)), 1) if 0 < wr < 1 else 0.0
    print(f"    [anchor {num_sims}sims] win%={wr*100:.1f}%, ELO Δ={elo:+.1f}")
    return {"win_pct": round(wr * 100, 1), "wins": wins_probe, "total": total, "elo_delta": elo}


def eval_partnership_suite(model_probe):
    """Run partnership suite if available. Returns score or None."""
    try:
        tests_dir = os.path.join(_THIS_DIR, "tests")
        sys.path.insert(0, tests_dir)
        from test_partnership_suite import evaluate_suite, make_engine_fn, SUITE_PATH
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_probe_dev = model_probe.to(device)
        eng = make_engine_fn(model_probe_dev, sims=0, device=str(device))
        report = evaluate_suite(eng, SUITE_PATH)
        score = report["avg_score"]
        print(f"    [partnership suite] avg_score={score:.4f}")
        return score
    except Exception as e:
        print(f"    [partnership suite] skipped: {e}")
        return None


# ─── single probe run ─────────────────────────────────────────────────────────

def run_one_probe(lambda_belief, base_ckpt_path, args, probe_label):
    """
    Run a single 5-gen training probe with given λ_belief.
    Returns dict with final eval results.
    """
    import torch.multiprocessing as mp
    from collections import deque
    from torch.utils.data import DataLoader

    probe_dir = os.path.join(CHECKPOINTS_DIR, f"phase6_probe_{probe_label}")
    os.makedirs(probe_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Phase 6 Probe: λ_belief = {lambda_belief} ({probe_label})")
    print(f"  Base checkpoint: {os.path.basename(base_ckpt_path)}")
    print(f"  Gens: {args.gens_per_probe}, Workers: {args.workers}, "
          f"Games/worker: {args.games_per_worker}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Load model from base checkpoint
    model = DominoNet().to(device)
    base_ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=True)
    base_sd = base_ckpt.get("model_state_dict", base_ckpt)
    missing, _ = model.load_state_dict(base_sd, strict=False)
    if missing:
        print(f"  [probe] Missing keys (belief head init): {missing[:4]}")

    trainer = Trainer(model, lr=1e-4, belief_weight=lambda_belief)
    replay_buffer = deque(maxlen=100000)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    t_probe_start = time.time()
    final_ckpt_path = None

    if not args.dry_run:
        for gen in range(1, args.gens_per_probe + 1):
            print(f"\n  --- Probe gen {gen}/{args.gens_per_probe} ---")
            t0 = time.time()

            shared_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ctx = mp.get_context('spawn')
            result_queue = ctx.Queue()
            procs = []

            for w_id in range(args.workers):
                p = ctx.Process(
                    target=self_play_worker,
                    args=(w_id, shared_weights, args.games_per_worker,
                          True,   # use_mcts
                          100,    # mcts_sims (moderate for probe)
                          result_queue,
                          'me',   # value_target
                          'visits',  # policy_target
                          0.1,    # high_sim_fraction
                          4,      # high_sim_multiplier
                          True,   # use_belief_head
                    )
                )
                p.start()
                procs.append(p)

            collected = 0
            per_game_s = 90
            timeout = max(3600, args.games_per_worker * per_game_s)
            while collected < args.workers:
                try:
                    data = result_queue.get(timeout=timeout)
                    replay_buffer.extend(data)
                    collected += 1
                    print(f"    Worker {collected}/{args.workers}: {len(data)} samples")
                except Exception as e:
                    alive = sum(1 for p in procs if p.is_alive())
                    print(f"    Worker collection error: {e} ({alive} still alive)")
                    if alive == 0:
                        break
                    collected += 1

            for p in procs:
                p.join(timeout=30)

            if len(replay_buffer) >= 2000:
                dataset = ReplayDataset(list(replay_buffer))
                loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
                for epoch in range(5):
                    loss, v_loss, p_loss, b_loss = trainer.train_epoch(loader)
                    print(f"    Epoch {epoch+1}/5 | "
                          f"Loss={loss:.4f} (V={v_loss:.4f}, P={p_loss:.4f}, B={b_loss:.4f})")
            else:
                print(f"    Buffer too small ({len(replay_buffer)}), skipping training")

            elapsed = time.time() - t0
            print(f"    Gen {gen} done in {elapsed:.1f}s")

        # Save final checkpoint
        final_ckpt_name = f"phase6_{probe_label}_final.pt"
        final_ckpt_path = os.path.join(probe_dir, final_ckpt_name)
        torch.save({
            'lambda_belief': lambda_belief,
            'probe_label': probe_label,
            'gens': args.gens_per_probe,
            'model_state_dict': model.state_dict(),
            'base_checkpoint': base_ckpt_path,
        }, final_ckpt_path)
        print(f"\n  Saved probe checkpoint: {final_ckpt_path}")
    else:
        print(f"  [DRY RUN] Skipping training, using base model for evaluation.")
        final_ckpt_path = base_ckpt_path

    probe_elapsed = time.time() - t_probe_start

    # ── Evaluate probe model ──────────────────────────────────────────────────
    print(f"\n  Evaluating probe (λ={lambda_belief})...")
    model_probe = model if not args.dry_run else load_model(base_ckpt_path)
    model_probe.eval()
    model_base  = load_model(base_ckpt_path)

    # Partnership suite
    print("  [1/3] Partnership suite...")
    partnership_score = eval_partnership_suite(model_probe)

    # Search scaling vs base
    print("  [2/3] Search scaling vs base checkpoint...")
    model_probe_cpu = model_probe.cpu()
    scaling_results = eval_search_scaling(
        model_probe_cpu, model_base,
        sim_list=EVAL_SIMS_LIST,
        num_pairs=EVAL_PAIRS,
        seed_base=70000 + int(lambda_belief * 1000),
    )

    # Anchor eval vs gen50
    print(f"  [3/3] Anchor eval (vs gen {ANCHOR_GEN}) at 100 sims...")
    anchor_path = os.path.join(CHECKPOINTS_DIR, f"domino_gen_{ANCHOR_GEN:04d}.pt")
    anchor_result = eval_anchor(
        model_probe_cpu, anchor_path,
        num_sims=100,
        num_pairs=ANCHOR_PAIRS,
        seed_base=80000 + int(lambda_belief * 1000),
    )

    result = {
        "probe_label": probe_label,
        "lambda_belief": lambda_belief,
        "gens_trained": args.gens_per_probe if not args.dry_run else 0,
        "training_time_s": round(probe_elapsed, 1),
        "base_checkpoint": os.path.basename(base_ckpt_path),
        "final_checkpoint": os.path.basename(final_ckpt_path) if final_ckpt_path else None,
        "partnership_score": partnership_score,
        "search_scaling": scaling_results,
        "anchor_eval_gen50_100sims": anchor_result,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Print summary
    print(f"\n  === Summary (λ={lambda_belief}) ===")
    if partnership_score is not None:
        print(f"  Partnership: {partnership_score:.4f}")
    print(f"  Search scaling:")
    for sr in scaling_results:
        print(f"    {sr['sims']:>4} sims: {sr['win_pct']:.1f}%")
    if anchor_result:
        print(f"  Anchor (gen{ANCHOR_GEN}): {anchor_result['win_pct']:.1f}% "
              f"(ELO Δ={anchor_result['elo_delta']:+.1f})")

    return result


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 belief-head λ probe experiment (3×5-gen runs)"
    )
    parser.add_argument("--base-checkpoint", type=str, default=None,
                        help="Path to starting checkpoint. Default: best_100sims.pt or latest gen.")
    parser.add_argument("--lambda-values", type=str, default=",".join(map(str, LAMBDA_VALUES)),
                        help=f"Comma-separated λ values (default: {','.join(map(str, LAMBDA_VALUES))})")
    parser.add_argument("--gens-per-probe", type=int, default=DEFAULT_GENS,
                        help=f"Training generations per probe (default: {DEFAULT_GENS})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Self-play workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--games-per-worker", type=int, default=DEFAULT_GAMES,
                        help=f"Games per worker per gen (default: {DEFAULT_GAMES})")
    parser.add_argument("--output", type=str, default=RESULTS_PATH,
                        help=f"Output JSON path (default: {RESULTS_PATH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip training; evaluate base checkpoint only (for testing)")
    args = parser.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)

    # Resolve base checkpoint
    if args.base_checkpoint:
        base_path = args.base_checkpoint
        if not os.path.exists(base_path):
            print(f"ERROR: base checkpoint not found: {base_path}")
            sys.exit(1)
        base_label = os.path.basename(base_path)
    else:
        base_path, base_label = find_best_base_checkpoint()

    print(f"\nPhase 6 Probe — λ sweep experiment")
    print(f"Base checkpoint: {base_path}")
    print(f"λ values: {args.lambda_values}")
    print(f"Gens per probe: {args.gens_per_probe}")
    print(f"Workers: {args.workers}, Games/worker: {args.games_per_worker}")
    print(f"Output: {args.output}")
    if args.dry_run:
        print("*** DRY RUN MODE — no training ***")

    lambda_list = [float(x) for x in args.lambda_values.split(",")]
    labels = [f"lambda_{str(lv).replace('.', 'p')}" for lv in lambda_list]

    all_results = []
    t_total = time.time()

    for lv, label in zip(lambda_list, labels):
        probe_result = run_one_probe(lv, base_path, args, label)
        all_results.append(probe_result)

        # Write partial results after each probe
        payload = {
            "meta": {
                "script": "phase6_probe.py",
                "base_checkpoint": base_label,
                "lambda_values": lambda_list,
                "gens_per_probe": args.gens_per_probe,
                "dry_run": args.dry_run,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "probes": all_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Partial results saved to: {args.output}")

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"All probes complete in {total_elapsed/60:.1f} min")
    print(f"\nResults comparison:")
    print(f"  {'λ':<6}  {'Partnership':>12}  {'Scaling@100':>12}  {'Anchor@100':>11}  {'ELO Δ':>8}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*11}  {'─'*8}")
    for r in all_results:
        ps = f"{r['partnership_score']:.4f}" if r['partnership_score'] else "  N/A"
        sc100 = next((f"{s['win_pct']:.1f}%" for s in r['search_scaling'] if s['sims'] == 100), "N/A")
        ar = r['anchor_eval_gen50_100sims']
        ac = f"{ar['win_pct']:.1f}%" if ar else "   N/A"
        elo = f"{ar['elo_delta']:+.1f}" if ar else "   N/A"
        print(f"  {r['lambda_belief']:<6}  {ps:>12}  {sc100:>12}  {ac:>11}  {elo:>8}")

    print(f"\nFull results: {args.output}")

    # Recommend best λ based on anchor eval + scaling at 100 sims
    best_lambda = None
    best_score = -999
    for r in all_results:
        ar = r.get('anchor_eval_gen50_100sims')
        elo = ar['elo_delta'] if ar else 0
        ps = r.get('partnership_score') or 0
        score = elo + ps * 100  # simple composite
        if score > best_score:
            best_score = score
            best_lambda = r['lambda_belief']

    if best_lambda is not None:
        print(f"\nRecommended λ_belief for Phase 6 full run: {best_lambda}")
        # Update results with recommendation
        payload["recommendation"] = {
            "best_lambda": best_lambda,
            "reasoning": "highest composite score (anchor ELO + 100×partnership_score)"
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
