"""
particle_disagreement_eval.py -- Measure policy sensitivity to hidden-info belief.

Test 4 from POST_GEN50_EXPERIMENTS.md.

For each sampled public state, generates K plausible hand assignments (particles),
encodes each with a deterministic belief encoder, and compares the resulting
policy distributions. High disagreement = belief bottleneck.

Results broken down by: phase (early/mid/late), forced vs non-forced.

Sub-signals:
  - MCTS policy disagreement  (search + belief)
  - Raw prior disagreement    (network only, sims=0)
  If prior is worse than MCTS: search partially compensates for belief noise.

Usage:
    python particle_disagreement_eval.py \\
        --model checkpoints/domino_gen_0050.pt \\
        --state-source replay \\
        --state-count 200 \\
        --particles 8 \\
        --sims 200 \\
        --phase-buckets early,mid,late \\
        --seed-base 9000 \\
        --output-json results/particle_disagreement_gen50.json \\
        --output-csv  results/particle_disagreement_gen50.csv \\
        --tag gen50_particle_disagreement

    # Prior-only (0 MCTS sims, fast)
    python particle_disagreement_eval.py --sims 0 --state-count 200

    # Non-forced states only
    python particle_disagreement_eval.py --non-forced-only --state-count 200
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domino_env import DominoEnv, TILES, NUM_TILES
from domino_encoder import DominoEncoder
from domino_net import DominoNet
from domino_mcts import DominoMCTS

CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ============================================================
# Data models
# ============================================================

@dataclass
class PerParticleSearchResult:
    particle_idx: int
    top_action: int
    root_value: float
    root_entropy: float
    root_policy: list[float]


@dataclass
class StateDisagreementRow:
    state_id: int
    phase: str
    forced: bool
    legal_count: int
    top1_agreement: float        # plurality: fraction of particles on most-voted action
    pairwise_jsd_mean: float
    value_mean: float
    value_std: float
    entropy_mean: float
    entropy_std: float


# ============================================================
# Model helpers
# ============================================================

def load_model(path: str) -> DominoNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    input_dim = sd["input_fc.weight"].shape[1]
    m = DominoNet(input_dim=input_dim, hidden_dim=256, num_actions=57, num_blocks=4)
    m.load_state_dict(sd)
    m.eval()
    return m


def latest_checkpoint():
    pattern = os.path.join(CHECKPOINTS_DIR, "domino_gen_????.pt")
    files = [f for f in glob.glob(pattern) if "BACKUP" not in f]
    if not files:
        raise FileNotFoundError("No checkpoints found")
    p = max(files, key=os.path.getmtime)
    gen = int(os.path.basename(p).replace("domino_gen_", "").replace(".pt", ""))
    return gen, p


# ============================================================
# Statistics (stdlib-first, matching skeleton)
# ============================================================

def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _stddev(xs: list[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) >= 2 else 0.0


def policy_entropy(pi: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(pi, dtype=np.float64)
    p = p[p > eps]
    return float(-(p * np.log(p)).sum()) if p.size > 0 else 0.0


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64); p = p / max(np.sum(p), eps)
    q = np.asarray(q, dtype=np.float64); q = q / max(np.sum(q), eps)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > eps
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], eps, None))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pairwise_jsd_mean(policies: list[np.ndarray]) -> float:
    if len(policies) < 2:
        return 0.0
    vals = [js_divergence(policies[i], policies[j])
            for i, j in itertools.combinations(range(len(policies)), 2)]
    return _mean(vals)


def top1_agreement(actions: list[int]) -> float:
    """Plurality agreement: fraction of particles that chose the most-voted action."""
    if not actions:
        return 0.0
    counts = Counter(actions)
    return float(max(counts.values()) / len(actions))


# ============================================================
# Game phase / forced detection
# ============================================================

def game_phase(obs: dict) -> str:
    n = len(obs.get("played", []))
    return "early" if n <= 7 else ("mid" if n <= 15 else "late")


def is_forced(mask: np.ndarray) -> bool:
    return int(mask.sum()) == 1


# ============================================================
# Particle sampling + encoding
# ============================================================

def sample_particles(obs: dict, n_particles: int, rng) -> list[dict]:
    """
    Sample N plausible hand assignments consistent with:
      known hand, played tiles, cant_have, hand_sizes.
    Returns list of {partner, lho, rho} -> set of tile indices.
    """
    me = obs["player"]
    partner, lho, rho = (me + 2) % 4, (me + 1) % 4, (me + 3) % 4
    my_hand = set(obs["hand"])
    played  = set(obs["played"])
    cant_have  = obs["cant_have"]
    hand_sizes = obs["hand_sizes"]
    unknown = [t for t in range(NUM_TILES) if t not in my_hand and t not in played]
    player_names = ["partner", "lho", "rho"]
    player_abs   = [partner, lho, rho]
    targets = {"partner": hand_sizes[partner], "lho": hand_sizes[lho], "rho": hand_sizes[rho]}

    result, attempts = [], 0
    while len(result) < n_particles and attempts < n_particles * 40:
        attempts += 1
        asgn = {"partner": set(), "lho": set(), "rho": set()}
        tiles = list(unknown); rng.shuffle(tiles)
        ok = True
        for tile in tiles:
            left, right = TILES[tile]
            eligible = [n for n, p in zip(player_names, player_abs)
                        if left  not in cant_have[p] and
                           right not in cant_have[p] and
                           len(asgn[n]) < targets[n]]
            eligible.append("dorme")
            weights = [max(0.0, float(targets[n] - len(asgn[n])))
                       if n != "dorme" else 0.5 for n in eligible]
            total_w = sum(weights)
            if total_w == 0:
                ok = False; break
            weights = [w / total_w for w in weights]
            chosen = rng.choice(eligible, p=weights)
            if chosen != "dorme":
                asgn[chosen].add(tile)
        if ok:
            result.append(asgn)
    return result


def encoder_for_particle(obs: dict, assignment: dict) -> DominoEncoder:
    enc = DominoEncoder(); enc.reset()
    enc.belief[:, :] = 0.0
    my_hand = set(obs["hand"]); played = set(obs["played"])
    for t in range(NUM_TILES):
        if t in my_hand or t in played:
            continue
        if   t in assignment.get("partner", set()): enc.belief[t, 0] = 1.0
        elif t in assignment.get("lho",     set()): enc.belief[t, 1] = 1.0
        elif t in assignment.get("rho",     set()): enc.belief[t, 2] = 1.0
        else:                                        enc.belief[t, 3] = 1.0
    return enc


# ============================================================
# State sampling
# ============================================================

def load_public_states(state_source: str, state_count: int,
                        phase_buckets: list[str], seed_base: int,
                        non_forced_only: bool = False) -> list[tuple]:
    """
    Sample game states via random play, stratified by phase.
    Returns list of (state_id, env_snap, obs, mask).
    """
    if state_source not in ("replay", "random"):
        raise ValueError(f"Unsupported state-source: {state_source}")

    rng_np = np.random.default_rng(42)
    per_phase = {p: state_count // len(phase_buckets) for p in phase_buckets}
    remainder = state_count - sum(per_phase.values())
    for p in list(per_phase.keys())[:remainder]:
        per_phase[p] += 1

    collected = {p: [] for p in phase_buckets}
    total_needed = state_count
    game_idx = 0

    print(f"Sampling {state_count} states (phases: {phase_buckets})...", flush=True)

    while sum(len(v) for v in collected.values()) < total_needed:
        env = DominoEnv()
        env.reset(seed=seed_base + game_idx)
        game_idx += 1
        while not env.is_over():
            mask = env.get_legal_moves_mask()
            legal = np.where(mask > 0)[0]
            obs = env.get_obs()
            ph = game_phase(obs)
            forced = is_forced(mask)
            if ph in phase_buckets and len(collected[ph]) < per_phase[ph]:
                if (not non_forced_only) or (not forced):
                    collected[ph].append((env.clone(), obs, mask.copy()))
            env.step(int(rng_np.choice(legal)))
        if game_idx % 30 == 0:
            done = sum(len(v) for v in collected.values())
            print(f"  {done}/{total_needed} ({game_idx} games)", flush=True)

    states = []
    sid = 0
    for ph in phase_buckets:
        for env_snap, obs, mask in collected[ph]:
            states.append((sid, env_snap, obs, mask))
            sid += 1
    print(f"  Done: {len(states)} states collected", flush=True)
    return states


# ============================================================
# Per-particle search
# ============================================================

def run_search_on_particle(model: DominoNet, mcts,
                            env_snap, obs: dict, mask: np.ndarray,
                            assignment: dict, particle_idx: int,
                            num_sims: int) -> PerParticleSearchResult:
    enc = encoder_for_particle(obs, assignment)
    state_np = enc.encode(obs)

    if num_sims > 0 and mcts is not None:
        pi = mcts.get_action_probs(env_snap.clone(), enc, temperature=1.0)
        raw_pi, raw_v = model.predict(state_np, mask)
        value = float(raw_v)
    else:
        raw_pi, raw_v = model.predict(state_np, mask)
        pi = raw_pi.copy()
        value = float(raw_v)

    pi_clean = pi[pi > 1e-9]
    ent = float(-(pi_clean * np.log(pi_clean)).sum()) if pi_clean.size > 0 else 0.0

    return PerParticleSearchResult(
        particle_idx=particle_idx,
        top_action=int(np.argmax(pi)),
        root_value=value,
        root_entropy=ent,
        root_policy=pi.tolist(),
    )


# ============================================================
# Per-state evaluation
# ============================================================

def evaluate_state_particle_disagreement(state_id: int,
                                          model: DominoNet, mcts,
                                          env_snap, obs: dict, mask: np.ndarray,
                                          particles: int, sims: int,
                                          rng) -> StateDisagreementRow | None:
    forced  = is_forced(mask)
    phase   = game_phase(obs)
    legal   = int(mask.sum())

    asgns = sample_particles(obs, particles, rng)
    if len(asgns) < 2:
        return None

    results = [run_search_on_particle(model, mcts, env_snap, obs, mask,
                                       asgn, pidx, sims)
               for pidx, asgn in enumerate(asgns)]

    top_actions = [r.top_action for r in results]
    values      = [r.root_value for r in results]
    entropies   = [r.root_entropy for r in results]
    policies    = [np.asarray(r.root_policy, dtype=np.float64) for r in results]

    return StateDisagreementRow(
        state_id=state_id,
        phase=phase,
        forced=forced,
        legal_count=legal,
        top1_agreement=top1_agreement(top_actions),
        pairwise_jsd_mean=pairwise_jsd_mean(policies),
        value_mean=_mean(values),
        value_std=_stddev(values),
        entropy_mean=_mean(entropies),
        entropy_std=_stddev(entropies),
    )


# ============================================================
# Aggregation
# ============================================================

def aggregate_bucket(rows: list[StateDisagreementRow]) -> dict[str, Any]:
    if not rows:
        return {"states": 0}
    t1_vals  = [r.top1_agreement   for r in rows]
    jsd_vals = [r.pairwise_jsd_mean for r in rows]
    v_stds   = [r.value_std        for r in rows]
    e_stds   = [r.entropy_std      for r in rows]

    # Top-1 histogram
    hist = {"100pct": 0, "75_99pct": 0, "50_74pct": 0, "below_50pct": 0}
    for v in t1_vals:
        vp = v * 100
        if vp >= 99.9:   hist["100pct"]     += 1
        elif vp >= 75:   hist["75_99pct"]   += 1
        elif vp >= 50:   hist["50_74pct"]   += 1
        else:            hist["below_50pct"] += 1

    return {
        "states": len(rows),
        "top1_agreement_mean": _mean(t1_vals),
        "pairwise_jsd_mean":   _mean(jsd_vals),
        "value_std_mean":      _mean(v_stds),
        "entropy_std_mean":    _mean(e_stds),
        "top1_histogram":      hist,
    }


def build_summary(rows: list[StateDisagreementRow],
                   phase_buckets: list[str]) -> dict[str, Any]:
    valid = [r for r in rows if r is not None]
    return {
        "all":        aggregate_bucket(valid),
        "non_forced": aggregate_bucket([r for r in valid if not r.forced]),
        "forced":     aggregate_bucket([r for r in valid if r.forced]),
        "by_phase": {
            ph: aggregate_bucket([r for r in valid if r.phase == ph])
            for ph in phase_buckets
        },
        "by_phase_non_forced": {
            ph: aggregate_bucket([r for r in valid if r.phase == ph and not r.forced])
            for ph in phase_buckets
        },
    }


# ============================================================
# Verdict
# ============================================================

def disagreement_verdict(summary: dict[str, Any]) -> dict[str, Any]:
    nf = summary.get("non_forced", {})
    top1 = nf.get("top1_agreement_mean", 0.0)
    jsd  = nf.get("pairwise_jsd_mean",   0.0)
    n    = nf.get("states", 0)

    if n == 0:
        return {"label": "NO_DATA", "detail": "No non-forced states"}

    if top1 < 0.65:
        label  = "BELIEF_INSTABILITY_SEVERE"
        detail = (f"Non-forced top-1 agreement = {top1*100:.1f}% "
                  f"-- policy targets severely smeared. Belief modeling is the primary bottleneck.")
    elif top1 < 0.70 or jsd > 0.20:
        label  = "BELIEF_INSTABILITY_HIGH"
        detail = (f"Non-forced top-1 = {top1*100:.1f}%, JSD = {jsd:.4f} "
                  f"-- high belief instability. Fix hidden-info inference before other changes.")
    elif top1 < 0.85 or jsd > 0.10:
        label  = "BELIEF_INSTABILITY_MODERATE"
        detail = (f"Non-forced top-1 = {top1*100:.1f}%, JSD = {jsd:.4f} "
                  f"-- moderate instability. Some policy noise but probably not the primary bottleneck.")
    else:
        label  = "BELIEF_INSTABILITY_MANAGEABLE"
        detail = (f"Non-forced top-1 = {top1*100:.1f}%, JSD = {jsd:.4f} "
                  f"-- belief instability is manageable.")

    jsd_level = "HIGH" if jsd > 0.20 else ("MODERATE" if jsd > 0.10 else "LOW")

    return {
        "label": label,
        "detail": detail,
        "non_forced_top1_agreement": round(top1, 4),
        "non_forced_jsd_mean": round(jsd, 5),
        "jsd_level": jsd_level,
    }


# ============================================================
# Output
# ============================================================

def print_summary(summary: dict[str, Any], verdict: dict[str, Any],
                   gen: int, sims: int, particles: int):
    print(f"\n{'='*72}")
    print(f"  Particle Disagreement: Gen {gen}  ({particles} particles, {sims} sims)")
    print(f"{'='*72}")
    print(f"\n  {'Bucket':<22} {'N':>5} {'Top1 Agree':>12} {'Mean JSD':>10} "
          f"{'Val Std':>9} {'Ent Std':>9}")
    print(f"  {'-'*22} {'-'*5} {'-'*12} {'-'*10} {'-'*9} {'-'*9}")

    for label, key in [("All", "all"), ("Non-forced", "non_forced"), ("Forced", "forced")]:
        g = summary.get(key, {})
        _print_bucket_row(label, g)

    for ph in ["early", "mid", "late"]:
        g = summary.get("by_phase", {}).get(ph, {})
        if g.get("states", 0) > 0:
            _print_bucket_row(ph.capitalize(), g)
        g_nf = summary.get("by_phase_non_forced", {}).get(ph, {})
        if g_nf.get("states", 0) > 0:
            _print_bucket_row(f"{ph.capitalize()} (non-forced)", g_nf)

    # Histogram for non-forced
    nf = summary.get("non_forced", {})
    hist = nf.get("top1_histogram", {})
    nf_n = nf.get("states", 0)
    if hist and nf_n > 0:
        print(f"\n  Top-1 agreement histogram (non-forced, n={nf_n}):")
        for hlabel, k in [("100%", "100pct"), ("75-99%", "75_99pct"),
                           ("50-74%", "50_74pct"), ("<50%", "below_50pct")]:
            count = hist.get(k, 0)
            pct = count / nf_n * 100
            bar = "#" * int(pct / 2)
            print(f"    {hlabel:>8}: {bar:<30} {count:>4} ({pct:.0f}%)")

    print(f"\n  INTERPRETATION")
    print(f"  Verdict: {verdict['label']}")
    print(f"  {verdict['detail']}")
    print(f"\n  Thresholds (non-forced top-1 agreement):")
    print("    > 85%  = BELIEF_INSTABILITY_MANAGEABLE")
    print("    70-85% = BELIEF_INSTABILITY_MODERATE")
    print("    65-70% = BELIEF_INSTABILITY_HIGH")
    print("    < 65%  = BELIEF_INSTABILITY_SEVERE")


def _print_bucket_row(label: str, g: dict):
    n  = g.get("states", 0)
    if n == 0:
        return
    t1 = g.get("top1_agreement_mean")
    jv = g.get("pairwise_jsd_mean")
    vs = g.get("value_std_mean")
    es = g.get("entropy_std_mean")
    print(f"  {label:<22} {n:>5} "
          f"{'--' if t1 is None else f'{t1*100:.1f}%':>12} "
          f"{'--' if jv is None else f'{jv:.5f}':>10} "
          f"{'--' if vs is None else f'{vs:.4f}':>9} "
          f"{'--' if es is None else f'{es:.4f}':>9}")


def write_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csvs(output_csv: str, rows: list[StateDisagreementRow],
                summary: dict[str, Any]) -> tuple[str, str]:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_rows = [
        ("all",               summary["all"]),
        ("non_forced",        summary["non_forced"]),
        ("forced",            summary["forced"]),
    ]
    for ph in ["early", "mid", "late"]:
        summary_rows.append((ph, summary["by_phase"].get(ph, {})))
        summary_rows.append((f"{ph}_non_forced",
                              summary["by_phase_non_forced"].get(ph, {})))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "states", "top1_agreement_mean",
                    "pairwise_jsd_mean", "value_std_mean", "entropy_std_mean"])
        for bucket, g in summary_rows:
            w.writerow([bucket, g.get("states", 0),
                        g.get("top1_agreement_mean"), g.get("pairwise_jsd_mean"),
                        g.get("value_std_mean"), g.get("entropy_std_mean")])

    # Per-state CSV
    per_state_csv = output_csv.replace(".csv", "_per_state.csv")
    valid = [r for r in rows if r is not None]
    if valid:
        with open(per_state_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(valid[0]).keys()))
            w.writeheader()
            w.writerows([asdict(r) for r in valid])

    return output_csv, per_state_csv


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Particle disagreement evaluation")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--state-source", type=str, default="replay",
                        choices=["replay", "random"])
    parser.add_argument("--state-count", type=int, default=100)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS sims per particle (0 = prior only)")
    parser.add_argument("--phase-buckets", type=str, default="early,mid,late")
    parser.add_argument("--non-forced-only", action="store_true")
    parser.add_argument("--max-forced-frac", type=float, default=1.0)
    parser.add_argument("--state-file", default="")
    parser.add_argument("--seed-base", type=int, default=9000)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv",  type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    phase_buckets = [p.strip() for p in args.phase_buckets.split(",")]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")

    if args.model:
        ckpt_path = args.model
        current_gen = int(os.path.basename(ckpt_path).replace("domino_gen_", "").replace(".pt", ""))
    else:
        current_gen, ckpt_path = latest_checkpoint()
        print(f"Auto-detected: gen {current_gen}")

    if not args.output_json:
        args.output_json = os.path.join(
            RESULTS_DIR, f"particle_disagreement_gen{current_gen:04d}_{ts}.json")
    if not args.output_csv:
        args.output_csv = args.output_json.replace(".json", ".csv")

    print(f"Model: gen {current_gen}")
    print(f"States: {args.state_count}, Particles: {args.particles}, Sims: {args.sims}")
    print(f"Phases: {phase_buckets}, Non-forced-only: {args.non_forced_only}")
    print()

    model = load_model(ckpt_path)
    mcts  = DominoMCTS(model, num_simulations=args.sims) if args.sims > 0 else None

    states = load_public_states(args.state_source, args.state_count,
                                 phase_buckets, args.seed_base, args.non_forced_only)

    print("Running particle disagreement eval...", flush=True)
    rng = np.random.default_rng(77)
    rows: list[StateDisagreementRow | None] = []

    for sid, env_snap, obs, mask in states:
        r = evaluate_state_particle_disagreement(
            sid, model, mcts, env_snap, obs, mask,
            args.particles, args.sims, rng)
        rows.append(r)
        if (sid + 1) % 25 == 0:
            valid = [x for x in rows if x is not None]
            if valid:
                t1_mean = _mean([x.top1_agreement for x in valid])
                print(f"  {sid+1}/{args.state_count} done  "
                      f"mean_top1_agree={t1_mean*100:.1f}%", flush=True)

    summary = build_summary(rows, phase_buckets)
    verdict = disagreement_verdict(summary)
    print_summary(summary, verdict, current_gen, args.sims, args.particles)

    # Save
    valid_rows = [r for r in rows if r is not None]
    payload = {
        "meta": {
            "script": "particle_disagreement_eval.py",
            "model": ckpt_path,
            "gen": current_gen,
            "state_source": args.state_source,
            "state_count": len(valid_rows),
            "particles": args.particles,
            "sims": args.sims,
            "phase_buckets": phase_buckets,
            "seed_base": args.seed_base,
            "non_forced_only": args.non_forced_only,
            "tag": args.tag or f"gen{current_gen:04d}_particle_disagreement",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": summary,
        "interpretation": {"verdict": verdict},
        "state_rows": [asdict(r) for r in valid_rows],
    }
    write_json(args.output_json, payload)
    csv_path, per_state_csv = write_csvs(args.output_csv, rows, summary)
    print(f"\nSaved JSON:          {args.output_json}")
    print(f"Saved summary CSV:   {csv_path}")
    print(f"Saved per-state CSV: {per_state_csv}")


if __name__ == "__main__":
    main()
