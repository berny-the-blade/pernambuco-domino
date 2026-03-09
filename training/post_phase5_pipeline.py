"""
post_phase5_pipeline.py — Auto-runs after Phase 5 Gen 20 completes.

Steps:
  1. Export domino_model_gen20.bin
  2. Search-scaling benchmark (Gen 20 vs Gen 50, 50/100/200/400 sims)
  3. Anchor eval (Gen 20 vs Gen 1, 5, 10, 46, 50)
  4. Backward-compat validation (resume from Gen 50 for 1 gen)
  5. Write RESULTS.md summary

Usage:
  python training/post_phase5_pipeline.py
"""

import subprocess
import sys
import os
import json
import datetime

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR     = os.path.join(TRAINING_DIR, 'checkpoints')
GEN20_PT     = os.path.join(CKPT_DIR, 'domino_gen_0020.pt')
GEN50_PT     = os.path.join(CKPT_DIR, 'domino_gen_0050.pt')
RESULTS_MD   = os.path.join(TRAINING_DIR, 'PHASE5_RESULTS.md')

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"CMD: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=TRAINING_DIR)
    if result.returncode != 0:
        print(f"WARNING: {label} exited with code {result.returncode}")
    return result.returncode

def main():
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"\nPost-Phase-5 pipeline starting at {ts}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    if not os.path.exists(GEN20_PT):
        print(f"ERROR: Gen 20 checkpoint not found at {GEN20_PT}")
        print("Waiting for training to finish...")
        sys.exit(1)

    if not os.path.exists(GEN50_PT):
        print(f"WARNING: Gen 50 checkpoint not found — skipping comparisons against it")

    # ── Step 1: Export browser model ──────────────────────────────────────────
    export_out = os.path.join(TRAINING_DIR, '..', 'domino_model_gen20.bin')
    run([
        sys.executable, os.path.join(TRAINING_DIR, 'export_model.py'),
        GEN20_PT,
        '--output', export_out,
        '--format', 'binary',
    ], "STEP 1: Export domino_model_gen20.bin")

    # ── Step 2: Search-scaling benchmark ─────────────────────────────────────
    scaling_out = os.path.join(TRAINING_DIR, 'logs', 'search_scaling_phase5.json')
    os.makedirs(os.path.join(TRAINING_DIR, 'logs'), exist_ok=True)
    run([
        sys.executable, os.path.join(TRAINING_DIR, 'search_scaling_eval.py'),
        '--model-a', GEN20_PT,
        '--model-b', GEN50_PT,
        '--gen-a', '20',
        '--gen-b', '50',
        '--sim-list', '50,100,200,400',
        '--deal-pairs', '200',
        '--duplicate-deals',
        '--output-json', scaling_out,
        '--tag', 'phase5_post',
    ], "STEP 2: Search-scaling benchmark (Gen20 vs Gen50 @ 50/100/200/400 sims)")

    # ── Step 3: Anchor eval ───────────────────────────────────────────────────
    run([
        sys.executable, os.path.join(TRAINING_DIR, 'anchor_eval.py'),
        '--current', GEN20_PT,
        '--current-gen', '20',
        '--anchors', '1', '5', '10', '46', '50',
        '--games', '400',
    ], "STEP 3: Anchor eval (Gen20 vs Gen1/5/10/46/50)")

    # ── Step 4: Backward-compat validation ───────────────────────────────────
    # Quick smoke test: load Gen50, 1 worker × 5 games = 5 games total.
    # Verifies strict=False loading + belief-head additions don't break old ckpts.
    run([
        sys.executable, os.path.join(TRAINING_DIR, 'orchestrator.py'),
        '--resume', GEN50_PT,
        '--generations', '1',
        '--workers', '1',
        '--games-per-worker', '5',
    ], "STEP 4: Backward-compat smoke test (load Gen50, 5 games)")

    # ── Step 5: Write results summary ─────────────────────────────────────────
    summary_lines = [
        f"# Phase 5 Post-Training Results",
        f"",
        f"**Generated:** {ts}",
        f"",
        f"## Checkpoints",
        f"- Gen 20: `{GEN20_PT}`",
        f"- Gen 50 (Phase 4 best): `{GEN50_PT}`",
        f"",
        f"## Search Scaling",
    ]

    if os.path.exists(scaling_out):
        try:
            with open(scaling_out) as f:
                data = json.load(f)
            summary_lines.append("")
            summary_lines.append("| Sims | Gen20 Win% | Gen50 Win% |")
            summary_lines.append("|------|------------|------------|")
            for entry in data.get('results', []):
                sims = entry.get('sims', '?')
                wa   = entry.get('win_rate_a', entry.get('win_rate', '?'))
                wb   = entry.get('win_rate_b', 1 - wa if isinstance(wa, float) else '?')
                if isinstance(wa, float):
                    monotone_note = ""
                    summary_lines.append(f"| {sims} | {wa*100:.1f}% | {wb*100:.1f}% |")
        except Exception as e:
            summary_lines.append(f"\n_(could not parse scaling results: {e})_")
    else:
        summary_lines.append("\n_(search scaling results not found)_")

    summary_lines += [
        f"",
        f"## Files",
        f"- Search scaling: `training/logs/search_scaling_phase5.json`",
        f"- Browser model: `domino_model_gen20.bin`",
        f"",
        f"## Next Steps",
        f"- [ ] Check search-scaling monotonicity (goal: 50%→55%→57%→59%)",
        f"- [ ] If monotone → start Phase 6 probe (belief head λ=0.1/0.2/0.3)",
        f"- [ ] If non-monotone → diagnose (PUCT priors, more gens)",
    ]

    with open(RESULTS_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines) + '\n')

    print(f"\n{'='*60}")
    print(f"  Pipeline complete! Results: {RESULTS_MD}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
