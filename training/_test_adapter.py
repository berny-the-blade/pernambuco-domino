from tools.eval_adapter import (
    load_checkpoint_model, build_eval_agent,
    run_duplicate_pair, sample_public_states, run_particle_search
)

# Load models
m46 = load_checkpoint_model('checkpoints/domino_gen_0046.pt', 'cpu')
m15 = load_checkpoint_model('checkpoints/domino_gen_0015.pt', 'cpu')
print(f"Loaded gen {m46.generation} (dim={m46.extra['input_dim']}) and gen {m15.generation}")

a46 = build_eval_agent(m46, sims=5, temperature=0.1)
a15 = build_eval_agent(m15, sims=5, temperature=0.1)
print(f"Agents: a15.sims={a15.sims}, a46.sims={a46.sims}")

# Duplicate pair
pair = run_duplicate_pair(a15, a46, seed=42)
print(f"Pair: game1={pair.game1.margin_a:+.1f}  game2={pair.game2.margin_a:+.1f}  "
      f"pair={pair.pair_margin_a:+.3f}  wins_a={pair.wins_a}/2")
forced_pct = sum(pair.game1.forced_move_flags) / pair.game1.game_length * 100
print(f"Game1: len={pair.game1.game_length}  forced={forced_pct:.0f}%  "
      f"winner={pair.game1.winner}")

# State sampling
states = sample_public_states('replay', 6, ['early', 'mid', 'late'], seed_base=9000)
print(f"\nStates: {len(states)} sampled")
s = states[2]
print(f"  id={s.state_id} phase={s.phase} forced={s.forced} legal={s.legal_count}")

# Particle search
r0 = run_particle_search(a15, s, particle_idx=0, seed=9000)
r1 = run_particle_search(a15, s, particle_idx=1, seed=9000)
r2 = run_particle_search(a15, s, particle_idx=2, seed=9000)
print(f"\nParticles (3 of them):")
print(f"  p0: top={r0.top_action}  v={r0.root_value:.4f}  ent={r0.root_entropy:.4f}")
print(f"  p1: top={r1.top_action}  v={r1.root_value:.4f}  ent={r1.root_entropy:.4f}")
print(f"  p2: top={r2.top_action}  v={r2.root_value:.4f}  ent={r2.root_entropy:.4f}")
actions = [r0.top_action, r1.top_action, r2.top_action]
from collections import Counter
counts = Counter(actions)
agree = max(counts.values()) / len(actions)
print(f"  top1 agreement: {agree:.2f}")
print("\nAdapter OK")
