"""
Search-quality diagnostics for MCTS.

Metrics:
  - root policy entropy (higher = more uncertain)
  - top-1 action mass (visit share of best move)
  - top-2 gap (visit difference between best and second-best)
"""

import numpy as np


def policy_entropy(visit_counts, eps=1e-12):
    """Shannon entropy of the visit distribution."""
    total = visit_counts.sum()
    if total < 1:
        return 0.0
    p = visit_counts / total
    p = p[p > eps]
    return float(-(p * np.log(p)).sum())


def top1_mass(visit_counts):
    """Fraction of visits going to the most-visited action."""
    total = visit_counts.sum()
    if total < 1:
        return 0.0
    return float(visit_counts.max() / total)


def top2_gap(visit_counts):
    """Difference in visit fraction between #1 and #2 actions."""
    total = visit_counts.sum()
    if total < 1:
        return 0.0
    sorted_v = np.sort(visit_counts)[::-1]
    if len(sorted_v) < 2:
        return float(sorted_v[0] / total)
    return float((sorted_v[0] - sorted_v[1]) / total)


def summarize_search(visit_counts):
    """Return dict of all search quality metrics."""
    vc = np.asarray(visit_counts, dtype=np.float64)
    return {
        'entropy': policy_entropy(vc),
        'top1_mass': top1_mass(vc),
        'top2_gap': top2_gap(vc),
    }
