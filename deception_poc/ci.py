#!/usr/bin/env python3
"""Shared bootstrap confidence intervals for the deception PoC.

The Phase-1 audit's #2: every gain number needs a CI so "separates from variance" is checkable,
not asserted. We resample whole GAMES with replacement (a clustered / block bootstrap), because
decisions within a game share the same deck / opponents and are not independent — a naive
per-decision bootstrap would understate the variance the audit is worried about.
"""
import random as _random


def clustered_bootstrap(items, group_fn, value_fn, n_boot=2000, seed=0, alpha=0.05):
    """Mean of value_fn over items, with a CI from resampling groups (games) with replacement.

    items     : iterable of records
    group_fn  : record -> hashable game id (the resampling cluster)
    value_fn  : record -> float, or None to drop the record
    returns   : {"mean","lo","hi","se","n_groups","n_items"} or None if empty
    """
    groups = {}
    for it in items:
        v = value_fn(it)
        if v is None:
            continue
        groups.setdefault(group_fn(it), []).append(v)
    gkeys = list(groups)
    n_items = sum(len(v) for v in groups.values())
    if not gkeys or n_items == 0:
        return None

    def stat(keys):
        vals = [v for g in keys for v in groups[g]]
        return sum(vals) / len(vals) if vals else float("nan")

    point = stat(gkeys)
    rng = _random.Random(seed)
    boots = sorted(stat([rng.choice(gkeys) for _ in gkeys]) for _ in range(n_boot))
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    mean_b = sum(boots) / len(boots)
    se = (sum((b - mean_b) ** 2 for b in boots) / (len(boots) - 1)) ** 0.5
    return {"mean": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "se": round(se, 4), "n_groups": len(gkeys), "n_items": n_items}


def fmt(ci, prec=1):
    """'12.3 [1.0, 23.4]' for a CI dict, or 'n/a'."""
    if not ci:
        return "n/a"
    return f"{ci['mean']:.{prec}f} [{ci['lo']:.{prec}f}, {ci['hi']:.{prec}f}]"


def separates_from_zero(ci):
    """True if the 95% CI excludes 0 (both bounds same sign)."""
    if not ci:
        return None
    return ci["lo"] > 0 or ci["hi"] < 0
