"""Small, dependency-free statistics so run-to-run comparisons are honest.

With only a handful of trials per case, one flaky failure should not break your
build, and a real drop should not hide. These helpers give confidence intervals
and a significance test using nothing but the standard library.
"""
from __future__ import annotations

import math
from typing import Tuple


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a pass rate. Works well for small n and rates near 0 or 1."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021).

    n trials were run, c of them passed. Returns the chance that at least one of
    k randomly chosen trials passes.
    """
    if k <= 0 or n <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    prob_all_fail = 1.0
    for i in range(n - c + 1, n + 1):
        prob_all_fail *= 1 - k / i
    return 1 - prob_all_fail


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k estimator: chance that k randomly chosen trials ALL pass (reliability)."""
    if k <= 0 or n <= 0 or c < k:
        return 0.0 if k > 0 else 1.0
    prob = 1.0
    for i in range(k):
        prob *= (c - i) / (n - i)
    return prob


def two_proportion_p(s1: int, n1: int, s2: int, n2: int) -> float:
    """Two-sided p-value that two pass rates differ (pooled z-test, with continuity fix).

    Returns 1.0 when there is not enough data to tell.
    """
    if n1 <= 0 or n2 <= 0:
        return 1.0
    p1, p2 = s1 / n1, s2 / n2
    pool = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0 if p1 == p2 else 0.0
    cc = 0.5 * (1 / n1 + 1 / n2)
    z = max(0.0, abs(p1 - p2) - cc) / se
    return math.erfc(z / math.sqrt(2))
