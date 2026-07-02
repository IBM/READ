# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Statistical helpers for READ metrics.

Two routines used throughout summarize_results.py:

    clopper_pearson_ci(k, n)         exact binomial CI for proportions
                                     (PIR, CSR, ASR, CCR; Tables S2 / S3
                                     of the paper)
    bootstrap_ci_from_binary(...)    percentile bootstrap on a list of binary
                                     indicators (B = 1000, seed 42 are the
                                     defaults the paper pins for PC-CSR)
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from scipy.stats import beta


def clopper_pearson_ci(
    successes: int,
    n: int,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Exact Clopper-Pearson (beta) confidence interval for a binomial proportion.

    Args:
        successes: number of observed successes (0 ≤ successes ≤ n).
        n: total trials.
        ci_level: confidence level (default 0.95, two-sided).

    Returns:
        ``(lower, upper)`` with both bounds in ``[0, 1]``. Conventions on
        boundary cases:

            - ``n == 0``      → ``(0.0, 1.0)`` (no information; return the trivial CI)
            - ``successes == 0``  → lower bound exactly 0
            - ``successes == n``  → upper bound exactly 1

    Raises:
        ValueError if ``successes`` is outside ``[0, n]``.
    """
    if n == 0:
        return (0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError(
            f"successes must be in [0, n]; got successes={successes}, n={n}"
        )
    alpha = 1 - ci_level
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    )
    upper = (
        1.0
        if successes == n
        else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    )
    return (lower, upper)


def bootstrap_ci_from_binary(
    indicators: Sequence[bool] | Sequence[int] | Iterable[bool] | Iterable[int],
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of a binary sample.

    Paper-pinned defaults for PC-CSR (Table S2 caption): ``B = 1000``, ``seed = 42``.
    """
    bits = [int(bool(x)) for x in indicators]
    n = len(bits)
    if n == 0:
        return (0.0, 1.0)
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choices(bits, k=n)) / n for _ in range(n_boot))
    alpha = (1 - ci_level) / 2
    lower = estimates[max(0, int(alpha * n_boot))]
    upper = estimates[min(n_boot - 1, int((1 - alpha) * n_boot))]
    return (float(lower), float(upper))
