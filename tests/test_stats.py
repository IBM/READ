# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Tests for src/stats.py.

Clopper-Pearson edge cases match the textbook construction (Brown, Cai & DasGupta
2001). Bootstrap tests verify reproducibility under the paper's pinned seed.
"""

from __future__ import annotations

import pytest

from read_toolkit.stats import bootstrap_ci_from_binary, clopper_pearson_ci


class TestClopperPearson:

    def test_n_zero_returns_trivial_ci(self):
        """No trials → no information → trivial [0, 1] CI by convention."""
        assert clopper_pearson_ci(0, 0) == (0.0, 1.0)

    def test_k_zero_lower_bound_is_exactly_zero(self):
        """k == 0: lower bound pinned at 0, upper bound = Beta(1, n) quantile."""
        lower, upper = clopper_pearson_ci(0, 100, ci_level=0.95)
        assert lower == 0.0
        # Wilson rule-of-three approximation: upper ≈ 3/n for k=0
        assert 0.02 < upper < 0.05

    def test_k_equal_n_upper_bound_is_exactly_one(self):
        """k == n: upper bound pinned at 1, lower bound = Beta(n, 1) quantile."""
        lower, upper = clopper_pearson_ci(50, 50, ci_level=0.95)
        assert upper == 1.0
        assert 0.92 < lower < 0.94

    def test_textbook_10_out_of_100(self):
        """10/100 at 95% CI ≈ [0.049, 0.176] per published Clopper-Pearson tables."""
        lower, upper = clopper_pearson_ci(10, 100, ci_level=0.95)
        assert 0.048 < lower < 0.051
        assert 0.174 < upper < 0.178

    def test_one_out_of_two_is_wide(self):
        """k=1/n=2 yields wide CI [~0.013, ~0.987]."""
        lower, upper = clopper_pearson_ci(1, 2, ci_level=0.95)
        assert 0.01 < lower < 0.03
        assert 0.97 < upper < 0.99

    def test_raises_on_invalid_successes(self):
        with pytest.raises(ValueError):
            clopper_pearson_ci(11, 10)
        with pytest.raises(ValueError):
            clopper_pearson_ci(-1, 10)

    def test_different_confidence_levels_widen_or_narrow(self):
        """Higher confidence → wider interval."""
        l90, u90 = clopper_pearson_ci(50, 100, ci_level=0.90)
        l99, u99 = clopper_pearson_ci(50, 100, ci_level=0.99)
        assert (u99 - l99) > (u90 - l90)


class TestBootstrapFromBinary:

    def test_empty_returns_trivial_ci(self):
        assert bootstrap_ci_from_binary([]) == (0.0, 1.0)

    def test_all_true_collapses_to_one(self):
        """A vector of all True returns CI [1.0, 1.0] (no resample variance)."""
        lo, hi = bootstrap_ci_from_binary([True] * 50)
        assert lo == 1.0
        assert hi == 1.0

    def test_all_false_collapses_to_zero(self):
        lo, hi = bootstrap_ci_from_binary([False] * 50)
        assert lo == 0.0
        assert hi == 0.0

    def test_seed_42_deterministic(self):
        """Paper-pinned seed=42 must be reproducible across calls."""
        indicators = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 10
        a = bootstrap_ci_from_binary(indicators, n_boot=1000, seed=42)
        b = bootstrap_ci_from_binary(indicators, n_boot=1000, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        """Sanity check that the seed is actually consumed."""
        indicators = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 10
        a = bootstrap_ci_from_binary(indicators, n_boot=1000, seed=42)
        b = bootstrap_ci_from_binary(indicators, n_boot=1000, seed=99)
        assert a != b

    def test_ci_contains_observed_proportion(self):
        """The point estimate must lie inside the bootstrap CI."""
        indicators = [1] * 70 + [0] * 30
        lo, hi = bootstrap_ci_from_binary(indicators, n_boot=1000, seed=42)
        assert lo < 0.70 < hi
