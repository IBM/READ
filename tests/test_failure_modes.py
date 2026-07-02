# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Tests for the failure-mode classifier (src/failure_modes.py).

The 14 cases below come from the paper's Figure 3 decision tree (READ paper,
sections 3 and 5.2) and from the transfer-plan design doc. Each per-instance
test pins a single branch of the tree; the dataset-level tests cover the
long-format CSV → per-instance bundle pivot and the aggregation pipeline.
"""

from __future__ import annotations

import pytest

from read_toolkit.failure_modes import (
    DEFAULT_THRESHOLDS,
    FailureMode,
    _bundle_per_instance,
    classify_dataset,
    classify_failure_mode,
    degenerate_flag,
    failure_mode_distribution,
)


# row-builder helpers (mirror the run_benchmark.py CSV schema)

def _row(
    instance_id: str,
    model: str,
    perturbation: str,
    ground_truth: str,
    predicted_label: str,
    violated_rules: str = "",
    domain: str = "ibm_hr",
) -> dict:
    return {
        "instance_id": instance_id,
        "domain": domain,
        "model": model,
        "perturbation": perturbation,
        "repeat_id": "0",
        "ground_truth": ground_truth,
        "violated_rules": violated_rules,
        "predicted_label": predicted_label,
        "is_flip": predicted_label != ground_truth,
        "seed": 42,
    }


def _make_v_rows(instance_id: str, model: str, kind: str) -> list[dict]:
    """Build CSV rows for one V-branch (model, instance) in a specific failure-mode shape."""
    if kind == "faithful":
        # y_P=V, y_0=COMPLIES, drop(R1)=COMPLIES (flips), drops(R2,R3)=V (stable)
        return [
            _row(instance_id, model, "base", "VIOLATES", "VIOLATES", "R1"),
            _row(instance_id, model, "remove_policy", "VIOLATES", "COMPLIES", "R1"),
            _row(instance_id, model, "drop_clause:R1", "VIOLATES", "COMPLIES", "R1"),
            _row(instance_id, model, "drop_clause:R2", "VIOLATES", "VIOLATES", "R1"),
            _row(instance_id, model, "drop_clause:R3", "VIOLATES", "VIOLATES", "R1"),
        ]
    if kind == "confused":
        # y_P=V, y_0=COMPLIES, drop(R1)=V (no flip on violated clause), others=V
        return [
            _row(instance_id, model, "base", "VIOLATES", "VIOLATES", "R1"),
            _row(instance_id, model, "remove_policy", "VIOLATES", "COMPLIES", "R1"),
            _row(instance_id, model, "drop_clause:R1", "VIOLATES", "VIOLATES", "R1"),
            _row(instance_id, model, "drop_clause:R2", "VIOLATES", "VIOLATES", "R1"),
            _row(instance_id, model, "drop_clause:R3", "VIOLATES", "VIOLATES", "R1"),
        ]
    if kind == "prior":
        # y_P=COMPLIES, y_0=COMPLIES → Prior-Based (false negative both ways)
        return [
            _row(instance_id, model, "base", "VIOLATES", "COMPLIES", "R1"),
            _row(instance_id, model, "remove_policy", "VIOLATES", "COMPLIES", "R1"),
            _row(instance_id, model, "drop_clause:R1", "VIOLATES", "COMPLIES", "R1"),
            _row(instance_id, model, "drop_clause:R2", "VIOLATES", "VIOLATES", "R1"),
            _row(instance_id, model, "drop_clause:R3", "VIOLATES", "VIOLATES", "R1"),
        ]
    if kind == "policy_degraded":
        # y_P=COMPLIES (wrong with policy), y_0=VIOLATES (correct without)
        return [
            _row(instance_id, model, "base", "VIOLATES", "COMPLIES", "R1"),
            _row(instance_id, model, "remove_policy", "VIOLATES", "VIOLATES", "R1"),
            _row(instance_id, model, "drop_clause:R1", "VIOLATES", "COMPLIES", "R1"),
            _row(instance_id, model, "drop_clause:R2", "VIOLATES", "VIOLATES", "R1"),
            _row(instance_id, model, "drop_clause:R3", "VIOLATES", "VIOLATES", "R1"),
        ]
    raise ValueError(f"unknown V-kind {kind!r}")


# 1-5: V-branch cases

def test_v_branch_prior_based_false_negative():
    """V instance, both with and without policy say COMPLIES → Prior-Based (false negative)."""
    verdicts = {
        "ground_truth": "VIOLATES",
        "y_P": "COMPLIES",
        "y_0": "COMPLIES",
        "y_P_minus_rstar": "COMPLIES",
        "y_P_minus_others": ["VIOLATES", "VIOLATES", "VIOLATES"],
    }
    assert classify_failure_mode(verdicts) == FailureMode.PRIOR_BASED


def test_v_branch_policy_degraded():
    """V instance: y_P=COMPLIES (wrong with policy), y_0=VIOLATES (correct without) → Policy-Degraded."""
    verdicts = {
        "ground_truth": "VIOLATES",
        "y_P": "COMPLIES",
        "y_0": "VIOLATES",
        "y_P_minus_rstar": "COMPLIES",
        "y_P_minus_others": ["VIOLATES", "VIOLATES", "VIOLATES"],
    }
    assert classify_failure_mode(verdicts) == FailureMode.POLICY_DEGRADED


def test_v_branch_faithful_clause_grounded():
    """V instance: y_P=V correct, y_0=C (policy needed), drop-violated flips, drop-others stable → Faithful."""
    verdicts = {
        "ground_truth": "VIOLATES",
        "y_P": "VIOLATES",
        "y_0": "COMPLIES",
        "y_P_minus_rstar": "COMPLIES",
        "y_P_minus_others": ["VIOLATES", "VIOLATES", "VIOLATES"],
    }
    assert classify_failure_mode(verdicts) == FailureMode.FAITHFUL


def test_v_branch_confused_no_flip_on_violated_clause():
    """V instance: y_P=V correct, y_0=C (policy needed), but drop-violated stays V → Confused."""
    verdicts = {
        "ground_truth": "VIOLATES",
        "y_P": "VIOLATES",
        "y_0": "COMPLIES",
        "y_P_minus_rstar": "VIOLATES",
        "y_P_minus_others": ["VIOLATES", "VIOLATES", "VIOLATES"],
    }
    assert classify_failure_mode(verdicts) == FailureMode.CONFUSED


def test_v_branch_confused_flip_but_collateral_flip():
    """V instance: drop-violated flips, but some non-violated drop also flips → not specific → Confused."""
    verdicts = {
        "ground_truth": "VIOLATES",
        "y_P": "VIOLATES",
        "y_0": "COMPLIES",
        "y_P_minus_rstar": "COMPLIES",
        "y_P_minus_others": ["VIOLATES", "COMPLIES", "VIOLATES"],
    }
    assert classify_failure_mode(verdicts) == FailureMode.CONFUSED


# 6-9: C-branch cases

def test_c_branch_faithful_stable():
    """C instance: y_P=C correct, all clause drops stay C → Faithful."""
    verdicts = {
        "ground_truth": "COMPLIES",
        "y_P": "COMPLIES",
        "y_0": "COMPLIES",
        "y_P_minus_rstar": None,
        "y_P_minus_others": ["COMPLIES", "COMPLIES", "COMPLIES"],
    }
    assert classify_failure_mode(verdicts) == FailureMode.FAITHFUL


def test_c_branch_prior_based_false_positive():
    """C instance: y_P=V (false positive), y_0=V (also false positive) → Prior-Based."""
    verdicts = {
        "ground_truth": "COMPLIES",
        "y_P": "VIOLATES",
        "y_0": "VIOLATES",
        "y_P_minus_rstar": None,
        "y_P_minus_others": ["VIOLATES", "VIOLATES", "VIOLATES"],
    }
    assert classify_failure_mode(verdicts) == FailureMode.PRIOR_BASED


def test_c_branch_policy_degraded_spurious_violation():
    """C instance: y_P=V (false positive only with policy), y_0=C (correct without) → Policy-Degraded."""
    verdicts = {
        "ground_truth": "COMPLIES",
        "y_P": "VIOLATES",
        "y_0": "COMPLIES",
        "y_P_minus_rstar": None,
        "y_P_minus_others": ["VIOLATES", "VIOLATES", "VIOLATES"],
    }
    assert classify_failure_mode(verdicts) == FailureMode.POLICY_DEGRADED


def test_c_branch_confused_fragile_under_ablation():
    """C instance: y_P=C correct, but at least one drop flips to V → Confused (fragile)."""
    verdicts = {
        "ground_truth": "COMPLIES",
        "y_P": "COMPLIES",
        "y_0": "COMPLIES",
        "y_P_minus_rstar": None,
        "y_P_minus_others": ["COMPLIES", "VIOLATES", "COMPLIES"],
    }
    assert classify_failure_mode(verdicts) == FailureMode.CONFUSED


# 10-11: UNKNOWN propagation

def test_unknown_y_P_propagates_to_unclassified():
    """UNKNOWN on the base verdict → cannot classify → Unclassified."""
    verdicts = {
        "ground_truth": "VIOLATES",
        "y_P": "UNKNOWN",
        "y_0": "COMPLIES",
        "y_P_minus_rstar": "COMPLIES",
        "y_P_minus_others": ["VIOLATES", "VIOLATES", "VIOLATES"],
    }
    assert classify_failure_mode(verdicts) == FailureMode.UNCLASSIFIED


def test_v_branch_unknown_in_others_dropped_still_faithful():
    """UNKNOWN siblings in y_P_minus_others are dropped from the specificity check, not a disqualifier."""
    verdicts = {
        "ground_truth": "VIOLATES",
        "y_P": "VIOLATES",
        "y_0": "COMPLIES",
        "y_P_minus_rstar": "COMPLIES",
        "y_P_minus_others": ["VIOLATES", "UNKNOWN", "VIOLATES"],
    }
    assert classify_failure_mode(verdicts) == FailureMode.FAITHFUL


# 12-13: dataset-level pivot + distribution

def test_classify_dataset_handles_long_format_csv_pivot():
    """classify_dataset assembles per-instance bundles from long-format rows."""
    rows = _make_v_rows("i1", "judge1", "faithful")
    result = classify_dataset(rows)
    assert ("judge1", "i1") in result
    assert result[("judge1", "i1")] == FailureMode.FAITHFUL


def test_failure_mode_distribution_modal_label_and_share():
    """Distribution computes modal V-Mode and its share. 2 Confused + 1 Faithful → Confused 67%."""
    rows: list[dict] = []
    rows.extend(_make_v_rows("i1", "judge_x", "confused"))
    rows.extend(_make_v_rows("i2", "judge_x", "confused"))
    rows.extend(_make_v_rows("i3", "judge_x", "faithful"))
    out = failure_mode_distribution(rows, group_key="model")
    assert "judge_x" in out
    assert out["judge_x"]["v_mode"]["label"] == "Confused"
    assert abs(out["judge_x"]["v_mode"]["share"] - 2 / 3) < 1e-3
    assert out["judge_x"]["n_violation"] == 3
    assert out["judge_x"]["v_mode"]["distribution"]["Confused"] > out["judge_x"]["v_mode"]["distribution"]["Faithful"]


# 14: Degenerate threshold (paper C8 = 0.05, strict less-than)

def test_degenerate_flag_at_0_05_threshold():
    """Paper Table 1 footnote: cells with F1<0.05 in either class get the † flag."""
    assert degenerate_flag(f1_violates=0.04, f1_complies=0.90) is True
    assert degenerate_flag(f1_violates=0.06, f1_complies=0.90) is False
    assert degenerate_flag(f1_violates=0.90, f1_complies=0.04) is True
    # Strict less-than: exact 0.05 is not flagged
    assert degenerate_flag(f1_violates=0.05, f1_complies=0.05) is False
    # Custom threshold override (sensitivity analysis)
    assert degenerate_flag(f1_violates=0.08, f1_complies=0.90, threshold=0.10) is True


# additional safety: multi-violation case (combined clause perturbation)

def test_multi_violation_routes_combined_drop_as_rstar():
    """For multi-violation instances (violated_rules='R1,R2'), the combined drop_clause:R1+R2
    perturbation is the r* removal; individual R1 / R2 drops are collateral."""
    rows = [
        _row("i1", "j", "base", "VIOLATES", "VIOLATES", "R1,R2"),
        _row("i1", "j", "remove_policy", "VIOLATES", "COMPLIES", "R1,R2"),
        _row("i1", "j", "drop_clause:R1+R2", "VIOLATES", "COMPLIES", "R1,R2"),
        _row("i1", "j", "drop_clause:R1", "VIOLATES", "VIOLATES", "R1,R2"),  # single is collateral
        _row("i1", "j", "drop_clause:R2", "VIOLATES", "VIOLATES", "R1,R2"),
        _row("i1", "j", "drop_clause:R3", "VIOLATES", "VIOLATES", "R1,R2"),
    ]
    bundles = _bundle_per_instance(rows)
    bundle = bundles[("j", "i1")]
    assert bundle["y_P_minus_rstar"] == "COMPLIES"
    # All four non-combined drops live in y_P_minus_others
    assert len(bundle["y_P_minus_others"]) == 3
    # And classification = FAITHFUL (combined drop flips, all single drops stable)
    assert classify_failure_mode(bundle) == FailureMode.FAITHFUL


def test_compliant_with_no_violated_rules_routes_all_drops_to_others():
    """COMPLIES instance has violated_rules='', so every drop_clause is a non-violated ablation."""
    rows = [
        _row("c1", "j", "base", "COMPLIES", "COMPLIES", ""),
        _row("c1", "j", "remove_policy", "COMPLIES", "COMPLIES", ""),
        _row("c1", "j", "drop_clause:R1", "COMPLIES", "COMPLIES", ""),
        _row("c1", "j", "drop_clause:R2", "COMPLIES", "COMPLIES", ""),
        _row("c1", "j", "drop_clause:R3", "COMPLIES", "VIOLATES", ""),  # fragile under one ablation
    ]
    bundles = _bundle_per_instance(rows)
    bundle = bundles[("j", "c1")]
    assert bundle["y_P_minus_rstar"] is None  # no violated rules → no r*
    assert len(bundle["y_P_minus_others"]) == 3
    assert classify_failure_mode(bundle) == FailureMode.CONFUSED
