# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Tests for the helpers reproduce_paper.py builds on top of:
    - src/evaluator.with_vs_no_policy_accuracy
    - src/evaluator.pool_metrics_across_domains

Plus a tiny end-to-end smoke test that drives the script against a
synthetic two-domain × two-model CSV.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from read_toolkit.evaluator import pool_metrics_across_domains, with_vs_no_policy_accuracy


# row-builder (shared shape with run_benchmark.py CSV)

def _row(instance_id, model, perturbation, gt, pred, violated="", domain="ibm_hr"):
    return {
        "instance_id": instance_id,
        "domain": domain,
        "model": model,
        "perturbation": perturbation,
        "repeat_id": "0",
        "ground_truth": gt,
        "violated_rules": violated,
        "predicted_label": pred,
        "is_flip": pred != gt,
        "seed": 42,
    }


# with_vs_no_policy_accuracy

class TestWithVsNoPolicy:

    def test_gap_sign_positive_when_no_policy_more_accurate(self):
        """3/4 right with policy, 4/4 right without → gap = +0.25 (judge better off without policy)."""
        rows = []
        # Instance 1: V correct with policy, V correct without
        rows += [
            _row("i1", "m", "base", "VIOLATES", "VIOLATES"),
            _row("i1", "m", "remove_policy", "VIOLATES", "VIOLATES"),
        ]
        # Instance 2: V correct with policy, V correct without
        rows += [
            _row("i2", "m", "base", "VIOLATES", "VIOLATES"),
            _row("i2", "m", "remove_policy", "VIOLATES", "VIOLATES"),
        ]
        # Instance 3: V correct with policy, V correct without
        rows += [
            _row("i3", "m", "base", "VIOLATES", "VIOLATES"),
            _row("i3", "m", "remove_policy", "VIOLATES", "VIOLATES"),
        ]
        # Instance 4: V wrong with policy (the policy actively misled it), V correct without
        rows += [
            _row("i4", "m", "base", "VIOLATES", "COMPLIES"),
            _row("i4", "m", "remove_policy", "VIOLATES", "VIOLATES"),
        ]
        out = with_vs_no_policy_accuracy(rows, group_key="model")
        assert "m" in out
        assert out["m"]["with_policy_acc"] == 0.75
        assert out["m"]["no_policy_acc"] == 1.0
        assert out["m"]["gap"] == 0.25
        assert out["m"]["n"] == 4

    def test_gap_sign_negative_when_with_policy_more_accurate(self):
        """The expected case: policy helps. gap < 0."""
        rows = [
            _row("i1", "m", "base", "VIOLATES", "VIOLATES"),
            _row("i1", "m", "remove_policy", "VIOLATES", "COMPLIES"),
            _row("i2", "m", "base", "VIOLATES", "VIOLATES"),
            _row("i2", "m", "remove_policy", "VIOLATES", "COMPLIES"),
        ]
        out = with_vs_no_policy_accuracy(rows, group_key="model")
        assert out["m"]["with_policy_acc"] == 1.0
        assert out["m"]["no_policy_acc"] == 0.0
        assert out["m"]["gap"] == -1.0
        assert out["m"]["n"] == 2

    def test_excludes_instances_without_both_variants(self):
        """An instance with only a base row (no remove_policy) is dropped from the comparison."""
        rows = [
            _row("i1", "m", "base", "VIOLATES", "VIOLATES"),
            _row("i1", "m", "remove_policy", "VIOLATES", "VIOLATES"),
            _row("i2", "m", "base", "VIOLATES", "VIOLATES"),  # no remove_policy for i2
        ]
        out = with_vs_no_policy_accuracy(rows, group_key="model")
        assert out["m"]["n"] == 1

    def test_groups_separately_when_group_key_set(self):
        """Two judges → two entries; the second's data does not contaminate the first."""
        rows = [
            _row("i1", "phi-4", "base", "VIOLATES", "VIOLATES"),
            _row("i1", "phi-4", "remove_policy", "VIOLATES", "COMPLIES"),
            _row("i1", "llama-70b", "base", "VIOLATES", "COMPLIES"),
            _row("i1", "llama-70b", "remove_policy", "VIOLATES", "VIOLATES"),
        ]
        out = with_vs_no_policy_accuracy(rows, group_key="model")
        assert out["phi-4"]["gap"] == -1.0  # policy was helpful for phi-4
        assert out["llama-70b"]["gap"] == 1.0  # policy hurt llama-70b

    def test_no_eligible_rows_returns_empty(self):
        out = with_vs_no_policy_accuracy([])
        assert out == {}


# pool_metrics_across_domains

class TestPoolMetricsAcrossDomains:

    def test_pool_matches_raw_sum(self):
        """Pooled rate = sum(numerator) / sum(denominator), NOT mean of per-domain rates."""
        per_domain = {
            "ibm_hr": {"numerator": 50, "denominator": 100},   # 50%
            "edu": {"numerator": 10, "denominator": 100},      # 10%
        }
        # Mean of rates would be 30%; pooled rate is 60/200 = 30% here by coincidence
        # so use asymmetric counts:
        per_domain = {
            "ibm_hr": {"numerator": 80, "denominator": 100},   # 80%
            "edu": {"numerator": 0, "denominator": 900},       # 0% but huge denom
        }
        out = pool_metrics_across_domains(per_domain)
        assert out["numerator"] == 80
        assert out["denominator"] == 1000
        assert out["rate"] == 0.08  # NOT 40% (mean of 80% + 0%)

    def test_empty_returns_zero_rate(self):
        out = pool_metrics_across_domains({})
        assert out["numerator"] == 0
        assert out["denominator"] == 0
        assert out["rate"] == 0.0

    def test_zero_denominator_returns_zero_rate(self):
        out = pool_metrics_across_domains({"d": {"numerator": 0, "denominator": 0}})
        assert out["rate"] == 0.0

    def test_three_domain_pool(self):
        per_domain = {
            "a": {"numerator": 10, "denominator": 100},
            "b": {"numerator": 20, "denominator": 200},
            "c": {"numerator": 30, "denominator": 300},
        }
        out = pool_metrics_across_domains(per_domain)
        assert out["numerator"] == 60
        assert out["denominator"] == 600
        assert out["rate"] == 0.1


# End-to-end smoke test for reproduce_paper.py

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _faithful_v_rows(iid, model, domain):
    return [
        _row(iid, model, "base", "VIOLATES", "VIOLATES", "R1", domain=domain),
        _row(iid, model, "remove_policy", "VIOLATES", "COMPLIES", "R1", domain=domain),
        _row(iid, model, "drop_clause:R1", "VIOLATES", "COMPLIES", "R1", domain=domain),
        _row(iid, model, "drop_clause:R2", "VIOLATES", "VIOLATES", "R1", domain=domain),
        _row(iid, model, "drop_clause:R3", "VIOLATES", "VIOLATES", "R1", domain=domain),
    ]


def _compliant_faithful_rows(iid, model, domain):
    return [
        _row(iid, model, "base", "COMPLIES", "COMPLIES", "", domain=domain),
        _row(iid, model, "remove_policy", "COMPLIES", "COMPLIES", "", domain=domain),
        _row(iid, model, "drop_clause:R1", "COMPLIES", "COMPLIES", "", domain=domain),
        _row(iid, model, "drop_clause:R2", "COMPLIES", "COMPLIES", "", domain=domain),
        _row(iid, model, "drop_clause:R3", "COMPLIES", "COMPLIES", "", domain=domain),
    ]


def test_reproduce_paper_smoke(tmp_path):
    """Build a tiny two-domain × one-model CSV and verify the script writes
    table_s1.csv, table_s4.csv, table_s5.csv, figure_data.json without crashing
    and produces a plausibly-shaped output."""
    rows_a = (
        _faithful_v_rows("a1", "judge", "ibm_hr")
        + _compliant_faithful_rows("a2", "judge", "ibm_hr")
    )
    rows_b = (
        _faithful_v_rows("b1", "judge", "edu")
        + _compliant_faithful_rows("b2", "judge", "edu")
    )

    csv_a = tmp_path / "ibm_hr.csv"
    csv_b = tmp_path / "edu.csv"
    _write_csv(rows_a, csv_a)
    _write_csv(rows_b, csv_b)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "reproduce_paper.py"),
            "--domain-csvs", f"ibm_hr={csv_a},edu={csv_b}",
            "--output-dir", str(out_dir),
            "--tables", "s1,s4,s5,figure",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"reproduce_paper failed:\n{result.stderr}"

    # All four artefacts exist
    assert (out_dir / "table_s1.csv").exists()
    assert (out_dir / "table_s4.csv").exists()
    assert (out_dir / "table_s5.csv").exists()
    assert (out_dir / "figure_data.json").exists()

    # Table S1 has 2 rows (2 domains × 1 model)
    s1_content = (out_dir / "table_s1.csv").read_text().strip().splitlines()
    assert len(s1_content) == 3  # header + 2 data rows

    # Table S4 has 1 row (1 model pooled)
    s4_content = (out_dir / "table_s4.csv").read_text().strip().splitlines()
    assert len(s4_content) == 2  # header + 1 data row

    # Figure data is keyed by domain → model
    fig = json.loads((out_dir / "figure_data.json").read_text())
    assert "ibm_hr" in fig and "edu" in fig
    assert "judge" in fig["ibm_hr"]
