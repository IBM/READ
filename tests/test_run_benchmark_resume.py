# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for run_benchmark resume behavior."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_benchmark as rb
from read_toolkit.llm_client import JudgeResult


def _stub_summary_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "baseline_accuracy_by_model",
        "model_classification_metrics",
        "baseline_accuracy_by_domain",
        "base_only_accuracy_by_model",
        "base_only_model_classification_metrics",
        "base_only_accuracy_by_domain",
        "confusion_counts",
        "average_flip_rate_by_model",
        "average_flip_rate_by_perturbation",
        "policy_influence_rate_by_model",
        "policy_influence_rate_by_domain",
        "clause_sensitivity_rate_by_model",
        "clause_sensitivity_rate_by_domain",
        "ablation_specificity_rate",
        "complies_clause_robustness",
        "expected_direction_match_rate",
        "counterfactual_consistency",
        "outcome_consistency",
    ]:
        monkeypatch.setattr(rb, name, lambda *args, **kwargs: {})


def _write_dataset(path: Path) -> None:
    row = {
        "instance_id": "inst_001",
        "domain": "edu",
        "ground_truth_label": "VIOLATES",
        "policy_base_nl": "policy text",
        "test_query_input": "query text",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _result(label: str, raw: str) -> JudgeResult:
    binary = "YES" if label == "VIOLATES" else "NO"
    return JudgeResult(
        label=label,
        raw_response=raw,
        logprob=None,
        binary_label=binary,
        trace=None,
    )


def test_resume_fills_missing_perturbations_after_mid_instance_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "resume.csv"
    _write_dataset(dataset_path)

    monkeypatch.setattr(rb, "apply_all", lambda policy: {"remove_policy": "remove"})
    monkeypatch.setattr(
        rb,
        "parse_policy_clauses",
        lambda policy: [
            {"id": 1, "rule_id": "R1"},
            {"id": 2, "rule_id": "R2"},
        ],
    )
    monkeypatch.setattr(rb, "drop_clause", lambda policy, cid: f"drop-{cid}")
    monkeypatch.setattr(rb, "mask_clause", lambda policy, cid: f"mask-{cid}")
    monkeypatch.setattr(rb, "drop_clauses", lambda policy, cids: f"drop-combo-{cids}")
    monkeypatch.setattr(rb, "mask_clauses", lambda policy, cids: f"mask-combo-{cids}")
    monkeypatch.setattr(rb, "remove_policy", lambda policy: "remove-all")
    monkeypatch.setattr(rb, "_build_clients", lambda *args, **kwargs: {"mock-model": object()})
    _stub_summary_functions(monkeypatch)

    first_run_calls = iter(
        [
            (_result("VIOLATES", "base"), 10),
            (_result("COMPLIES", "remove"), 11),
            RuntimeError("temporary DNS failure"),
        ]
    )

    def crash_mid_instance(*args, **kwargs):
        value = next(first_run_calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(rb, "_judge_and_time", crash_mid_instance)

    with pytest.raises(RuntimeError, match="temporary DNS failure"):
        rb.run_benchmark(
            dataset_path=dataset_path,
            output_path=output_path,
            models=["mock-model"],
            domains=["edu"],
            limit=1,
            timeout=1,
            retries=1,
            seed=42,
        )

    partial_rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert [row["perturbation"] for row in partial_rows] == ["base", "remove_policy"]

    second_run_calls = iter(
        [
            (_result("COMPLIES", "drop-r1"), 12),
            (_result("VIOLATES", "mask-r1"), 13),
            (_result("COMPLIES", "drop-r2"), 14),
            (_result("VIOLATES", "mask-r2"), 15),
        ]
    )
    monkeypatch.setattr(rb, "_judge_and_time", lambda *args, **kwargs: next(second_run_calls))

    exit_code = rb.run_benchmark(
        dataset_path=dataset_path,
        output_path=output_path,
        models=["mock-model"],
        domains=["edu"],
        limit=1,
        timeout=1,
        retries=1,
        seed=42,
    )

    assert exit_code == 0

    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert [row["perturbation"] for row in rows] == [
        "base",
        "remove_policy",
        "drop_clause:R1",
        "mask_clause:R1",
        "drop_clause:R2",
        "mask_clause:R2",
    ]
    assert len(rows) == 6
    assert rows[0]["raw_response"] == "base"
    assert rows[1]["raw_response"] == "remove"
