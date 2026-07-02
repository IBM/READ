# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for READ metric computation.

Tests cover all metrics in both src/evaluator.py and src/summarize_results.py
using synthetic row dicts; no API calls or file I/O required.
"""
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from read_toolkit.evaluator import (
    _normalize_label,
    _is_flip,
    _clause_sensitivity_rate,
    ablation_specificity_rate,
    complies_clause_robustness,
    policy_influence_rate,
    average_flip_rate_by_model,
    model_classification_metrics,
    per_instance_csr,
    per_rule_csr,
    policy_corrected_csr,
)
from read_toolkit.summarize_results import (
    compute_fidelity,
    compute_pir,
    compute_metrics,
    run_sanity_checks,
    norm,
)


# Test data factory

def _make_row(
    instance_id="inst_001",
    model="model_a",
    perturbation="base",
    ground_truth="VIOLATES",
    predicted_label="VIOLATES",
    violated_rules="",
    is_flip=False,
):
    return {
        "instance_id": instance_id,
        "domain": "polyguard",
        "model": model,
        "perturbation": perturbation,
        "repeat_id": "0",
        "ground_truth": ground_truth,
        "violated_rules": violated_rules,
        "predicted_label": predicted_label,
        "raw_response": "",
        "trace": "",
        "binary_label": "",
        "logprob": "",
        "latency_ms": "100",
        "is_flip": str(is_flip).upper(),
        "seed": "42",
    }


# CSR tests (the bug fix, most critical)

class TestCSR:
    """CSR must only count removals of violated clauses."""

    def test_csr_only_counts_violated_clause(self):
        """1 instance violates R3 only. Model flips on R1 and R3.
        Correct CSR = 1.0 (only R3 eligible, it flips).
        Buggy code would give 2/3 = 0.667 (all three eligible).
        """
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R1", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
            _make_row(perturbation="drop_clause:R2", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
        ]
        # evaluator.py path
        result = _clause_sensitivity_rate(rows, "model")
        assert result["model_a"] == 1.0, f"Expected CSR=1.0, got {result['model_a']}"

        # summarize_results.py path
        fidelity = compute_fidelity(rows)
        assert fidelity is not None
        assert fidelity["targeted_csr"] == 1.0, f"Expected targeted_csr=1.0, got {fidelity['targeted_csr']}"

    def test_csr_multi_violation_individual_excluded(self):
        """Removing a single clause from a multi-violation case is NOT eligible.
        Only the combined removal (drop_clause:R3+R5) counts.
        """
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R3,R5"),
            _make_row(perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3,R5", is_flip=True),
            _make_row(perturbation="drop_clause:R5", predicted_label="VIOLATES", violated_rules="R3,R5"),
        ]
        # Individual clause removals on multi-violation → 0 eligible
        result = _clause_sensitivity_rate(rows, "model")
        assert result == {}, f"Expected empty (individual removals excluded), got {result}"

    def test_csr_multi_violation_combined_included(self):
        """Combined removal (drop_clause:R3+R5) on multi-violation case IS eligible.
        If model flips to COMPLIES, CSR counts it.
        """
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R3,R5"),
            _make_row(perturbation="drop_clause:R3+R5", predicted_label="COMPLIES", violated_rules="R3,R5", is_flip=True),
        ]
        result = _clause_sensitivity_rate(rows, "model")
        assert result["model_a"] == 1.0

        fidelity = compute_fidelity(rows)
        assert fidelity["targeted_csr"] == 1.0
        assert fidelity["n_eligible_ablations"] == 1

    def test_csr_single_and_multi_violation_mixed(self):
        """Two instances: one single-violation (R3), one multi (R3,R5) with combined removal.
        Both count toward CSR.
        """
        rows = [
            # Single-violation instance (R3 only), individual drop eligible
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
            # Multi-violation instance (R3,R5), only combined drop eligible
            _make_row(instance_id="i2", perturbation="base", predicted_label="VIOLATES", violated_rules="R3,R5"),
            _make_row(instance_id="i2", perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3,R5", is_flip=True),
            _make_row(instance_id="i2", perturbation="drop_clause:R3+R5", predicted_label="COMPLIES", violated_rules="R3,R5", is_flip=True),
        ]
        result = _clause_sensitivity_rate(rows, "model")
        assert result["model_a"] == 1.0  # 2 eligible (i1:R3 + i2:R3+R5), both flip

        fidelity = compute_fidelity(rows)
        assert fidelity["targeted_csr"] == 1.0
        assert fidelity["n_eligible_ablations"] == 2

    def test_csr_empty_violated_rules_skipped(self):
        """Instance is VIOLATES but violated_rules is empty (no checker).
        Should contribute 0 to CSR denominator.
        """
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules=""),
            _make_row(perturbation="drop_clause:R1", predicted_label="COMPLIES", violated_rules="", is_flip=True),
            _make_row(perturbation="drop_clause:R2", predicted_label="COMPLIES", violated_rules="", is_flip=True),
        ]
        result = _clause_sensitivity_rate(rows, "model")
        assert result == {}, f"Expected empty dict (no eligible rows), got {result}"

        fidelity = compute_fidelity(rows)
        # With no eligible CSR rows, compute_fidelity still returns but CSR-related fields are 0
        assert fidelity is not None
        assert fidelity["n_eligible_ablations"] == 0

    def test_csr_never_flips(self):
        """Model stays VIOLATES even when violated clause removed. CSR = 0.0."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R3", predicted_label="VIOLATES", violated_rules="R3"),
        ]
        result = _clause_sensitivity_rate(rows, "model")
        assert result["model_a"] == 0.0

    def test_csr_per_clause_only_contains_violated(self):
        """After fix, csr_per_clause should only have keys for violated clauses."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R1", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
            _make_row(perturbation="drop_clause:R2", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
        ]
        fidelity = compute_fidelity(rows)
        csr_keys = set(fidelity["csr_per_clause"].keys())
        assert csr_keys == {"R3"}, f"Expected only R3 in csr_per_clause, got {csr_keys}"

    def test_csr_mask_clause_also_works(self):
        """mask_clause: perturbations should be handled identically to drop_clause:."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R2"),
            _make_row(perturbation="mask_clause:R2", predicted_label="COMPLIES", violated_rules="R2", is_flip=True),
        ]
        result = _clause_sensitivity_rate(rows, "model")
        assert result["model_a"] == 1.0


# ASR tests (already correct; regression protection)

class TestASR:
    """ASR must only count removals of NON-violated clauses."""

    def test_asr_only_non_violated(self):
        """Instance violates R3. Removals of R1, R2 (non-violated) both stay VIOLATES.
        ASR = 2/2 = 1.0.
        """
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R1", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R2", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
        ]
        result = ablation_specificity_rate(rows, "model")
        assert result["model_a"] == 1.0

    def test_asr_fragile_model(self):
        """Model flips on non-violated clause removals too. ASR < 1.0."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R1", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
            _make_row(perturbation="drop_clause:R2", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
        ]
        result = ablation_specificity_rate(rows, "model")
        assert result["model_a"] == 0.5  # 1 stable out of 2 non-violated

    def test_asr_empty_violated_rules_skipped(self):
        """No violated_rules = row skipped entirely."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules=""),
            _make_row(perturbation="drop_clause:R1", predicted_label="VIOLATES", violated_rules=""),
        ]
        result = ablation_specificity_rate(rows, "model")
        assert result == {}


# CCR tests (already correct; regression protection)

class TestCCR:
    """CCR measures stability of COMPLIES verdicts under clause removal."""

    def test_ccr_complies_stays(self):
        """COMPLIES baseline, clause removal keeps COMPLIES. CCR = 1.0."""
        rows = [
            _make_row(perturbation="base", ground_truth="COMPLIES", predicted_label="COMPLIES"),
            _make_row(perturbation="drop_clause:R1", ground_truth="COMPLIES", predicted_label="COMPLIES"),
            _make_row(perturbation="drop_clause:R2", ground_truth="COMPLIES", predicted_label="COMPLIES"),
        ]
        result = complies_clause_robustness(rows, "model")
        assert result["model_a"] == 1.0

    def test_ccr_fabricated_violation(self):
        """Model fabricates VIOLATES after clause removal on COMPLIES case. CCR < 1.0."""
        rows = [
            _make_row(perturbation="base", ground_truth="COMPLIES", predicted_label="COMPLIES"),
            _make_row(perturbation="drop_clause:R1", ground_truth="COMPLIES", predicted_label="COMPLIES"),
            _make_row(perturbation="drop_clause:R2", ground_truth="COMPLIES", predicted_label="VIOLATES", is_flip=True),
        ]
        result = complies_clause_robustness(rows, "model")
        assert result["model_a"] == 0.5  # 1 stable out of 2


# PIR tests (already correct; regression protection)

class TestPIR:
    """PIR measures how much the policy influences verdicts."""

    def test_pir_flips(self):
        """Remove policy changes verdict. PIR = 1.0."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES"),
            _make_row(perturbation="remove_policy", predicted_label="COMPLIES", is_flip=True),
        ]
        # evaluator.py
        result = policy_influence_rate(rows, "model")
        assert result["model_a"]["pir"] == 1.0
        assert result["model_a"]["v_to_c"] == 1

        # summarize_results.py
        pir = compute_pir(rows)
        assert pir["pir"] == 1.0

    def test_pir_no_change(self):
        """Same verdict with and without policy. PIR = 0.0."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES"),
            _make_row(perturbation="remove_policy", predicted_label="VIOLATES"),
        ]
        result = policy_influence_rate(rows, "model")
        assert result["model_a"]["pir"] == 0.0

        pir = compute_pir(rows)
        assert pir["pir"] == 0.0

    def test_pir_mixed(self):
        """2 instances, 1 flips, 1 doesn't. PIR = 0.5."""
        rows = [
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES"),
            _make_row(instance_id="i1", perturbation="remove_policy", predicted_label="COMPLIES", is_flip=True),
            _make_row(instance_id="i2", perturbation="base", predicted_label="VIOLATES"),
            _make_row(instance_id="i2", perturbation="remove_policy", predicted_label="VIOLATES"),
        ]
        result = policy_influence_rate(rows, "model")
        assert result["model_a"]["pir"] == 0.5


# Accuracy / F1 / flip rate tests

class TestBasicMetrics:

    def test_accuracy_perfect(self):
        rows = [
            _make_row(perturbation="base", ground_truth="VIOLATES", predicted_label="VIOLATES"),
            _make_row(perturbation="base", ground_truth="COMPLIES", predicted_label="COMPLIES", instance_id="i2"),
        ]
        metrics = compute_metrics(rows)
        assert metrics["accuracy"] == 1.0

    def test_f1_degenerate_classifier(self):
        """Model always predicts COMPLIES. F1-VIOLATES = 0.0."""
        rows = [
            _make_row(perturbation="base", ground_truth="VIOLATES", predicted_label="COMPLIES"),
            _make_row(perturbation="base", ground_truth="COMPLIES", predicted_label="COMPLIES", instance_id="i2"),
        ]
        metrics = compute_metrics(rows)
        assert metrics["f1_violates"] == 0.0
        assert metrics["f1_complies"] > 0.0

    def test_flip_rate(self):
        """2 out of 4 perturbed rows flip. flip_rate = 0.5."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES"),
            _make_row(perturbation="typos", predicted_label="COMPLIES", is_flip=True),
            _make_row(perturbation="bullet_points", predicted_label="VIOLATES", is_flip=False),
            _make_row(perturbation="json_format", predicted_label="COMPLIES", is_flip=True),
            _make_row(perturbation="reorder_sentences", predicted_label="VIOLATES", is_flip=False),
        ]
        metrics = compute_metrics(rows)
        assert metrics["flip_rate"] == 0.5


# Cross-validation: evaluator.py vs summarize_results.py

class TestCrossValidation:

    def test_csr_evaluator_matches_summarizer(self):
        """Same single-violation data through both code paths must yield the same CSR."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R1", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
        ]
        evaluator_csr = _clause_sensitivity_rate(rows, "model")["model_a"]
        fidelity = compute_fidelity(rows)
        summarizer_csr = fidelity["targeted_csr"]
        assert evaluator_csr == summarizer_csr, (
            f"evaluator CSR={evaluator_csr} != summarizer CSR={summarizer_csr}"
        )


# Sanity check tests

class TestSanityChecks:

    def test_sanity_degenerate_classifier(self):
        """Flag fires when F1-VIOLATES = 0.0."""
        metrics = {"f1_violates": 0.0, "f1_complies": 0.67, "accuracy": 0.5}
        flags = run_sanity_checks([], metrics)
        degenerate = [f for f in flags if "DEGENERATE_CLASSIFIER" in f]
        assert len(degenerate) == 1

    def test_sanity_coin_flip(self):
        """Flag fires when accuracy is 50%."""
        metrics = {"f1_violates": 0.5, "f1_complies": 0.5, "accuracy": 0.50}
        flags = run_sanity_checks([], metrics)
        coin_flip = [f for f in flags if "COIN_FLIP" in f]
        assert len(coin_flip) == 1

    def test_sanity_no_false_positive(self):
        """No flags on a healthy model."""
        metrics = {"f1_violates": 0.85, "f1_complies": 0.90, "accuracy": 0.88}
        flags = run_sanity_checks([], metrics)
        assert len(flags) == 0


# Edge cases

class TestEdgeCases:

    def test_label_normalization(self):
        assert norm("YES") == "VIOLATES"
        assert norm("NO") == "COMPLIES"
        assert norm("UNSAFE") == "VIOLATES"
        assert norm("SAFE") == "COMPLIES"
        assert norm("violates") == "VIOLATES"
        assert norm("complies") == "COMPLIES"
        assert _normalize_label("YES") == "VIOLATES"
        assert _normalize_label("NO") == "COMPLIES"

    def test_empty_rows_metrics(self):
        result = compute_metrics([])
        assert result == {}

    def test_counterfactual_consistency(self):
        """drop_clause and mask_clause for same clause agree -> consistency = 1.0."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
            _make_row(perturbation="mask_clause:R3", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
        ]
        fidelity = compute_fidelity(rows)
        assert fidelity["counterfactual_consistency"] == 1.0

    def test_counterfactual_inconsistency(self):
        """drop_clause and mask_clause disagree -> consistency = 0.0."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3", is_flip=True),
            _make_row(perturbation="mask_clause:R3", predicted_label="VIOLATES", violated_rules="R3"),
        ]
        fidelity = compute_fidelity(rows)
        assert fidelity["counterfactual_consistency"] == 0.0

    def test_csr_combined_cross_validation(self):
        """Combined multi-violation CSR matches between evaluator and summarizer."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R20,R24"),
            _make_row(perturbation="drop_clause:R20+R24", predicted_label="COMPLIES", violated_rules="R20,R24", is_flip=True),
        ]
        evaluator_csr = _clause_sensitivity_rate(rows, "model")["model_a"]
        fidelity = compute_fidelity(rows)
        summarizer_csr = fidelity["targeted_csr"]
        assert evaluator_csr == summarizer_csr == 1.0


# Regression: bug-fix validation

class TestBugRegressions:

    def test_asr_combined_clause_excluded(self):
        """OLD BUG: 'R20+R24' not in {'R20','R24'} → combined removals leaked
        into ASR denominator. Now uses set-intersection."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R20,R24"),
            # Combined removal of both violated clauses; should be CSR, not ASR
            _make_row(perturbation="drop_clause:R20+R24", predicted_label="COMPLIES", violated_rules="R20,R24", is_flip=True),
            # Non-violated clause removal; this IS ASR territory
            _make_row(perturbation="drop_clause:R5", predicted_label="VIOLATES", violated_rules="R20,R24"),
        ]
        asr = ablation_specificity_rate(rows, "model")["model_a"]
        # Only R5 in denominator (1 row), stable → ASR=1.0
        assert asr == 1.0

    def test_asr_combined_partial_overlap_excluded(self):
        """Combined clause R20+R5 where R20 is violated; should be excluded from ASR."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES", violated_rules="R20"),
            _make_row(perturbation="drop_clause:R20+R5", predicted_label="COMPLIES", violated_rules="R20", is_flip=True),
            _make_row(perturbation="drop_clause:R5", predicted_label="VIOLATES", violated_rules="R20"),
        ]
        asr = ablation_specificity_rate(rows, "model")["model_a"]
        # R20+R5 excluded (R20 overlaps violated), only R5 in denominator → ASR=1.0
        assert asr == 1.0

    def test_is_flip_string_false(self):
        """OLD BUG: bool('False') == True. CSV-loaded rows had string is_flip."""
        assert _is_flip({"is_flip": False}) is False
        assert _is_flip({"is_flip": "False"}) is False
        assert _is_flip({"is_flip": "false"}) is False
        assert _is_flip({"is_flip": ""}) is False
        assert _is_flip({}) is False

    def test_is_flip_string_true(self):
        assert _is_flip({"is_flip": True}) is True
        assert _is_flip({"is_flip": "True"}) is True
        assert _is_flip({"is_flip": "TRUE"}) is True
        assert _is_flip({"is_flip": "1"}) is True

    def test_flip_rate_csv_strings(self):
        """flip_rate must handle string is_flip from CSV-loaded data."""
        rows = [
            _make_row(perturbation="base", predicted_label="VIOLATES"),
            _make_row(perturbation="typos", predicted_label="COMPLIES", is_flip="True"),
            _make_row(perturbation="bullet_points", predicted_label="VIOLATES", is_flip="False"),
        ]
        result = average_flip_rate_by_model(rows)
        assert result["model_a"] == 0.5  # 1 of 2 perturbed flips


# Per-Instance CSR tests

class TestPerInstanceCSR:
    """per_instance_csr returns {instance_id: bool} for each VIOLATES instance."""

    def test_basic_flip(self):
        """Instance violates R3, dropping R3 flips to COMPLIES -> True."""
        rows = [
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3"),
        ]
        result = per_instance_csr(rows)
        assert result["i1"] is True

    def test_no_flip(self):
        """Instance violates R3, dropping R3 stays VIOLATES -> False."""
        rows = [
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="drop_clause:R3", predicted_label="VIOLATES", violated_rules="R3"),
        ]
        result = per_instance_csr(rows)
        assert result["i1"] is False

    def test_multiple_instances(self):
        """Two instances: one flips, one doesn't."""
        rows = [
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3"),
            _make_row(instance_id="i2", perturbation="base", predicted_label="VIOLATES", violated_rules="R5"),
            _make_row(instance_id="i2", perturbation="drop_clause:R5", predicted_label="VIOLATES", violated_rules="R5"),
        ]
        result = per_instance_csr(rows)
        assert result["i1"] is True
        assert result["i2"] is False

    def test_complies_excluded(self):
        """base=COMPLIES instances should not appear in output."""
        rows = [
            _make_row(instance_id="i1", perturbation="base", predicted_label="COMPLIES", violated_rules=""),
        ]
        result = per_instance_csr(rows)
        assert "i1" not in result

    def test_model_filter(self):
        """Filter to specific model."""
        rows = [
            _make_row(instance_id="i1", model="m1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", model="m1", perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3"),
            _make_row(instance_id="i1", model="m2", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", model="m2", perturbation="drop_clause:R3", predicted_label="VIOLATES", violated_rules="R3"),
        ]
        result_m1 = per_instance_csr(rows, model="m1")
        result_m2 = per_instance_csr(rows, model="m2")
        assert result_m1["i1"] is True
        assert result_m2["i1"] is False


# Per-Rule CSR tests

class TestPerRuleCSR:
    """per_rule_csr returns {rule_id: {csr, n, flipped}} for single-violation instances."""

    def test_basic(self):
        """Two instances violate R3: one flips, one doesn't. CSR(R3)=0.5."""
        rows = [
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3"),
            _make_row(instance_id="i2", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i2", perturbation="drop_clause:R3", predicted_label="VIOLATES", violated_rules="R3"),
        ]
        result = per_rule_csr(rows)
        assert result["R3"]["csr"] == 0.5
        assert result["R3"]["n"] == 2
        assert result["R3"]["flipped"] == 1

    def test_multi_violation_excluded(self):
        """Multi-violation instances (R3,R5) should be excluded from per-rule."""
        rows = [
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3,R5"),
            _make_row(instance_id="i1", perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3,R5"),
        ]
        result = per_rule_csr(rows)
        assert len(result) == 0

    def test_multiple_rules(self):
        """Different rules tracked separately."""
        rows = [
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3"),
            _make_row(instance_id="i2", perturbation="base", predicted_label="VIOLATES", violated_rules="R5"),
            _make_row(instance_id="i2", perturbation="drop_clause:R5", predicted_label="VIOLATES", violated_rules="R5"),
        ]
        result = per_rule_csr(rows)
        assert result["R3"]["csr"] == 1.0
        assert result["R5"]["csr"] == 0.0


# Policy-Corrected CSR tests

class TestPolicyCorrectedCSR:
    """PC-CSR filters to policy-dependent instances before computing CSR."""

    def test_basic_pc_csr(self):
        """Instance is policy-dependent (base=V, remove_policy=C) and clause-sensitive."""
        rows = [
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="remove_policy", predicted_label="COMPLIES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3"),
        ]
        result = policy_corrected_csr(rows)
        assert result["model_a"]["pc_csr"] == 1.0
        assert result["model_a"]["n_eligible"] == 1

    def test_prior_driven_excluded(self):
        """Instance where remove_policy=VIOLATES is excluded (model uses priors)."""
        rows = [
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="remove_policy", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3"),
        ]
        result = policy_corrected_csr(rows)
        assert result.get("model_a", {}).get("n_eligible", 0) == 0

    def test_mixed_instances(self):
        """One policy-dependent, one prior-driven. PC-CSR only counts the first."""
        rows = [
            # i1: policy-dependent, clause-sensitive
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="remove_policy", predicted_label="COMPLIES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="drop_clause:R3", predicted_label="COMPLIES", violated_rules="R3"),
            # i2: prior-driven (remove_policy still VIOLATES)
            _make_row(instance_id="i2", perturbation="base", predicted_label="VIOLATES", violated_rules="R5"),
            _make_row(instance_id="i2", perturbation="remove_policy", predicted_label="VIOLATES", violated_rules="R5"),
            _make_row(instance_id="i2", perturbation="drop_clause:R5", predicted_label="COMPLIES", violated_rules="R5"),
        ]
        result = policy_corrected_csr(rows)
        assert result["model_a"]["pc_csr"] == 1.0
        assert result["model_a"]["n_eligible"] == 1  # only i1

    def test_policy_dependent_but_not_clause_sensitive(self):
        """Instance needs policy (V->C on remove) but drop_clause doesn't flip."""
        rows = [
            _make_row(instance_id="i1", perturbation="base", predicted_label="VIOLATES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="remove_policy", predicted_label="COMPLIES", violated_rules="R3"),
            _make_row(instance_id="i1", perturbation="drop_clause:R3", predicted_label="VIOLATES", violated_rules="R3"),
        ]
        result = policy_corrected_csr(rows)
        assert result["model_a"]["pc_csr"] == 0.0
        assert result["model_a"]["n_eligible"] == 1
