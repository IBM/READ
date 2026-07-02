# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Metric computation for the READ benchmark.

All public functions operate on the flat list[dict] produced by run_benchmark
(one dict per CSV row). Metrics are grouped into three families:

  - Accuracy / classification  (accuracy, F1, confusion counts)
  - Robustness                 (flip rate by model, by perturbation)
  - Fidelity                   (clause sensitivity rate)
"""

from __future__ import annotations


def _is_flip(row: dict) -> bool:
    """Handle is_flip from both in-memory (bool) and CSV-loaded (string) rows."""
    v = row.get("is_flip", "")
    if isinstance(v, bool):
        return v
    return str(v).strip().upper() in ("TRUE", "1", "YES")


def _normalize_label(label: str) -> str:
    normalized = (label or "").strip().upper()
    mapping = {
        "YES": "VIOLATES",
        "NO": "COMPLIES",
        "UNSAFE": "VIOLATES",
        "SAFE": "COMPLIES",
        "COMPLIES": "COMPLIES",
        "VIOLATES": "VIOLATES",
    }
    return mapping.get(normalized, normalized)


def _group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def _accuracy_for_rows(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    correct = sum(
        1 for r in rows
        if _normalize_label(r["predicted_label"]) == _normalize_label(r["ground_truth"])
    )
    return correct / len(rows)


def _binary_prf(rows: list[dict], positive_label: str) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for row in rows:
        gt = _normalize_label(row["ground_truth"])
        pred = _normalize_label(row["predicted_label"])
        if pred == positive_label and gt == positive_label:
            tp += 1
        elif pred == positive_label and gt != positive_label:
            fp += 1
        elif pred != positive_label and gt == positive_label:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _is_clause_perturbation(name: str) -> bool:
    return name.startswith("drop_clause:") or name.startswith("mask_clause:")


# accuracy / classification metrics

def baseline_accuracy_by_model(rows: list[dict]) -> dict[str, float]:
    return {model: _accuracy_for_rows(vals) for model, vals in _group_by(rows, "model").items()}


def baseline_accuracy_by_domain(rows: list[dict]) -> dict[str, float]:
    return {domain: _accuracy_for_rows(vals) for domain, vals in _group_by(rows, "domain").items()}


def base_only_accuracy_by_model(rows: list[dict]) -> dict[str, float]:
    return baseline_accuracy_by_model([r for r in rows if r.get("perturbation") == "base"])


def base_only_accuracy_by_domain(rows: list[dict]) -> dict[str, float]:
    return baseline_accuracy_by_domain([r for r in rows if r.get("perturbation") == "base"])


def model_classification_metrics(rows: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model, values in _group_by(rows, "model").items():
        _, _, f1_violates = _binary_prf(values, "VIOLATES")
        _, _, f1_complies = _binary_prf(values, "COMPLIES")
        macro_f1 = (f1_violates + f1_complies) / 2
        out[model] = {
            "accuracy": _accuracy_for_rows(values),
            "f1_violates": f1_violates,
            "f1_complies": f1_complies,
            "macro_f1": macro_f1,
        }
    return out


def base_only_model_classification_metrics(rows: list[dict]) -> dict[str, dict[str, float]]:
    return model_classification_metrics([r for r in rows if r.get("perturbation") == "base"])


def confusion_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"tp_violates": 0, "tn_complies": 0, "fp_violates": 0, "fn_violates": 0}
    for row in rows:
        gt = _normalize_label(row["ground_truth"])
        pred = _normalize_label(row["predicted_label"])
        if gt == "VIOLATES" and pred == "VIOLATES":
            counts["tp_violates"] += 1
        elif gt == "COMPLIES" and pred == "COMPLIES":
            counts["tn_complies"] += 1
        elif gt == "COMPLIES" and pred == "VIOLATES":
            counts["fp_violates"] += 1
        elif gt == "VIOLATES" and pred == "COMPLIES":
            counts["fn_violates"] += 1
    return counts


# robustness metrics

def average_flip_rate_by_model(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for model, values in _group_by(rows, "model").items():
        perturbed = [r for r in values if r.get("perturbation") != "base"]
        if not perturbed:
            out[model] = 0.0
            continue
        out[model] = sum(1 for r in perturbed if _is_flip(r)) / len(perturbed)
    return out


def average_flip_rate_by_perturbation(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        p = row.get("perturbation", "base")
        if p != "base":
            grouped.setdefault(p, []).append(row)
    return {
        p: (sum(1 for r in vals if _is_flip(r)) / len(vals)) if vals else 0.0
        for p, vals in grouped.items()
    }


# fidelity metrics

def _build_base_labels(rows: list[dict]) -> dict[tuple[str, str], str]:
    """Map (instance_id, model) -> normalized baseline label."""
    labels: dict[tuple[str, str], str] = {}
    for row in rows:
        if row.get("perturbation") != "base":
            continue
        key = (str(row.get("instance_id")), str(row.get("model")))
        labels[key] = _normalize_label(str(row.get("predicted_label", "")))
    return labels


# PIR (Policy Influence Rate), Level 1 fidelity

def policy_influence_rate(rows: list[dict], group_key: str = "model") -> dict[str, dict[str, float | int]]:
    """PIR = |verdict(policy) != verdict(no-policy)| / N. PIR=0 means pure prior. Reports V→C and C→V splits."""
    base_labels = _build_base_labels(rows)
    nopolicy_labels: dict[tuple[str, str], str] = {}
    nopolicy_groups: dict[tuple[str, str], str] = {}

    for row in rows:
        if row.get("perturbation") != "remove_policy":
            continue
        key = (str(row.get("instance_id")), str(row.get("model", "")))
        nopolicy_labels[key] = _normalize_label(str(row.get("predicted_label", "")))
        nopolicy_groups[key] = str(row.get(group_key, ""))

    counts: dict[str, dict[str, int]] = {}
    for key, no_pol_label in nopolicy_labels.items():
        base_label = base_labels.get(key)
        if base_label not in ("VIOLATES", "COMPLIES"):
            continue
        if no_pol_label not in ("VIOLATES", "COMPLIES"):
            continue
        group = nopolicy_groups.get(key, "")
        stats = counts.setdefault(group, {"total": 0, "flipped": 0, "v_to_c": 0, "c_to_v": 0})
        stats["total"] += 1
        if base_label != no_pol_label:
            stats["flipped"] += 1
            if base_label == "VIOLATES" and no_pol_label == "COMPLIES":
                stats["v_to_c"] += 1
            elif base_label == "COMPLIES" and no_pol_label == "VIOLATES":
                stats["c_to_v"] += 1

    result: dict[str, dict[str, float | int]] = {}
    for g, s in counts.items():
        result[g] = {
            "pir": s["flipped"] / s["total"] if s["total"] else 0.0,
            "n": s["total"],
            "v_to_c": s["v_to_c"],
            "c_to_v": s["c_to_v"],
        }
    return result


def policy_influence_rate_by_model(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    return policy_influence_rate(rows, "model")


def policy_influence_rate_by_domain(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    return policy_influence_rate(rows, "domain")


def _clause_sensitivity_rate(rows: list[dict], group_key: str) -> dict[str, float]:
    """CSR grouped by ``group_key``: among base=VIOLATES cases, share that flip to COMPLIES
    when *all* violated clauses are removed (so combined drops like ``R20+R24`` are required
    for multi-violation instances; avoids the partial-removal confound)."""
    base_labels = _build_base_labels(rows)
    grouped_counts: dict[str, dict[str, int]] = {}

    for row in rows:
        perturbation = str(row.get("perturbation", ""))
        if not _is_clause_perturbation(perturbation):
            continue
        removed_ids = _extract_clause_ids(perturbation)
        if removed_ids is None:
            continue
        key = (str(row.get("instance_id")), str(row.get("model", "")))
        if base_labels.get(key) != "VIOLATES":
            continue
        # Need violated_rules to know which clauses ARE violated
        vr = str(row.get("violated_rules", ""))
        if not vr:
            continue
        violated = set(vr.split(","))
        if removed_ids != violated:
            continue
        perturbed = _normalize_label(str(row.get("predicted_label", "")))
        if perturbed not in ("VIOLATES", "COMPLIES"):
            continue  # exclude UNKNOWN from both numerator and denominator
        group = str(row.get(group_key, ""))
        stats = grouped_counts.setdefault(group, {"eligible": 0, "sensitive": 0})
        stats["eligible"] += 1
        if perturbed == "COMPLIES":
            stats["sensitive"] += 1

    return {
        g: (s["sensitive"] / s["eligible"]) if s["eligible"] else 0.0
        for g, s in grouped_counts.items()
    }


def clause_sensitivity_rate_by_model(rows: list[dict]) -> dict[str, float]:
    return _clause_sensitivity_rate(rows, "model")


def clause_sensitivity_rate_by_domain(rows: list[dict]) -> dict[str, float]:
    return _clause_sensitivity_rate(rows, "domain")


def ablation_specificity_rate(rows: list[dict], group_key: str = "model") -> dict[str, float]:
    """ASR: among base=VIOLATES cases, share that *stay* at VIOLATES when a non-violated clause is removed.
    High CSR + high ASR = clause-grounded; high CSR + low ASR = fragile. Requires ``violated_rules``."""
    base_labels = _build_base_labels(rows)
    counts: dict[str, dict[str, int]] = {}

    for row in rows:
        perturbation = str(row.get("perturbation", ""))
        if not _is_clause_perturbation(perturbation):
            continue
        clause_id = _extract_clause_id(perturbation)
        if clause_id is None:
            continue

        key = (str(row.get("instance_id")), str(row.get("model", "")))
        if base_labels.get(key) != "VIOLATES":
            continue

        # need violated_rules to know which clauses ARE violated
        vr = str(row.get("violated_rules", ""))
        if not vr:
            continue  # no checker data, skip
        violated = set(vr.split(","))

        # only count non-violated clause removals
        # Use set-intersection for combined clauses like "R20+R24"
        removed_ids = set(clause_id.split("+"))
        if removed_ids & violated:
            continue

        perturbed = _normalize_label(str(row.get("predicted_label", "")))
        if perturbed not in ("VIOLATES", "COMPLIES"):
            continue  # exclude UNKNOWN from both numerator and denominator
        group = str(row.get(group_key, ""))
        stats = counts.setdefault(group, {"total": 0, "stable": 0})
        stats["total"] += 1
        if perturbed == "VIOLATES":
            stats["stable"] += 1

    return {
        g: (s["stable"] / s["total"]) if s["total"] else 0.0
        for g, s in counts.items()
    }


def complies_clause_robustness(rows: list[dict], group_key: str = "model") -> dict[str, float]:
    """CCR: among gt=COMPLIES and base=COMPLIES cases, share that stay COMPLIES under clause removal.
    Should be near 100%; anything less means the model fabricates violations when the policy text changes."""
    base_labels = _build_base_labels(rows)
    counts: dict[str, dict[str, int]] = {}

    for row in rows:
        perturbation = str(row.get("perturbation", ""))
        if not _is_clause_perturbation(perturbation):
            continue
        key = (str(row.get("instance_id")), str(row.get("model", "")))

        gt = _normalize_label(str(row.get("ground_truth", "")))
        if gt != "COMPLIES":
            continue
        if base_labels.get(key) != "COMPLIES":
            continue

        perturbed = _normalize_label(str(row.get("predicted_label", "")))
        if perturbed not in ("VIOLATES", "COMPLIES"):
            continue  # exclude UNKNOWN from both numerator and denominator
        group = str(row.get(group_key, ""))
        stats = counts.setdefault(group, {"total": 0, "stable": 0})
        stats["total"] += 1
        if perturbed == "COMPLIES":
            stats["stable"] += 1

    return {
        g: (s["stable"] / s["total"]) if s["total"] else 0.0
        for g, s in counts.items()
    }


# per-instance & policy-corrected fidelity (see paper §3.4)

def per_instance_csr(rows: list[dict], model: str | None = None) -> dict[str, bool]:
    """Per-instance CSR: ``{instance_id: True}`` iff dropping the violated rule flipped to COMPLIES."""
    filtered = rows if model is None else [r for r in rows if r.get("model") == model]
    base_labels = _build_base_labels(filtered)

    # Map (instance_id, model) -> violated_rules from base rows
    violated_map: dict[tuple[str, str], set[str]] = {}
    for row in filtered:
        if row.get("perturbation") != "base":
            continue
        vr = str(row.get("violated_rules", ""))
        if vr:
            key = (str(row.get("instance_id")), str(row.get("model", "")))
            violated_map[key] = set(vr.split(","))

    # Collect drop_clause results
    drop_results: dict[tuple[str, str], dict[str, str]] = {}
    for row in filtered:
        perturbation = str(row.get("perturbation", ""))
        if not _is_clause_perturbation(perturbation):
            continue
        removed = _extract_clause_ids(perturbation)
        if removed is None:
            continue
        key = (str(row.get("instance_id")), str(row.get("model", "")))
        label = _normalize_label(str(row.get("predicted_label", "")))
        drop_results.setdefault(key, {})[perturbation] = label

    result: dict[str, bool] = {}
    for key, violated in violated_map.items():
        if base_labels.get(key) != "VIOLATES":
            continue
        iid = key[0]
        # Find the perturbation that removes exactly the violated rules
        flipped = False
        for pert, label in drop_results.get(key, {}).items():
            removed = _extract_clause_ids(pert)
            if removed == violated and label == "COMPLIES":
                flipped = True
                break
        result[iid] = flipped

    return result


def per_rule_csr(rows: list[dict], model: str | None = None) -> dict[str, dict]:
    """CSR per rule on single-violation instances: ``{rule_id: {"csr", "n", "flipped"}}``."""
    filtered = rows if model is None else [r for r in rows if r.get("model") == model]
    base_labels = _build_base_labels(filtered)

    # Map (instance_id, model) -> single violated rule
    single_violation: dict[tuple[str, str], str] = {}
    for row in filtered:
        if row.get("perturbation") != "base":
            continue
        vr = str(row.get("violated_rules", ""))
        if vr and "," not in vr and "+" not in vr:
            key = (str(row.get("instance_id")), str(row.get("model", "")))
            if base_labels.get(key) == "VIOLATES":
                single_violation[key] = vr

    # Collect drop_clause results for single-violation instances
    rule_counts: dict[str, dict[str, int]] = {}
    for row in filtered:
        perturbation = str(row.get("perturbation", ""))
        if not perturbation.startswith("drop_clause:"):
            continue
        removed = _extract_clause_id(perturbation)
        if removed is None:
            continue
        key = (str(row.get("instance_id")), str(row.get("model", "")))
        if key not in single_violation:
            continue
        if single_violation[key] != removed:
            continue
        label = _normalize_label(str(row.get("predicted_label", "")))
        if label not in ("VIOLATES", "COMPLIES"):
            continue
        stats = rule_counts.setdefault(removed, {"n": 0, "flipped": 0})
        stats["n"] += 1
        if label == "COMPLIES":
            stats["flipped"] += 1

    return {
        rule: {"csr": s["flipped"] / s["n"] if s["n"] else 0.0, **s}
        for rule, s in sorted(rule_counts.items())
    }


def policy_corrected_csr(rows: list[dict], group_key: str = "model") -> dict[str, dict]:
    """PC-CSR: CSR restricted to policy-dependent instances (base=VIOLATES ∧ remove_policy=COMPLIES).
    Strips out the prior-driven confound: measures clause attribution only where the policy
    causally drives the verdict. Returns ``{group: {"pc_csr", "n_eligible", "n_flipped"}}``."""
    base_labels = _build_base_labels(rows)

    # Step 1: Find instances where base=VIOLATES AND remove_policy=COMPLIES
    policy_dependent: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("perturbation") != "remove_policy":
            continue
        key = (str(row.get("instance_id")), str(row.get("model", "")))
        if base_labels.get(key) != "VIOLATES":
            continue
        no_pol = _normalize_label(str(row.get("predicted_label", "")))
        if no_pol == "COMPLIES":
            policy_dependent.add(key)

    # Step 2: For these instances, compute CSR (drop violated clause -> COMPLIES?)
    violated_map: dict[tuple[str, str], set[str]] = {}
    group_map: dict[tuple[str, str], str] = {}
    for row in rows:
        if row.get("perturbation") != "base":
            continue
        key = (str(row.get("instance_id")), str(row.get("model", "")))
        if key not in policy_dependent:
            continue
        vr = str(row.get("violated_rules", ""))
        if vr:
            violated_map[key] = set(vr.split(","))
            group_map[key] = str(row.get(group_key, ""))

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        perturbation = str(row.get("perturbation", ""))
        if not _is_clause_perturbation(perturbation):
            continue
        removed = _extract_clause_ids(perturbation)
        if removed is None:
            continue
        key = (str(row.get("instance_id")), str(row.get("model", "")))
        if key not in violated_map:
            continue
        if removed != violated_map[key]:
            continue
        label = _normalize_label(str(row.get("predicted_label", "")))
        if label not in ("VIOLATES", "COMPLIES"):
            continue
        group = group_map.get(key, "")
        stats = counts.setdefault(group, {"eligible": 0, "flipped": 0})
        stats["eligible"] += 1
        if label == "COMPLIES":
            stats["flipped"] += 1

    return {
        g: {
            "pc_csr": s["flipped"] / s["eligible"] if s["eligible"] else 0.0,
            "n_eligible": s["eligible"],
            "n_flipped": s["flipped"],
        }
        for g, s in counts.items()
    }


def _extract_clause_id(perturbation: str) -> str | None:
    """``'drop_clause:R3'`` → ``'R3'``; ``'drop_clause:R20+R24'`` → ``'R20+R24'``."""
    if ":" not in perturbation:
        return None
    return perturbation.split(":", 1)[1]


def _extract_clause_ids(perturbation: str) -> set[str] | None:
    """``'drop_clause:R20+R24'`` → ``{'R20', 'R24'}``."""
    raw = _extract_clause_id(perturbation)
    if raw is None:
        return None
    return set(raw.split("+"))


def expected_direction_match_rate(rows: list[dict], group_key: str = "model") -> dict[str, float]:
    """Among clause-ablation flips, share that go V→C (expected) rather than C→V (logically wrong).
    Removing a restrictive clause can only make a policy more permissive."""
    base_labels = _build_base_labels(rows)
    counts: dict[str, dict[str, int]] = {}

    for row in rows:
        perturbation = str(row.get("perturbation", ""))
        if not _is_clause_perturbation(perturbation):
            continue
        key = (str(row.get("instance_id")), str(row.get("model", "")))
        baseline = base_labels.get(key)
        if baseline is None:
            continue
        perturbed = _normalize_label(str(row.get("predicted_label", "")))
        if perturbed == baseline:
            continue  # no flip, skip
        group = str(row.get(group_key, ""))
        stats = counts.setdefault(group, {"expected": 0, "unexpected": 0})
        if baseline == "VIOLATES" and perturbed == "COMPLIES":
            stats["expected"] += 1
        elif baseline == "COMPLIES" and perturbed == "VIOLATES":
            stats["unexpected"] += 1

    return {
        g: (s["expected"] / (s["expected"] + s["unexpected"]))
        if (s["expected"] + s["unexpected"]) else 0.0
        for g, s in counts.items()
    }


def counterfactual_consistency(rows: list[dict], group_key: str = "model") -> dict[str, float]:
    """Per-clause agreement between ``drop_clause:Rn`` and ``mask_clause:Rn`` on the same instance."""
    # collect labels keyed by (instance_id, model, clause_id, op_type)
    clause_labels: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        perturbation = str(row.get("perturbation", ""))
        if not _is_clause_perturbation(perturbation):
            continue
        clause_id = _extract_clause_id(perturbation)
        if clause_id is None:
            continue
        op = "drop" if perturbation.startswith("drop_clause:") else "mask"
        triple = (str(row.get("instance_id")), str(row.get("model", "")), clause_id)
        label = _normalize_label(str(row.get("predicted_label", "")))
        clause_labels.setdefault(triple, {})[op] = label

    # group by group_key; need to look up group from rows
    instance_group: dict[tuple[str, str], str] = {}
    for row in rows:
        ig_key = (str(row.get("instance_id")), str(row.get("model", "")))
        if ig_key not in instance_group:
            instance_group[ig_key] = str(row.get(group_key, ""))

    counts: dict[str, dict[str, int]] = {}
    for (iid, model, _cid), ops in clause_labels.items():
        if "drop" not in ops or "mask" not in ops:
            continue
        group = instance_group.get((iid, model), "")
        stats = counts.setdefault(group, {"consistent": 0, "total": 0})
        stats["total"] += 1
        if ops["drop"] == ops["mask"]:
            stats["consistent"] += 1

    return {
        g: (s["consistent"] / s["total"]) if s["total"] else 0.0
        for g, s in counts.items()
    }


# stochastic reliability (requires --num-repeats > 1)

def outcome_consistency(rows: list[dict], group_key: str = "model") -> dict[str, float]:
    """Agreement across repeats of the same (instance, model, perturbation). Needs ``--num-repeats > 1``."""
    # group by (instance_id, model, perturbation)
    repeat_groups: dict[tuple[str, str, str], list[str]] = {}
    group_lookup: dict[tuple[str, str, str], str] = {}
    for row in rows:
        triple = (
            str(row.get("instance_id")),
            str(row.get("model", "")),
            str(row.get("perturbation", "")),
        )
        label = _normalize_label(str(row.get("predicted_label", "")))
        repeat_groups.setdefault(triple, []).append(label)
        if triple not in group_lookup:
            group_lookup[triple] = str(row.get(group_key, ""))

    counts: dict[str, dict[str, int]] = {}
    for triple, labels in repeat_groups.items():
        if len(labels) < 2:
            continue  # single run, skip
        group = group_lookup.get(triple, "")
        stats = counts.setdefault(group, {"consistent": 0, "total": 0})
        stats["total"] += 1
        if len(set(labels)) == 1:
            stats["consistent"] += 1

    return {
        g: (s["consistent"] / s["total"]) if s["total"] else 0.0
        for g, s in counts.items()
    }


# statistical tests

def with_vs_no_policy_accuracy(
    rows: list[dict], group_key: str = "model",
) -> dict[str, dict[str, float | int]]:
    """Paired with-policy vs no-policy accuracy on the same instance set (paper Table S5).
    ``gap = no_policy - with_policy``; positive means the judge is *more* accurate without the policy."""
    per_group: dict[str, dict[tuple[str, str], dict[str, str]]] = {}

    for row in rows:
        perturbation = str(row.get("perturbation", ""))
        if perturbation not in ("base", "remove_policy"):
            continue
        iid = str(row.get("instance_id", ""))
        model = str(row.get("model", ""))
        if not iid or not model:
            continue
        group = str(row.get(group_key, ""))
        gt = _normalize_label(str(row.get("ground_truth", "")))
        pred = _normalize_label(str(row.get("predicted_label", "")))
        if gt not in ("VIOLATES", "COMPLIES") or pred not in ("VIOLATES", "COMPLIES"):
            continue
        bucket = per_group.setdefault(group, {})
        entry = bucket.setdefault((iid, model), {"ground_truth": gt})
        entry[perturbation] = pred

    out: dict[str, dict[str, float | int]] = {}
    for group, instances in per_group.items():
        with_correct = 0
        no_correct = 0
        n = 0
        for (_, _), entry in instances.items():
            if "base" not in entry or "remove_policy" not in entry:
                continue
            gt = entry["ground_truth"]
            if entry["base"] == gt:
                with_correct += 1
            if entry["remove_policy"] == gt:
                no_correct += 1
            n += 1
        if n == 0:
            continue
        with_acc = with_correct / n
        no_acc = no_correct / n
        out[group] = {
            "with_policy_acc": round(with_acc, 4),
            "no_policy_acc": round(no_acc, 4),
            "gap": round(no_acc - with_acc, 4),
            "n": n,
        }
    return out


def pool_metrics_across_domains(
    per_domain_counts: dict[str, dict[str, int]],
) -> dict[str, float | int]:
    """Pool raw counts across domains: ``rate = Σ num / Σ den`` (not a mean of pre-computed rates)."""
    total_num = sum(int(v.get("numerator", 0)) for v in per_domain_counts.values())
    total_den = sum(int(v.get("denominator", 0)) for v in per_domain_counts.values())
    rate = total_num / total_den if total_den else 0.0
    return {"numerator": total_num, "denominator": total_den, "rate": round(rate, 4)}


def mcnemar_test(rows: list[dict], model_a: str, model_b: str) -> dict[str, float | int]:
    """McNemar's test on paired base-only verdicts. ``b`` = A right/B wrong, ``c`` = A wrong/B right;
    under H0, ``(b-c)²/(b+c) ~ χ²(1)``."""
    import math

    base_rows = [r for r in rows if r.get("perturbation") == "base"]

    preds_a: dict[str, bool] = {}
    preds_b: dict[str, bool] = {}
    for r in base_rows:
        iid = str(r.get("instance_id"))
        gt = _normalize_label(str(r.get("ground_truth", "")))
        pred = _normalize_label(str(r.get("predicted_label", "")))
        model = str(r.get("model", ""))
        correct = pred == gt
        if model == model_a:
            preds_a[iid] = correct
        elif model == model_b:
            preds_b[iid] = correct

    shared = set(preds_a.keys()) & set(preds_b.keys())
    if not shared:
        return {"n": 0, "b": 0, "c": 0, "chi2": 0.0, "p_value": 1.0}

    b = sum(1 for iid in shared if preds_a[iid] and not preds_b[iid])
    c = sum(1 for iid in shared if not preds_a[iid] and preds_b[iid])

    if b + c == 0:
        return {"n": len(shared), "b": b, "c": c, "chi2": 0.0, "p_value": 1.0}

    chi2 = (b - c) ** 2 / (b + c)
    z = math.sqrt(chi2)
    p_value = math.erfc(z / math.sqrt(2))

    return {"n": len(shared), "b": b, "c": c, "chi2": round(chi2, 4), "p_value": round(p_value, 6)}
