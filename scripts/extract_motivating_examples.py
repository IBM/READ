#!/usr/bin/env python3
# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Extract motivating examples for the READ paper.

Finds compelling counterfactual pairs showing failure modes:
1. FABRICATION: Clause removed → model still says VIOLATES with invented reason
2. POLICY-BLIND: Model ignores policy entirely (e.g. llama-guard on business rules)
3. PRIOR-DRIVEN: Model cites safety-adjacent rules instead of the actual violated rule
4. SUPPRESSION: Model flips COMPLIES→VIOLATES when irrelevant clause removed

Output: JSON + Markdown for paper figures.

Usage:
    python scripts/extract_motivating_examples.py results/exp10_*.csv
    python scripts/extract_motivating_examples.py results/exp10_qwen2-5-72b-instruct.csv --top 5
    python scripts/extract_motivating_examples.py results/exp10_*.csv --type fabrication
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

LABEL_MAP = {
    "YES": "VIOLATES", "NO": "COMPLIES",
    "UNSAFE": "VIOLATES", "SAFE": "COMPLIES",
    "COMPLIES": "COMPLIES", "VIOLATES": "VIOLATES",
}

def norm(label):
    return LABEL_MAP.get((label or "").strip().upper(), (label or "").strip().upper())


def load_clause_map(path="data/clause_maps/polyguard_hr_IBM.json"):
    try:
        with open(path) as f:
            data = json.load(f)
        return {c["rule_id"]: c["label"] for c in data["clauses"]}
    except FileNotFoundError:
        return {}


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _truncate(text, n=600):
    text = (text or "").replace("\\n", "\n").strip()
    if len(text) > n:
        return text[:n] + "..."
    return text


def _get_reasoning(row):
    """Get the best reasoning text from a row.

    Some models (gpt-oss) put terse YES/NO in raw_response but full CoT
    in the trace column.  Always prefer the longer text.
    """
    raw = (row.get("raw_response", "") or "").strip()
    trace = (row.get("trace", "") or "").strip()
    if len(trace) > len(raw):
        return trace
    return raw


# Minimum reasoning length to be considered paper-worthy.
# Filters out terse labels like "safe", "unsafe S5".
MIN_REASONING_LEN = 30


def _diversify(examples, top, max_per_model=None):
    """Re-rank examples to ensure model diversity.

    If max_per_model is None, allow at most ceil(top / num_models) per model,
    with a floor of 2.
    """
    if not examples:
        return examples[:top]

    models = {ex["model"] for ex in examples}
    if max_per_model is None:
        max_per_model = max(2, -(-top // len(models)))  # ceil division

    model_counts = defaultdict(int)
    result = []
    # First pass: pick up to max_per_model from each model, in score order
    for ex in examples:
        m = ex["model"]
        if model_counts[m] < max_per_model:
            result.append(ex)
            model_counts[m] += 1
        if len(result) >= top:
            break

    # Second pass: if we still have room, fill from remaining
    if len(result) < top:
        used = set(id(ex) for ex in result)
        for ex in examples:
            if id(ex) not in used:
                result.append(ex)
                if len(result) >= top:
                    break

    return result


# Example type 1: FABRICATION (CSR=0 cases with reasoning)

def find_fabrications(rows, clause_labels, top=10):
    """Find cases where violated clause is removed but model still says VIOLATES."""
    base_map = {}
    for r in rows:
        if r["perturbation"] == "base":
            base_map[(r["instance_id"], r.get("model", ""))] = r

    examples = []
    for r in rows:
        if not r["perturbation"].startswith("drop_clause:"):
            continue
        iid = r["instance_id"]
        model = r.get("model", "")
        key = (iid, model)
        if key not in base_map:
            continue
        base = base_map[key]
        if norm(base["predicted_label"]) != "VIOLATES":
            continue

        vr = r.get("violated_rules", "")
        if not vr:
            continue
        removed = r["perturbation"].split(":", 1)[1]
        removed_set = set(removed.split("+"))
        violated_set = set(vr.split(","))
        if removed_set != violated_set:
            continue

        # CSR=0: model still says VIOLATES after removing the violated clause
        if norm(r["predicted_label"]) != "VIOLATES":
            continue

        base_text = _get_reasoning(base)
        drop_text = _get_reasoning(r)
        if not base_text or not drop_text:
            continue

        # Filter terse responses (not paper-worthy)
        if len(drop_text) < MIN_REASONING_LEN:
            continue

        # Score: prefer cases where reasoning CHANGES (different justification)
        # Also bonus for longer reasoning (more paper-worthy)
        base_words = set(base_text.lower().split())
        drop_words = set(drop_text.lower().split())
        overlap = len(base_words & drop_words) / max(len(base_words | drop_words), 1)
        diversity_score = 1.0 - overlap  # higher = more different reasoning

        # Bonus for longer reasoning (normalized to 0-0.1 range)
        length_bonus = min(len(drop_text) / 5000, 0.1)

        rule_label = clause_labels.get(vr, vr)
        examples.append({
            "type": "FABRICATION",
            "instance_id": iid,
            "model": r.get("model", "unknown"),
            "violated_rule": vr,
            "rule_label": rule_label,
            "ground_truth": r["ground_truth"],
            "base_prediction": norm(base["predicted_label"]),
            "drop_prediction": norm(r["predicted_label"]),
            "base_reasoning": _truncate(base_text),
            "drop_reasoning": _truncate(drop_text),
            "reasoning_diversity": round(diversity_score, 3),
            "score": round(diversity_score + length_bonus, 3),
        })

    examples.sort(key=lambda x: x["score"], reverse=True)
    return _diversify(examples, top)


# Example type 2: POLICY-BLIND (model says COMPLIES on clear VIOLATES)

def find_policy_blind(rows, clause_labels, top=10):
    """Find base cases where GT=VIOLATES but model says COMPLIES."""
    examples = []
    for r in rows:
        if r["perturbation"] != "base":
            continue
        if norm(r["ground_truth"]) != "VIOLATES":
            continue
        if norm(r["predicted_label"]) != "COMPLIES":
            continue

        vr = r.get("violated_rules", "")
        rule_label = clause_labels.get(vr, vr)
        resp = _get_reasoning(r)

        # Score: prefer longer reasoning (more paper-worthy explanation of WHY it missed)
        length_score = min(len(resp) / 500, 1.0) if resp else 0.0

        examples.append({
            "type": "POLICY_BLIND",
            "instance_id": r["instance_id"],
            "model": r.get("model", "unknown"),
            "violated_rule": vr,
            "rule_label": rule_label,
            "ground_truth": "VIOLATES",
            "prediction": "COMPLIES",
            "reasoning": _truncate(resp, 400),
            "score": round(length_score, 3),
        })

    # Prefer diverse rules, then sort by score within each rule
    examples.sort(key=lambda x: x["score"], reverse=True)
    seen_rules = set()
    deduped = []
    for ex in examples:
        if ex["violated_rule"] not in seen_rules:
            seen_rules.add(ex["violated_rule"])
            deduped.append(ex)
    return _diversify(deduped, top)


# Example type 3: PRIOR-DRIVEN (cites wrong rule in reasoning)

def find_prior_driven(rows, clause_labels, top=10):
    """Find cases where model cites a DIFFERENT rule than the actual violated one."""
    base_map = {}
    for r in rows:
        if r["perturbation"] == "base":
            base_map[(r["instance_id"], r.get("model", ""))] = r

    examples = []
    for (iid, model), r in base_map.items():
        if norm(r["predicted_label"]) != "VIOLATES":
            continue
        if norm(r["ground_truth"]) != "VIOLATES":
            continue

        vr = r.get("violated_rules", "")
        if not vr:
            continue
        resp = _get_reasoning(r).lower()
        if not resp or len(resp) < MIN_REASONING_LEN:
            continue

        # Check if reasoning mentions the actual violated rule's content
        rule_label = clause_labels.get(vr, "").lower()
        actual_mentioned = rule_label and any(
            word in resp for word in rule_label.split() if len(word) > 4
        )

        # Check if reasoning mentions safety-adjacent rules (R1-R4: harassment/discrimination)
        safety_keywords = ["harassment", "discriminat", "bullying", "hostile", "sexual advance",
                          "offensive", "intimidat", "violence", "threaten"]
        safety_mentioned = any(kw in resp for kw in safety_keywords)

        # Best examples: actual rule NOT mentioned, but safety rules ARE mentioned
        if not actual_mentioned and safety_mentioned and vr not in ("R1", "R2", "R3", "R4", "R5", "R6"):
            # Score: bonus for longer reasoning + business rules
            length_score = min(len(resp) / 500, 1.0)
            business_bonus = 0.1 if vr in ("R7","R8","R9","R10","R11","R12","R13","R14",
                                            "R15","R16","R17","R18","R19","R20","R21",
                                            "R22","R23","R24","R25","R26") else 0.0
            examples.append({
                "type": "PRIOR_DRIVEN",
                "instance_id": iid,
                "model": r.get("model", "unknown"),
                "violated_rule": vr,
                "rule_label": clause_labels.get(vr, vr),
                "ground_truth": "VIOLATES",
                "prediction": "VIOLATES",
                "correct_but_wrong_reason": True,
                "reasoning": _truncate(_get_reasoning(r), 500),
                "score": round(length_score + business_bonus, 3),
            })

    examples.sort(key=lambda x: x["score"], reverse=True)
    seen_rules = set()
    deduped = []
    for ex in examples:
        if ex["violated_rule"] not in seen_rules:
            seen_rules.add(ex["violated_rule"])
            deduped.append(ex)
    return _diversify(deduped, top)


# Example type 4: SUPPRESSION (COMPLIES→VIOLATES on irrelevant clause removal)

def find_suppression(rows, clause_labels, top=10):
    """Find cases where removing an IRRELEVANT clause flips COMPLIES→VIOLATES."""
    base_map = {}
    for r in rows:
        if r["perturbation"] == "base":
            base_map[(r["instance_id"], r.get("model", ""))] = r

    examples = []
    for r in rows:
        if not r["perturbation"].startswith("drop_clause:"):
            continue
        iid = r["instance_id"]
        model = r.get("model", "")
        key = (iid, model)
        if key not in base_map:
            continue
        base = base_map[key]
        if norm(base["predicted_label"]) != "COMPLIES":
            continue
        if norm(r["predicted_label"]) != "VIOLATES":
            continue

        # This is an irrelevant clause removal that ADDED a violation
        vr = r.get("violated_rules", "")
        removed = r["perturbation"].split(":", 1)[1]

        resp = _get_reasoning(r)
        base_resp = _get_reasoning(base)

        # Score: prefer longer reasoning + cases where reasoning differs
        length_score = min(len(resp) / 500, 1.0) if resp else 0.0
        if base_resp and resp:
            base_words = set(base_resp.lower().split())
            drop_words = set(resp.lower().split())
            overlap = len(base_words & drop_words) / max(len(base_words | drop_words), 1)
            change_score = 1.0 - overlap
        else:
            change_score = 0.0

        examples.append({
            "type": "SUPPRESSION",
            "instance_id": iid,
            "model": model or "unknown",
            "removed_clause": removed,
            "removed_label": clause_labels.get(removed, removed),
            "ground_truth": r["ground_truth"],
            "base_prediction": "COMPLIES",
            "drop_prediction": "VIOLATES",
            "base_reasoning": _truncate(base_resp, 400),
            "reasoning": _truncate(resp, 400),
            "score": round(0.5 * length_score + 0.5 * change_score, 3),
        })

    examples.sort(key=lambda x: x["score"], reverse=True)
    # Deduplicate by (instance, model) to avoid same case with different clauses flooding
    seen = set()
    deduped = []
    for ex in examples:
        key = (ex["instance_id"], ex["model"])
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    return _diversify(deduped, top)


# Output formatting

def to_markdown(examples, clause_labels):
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"### Example {i}: {ex['type']} ({ex['model']})")
        lines.append(f"**Instance:** `{ex['instance_id']}`")

        if ex["type"] == "FABRICATION":
            lines.append(f"**Violated Rule:** {ex['violated_rule']} ({ex['rule_label']})")
            lines.append(f"**GT:** {ex['ground_truth']}  |  **Reasoning diversity:** {ex.get('reasoning_diversity', 'N/A')}")
            lines.append("")
            lines.append("| Policy intact (base) | Clause removed |")
            lines.append("|---------------------|----------------|")
            base_lines = ex["base_reasoning"].replace("\n", " ")[:300]
            drop_lines = ex["drop_reasoning"].replace("\n", " ")[:300]
            lines.append(f"| VIOLATES: {base_lines} | VIOLATES: {drop_lines} |")

        elif ex["type"] == "POLICY_BLIND":
            lines.append(f"**Violated Rule:** {ex['violated_rule']} ({ex['rule_label']})")
            lines.append(f"**GT:** VIOLATES → **Prediction:** COMPLIES")
            if ex.get("reasoning"):
                lines.append(f"\n> {ex['reasoning'][:300]}")

        elif ex["type"] == "PRIOR_DRIVEN":
            lines.append(f"**Actual Violated Rule:** {ex['violated_rule']} ({ex['rule_label']})")
            lines.append(f"**Model says VIOLATES**: correct verdict, but cites safety rules instead of the actual policy rule.")
            if ex.get("reasoning"):
                lines.append(f"\n> {ex['reasoning'][:400]}")

        elif ex["type"] == "SUPPRESSION":
            lines.append(f"**Removed (irrelevant) clause:** {ex['removed_clause']} ({ex['removed_label']})")
            lines.append(f"**GT:** {ex['ground_truth']}  |  **Base:** COMPLIES → **After removal:** VIOLATES")
            if ex.get("reasoning"):
                lines.append(f"\n> {ex['reasoning'][:300]}")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# Main

def main():
    parser = argparse.ArgumentParser(description="Extract motivating examples for READ paper")
    parser.add_argument("csv_files", nargs="+", type=Path, help="EXP-10 CSV files")
    parser.add_argument("--top", type=int, default=5, help="Top N examples per type")
    parser.add_argument("--type", choices=["fabrication", "policy_blind", "prior_driven", "suppression", "all"],
                       default="all", help="Which example type to extract")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file")
    parser.add_argument("--clause-map", type=Path, default=Path("data/clause_maps/polyguard_hr_IBM.json"))
    args = parser.parse_args()

    clause_labels = load_clause_map(args.clause_map)

    all_rows = []
    for csv_file in args.csv_files:
        if csv_file.exists():
            all_rows.extend(load_csv(csv_file))
    print(f"Loaded {len(all_rows)} rows from {len(args.csv_files)} files", file=sys.stderr)

    results = {}
    if args.type in ("fabrication", "all"):
        results["fabrication"] = find_fabrications(all_rows, clause_labels, args.top)
        print(f"  FABRICATION: {len(results['fabrication'])} examples", file=sys.stderr)

    if args.type in ("policy_blind", "all"):
        results["policy_blind"] = find_policy_blind(all_rows, clause_labels, args.top)
        print(f"  POLICY_BLIND: {len(results['policy_blind'])} examples", file=sys.stderr)

    if args.type in ("prior_driven", "all"):
        results["prior_driven"] = find_prior_driven(all_rows, clause_labels, args.top)
        print(f"  PRIOR_DRIVEN: {len(results['prior_driven'])} examples", file=sys.stderr)

    if args.type in ("suppression", "all"):
        results["suppression"] = find_suppression(all_rows, clause_labels, args.top)
        print(f"  SUPPRESSION: {len(results['suppression'])} examples", file=sys.stderr)

    # JSON output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"JSON written to {args.output}", file=sys.stderr)

    # Always print markdown to stdout
    all_examples = []
    for examples in results.values():
        all_examples.extend(examples)
    print(to_markdown(all_examples, clause_labels))


if __name__ == "__main__":
    main()
