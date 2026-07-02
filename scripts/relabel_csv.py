#!/usr/bin/env python3
# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Re-label a benchmark CSV using the fixed parser.

Reads raw_response from each row, re-runs _extract_binary_label() and
_binary_to_canonical(), then recomputes is_flip against the base row.

Usage:
    python3 scripts/relabel_csv.py results/exp10_llama-3-1-8b-instruct.csv
    # Writes to results/exp10_llama-3-1-8b-instruct.relabeled.csv
    # Prints diff stats (how many labels changed).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from read_toolkit.llm_client import JudgeClient


def relabel(input_path: str):
    rows = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Re-parse binary_label and predicted_label
    changed_binary = 0
    changed_predicted = 0
    for row in rows:
        raw = row.get("raw_response", "")
        old_binary = row.get("binary_label", "")
        old_predicted = row.get("predicted_label", "")

        new_binary = JudgeClient._extract_binary_label(raw)
        new_predicted = JudgeClient._binary_to_canonical(new_binary, family="risk_yes")

        if new_binary != old_binary:
            changed_binary += 1
        if new_predicted != old_predicted:
            changed_predicted += 1

        row["binary_label"] = new_binary
        row["predicted_label"] = new_predicted

    # Recompute is_flip: compare each perturbed row's predicted_label to its base row
    base_labels = {}
    for row in rows:
        if row.get("perturbation") == "base":
            key = (row["instance_id"], row["model"])
            base_labels[key] = row["predicted_label"]

    changed_flip = 0
    for row in rows:
        if row.get("perturbation") == "base":
            new_flip = "False"
        else:
            key = (row["instance_id"], row["model"])
            base_label = base_labels.get(key, "")
            new_flip = str(row["predicted_label"] != base_label)

        old_flip = row.get("is_flip", "")
        if new_flip != old_flip:
            changed_flip += 1
        row["is_flip"] = new_flip

    # Write output
    out_path = input_path.replace(".csv", ".relabeled.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"Total rows: {total}")
    print(f"binary_label changed: {changed_binary} ({changed_binary/total*100:.1f}%)")
    print(f"predicted_label changed: {changed_predicted} ({changed_predicted/total*100:.1f}%)")
    print(f"is_flip changed: {changed_flip} ({changed_flip/total*100:.1f}%)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/relabel_csv.py <input.csv>")
        sys.exit(1)
    relabel(sys.argv[1])
