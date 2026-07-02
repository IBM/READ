# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Tests for the clause-map derivation in polyguard_ingest.build_clause_map."""
import hashlib

from read_toolkit.polyguard_ingest import build_clause_map

POLICY = """Risk Category 1: Interaction Rules
1. Do not use discriminatory language.
2. Do not threaten violence.

Risk Category 2: Data Rules
3. Do not share personal data without consent.
"""


def test_numbered_lines_become_clauses():
    cm = build_clause_map("demo", POLICY.strip())
    assert cm is not None
    assert cm["domain"] == "demo"
    assert cm["num_clauses"] == 3
    assert [c["rule_id"] for c in cm["clauses"]] == ["R1", "R2", "R3"]
    assert cm["clauses"][0]["text"] == "1. Do not use discriminatory language."


def test_category_headers_are_not_clauses():
    cm = build_clause_map("demo", POLICY.strip())
    assert all("Risk Category" not in c["text"] for c in cm["clauses"])


def test_policy_hash_matches_perturbations_lookup():
    policy = POLICY.strip()
    cm = build_clause_map("demo", policy)
    expected = hashlib.sha256(policy.encode()).hexdigest()[:16]
    assert cm["policy_hash"] == expected


def test_clause_text_is_verbatim_policy_substring():
    # drop_clause/mask_clause remove clauses via str.replace, so every
    # clause text must appear verbatim in the policy.
    policy = POLICY.strip()
    cm = build_clause_map("demo", policy)
    for clause in cm["clauses"]:
        assert clause["text"] in policy


def test_unnumbered_policy_returns_none():
    assert build_clause_map("demo", "Be nice. Do not spam.") is None
