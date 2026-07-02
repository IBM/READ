# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Callable

# registry mapping perturbation name to function for easy iteration
PERTURBATIONS: dict[str, Callable[[str], str]] = {}


def _clause_map_dirs() -> list[Path]:
    """Candidate directories holding clause maps (first hit per policy wins).

    Checked in order: the READ_CLAUSE_MAP_DIR environment variable, then
    ``data/clause_maps`` under the current working directory (pip-installed
    usage), then ``data/clause_maps`` relative to the repository root
    (running from a clone). scripts/fetch_data.py derives the maps.
    """
    dirs = []
    env_dir = os.environ.get("READ_CLAUSE_MAP_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    dirs.append(Path.cwd() / "data" / "clause_maps")
    dirs.append(Path(__file__).resolve().parent.parent / "data" / "clause_maps")
    return dirs

# in-memory cache: policy_hash -> clause list
_clause_cache: dict[str, list[dict[str, str | int]]] = {}

# policy hashes already warned about (sentence-split fallback)
_fallback_warned: set[str] = set()


def register(name: str):
    def decorator(fn: Callable[[str], str]):
        PERTURBATIONS[name] = fn
        return fn
    return decorator


def _split_sentences(policy: str) -> list[str]:
    """split policy text into sentences, stripping empty results"""
    parts = re.split(r"(?<=[.!?])\s+", policy.strip())
    return [p.strip() for p in parts if p.strip()]


def _policy_hash(policy: str) -> str:
    return hashlib.sha256(policy.encode()).hexdigest()[:16]


def _load_clause_maps() -> None:
    """Load all clause map JSON files into cache (keyed by policy_hash)."""
    if _clause_cache:
        return
    for map_dir in _clause_map_dirs():
        if not map_dir.is_dir():
            continue
        for path in sorted(map_dir.glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            ph = doc.get("policy_hash", "")
            if ph and "clauses" in doc:
                _clause_cache.setdefault(ph, doc["clauses"])


def parse_policy_clauses(policy: str) -> list[dict[str, str | int]]:
    """Return testable clauses for a policy via clause-map lookup or sentence split."""
    _load_clause_maps()
    ph = _policy_hash(policy)
    if ph in _clause_cache:
        return _clause_cache[ph]

    # fallback: sentence-level splitting
    if ph not in _fallback_warned:
        _fallback_warned.add(ph)
        warnings.warn(
            f"no curated clause map found for policy hash {ph}; falling back to "
            "sentence-level clause splitting. Rule ids become sentence indices, so "
            "clause-level fidelity metrics (CSR/ASR) may not match curated rule ids.",
            stacklevel=2,
        )
    clauses: list[dict[str, str | int]] = []
    for idx, sentence in enumerate(_split_sentences(policy), start=1):
        clauses.append({"id": idx, "rule_id": f"R{idx}", "text": sentence})
    return clauses


def _validate_clause_id(clauses: list[dict[str, str | int]], clause_id: int) -> None:
    if clause_id < 1 or clause_id > len(clauses):
        raise ValueError(f"clause_id {clause_id} out of range 1..{len(clauses)}")


def drop_clause(policy: str, clause_id: int) -> str:
    """Return policy with one clause removed."""
    clauses = parse_policy_clauses(policy)
    if not clauses:
        return policy
    _validate_clause_id(clauses, clause_id)
    target = str(clauses[clause_id - 1]["text"])
    return policy.replace(target, "", 1).strip()


def mask_clause(policy: str, clause_id: int) -> str:
    """Return policy with one clause replaced by a suspension message."""
    clauses = parse_policy_clauses(policy)
    if not clauses:
        return policy
    _validate_clause_id(clauses, clause_id)
    clause = clauses[clause_id - 1]
    target = str(clause["text"])
    replacement = f"Rule {clause['rule_id']} is currently suspended."
    return policy.replace(target, replacement, 1).strip()


@register("bullet_points")
def to_bullet_points(policy: str) -> str:
    """convert prose policy into a markdown bullet list, one sentence per item"""
    sentences = _split_sentences(policy)
    if len(sentences) <= 1:
        return f"- {policy.strip()}"
    return "\n".join(f"- {s}" for s in sentences)


@register("json_format")
def to_json_format(policy: str) -> str:
    """wrap each policy sentence as an entry in a json rules array"""
    sentences = _split_sentences(policy)
    payload = {"rules": sentences}
    return json.dumps(payload, indent=2)


@register("typos")
def add_typos(policy: str) -> str:
    """introduce three character-swap typos at evenly spaced positions in the policy"""
    chars = list(policy)
    length = len(chars)
    if length < 6:
        return policy

    # pick three positions spread across the string, avoiding spaces
    positions = [length // 4, length // 2, (3 * length) // 4]
    for pos in positions:
        # move to the nearest non-space character
        while pos < length - 1 and chars[pos] == " ":
            pos += 1
        if pos + 1 < length and chars[pos + 1] != " ":
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

    return "".join(chars)


@register("reorder_sentences")
def reorder_sentences(policy: str) -> str:
    """swap the first and second sentences; return unchanged if fewer than two sentences"""
    sentences = _split_sentences(policy)
    if len(sentences) < 2:
        return policy
    sentences[0], sentences[1] = sentences[1], sentences[0]
    return " ".join(sentences)


@register("long_irrelevant_suffix")
def long_irrelevant_suffix(policy: str) -> str:
    """append benign administrative filler text to stress long-context policy ingestion"""
    filler = (
        "Additional Administrative Context: This policy framework was first archived by the internal "
        "governance committee in 1998 and has been revised across multiple record-management cycles. "
        "For audit continuity, all historical drafts, review notes, and procedural memos should be "
        "retained in the designated compliance repository. Teams should reference document control "
        "codes, preserve version history, and submit quarterly attestations of archival completeness. "
        "This context is operational guidance only and does not change decision logic, eligibility "
        "criteria, thresholds, exceptions, enforcement priorities, or case outcomes. Please confirm "
        "that records are indexed, time-stamped, and accessible for administrative review. Thank you "
        "for your adherence to documentation standards."
    )
    return f"{policy.rstrip()}\n\n{filler}"


def drop_clauses(policy: str, clause_ids: list[int]) -> str:
    """Return policy with multiple clauses removed (multi-violation CSR)."""
    clauses = parse_policy_clauses(policy)
    if not clauses:
        return policy
    for cid in clause_ids:
        _validate_clause_id(clauses, cid)
    result = policy
    for cid in clause_ids:
        target = str(clauses[cid - 1]["text"])
        result = result.replace(target, "", 1)
    return result.strip()


def mask_clauses(policy: str, clause_ids: list[int]) -> str:
    """Return policy with multiple clauses replaced by suspension text."""
    clauses = parse_policy_clauses(policy)
    if not clauses:
        return policy
    for cid in clause_ids:
        _validate_clause_id(clauses, cid)
    result = policy
    for cid in clause_ids:
        clause = clauses[cid - 1]
        target = str(clause["text"])
        replacement = f"Rule {clause['rule_id']} is currently suspended."
        result = result.replace(target, replacement, 1)
    return result.strip()


def remove_policy(policy: str) -> str:
    """Return an empty policy (PIR baseline)."""
    return ""


# Position-bias ablations

@register("reverse_clauses")
def reverse_clauses(policy: str) -> str:
    """reverse the order of all clauses in the policy text"""
    clauses = parse_policy_clauses(policy)
    if len(clauses) < 2:
        return policy
    reversed_texts = [str(c["text"]) for c in reversed(clauses)]
    return " ".join(reversed_texts)


@register("random_shuffle_clauses")
def random_shuffle_clauses(policy: str) -> str:
    """deterministic random shuffle of clause order (seed=73 for reproducibility)"""
    import random as _rng
    clauses = parse_policy_clauses(policy)
    if len(clauses) < 2:
        return policy
    texts = [str(c["text"]) for c in clauses]
    r = _rng.Random(73)
    r.shuffle(texts)
    return " ".join(texts)


# Recency-bias ablations

_SHORT_FILLER = (
    "For record-keeping purposes, this policy was last reviewed in Q3 2024."
)

_EXTRA_LONG_FILLER = (
    "Additional Administrative Context: This policy framework was first archived by the internal "
    "governance committee in 1998 and has been revised across multiple record-management cycles. "
    "For audit continuity, all historical drafts, review notes, and procedural memos should be "
    "retained in the designated compliance repository. Teams should reference document control "
    "codes, preserve version history, and submit quarterly attestations of archival completeness. "
    "This context is operational guidance only and does not change decision logic, eligibility "
    "criteria, thresholds, exceptions, enforcement priorities, or case outcomes. Please confirm "
    "that records are indexed, time-stamped, and accessible for administrative review. Thank you "
    "for your adherence to documentation standards. All records should be cross-referenced with "
    "the central management database. Periodic assessments of document integrity should be "
    "scheduled in accordance with the annual governance calendar. Stakeholders are reminded to "
    "verify that their departmental repositories are synchronised with the master archive."
)


@register("short_irrelevant_suffix")
def short_irrelevant_suffix(policy: str) -> str:
    """append a short (~30 token) filler sentence"""
    return f"{policy.rstrip()}\n\n{_SHORT_FILLER}"


@register("extra_long_irrelevant_suffix")
def extra_long_irrelevant_suffix(policy: str) -> str:
    """append an extra-long (~300 token) filler paragraph"""
    return f"{policy.rstrip()}\n\n{_EXTRA_LONG_FILLER}"


@register("irrelevant_prefix")
def irrelevant_prefix(policy: str) -> str:
    """prepend the standard filler text BEFORE the policy"""
    filler = (
        "Additional Administrative Context: This policy framework was first archived by the internal "
        "governance committee in 1998 and has been revised across multiple record-management cycles. "
        "For audit continuity, all historical drafts, review notes, and procedural memos should be "
        "retained in the designated compliance repository. Teams should reference document control "
        "codes, preserve version history, and submit quarterly attestations of archival completeness. "
        "This context is operational guidance only and does not change decision logic, eligibility "
        "criteria, thresholds, exceptions, enforcement priorities, or case outcomes.\n\n"
    )
    return f"{filler}{policy.lstrip()}"


@register("irrelevant_middle")
def irrelevant_middle(policy: str) -> str:
    """insert filler text in the middle of the policy"""
    clauses = parse_policy_clauses(policy)
    if len(clauses) < 2:
        return policy
    mid = len(clauses) // 2
    filler_sentence = (
        "Note: This clause is included for administrative completeness and does not "
        "affect eligibility criteria, decision logic, or case outcomes."
    )
    texts = [str(c["text"]) for c in clauses]
    texts.insert(mid, filler_sentence)
    return " ".join(texts)


# Tokenisation ablations

@register("numbered_list")
def to_numbered_list(policy: str) -> str:
    """convert prose policy into a numbered list"""
    sentences = _split_sentences(policy)
    if len(sentences) <= 1:
        return f"1. {policy.strip()}"
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))


@register("xml_format")
def to_xml_format(policy: str) -> str:
    """wrap each policy sentence in XML tags"""
    sentences = _split_sentences(policy)
    rules = "\n".join(f"  <rule>{s}</rule>" for s in sentences)
    return f"<rules>\n{rules}\n</rules>"


def apply_all(policy: str) -> dict[str, str]:
    """apply every registered perturbation and return a dict of name -> perturbed policy"""
    return {name: fn(policy) for name, fn in PERTURBATIONS.items()}
