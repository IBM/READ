# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Tests for _extract_binary_label() and _llama_guard_to_canonical() in llm_client.py.

Covers:
- Tier 1: Structured XML (<score> tags)
- Tier 2: First-token match
- Tier 3: Word-boundary keywords with negation guards
- Tier 4: Numeric / JSON
- Tier 5: Semantic fallback
- LlamaGuard canonical mapping
- Regression: old substring bugs (negation blindness)
"""

import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from read_toolkit.llm_client import JudgeClient

ebl = JudgeClient._extract_binary_label
lgc = JudgeClient._llama_guard_to_canonical


# ── Tier 1: Structured XML ──────────────────────────────────────────

class TestTier1XML:
    def test_score_yes(self):
        assert ebl("<score>yes</score>") == "YES"

    def test_score_no(self):
        assert ebl("<score>no</score>") == "NO"

    def test_score_whitespace(self):
        assert ebl("<score> yes </score>") == "YES"

    def test_score_case_insensitive(self):
        assert ebl("<Score>YES</Score>") == "YES"

    def test_score_with_surrounding_text(self):
        assert ebl("The answer is <score>no</score> because...") == "NO"

    def test_score_last_wins(self):
        """If multiple <score> tags, last one wins (existing behavior)."""
        assert ebl("<score>yes</score> wait <score>no</score>") == "NO"


# ── Tier 2: First-token match ───────────────────────────────────────

class TestTier2FirstToken:
    def test_yes_bare(self):
        assert ebl("YES") == "YES"

    def test_no_bare(self):
        assert ebl("NO") == "NO"

    def test_yes_with_explanation(self):
        assert ebl("Yes, the case violates the policy because...") == "YES"

    def test_no_with_explanation(self):
        assert ebl("No, the case complies with all requirements.") == "NO"

    def test_violates_first(self):
        assert ebl("Violates — multiple rules are broken.") == "YES"

    def test_complies_first(self):
        assert ebl("Complies. All criteria met.") == "NO"

    def test_unsafe_first(self):
        assert ebl("unsafe\nS1,S3") == "YES"

    def test_safe_first(self):
        assert ebl("safe") == "NO"

    def test_violation_first(self):
        assert ebl("Violation detected in rule 3.") == "YES"

    def test_compliant_first(self):
        assert ebl("Compliant with all rules.") == "NO"

    def test_leading_punctuation_ignored(self):
        assert ebl("**YES** — clear violation.") == "YES"

    def test_leading_newline_ignored(self):
        assert ebl("\nNo") == "NO"


# ── Tier 3: Word-boundary with negation ─────────────────────────────

class TestTier3NegationGuards:
    # -- SAFE / UNSAFE --
    def test_not_safe_is_violation(self):
        """THE critical regression test: 'not safe' must → YES (violates)."""
        assert ebl("This behavior is not safe for the organization.") == "YES"

    def test_isnt_safe(self):
        assert ebl("The action isn't safe under current policy.") == "YES"

    def test_never_safe(self):
        assert ebl("This is never safe to do.") == "YES"

    def test_cannot_be_safe(self):
        assert ebl("This cannot be considered safe.") == "YES"

    def test_safe_without_negation(self):
        assert ebl("The action is considered safe.") == "NO"

    def test_unsafe_always_yes(self):
        assert ebl("This is clearly unsafe behavior.") == "YES"

    # -- VIOLATE / COMPLY with negation --
    def test_does_not_violate(self):
        assert ebl("The case does not violate any policy rules.") == "NO"

    def test_doesnt_violate(self):
        assert ebl("It doesn't violate the requirements.") == "NO"

    def test_violates_without_negation(self):
        assert ebl("The case clearly violates rule 5.") == "YES"

    def test_violation_without_negation(self):
        assert ebl("A clear violation of company policy.") == "YES"

    def test_does_not_comply(self):
        assert ebl("The case does not comply with the policy.") == "YES"

    def test_doesnt_comply(self):
        assert ebl("It doesn't comply with guidelines.") == "YES"

    def test_complies_without_negation(self):
        assert ebl("The case fully complies with all rules.") == "NO"

    def test_non_compliant(self):
        """Tier 5 semantic: 'non-compliant' → YES."""
        assert ebl("The action is non-compliant.") == "YES"

    # -- YES / NO word boundary --
    def test_no_not_in_not(self):
        """\\bNO\\b must NOT match inside 'NOT'; the old substring bug."""
        # "NOT" contains "NO" as substring but \bNO\b won't match
        # because T is a word character following O.
        # This text has no standalone NO → falls through to VIOLAT match.
        assert ebl("The case does NOT meet requirements — a violation.") == "YES"

    def test_no_not_in_nothing(self):
        """'Nothing' doesn't trigger \\bNO\\b. 'violates' triggers VIOLAT → YES."""
        assert ebl("Nothing about this violates the policy.") == "YES"

    def test_nothing_wrong(self):
        """'Nothing' alone (no other keywords) → should not match NO."""
        assert ebl("Nothing wrong here, all good.") == "UNKNOWN"

    def test_no_not_in_know(self):
        """'KNOW' should not trigger NO match."""
        assert ebl("I know this is a breach of policy.") == "YES"

    def test_standalone_no(self):
        assert ebl("The answer is: NO.") == "NO"


# ── Tier 4: Numeric / JSON ──────────────────────────────────────────

class TestTier4Numeric:
    def test_bare_1(self):
        assert ebl("1") == "YES"

    def test_bare_0(self):
        assert ebl("0") == "NO"

    def test_quoted_1(self):
        assert ebl('"1"') == "YES"

    def test_quoted_0(self):
        assert ebl('"0"') == "NO"

    def test_json_violation_1(self):
        assert ebl('{"violation": 1}') == "YES"

    def test_json_violation_0(self):
        assert ebl('{"violation": 0}') == "NO"


# ── Tier 5: Semantic fallback ────────────────────────────────────────

class TestTier5Semantic:
    def test_breach(self):
        assert ebl("The policy is breached.") == "YES"

    def test_illegal(self):
        assert ebl("This action is illegal under the guidelines.") == "YES"

    def test_prohibited(self):
        assert ebl("Such behavior is prohibited.") == "YES"

    def test_rejected(self):
        assert ebl("The application is rejected.") == "YES"

    def test_fails(self):
        assert ebl("The case fails the policy check.") == "YES"

    def test_acceptable(self):
        assert ebl("The behavior is acceptable.") == "NO"

    def test_approved(self):
        assert ebl("The case is approved.") == "NO"

    def test_permitted(self):
        assert ebl("This action is permitted by the policy.") == "NO"

    def test_passes(self):
        assert ebl("The case passes all checks.") == "NO"

    def test_meets(self):
        assert ebl("The application meets all criteria.") == "NO"


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self):
        assert ebl("") == "UNKNOWN"

    def test_whitespace_only(self):
        assert ebl("   \n  ") == "UNKNOWN"

    def test_gibberish(self):
        assert ebl("asdfghjkl") == "UNKNOWN"

    def test_just_punctuation(self):
        assert ebl("...") == "UNKNOWN"

    def test_mixed_case(self):
        assert ebl("yEs") == "YES"

    def test_yes_and_no_in_explanation(self):
        """First-token wins when text contains both YES and NO."""
        assert ebl("YES — no doubt about it.") == "YES"

    def test_no_followed_by_violation_context(self):
        """First-token 'No' takes priority even if 'violation' appears later."""
        assert ebl("No, there is no violation here.") == "NO"


# ── LlamaGuard canonical mapping ────────────────────────────────────

class TestLlamaGuardCanonical:
    def test_safe(self):
        assert lgc("safe", "UNKNOWN") == "COMPLIES"

    def test_unsafe_with_categories(self):
        assert lgc("unsafe\nS1,S3", "UNKNOWN") == "VIOLATES"

    def test_unsafe_bare(self):
        assert lgc("unsafe", "UNKNOWN") == "VIOLATES"

    def test_first_token_safe(self):
        assert lgc("Safe — all good", "YES") == "COMPLIES"

    def test_first_token_unsafe(self):
        assert lgc("Unsafe behavior detected", "NO") == "VIOLATES"

    def test_fallback_to_binary(self):
        assert lgc("I don't know", "YES") == "VIOLATES"

    def test_fallback_unknown(self):
        assert lgc("gibberish", "UNKNOWN") == "UNKNOWN"


# ── Regression: old substring bugs ──────────────────────────────────

class TestRegressions:
    def test_not_safe_was_complies(self):
        """OLD BUG: 'not safe' → SAFE match → NO (COMPLIES). Now → YES."""
        assert ebl("This is not safe behavior.") == "YES"

    def test_no_in_not_was_no(self):
        """OLD BUG: 'NOT' → 'NO' substring match → NO. Now handled by \\b."""
        # Text with NOT but meaning violates; should not match NO
        assert ebl("The employee did NOT follow protocol, a clear violation.") == "YES"

    def test_safe_before_unsafe_check(self):
        """OLD BUG: 'SAFE' in 'UNSAFE' would match SAFE first if order was wrong.
        Now UNSAFE is checked before SAFE in Tier 3."""
        assert ebl("clearly unsafe") == "YES"

    def test_negation_misinversion_verbose(self):
        """OLD BUG: 'this case violates... does not directly violate...'
        Negation regex scanned entire text, found 'not...violate' late in
        response and wrongly returned NO. Now negation is windowed to first match."""
        text = (
            "Based on the policy, this case violates the following criteria: "
            "the employee engaged in prohibited moonlighting activity. "
            "While the policy does not directly violate any safety rules, "
            "it clearly breaches the employment terms."
        )
        assert ebl(text) == "YES"  # first VIOLAT at position ~40, not negated

    def test_compl_before_violat_position(self):
        """OLD BUG: Tier 3 checked VIOLAT* before COMPL* regardless of position.
        'this case complies... language that violates...' → VIOLAT fired first.
        Now position-based: earliest match wins."""
        text = (
            "This case complies with the company policy. "
            "Although the language used violates social norms, "
            "the policy requirements are met."
        )
        assert ebl(text) == "NO"  # COMPL appears before VIOLAT

    def test_violat_before_compl_position(self):
        """Position-based: when VIOLAT appears first, it should win."""
        text = (
            "This case violates the moonlighting policy. "
            "The employee complies with all other rules."
        )
        assert ebl(text) == "YES"  # VIOLAT appears before COMPL
