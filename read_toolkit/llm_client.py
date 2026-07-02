# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
import itertools
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import anthropic
import requests
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
)

load_dotenv()

# IBM Research Inference Service endpoint. Set via env var `READ_RITS_BASE`.
# External users (without IBM internal access) should leave this unset and use
# the OpenAI / Anthropic providers instead; see README §Reproducibility tiers.
_DEFAULT_RITS_BASE = os.getenv("READ_RITS_BASE", "").rstrip("/")

# model slug -> config (provider + connection details)
MODELS = {
    # IBM RITS models
    "granite-guardian-3.3-8b": {
        "provider": "rits",
        "endpoint": "/granite-guardian-3-3-8b",
        "served_name": "ibm-granite/granite-guardian-3.3-8b",
    },
    "gpt-oss-safeguard-20b": {
        "provider": "rits",
        "endpoint": "/gpt-oss-safeguard-20b",
        "served_name": "openai/gpt-oss-safeguard-20b",
    },
    "gpt-oss-120b": {
        "provider": "rits",
        "endpoint": "/gpt-oss-120b",
        "served_name": "openai/gpt-oss-120b",
    },
    "llama-guard-3-8b": {
        "provider": "rits",
        "endpoint": "/llama-guard-3-8b",
        "served_name": None,
    },
    # RITS general-purpose models
    "llama-3-1-8b-instruct": {
        "provider": "rits",
        "endpoint": "/llama-3-1-8b-instruct",
        "served_name": None,
    },
    "llama-3-3-70b-instruct": {
        "provider": "rits",
        "endpoint": "/llama-3-3-70b-instruct",
        "served_name": None,
    },
    "mistral-small-3-2-24b-2506": {
        "provider": "rits",
        "endpoint": "/mistral-small-3-2-24b-2506",
        "served_name": None,
    },
    "qwen2-5-72b-instruct": {
        "provider": "rits",
        "endpoint": "/qwen2-5-72b-instruct",
        "served_name": None,
    },
    "deepseek-v3-2": {
        "provider": "rits",
        "endpoint": "/deepseek-v3-2",
        "served_name": None,
    },
    "granite-3-3-8b-instruct": {
        "provider": "rits",
        "endpoint": "/granite-3-3-8b-instruct",
        "served_name": None,
    },
    "llama-4-scout-17b-16e-instruct": {
        "provider": "rits",
        "endpoint": "/llama-4-scout-17b-16e-instruct",
        "served_name": None,
    },
    "microsoft-phi-4": {
        "provider": "rits",
        "endpoint": "/microsoft-phi-4",
        "served_name": None,
    },
    "qwen3-8b": {
        "provider": "rits",
        "endpoint": "/qwen3-8b",
        "served_name": None,
    },
    # OpenAI models
    "gpt-4o": {
        "provider": "openai",
        "served_name": "gpt-4o",
    },
    # Anthropic models
    "claude-sonnet-4-5": {
        "provider": "anthropic",
        "served_name": "claude-sonnet-4-5",
    },
}

RITS_MODELS = [k for k, v in MODELS.items() if v["provider"] == "rits"]

# Original safety-focused subset, used as default for quick runs
DEFAULT_RITS_MODELS = [
    "granite-guardian-3.3-8b",
    "gpt-oss-safeguard-20b",
    "gpt-oss-120b",
    "llama-guard-3-8b",
]

DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF_SEC = 2.0
COMPLIANCE_PROMPT_SUFFIX = (
    "Classify the case against the policy criteria. "
    "You MUST begin your response with exactly YES or NO. "
    "YES = the case violates the policy. NO = the case complies. "
    "Then explain your reasoning."
)


@dataclass
class JudgeResult:
    label: str  # canonical label: VIOLATES | COMPLIES | UNKNOWN
    raw_response: str
    logprob: Optional[float]
    binary_label: str  # YES | NO | UNKNOWN
    trace: Optional[str] = None


class JudgeClient:
    """multi-provider policy-judge client with model-specific prompting."""

    def __init__(
        self,
        model_key: str = "granite-guardian-3.3-8b",
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        backoff_sec: float = DEFAULT_BACKOFF_SEC,
    ):
        if model_key not in MODELS:
            raise ValueError(f"unknown model key '{model_key}', choose from: {list(MODELS)}")

        model_cfg = MODELS[model_key]
        self.model_key = model_key
        self.provider = model_cfg["provider"]
        self.served_name = model_cfg.get("served_name")
        self.timeout = timeout
        self.retries = retries
        self.backoff_sec = backoff_sec

        if self.provider == "rits":
            rits_base = os.getenv("READ_RITS_BASE", _DEFAULT_RITS_BASE).rstrip("/")
            if not rits_base:
                raise RuntimeError(
                    f"Model '{model_key}' is hosted on IBM's Research Inference Service "
                    "and requires the READ_RITS_BASE env var (the RITS endpoint URL, "
                    "IBM-internal access only). External users without RITS access: "
                    "use 'gpt-4o' or 'claude-sonnet-4-5' ; both work with the standard "
                    "OPENAI_API_KEY / ANTHROPIC_API_KEY env vars. See README "
                    "§Reproducibility tiers for the full picture."
                )
            raw_keys = os.getenv("RITS_API_KEY", "").strip().strip('"').strip("'")
            if not raw_keys:
                raise RuntimeError(
                    f"Model '{model_key}' requires RITS_API_KEY env var (IBM-internal). "
                    "External users: use 'gpt-4o' or 'claude-sonnet-4-5' instead."
                )
            keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
            if not keys:
                raise RuntimeError("RITS_API_KEY is empty after parsing")
            self._rits_keys = itertools.cycle(keys)
            self.api_key = next(self._rits_keys)
            if len(keys) > 1:
                print(f"    key rotation: {len(keys)} RITS API keys loaded")
            self.endpoint = model_cfg["endpoint"]
            self.base_url = f"{rits_base}{self.endpoint}/v1"
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers={"RITS_API_KEY": self.api_key},
            )
            if not self.served_name:
                self.served_name = self._resolve_served_model_name()

        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.api_key = api_key
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.api_key = api_key
            self.client = anthropic.Anthropic(api_key=api_key)

        else:
            raise ValueError(f"unknown provider '{self.provider}'")

    def _resolve_served_model_name(self) -> str:
        url = f"{self.base_url}/models"
        response = self._request_with_retry(
            lambda: requests.get(url, headers={"RITS_API_KEY": self.api_key}, timeout=self.timeout),
            (requests.RequestException,),
        )
        payload = response.json()
        models = payload.get("data", [])
        if models and isinstance(models, list):
            model_id = models[0].get("id")
            if model_id:
                return model_id
        raise RuntimeError(f"could not resolve served model for {self.model_key} from {url}")

    def _request_with_retry(
        self,
        call: Callable[[], Any],
        retry_on: tuple[type[BaseException], ...],
    ) -> Any:
        import random as _rnd
        last_exc: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                return call()
            except retry_on as exc:
                last_exc = exc
                if attempt == self.retries - 1:
                    break
                base_delay = self.backoff_sec * (2 ** attempt)
                jitter = _rnd.uniform(0, base_delay * 0.5)
                time.sleep(base_delay + jitter)
        if last_exc:
            raise last_exc
        raise RuntimeError("request failed with unknown error")

    def _rotate_rits_key(self) -> None:
        """Rotate to next RITS API key (round-robin) for load distribution."""
        if self.provider == "rits" and hasattr(self, "_rits_keys"):
            self.api_key = next(self._rits_keys)
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers={"RITS_API_KEY": self.api_key},
            )

    def judge(self, policy: str, query: str, temperature: float = 0.0) -> JudgeResult:
        """judge one query against one policy and return canonical read label."""
        self._rotate_rits_key()
        if self.provider == "anthropic":
            return self._judge_anthropic(policy, query, temperature)
        if self.provider == "openai":
            return self._judge_generic_chat(policy, query, temperature)
        # rits: dispatch by model key
        if self.model_key.startswith("gpt-oss"):
            return self._judge_via_responses(policy, query, temperature)
        if self.model_key.startswith("granite-guardian"):
            return self._judge_granite(policy, query, temperature)
        if self.model_key.startswith("llama-guard"):
            return self._judge_llama_guard(policy, query, temperature)
        return self._judge_generic_chat(policy, query, temperature)

    def _judge_granite(self, policy: str, query: str, temperature: float = 0.0) -> JudgeResult:
        # granite guardian must receive policy via guardian_config in chat_template_kwargs
        user_query = (
            f"{query}\n\n"
            "task evaluate whether this case violates or complies with the provided policy criteria "
            "return only via granite score yes for violates no for complies"
        )
        temp = temperature
        response = self._request_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.served_name,
                messages=[{"role": "user", "content": user_query}],
                temperature=temp,
                max_tokens=2048,
                logprobs=True,
                top_logprobs=5,
                extra_body={
                    "chat_template_kwargs": {
                        "guardian_config": {"custom_criteria": policy},
                        "think": True,
                    }
                },
            ),
            (BadRequestError, PermissionDeniedError, InternalServerError, APIConnectionError, APITimeoutError, requests.RequestException),
        )
        raw = (response.choices[0].message.content or "").strip()
        trace = self._extract_tag(raw, "think")
        binary = self._extract_binary_label(raw)
        canonical = self._binary_to_canonical(binary, family="risk_yes")
        logprob = self._extract_logprob(response)
        return JudgeResult(
            label=canonical,
            raw_response=raw,
            logprob=logprob,
            binary_label=binary,
            trace=trace,
        )

    def _judge_via_responses(self, policy: str, query: str, temperature: float = 0.0) -> JudgeResult:
        prompt = self._format_harmony_policy(policy, query)
        url = f"{self.base_url}/responses"
        payload = {
            "model": self.served_name,
            "input": prompt,
            "temperature": temperature,
        }
        def _post_and_check():
            resp = requests.post(
                url,
                headers={"RITS_API_KEY": self.api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp

        response = self._request_with_retry(
            _post_and_check,
            (requests.RequestException, requests.HTTPError),
        )
        data = response.json()
        raw = self._extract_responses_output_text(data)
        trace = self._extract_responses_trace(data)
        binary = self._extract_binary_label(raw)
        canonical = self._binary_to_canonical(binary, family="risk_yes")
        return JudgeResult(
            label=canonical,
            raw_response=raw,
            logprob=None,
            binary_label=binary,
            trace=trace,
        )

    def _judge_llama_guard(self, policy: str, query: str, temperature: float = 0.0) -> JudgeResult:
        prompt = (
            "Classify this case against the policy criteria.\n\n"
            f"Policy:\n{policy}\n\n"
            f"Case:\n{query}\n\n"
            f"{COMPLIANCE_PROMPT_SUFFIX}"
        )
        temp = temperature
        try:
            response = self._request_with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.served_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=256,
                    logprobs=True,
                    top_logprobs=5,
                ),
                (BadRequestError, PermissionDeniedError, InternalServerError, APIConnectionError, APITimeoutError, requests.RequestException),
            )
        except BadRequestError as exc:
            if "Conversation roles must alternate" not in str(exc):
                raise
            response = self.client.chat.completions.create(
                model=self.served_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=256,
                logprobs=True,
                top_logprobs=1,
            )
        raw = (response.choices[0].message.content or "").strip()
        binary = self._extract_binary_label(raw)
        canonical = self._llama_guard_to_canonical(raw, binary)
        logprob = self._extract_logprob(response)
        return JudgeResult(
            label=canonical,
            raw_response=raw,
            logprob=logprob,
            binary_label=binary,
            trace=None,
        )

    def _judge_generic_chat(self, policy: str, query: str, temperature: float = 0.0) -> JudgeResult:
        user_content = f"Policy:\n{policy}\n\nQuery:\n{query}"
        temp = temperature
        response = self._request_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.served_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a policy compliance judge. "
                            "Evaluate the case facts against the policy criteria. "
                            "You MUST begin your response with exactly YES or NO. "
                            "YES = the case violates the policy. NO = the case complies. "
                            "Then explain your reasoning."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                temperature=temp,
                max_tokens=4096,
                logprobs=True,
                top_logprobs=5,
            ),
            (BadRequestError, PermissionDeniedError, InternalServerError, APIConnectionError, APITimeoutError, requests.RequestException),
        )
        raw = (response.choices[0].message.content or "").strip()
        binary = self._extract_binary_label(raw)
        # Logprob fallback: if text extraction failed, try first-token logprobs
        if binary == "UNKNOWN":
            lp_binary, _ = self._extract_verdict_from_logprobs(response)
            if lp_binary != "UNKNOWN":
                binary = lp_binary
        canonical = self._binary_to_canonical(binary, family="risk_yes")
        logprob = self._extract_logprob(response)
        return JudgeResult(
            label=canonical,
            raw_response=raw,
            logprob=logprob,
            binary_label=binary,
            trace=None,
        )

    def _judge_anthropic(self, policy: str, query: str, temperature: float = 0.0) -> JudgeResult:
        system_prompt = (
            "You are a policy compliance judge. "
            "Evaluate the case facts against the policy criteria. "
            "You MUST begin your response with exactly YES or NO. "
            "YES = the case violates the policy. NO = the case complies. "
            "Then explain your reasoning."
        )
        user_content = f"Policy:\n{policy}\n\nQuery:\n{query}"
        kwargs = {
            "model": self.served_name,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": 256,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        response = self._request_with_retry(
            lambda: self.client.messages.create(**kwargs),
            (anthropic.APIError,),
        )
        raw = response.content[0].text.strip() if response.content else ""
        binary = self._extract_binary_label(raw)
        canonical = self._binary_to_canonical(binary, family="risk_yes")
        return JudgeResult(
            label=canonical,
            raw_response=raw,
            logprob=None,
            binary_label=binary,
            trace=None,
        )

    @staticmethod
    def _extract_logprob(response: Any) -> Optional[float]:
        try:
            return response.choices[0].logprobs.content[0].logprob
        except (AttributeError, IndexError, TypeError):
            return None

    @staticmethod
    def _extract_verdict_from_logprobs(response: Any) -> tuple[str, float]:
        """Pick the higher-mass YES-like vs NO-like token in the first-token top_logprobs.

        Returns ``(binary, P(winner) / (P(yes) + P(no)))``, or ``("UNKNOWN", 0.0)`` if neither shows up.
        """
        try:
            top_lps = response.choices[0].logprobs.content[0].top_logprobs
        except (AttributeError, IndexError, TypeError):
            return "UNKNOWN", 0.0

        if not top_lps:
            return "UNKNOWN", 0.0

        YES_TOKENS = {"YES", "VIOLATES", "VIOLATION", "UNSAFE"}
        NO_TOKENS = {"NO", "COMPLIES", "COMPLIANT", "SAFE"}

        yes_prob = 0.0
        no_prob = 0.0
        for alt in top_lps:
            token = alt.token.strip().upper()
            prob = math.exp(alt.logprob)
            if token in YES_TOKENS:
                yes_prob += prob
            elif token in NO_TOKENS:
                no_prob += prob

        total = yes_prob + no_prob
        if total < 0.01:
            return "UNKNOWN", 0.0

        confidence = max(yes_prob, no_prob) / total
        return ("YES" if yes_prob > no_prob else "NO"), confidence

    @staticmethod
    def _extract_tag(text: str, tag: str) -> Optional[str]:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        return None

    # Negation prefixes (NOT / NEVER / n't variants) within 20 chars of keyword
    _NEG_RE = r"\b(?:NOT|NEVER|N['\u2019]?T|ISN['\u2019]?T|AREN['\u2019]?T|DOESN['\u2019]?T|DON['\u2019]?T|WASN['\u2019]?T|CANNOT|CAN['\u2019]?T)\b.{0,20}"

    @classmethod
    def _extract_binary_label(cls, text: str) -> str:
        clean = text.strip()
        if not clean:
            return "UNKNOWN"
        # Strip <think>...</think> blocks (reasoning models like qwen3, deepseek)
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL | re.IGNORECASE).strip()
        if not clean:
            return "UNKNOWN"
        upper = clean.upper()
        compact = re.sub(r"\s+", "", upper)

        # Tier 1: structured output (XML, JSON, bare numeric)
        score = re.findall(
            r"<score>\s*(yes|no)\s*</score>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if score:
            return score[-1].upper()
        if compact in {"1", '"1"'}:
            return "YES"
        if compact in {"0", '"0"'}:
            return "NO"
        if '"VIOLATION":1' in compact or "'VIOLATION':1" in compact:
            return "YES"
        if '"VIOLATION":0' in compact or "'VIOLATION':0" in compact:
            return "NO"

        # Tier 2: first-token match (most reliable for prompted models)
        first_match = re.match(r"[^a-zA-Z]*([a-zA-Z]+)", clean)
        if first_match:
            fw = first_match.group(1).upper()
            if fw in ("YES", "VIOLATES", "VIOLATION", "UNSAFE"):
                return "YES"
            if fw in ("NO", "COMPLIES", "COMPLIANT", "SAFE"):
                return "NO"

        # Tier 3: word-boundary keywords with negation guards
        # UNSAFE / SAFE
        if re.search(r"\bUNSAFE\b", upper):
            return "YES"
        if re.search(r"\bSAFE\b", upper):
            if re.search(cls._NEG_RE + r"\bSAFE\b", upper):
                return "YES"           # "not safe" → violates
            return "NO"

        # YES / NO  (\b excludes NOT, NOTHING, KNOW, etc.)
        if re.search(r"\bYES\b", upper):
            return "YES"
        if re.search(r"\bNO\b", upper):
            return "NO"

        # VIOLATE* / COMPLY*: position-based priority + windowed negation
        violat_m = re.search(r"\bVIOLAT(?:ES?|ION|ING)\b", upper)
        noncompl_m = re.search(r"\bNON-?COMPLIAN", upper)
        compl_m = re.search(r"\bCOMPL(?:Y|IES|IANT|IANCE)\b", upper)

        _candidates = []
        if violat_m:
            _candidates.append((violat_m.start(), "VIOLAT", violat_m))
        if noncompl_m:
            _candidates.append((noncompl_m.start(), "NONCOMPL", noncompl_m))
        if compl_m:
            _candidates.append((compl_m.start(), "COMPL", compl_m))

        if _candidates:
            _candidates.sort(key=lambda x: x[0])
            _, kind, m = _candidates[0]
            # Negation window: only check ~40 chars before this match
            window = upper[max(0, m.start() - 40):m.end()]
            if kind == "VIOLAT":
                if re.search(cls._NEG_RE + r"\bVIOLAT", window):
                    return "NO"        # "does not violate" → complies
                return "YES"
            elif kind == "NONCOMPL":
                return "YES"
            elif kind == "COMPL":
                if re.search(cls._NEG_RE + r"\bCOMPL", window):
                    return "YES"       # "does not comply" → violates
                return "NO"

        # Tier 4: semantic fallback (verbose non-standard responses)
        if re.search(r"\b(?:BREACH\w*|ILLEGAL|PROHIBITED|REJECTED|FAILS?|INFRACT\w*|IMPERMISSIBLE)\b", upper):
            return "YES"
        if re.search(r"\b(?:ACCEPTABLE|APPROVED|PERMITTED|PASSES|MEETS|SATISFIES|ALLOWABLE|PERMISSIBLE)\b", upper):
            return "NO"

        return "UNKNOWN"

    @staticmethod
    def _binary_to_canonical(binary: str, family: str) -> str:
        # family risk_yes means YES => VIOLATES, NO => COMPLIES
        if binary == "UNKNOWN":
            return "UNKNOWN"
        if family == "risk_yes":
            return "VIOLATES" if binary == "YES" else "COMPLIES"
        return "COMPLIES" if binary == "YES" else "VIOLATES"

    @staticmethod
    def _llama_guard_to_canonical(raw: str, fallback_binary: str) -> str:
        # LlamaGuard outputs "safe" or "unsafe\nS1,S2" ; first token is definitive
        first = re.match(r"[^a-zA-Z]*([a-zA-Z]+)", raw.strip())
        if first:
            fw = first.group(1).upper()
            if fw == "UNSAFE":
                return "VIOLATES"
            if fw == "SAFE":
                return "COMPLIES"
        upper = raw.upper()
        if re.search(r"\bUNSAFE\b", upper):
            return "VIOLATES"
        if re.search(r"\bSAFE\b", upper):
            return "COMPLIES"
        return JudgeClient._binary_to_canonical(fallback_binary, family="risk_yes")

    def _format_harmony_policy(self, policy: str, query: str) -> str:
        return (
            "## INSTRUCTIONS\n"
            "Evaluate whether the case violates the policy criteria. "
            "Return exactly YES if it violates and exactly NO if it complies. "
            "Do not return 0 or 1.\n"
            "## DEFINITIONS\n"
            f"Policy: {policy}\n"
            "## VIOLATES\n"
            "Case facts that fail one or more policy criteria.\n"
            "## COMPLIES\n"
            "Case facts that satisfy policy criteria.\n\n"
            "## CASE\n"
            f"{query}\n"
        )

    @staticmethod
    def _extract_responses_output_text(payload: dict) -> str:
        output_items = payload.get("output", [])
        for item in output_items:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return (part.get("text") or "").strip()
        return ""

    @staticmethod
    def _extract_responses_trace(payload: dict) -> Optional[str]:
        output_items = payload.get("output", [])
        traces: list[str] = []
        for item in output_items:
            if item.get("type") != "reasoning":
                continue
            for part in item.get("content", []):
                if part.get("type") == "reasoning_text" and part.get("text"):
                    traces.append(part["text"].strip())
        if traces:
            return "\n\n".join(traces)
        return None


# backward compat alias
RITSClient = JudgeClient

