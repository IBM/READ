# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Served model names can be overridden via env vars (enterprise gateways)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from read_toolkit.llm_client import JudgeClient


def test_anthropic_served_name_env_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("READ_ANTHROPIC_MODEL", "aws/claude-sonnet-4-5")
    client = JudgeClient(model_key="claude-sonnet-4-5")
    assert client.served_name == "aws/claude-sonnet-4-5"


def test_openai_served_name_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("READ_OPENAI_MODEL", "gateway/gpt-4o")
    client = JudgeClient(model_key="gpt-4o")
    assert client.served_name == "gateway/gpt-4o"


def test_served_name_unchanged_without_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("READ_OPENAI_MODEL", raising=False)
    client = JudgeClient(model_key="gpt-4o")
    assert client.served_name == "gpt-4o"
