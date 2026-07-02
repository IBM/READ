# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Console entry points print a short error and exit instead of raising a traceback."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_benchmark as rb
from read_toolkit import failure_modes, polyguard_ingest, summarize_results


def _main_exit(monkeypatch, module, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        module.main()
    return str(excinfo.value.code)


def test_benchmark_missing_dataset(monkeypatch, tmp_path):
    message = _main_exit(
        monkeypatch, rb,
        ["read-benchmark", "--dataset", str(tmp_path / "missing.jsonl"), "--models", "gpt-4o"],
    )
    assert "dataset not found" in message
    assert "fetch_data.py" in message


def test_benchmark_unknown_model(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    message = _main_exit(
        monkeypatch, rb,
        ["read-benchmark", "--dataset", str(dataset), "--models", "not-a-model"],
    )
    assert "unknown model keys" in message
    assert "gpt-4o" in message


def test_benchmark_missing_api_key(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({
            "instance_id": "inst_001",
            "domain": "edu",
            "ground_truth_label": "VIOLATES",
            "policy_base_nl": "policy text",
            "test_query_input": "query text",
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    message = _main_exit(
        monkeypatch, rb,
        [
            "read-benchmark",
            "--dataset", str(dataset),
            "--models", "gpt-4o",
            "--output", str(tmp_path / "out.csv"),
        ],
    )
    assert "OPENAI_API_KEY" in message


def test_failure_mode_missing_csv(monkeypatch, tmp_path):
    message = _main_exit(
        monkeypatch, failure_modes,
        ["read-failure-mode", str(tmp_path / "missing.csv")],
    )
    assert "results CSV not found" in message
    assert "read-benchmark" in message


def test_summarize_missing_csv(monkeypatch, tmp_path):
    message = _main_exit(
        monkeypatch, summarize_results,
        ["read-summarize", str(tmp_path / "missing.csv")],
    )
    assert "results CSV not found" in message
    assert "read-benchmark" in message


def test_ingest_list_domains_without_data(monkeypatch, tmp_path):
    monkeypatch.setattr(polyguard_ingest, "POLYGUARD_ROOT", tmp_path / "PolyGuard")
    message = _main_exit(monkeypatch, polyguard_ingest, ["read-ingest", "--list-domains"])
    assert "no PolyGuard data" in message
    assert "fetch_data.py" in message


def test_ingest_without_data(monkeypatch, tmp_path):
    monkeypatch.setattr(polyguard_ingest, "POLYGUARD_ROOT", tmp_path / "PolyGuard")
    message = _main_exit(
        monkeypatch, polyguard_ingest,
        ["read-ingest", "--output", str(tmp_path / "out.jsonl")],
    )
    assert "nothing to ingest" in message
