#!/usr/bin/env python3
# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Fetch the benchmark's input dataset into data/raw/.

The repo ships no data. After cloning, run this once to populate the
directory run_benchmark.py reads from. Re-runs are idempotent; already
present files are skipped unless ``--force`` is given.

Source:
    PolyGuard (Kang et al., NeurIPS 2025 Datasets & Benchmarks; v2 renamed
    to GuardSet-X). HuggingFace: AI-Secure/PolyGuard (the paper's official
    mirror; Virtue-AI-HUB/PolyGuard is a co-mirror with identical content).
    The HR, education and social-media subdomains (e.g. IBM HR, Apple HR,
    AI-for-Education, Reddit moderation) are converted to the READ JSONL
    schema by read_toolkit/polyguard_ingest.py; the cyber and code subsets
    ship no policy text on the mirror and are skipped. See the upstream
    HuggingFace page for the dataset license.

Usage:
    python scripts/fetch_data.py            # fetch if missing
    python scripts/fetch_data.py --force    # re-fetch
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
POLYGUARD_SRC = DATA_DIR / "PolyGuard"
POLYGUARD_OUT = DATA_DIR / "raw"
CLAUSE_MAP_DIR = DATA_DIR / "clause_maps"

POLYGUARD_HF_REPO = "AI-Secure/PolyGuard"


def _find_hf_cli() -> str | None:
    for binary in ("hf", "huggingface-cli"):
        if shutil.which(binary):
            return binary
    return None


_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _assert_no_lfs_pointers(target: Path) -> None:
    """A `git clone` without git-lfs leaves pointer stubs instead of data."""
    for path in target.rglob("*"):
        if ".git" in path.parts or not path.is_file() or path.stat().st_size > 512:
            continue
        with path.open("rb") as handle:
            if handle.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
                raise SystemExit(
                    f"error: {path} is a git-lfs pointer stub, not real data.\n"
                    "Install git-lfs (`git lfs install`) or the HuggingFace CLI "
                    '(`pip install -U "huggingface_hub[cli]"`), then re-run with --force.'
                )


def _hf_clone(repo_id: str, target: Path, force: bool) -> None:
    """Clone a HuggingFace dataset repo. Uses the hf CLI if available."""
    if target.exists() and not force:
        print(f"  [ok] {target} already present (use --force to re-fetch)")
        return
    if target.exists() and force:
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    hf_cli = _find_hf_cli()
    if hf_cli:
        print(f"  {hf_cli} download {repo_id} -> {target}")
        subprocess.check_call(
            [hf_cli, "download", repo_id, "--repo-type", "dataset", "--local-dir", str(target)],
        )
    else:
        url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"  git clone {url} -> {target}   (no hf/huggingface-cli found)")
        subprocess.check_call(["git", "clone", url, str(target)])
        _assert_no_lfs_pointers(target)


def _import_ingest():
    try:
        from read_toolkit import polyguard_ingest
    except ImportError:  # running from a clone without pip install
        sys.path.insert(0, str(REPO_ROOT))
        from read_toolkit import polyguard_ingest
    return polyguard_ingest


def fetch_polyguard(force: bool) -> None:
    print("PolyGuard")
    _hf_clone(POLYGUARD_HF_REPO, POLYGUARD_SRC, force)

    polyguard_ingest = _import_ingest()
    domains = polyguard_ingest.discover_all_domains()

    out_files = list(POLYGUARD_OUT.glob("polyguard_*.jsonl")) if POLYGUARD_OUT.exists() else []
    if out_files and not force:
        print(f"  [ok] data/raw/ already populated ({len(out_files)} jsonl files); skip ingest")
    else:
        print(f"  ingest via read_toolkit.polyguard_ingest -> {POLYGUARD_OUT}")
        POLYGUARD_OUT.mkdir(parents=True, exist_ok=True)
        for slug, config in sorted(domains.items()):
            instances = polyguard_ingest.ingest_domain(slug, config)
            out_path = POLYGUARD_OUT / f"{slug}.jsonl"
            polyguard_ingest.write_jsonl(instances, out_path)
            print(f"    {slug}: {len(instances)} instances -> {out_path.name}")

    # Clause maps are derived locally from the fetched policies (numbered
    # rules -> clauses R1..Rk); nothing from upstream is redistributed.
    CLAUSE_MAP_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for slug, config in sorted(domains.items()):
        full_policy = config.get("full_policy")
        clause_map = polyguard_ingest.build_clause_map(slug, full_policy) if full_policy else None
        if clause_map:
            path = CLAUSE_MAP_DIR / f"{slug}.json"
            path.write_text(json.dumps(clause_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written += 1
    print(f"  clause maps: {written} derived -> {CLAUSE_MAP_DIR.relative_to(REPO_ROOT)}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force", action="store_true", help="re-fetch even if files present")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fetch_polyguard(args.force)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"error: download failed (exit {exc.returncode}): {' '.join(map(str, exc.cmd))}\n"
            "Check network access to huggingface.co and re-run; finished files are kept."
        ) from exc

    print("\nDone. data/raw/ now contains:")
    if POLYGUARD_OUT.exists():
        n = sum(1 for _ in POLYGUARD_OUT.rglob("*") if _.is_file())
        print(f"  data/raw/  ({n} files)")


if __name__ == "__main__":
    main()
