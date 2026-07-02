# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.0a1] - 2026-07-02

### Added

- Initial public release: perturbation engine (clause erasure/masking +
  format perturbations), multi-provider judge client, PIR/CSR/ASR/CCR/PC-CSR
  metric library with confidence intervals, failure-mode classifier
  (Prior-Based / Policy-Degraded / Confused / Faithful), offline test suite,
  and a paper-reproduction driver.
- `scripts/fetch_data.py`: fetches PolyGuard from HuggingFace, converts it
  to the READ schema and derives per-domain clause maps locally.
- pip-installable as `read-toolkit` with `read-benchmark`,
  `read-failure-mode`, `read-summarize` and `read-ingest` console commands.
