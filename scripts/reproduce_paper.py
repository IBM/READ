#!/usr/bin/env python3
# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the AIES 2026 READ paper's main tables from raw benchmark CSVs.

Runs the same per-(model, domain) summary as summarize_results.py, then
assembles:

    table_s1.csv   cross-domain per-(model, domain) cells (Table 1 = the
                   ibm_hr slice; Table S1 = the full grid)
    table_s4.csv   per-model pooled across domains; numerators and
                   denominators are summed, NOT a mean of pre-computed rates
    table_s5.csv   with-policy vs no-policy accuracy and gap (positive when
                   the judge is more accurate without the policy)
    figure_data.json   V-Mode / C-Mode shares per (model, domain) and the
                       per-branch failure-mode distribution for Figure 4 / S1

Usage:

    python scripts/reproduce_paper.py \\
        --domain-csvs ibm_hr=results/exp14_ibm_hr.csv,edu=results/exp14_edu.csv,\\
reddit=results/exp14_reddit.csv,apple=results/exp14_apple.csv,\\
cyber=results/exp14_cyber_phishing.csv \\
        --output-dir results/reproduction/ \\
        --tables s1,s4,s5,figure
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as a top-level script
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from read_toolkit.evaluator import pool_metrics_across_domains, with_vs_no_policy_accuracy
from read_toolkit.failure_modes import failure_mode_distribution
from read_toolkit.stats import bootstrap_ci_from_binary, clopper_pearson_ci
from read_toolkit.summarize_results import compute_fidelity, compute_metrics, compute_pc_csr, compute_pir


# Per-domain extraction

def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _per_model_rows(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r.get("model", "unknown")].append(r)
    return out


def _model_cell(rows: list[dict]) -> dict:
    """Compute every metric + count the reproducer needs for one (model, domain) cell."""
    overall = compute_metrics(rows)
    pir = compute_pir(rows) or {}
    fidelity = compute_fidelity(rows) or {}
    pc_csr = compute_pc_csr(rows) or {}
    fm = failure_mode_distribution(rows, group_key="model") or {}
    # fm has one entry per model; for a single-model slice there is exactly one key
    model_fm = next(iter(fm.values()), {}) if fm else {}

    # Raw counts (numerator/denominator) for pool_metrics_across_domains
    n_total = len(rows)
    n_correct = sum(
        1 for r in rows
        if (r.get("predicted_label") or "").strip().upper() == (r.get("ground_truth") or "").strip().upper()
    )

    return {
        "n_rows": n_total,
        "acc": overall.get("accuracy"),
        "acc_correct": n_correct,
        "acc_n": n_total,
        "f1_v": overall.get("f1_violates"),
        "f1_c": overall.get("f1_complies"),
        "pir": pir.get("pir"),
        "pir_n": pir.get("n", 0),
        "pir_flipped": (pir.get("v_to_c", 0) + pir.get("c_to_v", 0)),
        "csr": fidelity.get("targeted_csr"),
        "csr_n_eligible": fidelity.get("n_eligible_ablations", 0),
        "csr_n_sensitive": fidelity.get("csr_n_sensitive", 0),
        "asr": fidelity.get("asr"),
        "asr_n": fidelity.get("asr_n", 0),
        "asr_n_stable": fidelity.get("asr_n_stable", 0),
        "ccr": fidelity.get("ccr"),
        "ccr_n": fidelity.get("ccr_n", 0),
        "ccr_n_stable": fidelity.get("ccr_n_stable", 0),
        "pc_csr": pc_csr.get("pc_csr"),
        "pc_csr_n_eligible": pc_csr.get("n_eligible", 0),
        "pc_csr_n_flipped": pc_csr.get("n_flipped", 0),
        "pc_csr_indicators": pc_csr.get("indicators", []),
        "v_mode_label": model_fm.get("v_mode", {}).get("label"),
        "v_mode_share": model_fm.get("v_mode", {}).get("share"),
        "v_mode_n": model_fm.get("v_mode", {}).get("n", 0),
        "v_branch_distribution": model_fm.get("v_mode", {}).get("distribution", {}),
        "c_mode_label": model_fm.get("c_mode", {}).get("label"),
        "c_mode_share": model_fm.get("c_mode", {}).get("share"),
        "c_mode_n": model_fm.get("c_mode", {}).get("n", 0),
        "c_branch_distribution": model_fm.get("c_mode", {}).get("distribution", {}),
    }


def _ci(num: int, den: int, ci_level: float = 0.95) -> tuple[float | None, float | None]:
    if den == 0:
        return (None, None)
    lo, hi = clopper_pearson_ci(num, den, ci_level=ci_level)
    return (round(lo, 4), round(hi, 4))


# Table generators

def build_table_s1(per_domain: dict[str, dict[str, dict]], ci_level: float = 0.95) -> list[dict]:
    """Cross-domain Table S1 (Table 1 = ibm_hr slice). Columns:
    domain | model | acc | acc_ci | f1_v | f1_c | pir | pir_ci | csr | csr_ci
          | asr | asr_ci | ccr | ccr_ci | pc_csr | pc_csr_n | v_mode | c_mode
    """
    rows = []
    for domain in sorted(per_domain.keys()):
        for model in sorted(per_domain[domain].keys()):
            c = per_domain[domain][model]
            acc_lo, acc_hi = _ci(c["acc_correct"], c["acc_n"], ci_level)
            pir_lo, pir_hi = _ci(c["pir_flipped"], c["pir_n"], ci_level)
            csr_lo, csr_hi = _ci(c["csr_n_sensitive"], c["csr_n_eligible"], ci_level)
            asr_lo, asr_hi = _ci(c["asr_n_stable"], c["asr_n"], ci_level)
            ccr_lo, ccr_hi = _ci(c["ccr_n_stable"], c["ccr_n"], ci_level)
            pc_lo = pc_hi = None
            if c["pc_csr_n_eligible"] and c["pc_csr_indicators"]:
                pc_lo, pc_hi = bootstrap_ci_from_binary(
                    c["pc_csr_indicators"], n_boot=1000, ci_level=ci_level, seed=42,
                )
                pc_lo, pc_hi = round(pc_lo, 4), round(pc_hi, 4)
            rows.append({
                "domain": domain,
                "model": model,
                "n": c["n_rows"],
                "acc": c["acc"],
                "acc_ci_lo": acc_lo,
                "acc_ci_hi": acc_hi,
                "f1_v": c["f1_v"],
                "f1_c": c["f1_c"],
                "pir": c["pir"],
                "pir_ci_lo": pir_lo,
                "pir_ci_hi": pir_hi,
                "csr": c["csr"],
                "csr_ci_lo": csr_lo,
                "csr_ci_hi": csr_hi,
                "asr": c["asr"],
                "asr_ci_lo": asr_lo,
                "asr_ci_hi": asr_hi,
                "ccr": c["ccr"],
                "ccr_ci_lo": ccr_lo,
                "ccr_ci_hi": ccr_hi,
                "pc_csr": c["pc_csr"],
                "pc_csr_n": c["pc_csr_n_eligible"],
                "pc_csr_ci_lo": pc_lo,
                "pc_csr_ci_hi": pc_hi,
                "v_mode": f"{c['v_mode_label']} {c['v_mode_share']:.2%}" if c["v_mode_label"] else None,
                "v_mode_n": c["v_mode_n"],
                "c_mode": f"{c['c_mode_label']} {c['c_mode_share']:.2%}" if c["c_mode_label"] else None,
                "c_mode_n": c["c_mode_n"],
            })
    return rows


def build_table_s4(per_domain: dict[str, dict[str, dict]], ci_level: float = 0.95) -> list[dict]:
    """Per-model pooled (Table S4). Pools numerators and denominators per metric
    across domains; PC-CSR pools indicator lists then re-bootstraps the CI.
    """
    models = sorted({m for dom in per_domain.values() for m in dom.keys()})
    rows = []
    for model in models:
        # Build per-domain count dicts per metric for this model
        domain_cells = {dom: per_domain[dom][model] for dom in per_domain if model in per_domain[dom]}
        acc_counts = {d: {"numerator": c["acc_correct"], "denominator": c["acc_n"]} for d, c in domain_cells.items()}
        pir_counts = {d: {"numerator": c["pir_flipped"], "denominator": c["pir_n"]} for d, c in domain_cells.items()}
        csr_counts = {d: {"numerator": c["csr_n_sensitive"], "denominator": c["csr_n_eligible"]} for d, c in domain_cells.items()}
        asr_counts = {d: {"numerator": c["asr_n_stable"], "denominator": c["asr_n"]} for d, c in domain_cells.items()}
        ccr_counts = {d: {"numerator": c["ccr_n_stable"], "denominator": c["ccr_n"]} for d, c in domain_cells.items()}
        pc_counts = {d: {"numerator": c["pc_csr_n_flipped"], "denominator": c["pc_csr_n_eligible"]} for d, c in domain_cells.items()}

        acc = pool_metrics_across_domains(acc_counts)
        pir = pool_metrics_across_domains(pir_counts)
        csr = pool_metrics_across_domains(csr_counts)
        asr = pool_metrics_across_domains(asr_counts)
        ccr = pool_metrics_across_domains(ccr_counts)
        pc = pool_metrics_across_domains(pc_counts)

        pc_indicators_pooled = []
        for c in domain_cells.values():
            pc_indicators_pooled.extend(c["pc_csr_indicators"])
        pc_lo = pc_hi = None
        if pc_indicators_pooled:
            pc_lo, pc_hi = bootstrap_ci_from_binary(
                pc_indicators_pooled, n_boot=1000, ci_level=ci_level, seed=42,
            )
            pc_lo, pc_hi = round(pc_lo, 4), round(pc_hi, 4)

        acc_lo, acc_hi = _ci(acc["numerator"], acc["denominator"], ci_level)
        pir_lo, pir_hi = _ci(pir["numerator"], pir["denominator"], ci_level)
        csr_lo, csr_hi = _ci(csr["numerator"], csr["denominator"], ci_level)
        asr_lo, asr_hi = _ci(asr["numerator"], asr["denominator"], ci_level)
        ccr_lo, ccr_hi = _ci(ccr["numerator"], ccr["denominator"], ci_level)

        rows.append({
            "model": model,
            "acc": acc["rate"],
            "acc_ci_lo": acc_lo,
            "acc_ci_hi": acc_hi,
            "pir": pir["rate"],
            "pir_ci_lo": pir_lo,
            "pir_ci_hi": pir_hi,
            "csr": csr["rate"],
            "csr_ci_lo": csr_lo,
            "csr_ci_hi": csr_hi,
            "asr": asr["rate"],
            "asr_ci_lo": asr_lo,
            "asr_ci_hi": asr_hi,
            "ccr": ccr["rate"],
            "ccr_ci_lo": ccr_lo,
            "ccr_ci_hi": ccr_hi,
            "pc_csr": pc["rate"],
            "pc_csr_n": pc["denominator"],
            "pc_csr_ci_lo": pc_lo,
            "pc_csr_ci_hi": pc_hi,
        })
    return rows


def build_table_s5(per_domain_rows: dict[str, list[dict]]) -> list[dict]:
    """With-policy vs no-policy accuracy (Table S5). Cells = (model, domain).
    Gap = no_policy_acc - with_policy_acc.
    """
    out_rows = []
    for domain in sorted(per_domain_rows.keys()):
        rows = per_domain_rows[domain]
        by_model = with_vs_no_policy_accuracy(rows, group_key="model")
        for model in sorted(by_model.keys()):
            stats = by_model[model]
            out_rows.append({
                "domain": domain,
                "model": model,
                "with_policy_acc": stats["with_policy_acc"],
                "no_policy_acc": stats["no_policy_acc"],
                "gap": stats["gap"],
                "n": stats["n"],
            })
    return out_rows


def build_figure_data(per_domain: dict[str, dict[str, dict]]) -> dict:
    """Per-(model, domain) V-branch and C-branch failure-mode distributions for
    Figure 4 (Table 1 model × ibm_hr) and Figure S1 (the full grid).
    """
    payload: dict[str, dict[str, dict]] = {}
    for domain in sorted(per_domain.keys()):
        for model in sorted(per_domain[domain].keys()):
            c = per_domain[domain][model]
            payload.setdefault(domain, {})[model] = {
                "v_branch": {
                    "n": c["v_mode_n"],
                    "distribution": c["v_branch_distribution"],
                },
                "c_branch": {
                    "n": c["c_mode_n"],
                    "distribution": c["c_branch_distribution"],
                },
            }
    return payload


# I/O

def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_domain_csvs(spec: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise SystemExit(f"--domain-csvs entry '{entry}' missing '='")
        key, path_str = entry.split("=", 1)
        path = Path(path_str.strip())
        if not path.exists():
            raise SystemExit(f"--domain-csvs path does not exist: {path}")
        out[key.strip()] = path
    if not out:
        raise SystemExit("--domain-csvs is empty")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--domain-csvs", required=True,
        help="Comma-separated key=path pairs (e.g. 'ibm_hr=results/exp14_ibm_hr.csv,edu=...').",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproduction"))
    parser.add_argument(
        "--tables", default="s1,s4,s5,figure",
        help="Which artefacts to emit (comma-separated subset of s1,s4,s5,figure). Default: all.",
    )
    parser.add_argument("--ci-level", type=float, default=0.95)
    args = parser.parse_args()

    requested = {t.strip() for t in args.tables.split(",") if t.strip()}
    valid = {"s1", "s4", "s5", "figure"}
    unknown = requested - valid
    if unknown:
        raise SystemExit(f"unknown table(s): {sorted(unknown)}; valid: {sorted(valid)}")

    domain_csvs = _parse_domain_csvs(args.domain_csvs)
    print(f"loading {len(domain_csvs)} domain CSVs:")
    for k, p in domain_csvs.items():
        print(f"  {k}: {p}")

    # Per-domain rows and per-domain per-model cells
    per_domain_rows: dict[str, list[dict]] = {}
    per_domain: dict[str, dict[str, dict]] = {}
    for domain, path in domain_csvs.items():
        rows = _load_csv(path)
        per_domain_rows[domain] = rows
        per_domain[domain] = {
            model: _model_cell(model_rows)
            for model, model_rows in _per_model_rows(rows).items()
        }
        print(f"  {domain}: {len(rows)} rows, {len(per_domain[domain])} models")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "s1" in requested:
        t = build_table_s1(per_domain, ci_level=args.ci_level)
        out = args.output_dir / "table_s1.csv"
        _write_csv(t, out)
        print(f"wrote {out} ({len(t)} rows)")

    if "s4" in requested:
        t = build_table_s4(per_domain, ci_level=args.ci_level)
        out = args.output_dir / "table_s4.csv"
        _write_csv(t, out)
        print(f"wrote {out} ({len(t)} rows)")

    if "s5" in requested:
        t = build_table_s5(per_domain_rows)
        out = args.output_dir / "table_s5.csv"
        _write_csv(t, out)
        print(f"wrote {out} ({len(t)} rows)")

    if "figure" in requested:
        payload = build_figure_data(per_domain)
        out = args.output_dir / "figure_data.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
