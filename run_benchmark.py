# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import json

csv.field_size_limit(10 * 1024 * 1024)  # 10 MB; raw_response fields can exceed the default 128 KB
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from read_toolkit.evaluator import (
    ablation_specificity_rate,
    average_flip_rate_by_model,
    average_flip_rate_by_perturbation,
    base_only_accuracy_by_domain,
    base_only_accuracy_by_model,
    base_only_model_classification_metrics,
    baseline_accuracy_by_domain,
    baseline_accuracy_by_model,
    clause_sensitivity_rate_by_domain,
    clause_sensitivity_rate_by_model,
    complies_clause_robustness,
    confusion_counts,
    counterfactual_consistency,
    expected_direction_match_rate,
    model_classification_metrics,
    outcome_consistency,
    policy_corrected_csr,
    policy_influence_rate_by_domain,
    policy_influence_rate_by_model,
)
from read_toolkit.llm_client import DEFAULT_RITS_MODELS, MODELS, RITS_MODELS, JudgeClient
from read_toolkit.perturbations import apply_all, drop_clause, drop_clauses, mask_clause, mask_clauses, parse_policy_clauses, remove_policy

DATASET_PATH = Path("data/raw/test_dataset.jsonl")
DEFAULT_OUTPUT = Path("results/benchmark_run.csv")

CSV_FIELDS = [
    "instance_id",
    "domain",
    "model",
    "perturbation",
    "repeat_id",
    "ground_truth",
    "violated_rules",
    "predicted_label",
    "raw_response",
    "trace",
    "binary_label",
    "logprob",
    "latency_ms",
    "is_flip",
    "seed",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _balanced_sample(
    rows: list[dict[str, Any]], limit: int, rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample up to `limit` records with equal representation of each ground_truth class."""
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(row.get("ground_truth_label", ""), []).append(row)
    for bucket in by_label.values():
        rng.shuffle(bucket)
    n_labels = len(by_label) or 1
    per_label = limit // n_labels
    sampled: list[dict[str, Any]] = []
    for bucket in by_label.values():
        sampled.extend(bucket[:per_label])
    # fill remaining slots from leftover records
    used = set(id(r) for r in sampled)
    remaining = [r for r in rows if id(r) not in used]
    rng.shuffle(remaining)
    sampled.extend(remaining[: limit - len(sampled)])
    rng.shuffle(sampled)
    return sampled


def _filter_records(
    records: list[dict[str, Any]],
    domains: list[str] | None,
    limit: int | None,
    seed: int | None = None,
    balanced: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    filtered = records
    if domains:
        ordered_domains = [d for d in domains if d]
        domain_set = set(ordered_domains)
        filtered = [row for row in filtered if row.get("domain") in domain_set]
        if limit is not None and limit >= 0:
            per_domain: list[dict[str, Any]] = []
            for domain in ordered_domains:
                domain_rows = [row for row in filtered if row.get("domain") == domain]
                if balanced:
                    per_domain.extend(_balanced_sample(domain_rows, limit, rng))
                else:
                    rng.shuffle(domain_rows)
                    per_domain.extend(domain_rows[:limit])
            return per_domain
    if limit is not None and limit >= 0:
        if balanced:
            return _balanced_sample(filtered, limit, rng)
        rng.shuffle(filtered)
        return filtered[:limit]
    return filtered


def _build_clients(
    models: list[str],
    timeout: int,
    retries: int,
) -> dict[str, JudgeClient]:
    return {
        model_key: JudgeClient(model_key=model_key, timeout=timeout, retries=retries)
        for model_key in models
    }


def _judge_and_time(
    client: JudgeClient, policy: str, query: str, temperature: float,
) -> tuple:
    started = time.perf_counter()
    result = client.judge(policy=policy, query=query, temperature=temperature)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return result, latency_ms


def _make_row(
    instance_id: str,
    domain: str,
    model_key: str,
    perturbation: str,
    repeat_id: int,
    ground_truth: str,
    result,
    latency_ms: int,
    is_flip: bool,
    violated_rules: str = "",
    seed: int | None = None,
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "domain": domain,
        "model": model_key,
        "perturbation": perturbation,
        "repeat_id": repeat_id,
        "ground_truth": ground_truth,
        "violated_rules": violated_rules,
        "predicted_label": result.label,
        "raw_response": (result.raw_response or "").replace("\n", "\\n").replace("\r", "\\r"),
        "trace": (result.trace or "").replace("\n", "\\n").replace("\r", "\\r"),
        "binary_label": result.binary_label,
        "logprob": result.logprob if result.logprob is not None else "",
        "latency_ms": latency_ms,
        "is_flip": is_flip,
        "seed": seed,
    }


FAST_MODELS = [
    "gpt-oss-safeguard-20b",
    "granite-guardian-3.3-8b",
    "llama-3-1-8b-instruct",
    "llama-3-3-70b-instruct",
    "llama-guard-3-8b",
]


def run_benchmark(
    dataset_path: Path,
    output_path: Path,
    models: list[str],
    domains: list[str] | None,
    limit: int | None,
    timeout: int,
    retries: int,
    num_repeats: int = 1,
    temperature: float = 0.0,
    seed: int | None = None,
    balanced: bool = False,
    no_mask: bool = False,
    workers: int = 1,
    perturbation_filter: set[str] | None = None,
) -> int:
    # ensure every run has a documented seed for reproducibility
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    print(f"  seed={seed}")

    records = _load_jsonl(dataset_path)
    records = _filter_records(records, domains, limit, seed=seed, balanced=balanced)
    if not records:
        print("no records matched filters", file=sys.stderr)
        return 1

    # diagnostic: show class balance of sampled records
    by_domain_label: dict[str, dict[str, int]] = {}
    for r in records:
        d = r.get("domain", "?")
        gt = r.get("ground_truth_label", "?")
        by_domain_label.setdefault(d, {}).setdefault(gt, 0)
        by_domain_label[d][gt] += 1
    for d, counts in sorted(by_domain_label.items()):
        parts = ", ".join(f"{lbl}={n}" for lbl, n in sorted(counts.items()))
        print(f"  sampled {d}: {parts}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clients = _build_clients(models, timeout=timeout, retries=retries)

    # resume support: load existing rows so we can fill only missing perturbations
    completed_keys: set[tuple[str, str, str, str]] = set()
    existing_rows_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    mode = "w"
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", encoding="utf-8") as existing:
            for row in csv.DictReader(existing):
                key = (row["instance_id"], row["model"], row["perturbation"], row["repeat_id"])
                completed_keys.add(key)
                existing_rows_by_key[key] = row
                output_rows.append(row)
        if completed_keys:
            mode = "a"
            print(f"  resuming: {len(completed_keys)} rows already completed")

    total = len(records) * len(models) * num_repeats
    done = 0

    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()

        for record in records:
            policy = record["policy_base_nl"]
            query = record["test_query_input"]
            ground_truth = record["ground_truth_label"]
            instance_id = record["instance_id"]
            domain = record["domain"]

            # pre-ingested per-instance violated rule IDs (empty for COMPLIES
            # or when the ingestor could not map a rule)
            raw_vr = record.get("violated_rule_ids") or []
            if isinstance(raw_vr, str):
                raw_vr = [r.strip() for r in raw_vr.split(",") if r.strip()]
            vr: list[str] = list(raw_vr)
            vr_str = ",".join(vr)

            # build perturbations: format perturbations + policy removal always,
            # clause ablation only for violated clauses (targeted)
            perturbed = apply_all(policy)
            # filter to requested perturbations if --perturbations is set
            if perturbation_filter:
                perturbed = {k: v for k, v in perturbed.items() if k in perturbation_filter}
            perturbed["remove_policy"] = remove_policy(policy)

            clauses = parse_policy_clauses(policy)
            clause_by_rid = {str(c["rule_id"]): c for c in clauses}

            # ablate every clause for every case (enables ASR computation)
            for clause in clauses:
                cid = int(clause["id"])
                rid = str(clause["rule_id"])
                perturbed[f"drop_clause:{rid}"] = drop_clause(policy, cid)
                if not no_mask:
                    perturbed[f"mask_clause:{rid}"] = mask_clause(policy, cid)

            # combined removal for multi-violation cases (enables CSR on those)
            if len(vr) > 1:
                combo_key = "+".join(sorted(vr))
                combo_ids = [int(clause_by_rid[r]["id"]) for r in sorted(vr) if r in clause_by_rid]
                if len(combo_ids) == len(vr):
                    perturbed[f"drop_clause:{combo_key}"] = drop_clauses(policy, combo_ids)
                    if not no_mask:
                        perturbed[f"mask_clause:{combo_key}"] = mask_clauses(policy, combo_ids)

            for model_key in models:
                client = clients[model_key]

                for repeat in range(num_repeats):
                    repeat_id = str(repeat)
                    expected_perturbations = ["base", *perturbed.keys()]
                    expected_keys = {
                        (instance_id, model_key, pert_name, repeat_id)
                        for pert_name in expected_perturbations
                    }
                    missing_keys = expected_keys - completed_keys
                    if not missing_keys:
                        done += 1
                        continue

                    base_key = (instance_id, model_key, "base", repeat_id)
                    base_row = existing_rows_by_key.get(base_key)
                    has_partial_rows = any(
                        key in completed_keys for key in expected_keys if key != base_key
                    )

                    if base_row is None and has_partial_rows:
                        raise RuntimeError(
                            "cannot safely resume because perturbation rows exist without "
                            f"a baseline row for {instance_id} / {model_key} / repeat {repeat}"
                        )

                    if base_row is None:
                        base_result, base_latency = _judge_and_time(
                            client, policy, query, temperature,
                        )
                        row = _make_row(
                            instance_id, domain, model_key, "base", repeat,
                            ground_truth, base_result, base_latency, False,
                            violated_rules=vr_str, seed=seed,
                        )
                        writer.writerow(row)
                        handle.flush()
                        completed_keys.add(base_key)
                        existing_rows_by_key[base_key] = row
                        output_rows.append(row)
                        base_label = base_result.label
                    else:
                        base_label = base_row["predicted_label"]
                        if has_partial_rows:
                            missing_count = len(missing_keys)
                            print(
                                f"  partial resume: {instance_id} / {model_key} / "
                                f"repeat {repeat} missing {missing_count} rows"
                            )

                    # perturbations, parallel when workers > 1
                    pending_perts = [
                        (pert_name, pert_policy)
                        for pert_name, pert_policy in perturbed.items()
                        if (instance_id, model_key, pert_name, repeat_id) not in completed_keys
                    ]

                    if workers <= 1 or len(pending_perts) <= 1:
                        # sequential path (original behavior)
                        for pert_name, pert_policy in pending_perts:
                            pert_result, pert_latency = _judge_and_time(
                                client, pert_policy, query, temperature,
                            )
                            flipped = pert_result.label != base_label
                            row = _make_row(
                                instance_id, domain, model_key, pert_name, repeat,
                                ground_truth, pert_result, pert_latency, flipped,
                                violated_rules=vr_str, seed=seed,
                            )
                            writer.writerow(row)
                            handle.flush()
                            pert_key = (instance_id, model_key, pert_name, repeat_id)
                            completed_keys.add(pert_key)
                            existing_rows_by_key[pert_key] = row
                            output_rows.append(row)
                    else:
                        # parallel path: submit all perturbations concurrently
                        futures = {}
                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            for pert_name, pert_policy in pending_perts:
                                future = pool.submit(
                                    _judge_and_time, client, pert_policy, query, temperature,
                                )
                                futures[future] = pert_name
                            for future in as_completed(futures):
                                pert_name = futures[future]
                                pert_result, pert_latency = future.result()
                                flipped = pert_result.label != base_label
                                row = _make_row(
                                    instance_id, domain, model_key, pert_name, repeat,
                                    ground_truth, pert_result, pert_latency, flipped,
                                    violated_rules=vr_str, seed=seed,
                                )
                                writer.writerow(row)
                                handle.flush()
                                pert_key = (instance_id, model_key, pert_name, repeat_id)
                                completed_keys.add(pert_key)
                                existing_rows_by_key[pert_key] = row
                                output_rows.append(row)

                    done += 1
                    if done % 5 == 0 or done == total:
                        print(f"  [{done}/{total}] {instance_id} / {model_key} / repeat {repeat}")

    # summary
    summary: dict[str, Any] = {
        "rows": len(output_rows),
        "num_repeats": num_repeats,
        "temperature": temperature,
        "seed": seed,
        # accuracy
        "accuracy_by_model": baseline_accuracy_by_model(output_rows),
        "accuracy_metrics_by_model": model_classification_metrics(output_rows),
        "accuracy_by_domain": baseline_accuracy_by_domain(output_rows),
        "baseline_accuracy_by_model": base_only_accuracy_by_model(output_rows),
        "baseline_metrics_by_model": base_only_model_classification_metrics(output_rows),
        "baseline_accuracy_by_domain": base_only_accuracy_by_domain(output_rows),
        "confusion_counts": confusion_counts(output_rows),
        # robustness
        "average_flip_rate_by_model": average_flip_rate_by_model(output_rows),
        "average_flip_rate_by_perturbation": average_flip_rate_by_perturbation(output_rows),
        # fidelity, Level 1: policy influence
        "policy_influence_rate_by_model": policy_influence_rate_by_model(output_rows),
        "policy_influence_rate_by_domain": policy_influence_rate_by_domain(output_rows),
        # fidelity, Level 2: clause attribution
        "clause_sensitivity_rate_by_model": clause_sensitivity_rate_by_model(output_rows),
        "clause_sensitivity_rate_by_domain": clause_sensitivity_rate_by_domain(output_rows),
        "ablation_specificity_rate_by_model": ablation_specificity_rate(output_rows, "model"),
        "complies_clause_robustness_by_model": complies_clause_robustness(output_rows, "model"),
        "expected_direction_match_rate_by_model": expected_direction_match_rate(output_rows, "model"),
        "counterfactual_consistency_by_model": counterfactual_consistency(output_rows, "model"),
        # fidelity, Level 3: policy-corrected clause sensitivity
        "policy_corrected_csr_by_model": policy_corrected_csr(output_rows, "model"),
    }
    # stochastic reliability (only meaningful with num_repeats > 1)
    if num_repeats > 1:
        summary["outcome_consistency_by_model"] = outcome_consistency(output_rows, "model")

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote results to {output_path}")
    print(f"wrote summary to {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="READ: run robustness and fidelity benchmark against policy-judge models",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", type=str, default=",".join(FAST_MODELS),
                        help="comma-separated model keys (default: 5 diagnostic models)")
    parser.add_argument("--all-models", action="store_true",
                        help="use all 9 RITS models instead of the 5 diagnostic defaults")
    parser.add_argument("--domains", type=str, default="")
    parser.add_argument("--limit", type=int, default=100,
                        help="instances per domain (default: 100; use 200 for full runs)")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--num-repeats", type=int, default=1,
                        help="run each inference K times for stochastic reliability")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sampling temperature (use >0 with --num-repeats for variance)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--balanced", action="store_true", default=True,
                        help="balance VIOLATES/COMPLIES when sampling (default: on)")
    parser.add_argument("--no-balanced", action="store_true",
                        help="disable balanced sampling")
    parser.add_argument("--mask", action="store_true",
                        help="include mask_clause perturbations (off by default for speed)")
    parser.add_argument("--no-mask", action="store_true", default=True,
                        help="skip mask_clause perturbations (default: on)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel API workers per instance (default: 1 = sequential)")
    parser.add_argument("--perturbations", type=str, default="",
                        help="comma-separated perturbation names to run (default: all registered)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all_models:
        models = list(RITS_MODELS)
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None

    unknown_models = [m for m in models if m not in MODELS]
    if unknown_models:
        raise SystemExit(f"error: unknown model keys: {unknown_models} valid: {list(MODELS)}")

    if not args.dataset.exists():
        raise SystemExit(
            f"error: dataset not found: {args.dataset}\n"
            "Run `python scripts/fetch_data.py` first to fetch PolyGuard into "
            "data/raw/ (see README Quickstart)."
        )

    # mask is off by default; --mask turns it on
    use_mask = args.mask
    no_mask = not use_mask

    # --no-balanced overrides default balanced=True
    balanced = args.balanced and not args.no_balanced

    # perturbation filter
    perturbation_filter: set[str] | None = None
    if args.perturbations:
        perturbation_filter = {p.strip() for p in args.perturbations.split(",") if p.strip()}
        print(f"  perturbation filter: {sorted(perturbation_filter)}")

    if no_mask:
        print("  no mask_clause perturbations (use --mask to include)")
    print(f"  models: {models}")
    print(f"  limit={args.limit}, balanced={balanced}, seed={args.seed}")

    try:
        exit_code = run_benchmark(
            dataset_path=args.dataset,
            output_path=args.output,
            models=models,
            domains=domains,
            limit=args.limit,
            timeout=args.timeout,
            retries=args.retries,
            num_repeats=args.num_repeats,
            temperature=args.temperature,
            seed=args.seed,
            balanced=balanced,
            no_mask=no_mask,
            workers=args.workers,
            perturbation_filter=perturbation_filter,
        )
    except (RuntimeError, ValueError) as exc:
        # setup errors the user can fix: missing API key or RITS config, bad dataset row, unsafe resume
        raise SystemExit(f"error: {exc}") from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
