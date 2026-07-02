# READ: Robustness through Erasure, Ablation and Diagnosis

A diagnostic framework that tells you whether an LLM-as-Judge actually reads the policy text it was shown, or just answers from pretraining priors. Companion repository to the AIES 2026 submission *"READ: Diagnosing Policy Grounding in LLM Judges via Causal Erasure"* (under review).

This repo ships the framework only: perturbation engine, judge client, metric library, failure-mode classifier, tests, and a paper-reproduction driver. No data, no experiment outputs. The dataset is fetched from HuggingFace on first run.

![Pipeline](docs/figures/pipeline.png)

---

## Quickstart guide

A first-time user, going from a fresh clone to a per-instance failure-mode report.

### 1. Clone and install (~2 min)

```bash
git clone https://github.com/IBM/READ.git
cd READ
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 2. Add at least one API key

```bash
cp env.example .env
```

Edit `.env` and set one of:

```
OPENAI_API_KEY=sk-...           # for --models gpt-4o
ANTHROPIC_API_KEY=sk-ant-...    # for --models claude-sonnet-4-5
```

(The 13 RITS-hosted judges are IBM-internal. External users ignore those.)

### 3. Fetch the dataset (~3 min, ~125 MB)

```bash
python scripts/fetch_data.py
```

This pulls `AI-Secure/PolyGuard` (Kang et al., NeurIPS 2025 D&B) from HuggingFace and converts it into the schema `read-benchmark` reads. Output lives in `data/raw/polyguard_*.jsonl` (gitignored). The script also derives per-domain clause maps (`data/clause_maps/*.json`) from the fetched policy texts; these drive the clause-ablation perturbations. Nothing from upstream is redistributed by this repository.

### 4. Run a benchmark (~5 min on `gpt-4o`)

```bash
read-benchmark \
    --dataset data/raw/polyguard_hr_IBM.jsonl \
    --models gpt-4o \
    --limit 50 --balanced --seed 42 \
    --output results/quickstart.csv
```

Each row in the output CSV is one (instance × perturbation × model) verdict. The script is resumable: if it crashes mid-run, re-running with the same `--output` path picks up where it left off.

### 5. Read a per-instance failure mode

This is the main analysis entry point. Pick the CSV from step 4, a model, and an instance id:

```bash
read-failure-mode results/quickstart.csv \
    --model gpt-4o --instance-id polyguard_hr_IBM_0042
```

```
model     instance_id              ground_truth  y_P       y_0       y_P_minus_rstar  n_others  failure_mode
gpt-4o    polyguard_hr_IBM_0042    VIOLATES      VIOLATES  COMPLIES  COMPLIES         11        Faithful
```

The columns are the verdict tuple the decision tree consumes: `y_P` is the verdict on the full policy, `y_0` on the empty policy, `y_P_minus_rstar` after dropping the violated clause, and `n_others` is how many drop-other-clause perturbations were available. The last column is the failure-mode label.

Drop `--instance-id` to dump every (model, instance_id) pair as TSV; add `--json` for machine-readable output.

### 6. Aggregate report (optional)

For the aggregate statistics used in the paper, computed across the whole CSV:

```bash
read-summarize results/quickstart.csv --bootstrap --failure-modes
```

JSON output: accuracy and per-class F1 with Clopper-Pearson and bootstrap CIs, the five fidelity metrics with CIs, pairwise McNemar's tests, and the four failure-mode distributions split by base verdict.

---

## What it does

READ probes each instance by issuing `k + 2` queries for every (scenario, judge) pair:

1. Full policy: the baseline verdict.
2. Empty policy: the same scenario with no policy in context, which exposes the prior-only verdict.
3. Drop the violated clause `r*`: a faithful judge should flip from VIOLATES to COMPLIES here.
4. Drop each non-violated clause in turn (`k − 1` ablations): a faithful judge should stay at VIOLATES.

The resulting verdict shifts feed five metrics:

| Metric | Reads |
|--------|-------|
| PIR (Policy Influence Rate) | fraction of verdicts that change when the policy is removed |
| CSR (Clause Substitution Rate) | fraction of correct violation verdicts that flip when the *violated* clause is removed |
| ASR (Ablation Specificity Rate) | fraction of correct violation verdicts that stay stable when a *non-violated* clause is removed |
| CCR (Context Consistency Rate) | fraction of correct compliant verdicts that stay stable under any clause removal |
| PC-CSR (Policy-Corrected CSR) | CSR restricted to instances where the judge demonstrably needs the policy (base = VIOLATES, no-policy = COMPLIES) |

The same verdict shifts get routed through a decision tree into four failure modes plus a Degenerate flag:

| Mode | What it means |
|------|---------------|
| Prior-Based | Verdict identical with and without the policy. Judge is policy-independent. |
| Policy-Degraded | The policy is the active cause of the wrong verdict. Would have been right without it. |
| Confused | Policy-reactive but not anchored in the correct clause. |
| Faithful | Correct verdict, specifically anchored to the right clause. |
| Degenerate (flag) | Per-class F1 below 0.05. The classifier collapsed. |

![Failure-mode decision tree](docs/figures/failure_modes.png)

---

## Data

The dataset is PolyGuard (Kang et al., NeurIPS 2025 D&B). Each instance is a `(policy, scenario, ground_truth_label, violated_rule)` tuple where the policy lists rules `R1, R2, …, Rk` and the violated rule is known by construction. PolyGuard covers eight high-stakes domains (social media, HR, finance, law, education, code generation, cybersecurity, general regulation). The fetch script converts the HR, education and social-media subdomains (e.g. IBM HR, Apple HR, AI-for-Education, Reddit moderation) into the schema `read-benchmark` reads; the cyber and code subsets ship no policy text on the HuggingFace mirror and are skipped.

The canonical JSONL contract is documented at `data/schemas/input_schema.json`.

---

## Environment variables

| Variable | Required for | Notes |
|----------|--------------|-------|
| `OPENAI_API_KEY` | `gpt-4o` | Standard OpenAI key. |
| `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` | Standard Anthropic key. |
| `READ_RITS_BASE` | Any RITS-hosted judge | IBM Research Inference Service endpoint URL. IBM-internal access only; external users use the OpenAI / Anthropic models instead. |
| `RITS_API_KEY` | Any RITS-hosted judge | Comma-separated keys are accepted and used in round-robin. |

If a RITS-only model is requested without `READ_RITS_BASE`, the client raises a clear `RuntimeError` that points you back to `gpt-4o` or `claude-sonnet-4-5`.

---

## Repository layout

```
read_toolkit/
    perturbations.py      clause-erasure + format perturbation engine
    llm_client.py         multi-provider judge client (OpenAI, Anthropic, RITS)
    evaluator.py          PIR / CSR / ASR / CCR / PC-CSR + McNemar + pooling helpers
    failure_modes.py      decision tree -> {Prior-Based, Policy-Degraded, Confused, Faithful} + CLI
    stats.py              Clopper-Pearson exact CI + binary bootstrap CI helpers
    summarize_results.py  CLI: benchmark CSV -> compact summary JSON with CIs + failure modes
    polyguard_ingest.py   raw PolyGuard -> READ JSONL converter
    trace_analyzer.py     optional regex / LLM-based reasoning-trace inspection
run_benchmark.py          CLI: JSONL + models -> CSV + summary JSON, parallel and resumable
scripts/
    fetch_data.py                 one-shot PolyGuard fetch from HuggingFace
    reproduce_paper.py            Table 1 / S1 / S4 / S5 + Figure 4/S1 inputs from CSVs
    build_final_plots.py          matplotlib figure rendering for the paper
    extract_motivating_examples.py  finds illustrative per-instance cases
    relabel_csv.py                re-applies the verdict parser to an existing CSV
data/
    schemas/input_schema.json     JSONL contract that run_benchmark.py reads (tracked)
    raw/                          PolyGuard JSONLs after fetch (gitignored)
docs/figures/                     README figures + the script that renders them
pyproject.toml                    package metadata; installs the read-* console commands
.github/workflows/tests.yml       CI runs the test suite on push and pull request
```

The three commands you'll actually use day to day:

```bash
python scripts/fetch_data.py                          # once
read-benchmark --dataset ... --models ... -o ...      # per experiment
read-failure-mode <csv> --model ... --instance-id ... # per instance
```

Everything else under `scripts/` is paper-specific tooling.

---

## Citation

If you use READ, please cite the paper (BibTeX will be updated once the AIES 2026 review concludes):

```bibtex
@misc{kahirov2026read,
  title  = {READ: Diagnosing Policy Grounding in LLM Judges via Causal Erasure},
  author = {Kahirov, Adam},
  year   = {2026},
  note   = {Under review (AIES 2026). Code: https://github.com/IBM/READ}
}
```

And the PolyGuard dataset the benchmark runs on:

```bibtex
@inproceedings{kang2025polyguard,
  title  = {PolyGuard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset},
  author = {Kang, Mintong and Chen, Zhaorun and Xu, Chejian and Zhang, Jiawei and
            Guo, Chengquan and Pan, Minzhou and Revilla, Ivan and Sun, Yu and Li, Bo},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year   = {2025},
  note   = {arXiv:2506.19054. Dataset: huggingface.co/datasets/AI-Secure/PolyGuard}
}
```

---

## Development notes

Parts of this codebase were developed with the assistance of AI coding tools. All code has been human-reviewed and is covered by the offline test suite (`pytest tests/`).

READ is distributed under the Apache License 2.0 (see `LICENSE` and `NOTICE`).
