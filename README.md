# FuzzyGen-BPD

FuzzyGen-BPD is a Streamlit prototype for ranking candidate vulnerability patches by behavioral disturbance. It applies build, proof-of-vulnerability (PoV), and functional-test eligibility gates, then scores eligible candidates with fuzzy TestDelta, CoverageDelta, and DiffSize features.

> **Research status:** The supplied paper reports a synthetic pipeline demonstration and leaves its Vul4J experiments unfinished. Included scores and charts are simulation results, not PatchEval or Vul4J findings (Reference: https://github.com/bytedance/PatchEval). BPD is a ranking heuristic, not a security or correctness proof.

## Try the app online

Open the deployed [FuzzyGen-BPD Streamlit app](https://fuzzygen-bpd-6uuqc6akmrbbcnxp2app9op.streamlit.app/).

## Install and run

Requires Python 3.10 or later. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run fuzzygen_bpd_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Upload candidate data

The app accepts CSV files with these columns:

| Column | Meaning | Values |
|---|---|---|
| `vulnerability_id` | Vulnerability identifier | Non-empty text |
| `candidate_id` | Patch candidate identifier, unique within vulnerability | Non-empty text |
| `TestDelta` | Fraction of legitimate test outcomes that changed | 0 to 1 |
| `CoverageDelta` | Absolute line-coverage ratio change | 0 to 1 |
| `DiffSize` | Added/removed non-comment lines divided by original lines, capped at 1 | 0 to 1 |
| `build_pass` | Build gate result | true/false or 1/0 |
| `pov_pass` | PoV gate result | true/false or 1/0 |
| `functional_pass` | Functional-test gate result | true/false or 1/0 |

Example input: `data/fuzzygen_bpd_example_candidates.csv`. The four rows use the paper's worked feature example and illustrative gate values; they are not benchmark observations.

PatchEval metadata is available as `data/patcheval_verified.csv`. It describes CVEs and repositories but does not include generated candidate patches, gate outcomes, measured BPD features, or independent acceptability labels. Those measurements cannot be inferred from the metadata alone. The included PatchEval source data is from [ByteDance/PatchEval](https://github.com/bytedance/PatchEval) and is distributed under Apache-2.0; see `data/PatchEval/LICENSE`.

## Score and ranking

The app uses the paper's illustrative trapezoidal membership functions and Sugeno output singletons (0.1, 0.5, 0.9):

```text
BPD = wT * F(TestDelta) + wC * F(CoverageDelta) + wD * F(DiffSize)
```

Feature weights must be finite and nonnegative, with at least one positive value; they are normalized to sum to 1. Any candidate that fails one of the three gates remains ineligible. Other candidates are ranked by ascending BPD within their vulnerability. The decision threshold labels disturbance and does not certify a patch.

## Run the evaluation

```bash
python scripts/run_fuzzygen_bpd_pipeline.py
```

This reproduces the paper's synthetic feature/label setup with grouped folds and compares tuned weights against equal weights. Outputs include held-out predictions, fold metrics, weights, confusion matrix, charts, and uncertainty summaries in `reports/fuzzygen_bpd/`.

## Compare optimization methods

```bash
python scripts/compare_fuzzygen_optimizers.py
```

The script compares Random Search, Genetic Algorithm (GA), Tree-structured Parzen Estimator (TPE), Gaussian-process Bayesian optimization (GP-BO), and SMAC-style random-forest expected improvement. Each optimizer gets 60 evaluations per outer fold. An equal-weight BPD baseline is included. The TPE, GP-BO, and SMAC implementations are lightweight approximations; see `optimizer_comparison.json` for details.

## Generate figures

```bash
python scripts/generate_fuzzygen_research_figures.py
```

The Streamlit app's **Research figure collection** includes the framework infographic and generated charts for membership curves, inference, weight effects, optimizer convergence, learned weights, consistency, ablation, feature distributions, BPD/label relationship, rankings, held-out folds, membership sensitivity, and Macro-F1.

## Project map

```text
fuzzygen_bpd_app.py                         Streamlit interface
src/fuzzygen_bpd.py                         Fuzzy membership and BPD scoring
scripts/run_fuzzygen_bpd_pipeline.py        Synthetic grouped evaluation
scripts/compare_fuzzygen_optimizers.py      Five-optimizer comparison
scripts/generate_fuzzygen_research_figures.py Figure generation
scripts/convert_patcheval_json_to_csv.py    PatchEval metadata converter
data/fuzzygen_bpd_example_candidates.csv    Uploadable example
data/patcheval_verified.csv                 PatchEval CVE metadata
reports/fuzzygen_bpd/                      Metrics and generated figures
```

For claims about real vulnerabilities, run candidate patches in the benchmark environments, measure the three features and eligibility gates, and obtain independent acceptability labels. Tune weights and thresholds without using held-out programs. The included simulation only demonstrates code and analysis mechanics.
