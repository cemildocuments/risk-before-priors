# Selective Risk without Prior Identification

Reproducible experiments for joint-mass certification of selective classification under label shift.

The implementation certifies the error rate conditional on accepting a prediction, while retaining uncertainty about target class proportions. It includes exact-binomial confidence regions, matched-policy comparisons, singular-channel simulations, and coverage-maximizing randomized acceptance policies.

## Study contents

- Eight attributed UCI datasets, 120 frozen classifier score populations, and original split indices.
- 2,880 main calibration scenarios and 8,640 operating settings, evaluated with nine methods.
- 3,600 independent synthetic calibration replications over singular and nearly singular channels.
- A 40-setting comparison of aggregate-event constraints and cellwise policy optimization.
- Measured outputs, numerical summaries, configuration files, integrity hashes, and verification tests.

## Installation

Use Python 3.12 and the pinned dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The reported environment used Python 3.12.1 on an ARM64 system, eight CPU workers for the main experiment, and 18 GB of shared system memory. The calibration experiments use CPU linear programming. Neural model training uses the MPS accelerator backend when available and otherwise uses CPU computation. Worker counts are configurable.

## Reproduce the statistical results

The frozen score populations allow reproduction of the calibration study directly:

```sh
python run_experiments.py --mode reproduced --reps 3 --workers 8
python run_synthetic.py --reps 100 --workers 4
python run_policy_study.py --workers 4
python analyze_results.py
```

`run_experiments.py` writes one CSV per scenario and an aggregate CSV. A new `--mode` creates a distinct results directory. Existing scenario files in that directory are reused, permitting interrupted sweeps to resume. The published main result files are under `results/main/` and `results/main_all.csv`; `analyze_results.py` summarizes that recorded main run and the synthetic and policy outputs. To summarize a new main run, replace `results/main_all.csv` with its aggregate file, keeping a copy of the original for comparison.

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python verify_reproduction.py
```

The verification command checks all eight raw-data hashes, all 120 score-population probability matrices, and the separation of training, design, and held-out record indices. It recomputes one main scenario per dataset and compares every substantive recorded outcome at relative tolerance `1e-8` and absolute tolerance `1e-9`. Use `--full` to recompute all 2,880 main scenarios. Elapsed-time fields vary with execution conditions and are excluded from outcome comparisons.

## Retrain the classifiers

`train_models.py` uses the included numerical data. To retrain into a fresh score-population directory, move the existing `data/processed/` directory to a backup and create an empty directory of the same name, then run:

```sh
python train_models.py --models lr rf --workers 6
python train_models.py --models mlp --workers 1
```

The models use fixed seeds 42–46. The neural models are trained sequentially on the available accelerator while CPU models can run in independent processes. Cached populations preserve the exact statistical inputs of the reported study. Retrained floating-point scores can differ slightly across accelerator kernels and software versions; repeating the calibration experiments from the cached populations is the direct numerical reproduction route.

## Files

| Path | Purpose |
| --- | --- |
| `src/rbp/core.py` | Clopper–Pearson intervals, sparse joint-mass regions, dual-checked certificates, and comparison selectors. |
| `src/rbp/optimize.py` | Constraint generation for cellwise randomized acceptance and a final optimization-gap check. |
| `train_models.py` | Stratified splitting, model fitting, independent confidence-bin design, and frozen score populations. |
| `run_experiments.py` | Main controlled label-shift experiment. |
| `run_synthetic.py` | Singular and nearly singular three-class channels. |
| `run_policy_study.py` | Aggregate-event ablation and cellwise optimization. |
| `analyze_results.py` | Numerical tables, paired differences, and dataset-level bootstrap summaries. |
| `verify_reproduction.py` | Data integrity, split separation, and seeded outcome reproduction. |
| `data/raw/` | Numerical public dataset inputs. |
| `data/processed/` | Compressed score populations, cell definitions, labels, original split indices, and neural parameters. |
| `data/manifest.json` | Source URLs, dataset DOIs, hashes, sizes, and preprocessing information. |
| `data/DATA_LICENSES.md` | Dataset attribution, source licenses, and transformation notes. |
| `configs/` | Experiment settings and seeds. |
| `results/` | Scenario-level outputs, aggregate results, training records, and verification reports. |
| `tests/` | Tests for interval endpoints, missing source classes, singular channels, dual bounds, dominance, infeasibility, and optimized-policy validity. |
| `CITATION.cff` | Machine-readable citation information. |

## Interpreting the experiment

The main calibration samples are independent draws with replacement from fixed empirical score populations constructed from held-out original records. Class proportions change while the class-conditional score distribution stays fixed. Target coverage and selective risk are evaluated exactly from those populations, which isolates calibration uncertainty from test-set measurement noise. The number of resampled observations is not the number of distinct original records.

Risk limits are 5%, 10%, and 20%, with a 5% calibration failure budget. All-reject policies have zero coverage and undefined selective risk. The three risk limits and nine methods share calibration samples; output rows are paired evaluations, not independent experiments. `matrix` and `box` are explicitly defined selective-risk adaptations of matrix-constraint and coordinate-region ideas. `ridge_plugin` is a regularized simplex least-squares comparison.

The main mean acceptance is 46.92% for joint-mass certification and 45.93% for the matrix-region comparison. The focused policy study increases mean acceptance from 66.34% to 83.18%. No joint-mass certificate in the recorded studies violates its requested risk limit. These empirical findings are conditional on the controlled study; the mathematical guarantee additionally requires the declared label-shift and sampling assumptions.

## License and citation

The software is licensed under MIT. Public datasets retain their separately documented source licenses. Cite the project using `CITATION.cff`, and retain the dataset attributions in `data/DATA_LICENSES.md` when redistributing data.
