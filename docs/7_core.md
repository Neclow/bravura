# Core library

Shared Python and R modules in `src/`. MATLAB source code for VBA and physiology
lives in subdirectories alongside its documentation.

## Python modules

### `src/_config.py`

Project-wide constants: data paths, clustering parameters, colour palettes,
outlier thresholds, and cluster definitions.

| Constant | Value |
| -------- | ----- |
| `DEFAULT_DATA_DIR` | `"data_v2"` |
| `DEFAULT_BRMS_DIR` | `"data_v2/brms"` |
| `DEFAULT_PROCESSED_DIR` | `"data_v2/processed"` |
| `RANDOM_SEED` | 42 |
| `DEFAULT_CLUSTERING_FEATURES` | Kp, Kr1, Krc, Kwc, R2, shock_opp1, shock_opp2, first_shock |
| `CLUSTER_NAMES` | {0: Reactive, 1: Non-aggressive, 2: Proactive} |
| `PALETTE` | {0: "#de8f05", 1: "#0173b2", 2: "#029e73"} |

### `src/preprocessing.py`

Utilities for building the behavioural feature matrix from VBA outputs and
questionnaire data.

| Function | Description |
| -------- | ----------- |
| `collect_metrics()` | Merge VBA fit metrics with per-subject AUC and Brier score |
| `load_behavioral_features()` | Build combined feature DataFrame from VBA coefficients, fit metrics, shock data, and beliefs |
| `sample_behavioral_features()` | Generator yielding MC samples by resampling VBA posterior coefficients from multivariate normals |
| `detect_outliers()` | Identify subjects with extreme shock counts AND low belief scores |
| `impute_missing()` | Fill `first_shock` NaN with 30; IterativeImputer (BayesianRidge) for belief columns |

### `src/cluster2.py`

Clustering algorithms and evaluation metrics.

| Function / Class | Description |
| ---------------- | ----------- |
| `CLUSTERERS` | Registry: k-means, k-medoids, GMM, HAC |
| `fit_predict()` | Fit a clusterer, return `ClusterResult` (labels, sizes) |
| `fuzzy_fit_predict()` | MC consensus clustering over paired sample streams for two cohorts |
| `ablate_k()` | Sweep k, compute gap statistic + silhouette per k |
| `ablate_solver()` | Compare solvers at fixed k |
| `ablate_X()` | Compare feature sets at fixed k and solver |
| `gap_score()` | Gap statistic (original log or "star" variant) |
| `confidence_ellipse()` | Draw 2D confidence ellipse on a matplotlib axis |

### `src/events.py`

Parse trial events from BioPac MatlabEvents files. Ports the canonical MATLAB
extraction logic from `aggression_choices_biopac.m`.

| Function | Description |
| -------- | ----------- |
| `parse_events()` | Extract 30 per-trial decisions from one `.mat` file |
| `load_all_trial_events()` | Parse all participants in a directory, return DataFrame |

Marker codes: 101 (experiment start), 107 (choice open), 108 (shock),
109 (ring), 110 (win), 111 (lose). Includes manual patches for subjects P102,
BF326, BF060 with anomalous event streams.

## R utilities

### `src/brms/utils.R`

Shared helpers for brms scripts. See [5_brms.md](5_brms.md) for usage.

| Function | Description |
| -------- | ----------- |
| `fit_or_load()` | Cache-aware `brm()` wrapper (saves/loads `.rds`) |
| `kfold_or_load()` | Cache-aware K-fold cross-validation |
| `save_diagnostics()` | Write summary, predictive checks, trace plots |
| `contrasts_eti()` | Equal-tailed 95% CrIs for emmeans contrasts |
| `bf_table()` | Contrast table with CrIs and Savage-Dickey BFs |
| `pairwise_bf()` | Pairwise BFs from posterior prediction draws |

## MATLAB source

### `src/vba/`

| File | Description |
| ---- | ----------- |
| `g_Aggression_short.m` | Observation function: sigmoid P(shock) from 4 parameters |
| `modelFit_short.m` | Fit VBA model for all subjects given prior means and SDs |
| `export.m` | Export BMA results to CSV and `.mat` for Python |

### `src/physio/`

| File | Description |
| ---- | ----------- |
| `My_Main_HRV_Analysis.m` | HRV analysis (PhysioNet Cardiovascular Signal Toolbox) |
| `My_InitializeHRVparams.m` | HRV parameter initialisation |

### `src/psap/`

Psychtoolbox task code for the PSAP concurrent-validity paradigm.

| File | Description |
| ---- | ----------- |
| `Run_Experiment.m` | Task script: earn (A), steal+protect (B), protect (C) |
| `PSAPResponses.m` | Score `.mat` files into proactive/reactive button proportions |
| `Functions/` | 16 Psychtoolbox helper functions |
