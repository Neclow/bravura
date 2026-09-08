# Data

All data for the Bravura analysis pipeline. Not tracked in git.

Paths are configured in `src/_config.py` (`DEFAULT_DATA_DIR = "data_v2"`).

## Raw data

### `raw/`

Original data from AggressionProjectDataShare.

| Path | Description |
| ---- | ----------- |
| `MatlabEvents/` | Per-subject BioPac event files (`.mat`, one per Cohort B subject) |
| `psap/` | PSAP task data (`.mat`, one per Cohort A subject) |
| `CortisolData.xlsx` | Salivary cortisol concentrations |
| `VR main-testosterone-september2019-longxlsx.xlsx` | Salivary testosterone concentrations |
| `additional.xlsx` | Additional participant metadata (PSAP button presses, questionnaires) |
| `Markers legend.xlsx` | BioPac event marker definitions |

### `shared/`

Files shared across both cohorts.

| File | Description |
| ---- | ----------- |
| `AggressionScript.xlsx` | Preprogrammed opponent behaviour (shocks, wins per trial) |
| `trial_events.csv` | Parsed per-trial data from BioPac events (duration, choice, outcome) |

## Per-cohort data

### `cohort_a/`

Cohort A (N=126, 114 after exclusion, 112 for physiology).

#### Task and questionnaire data

| File | Description |
| ---- | ----------- |
| `aggroPerformance.xlsx` | Per-trial shock decisions, latencies, choices (126 subjects x 30 trials) |
| `latPerformance.xlsx` | Clean latency data (124 subjects) |
| `beliefs.xlsx` | Belief ratings per opponent (0–10 scale) |
| `physPerformance.xlsx` | Cardiac features (HR, HRV) across experimental blocks |
| `physPerformanceAll.xlsx` | Full physiological features including respiration and EDA (archive) |

#### VBA model fitting

Produced by `pipeline/vba/` (MATLAB). See [VBA model outputs](#vba-model-outputs) for details.

| File | Description |
| ---- | ----------- |
| `vba_input.xlsx` | Transposed input matrix for MATLAB VBA |
| `vba_models/` | Per-grid-point model fits (256 `.mat` files) |
| `vba_bma/` | Bayesian model averaging (`bma_results.mat`, `free_energy.mat`) |
| `vba_posteriors.mat` | Posterior means (muPhi) and covariances (SigmaPhi) per subject |
| `outliers.mat` | 12 excluded subject IDs |

#### VBA-derived summaries

| File | Description |
| ---- | ----------- |
| `coefficients.csv` | BMA-averaged model coefficients [Kr1, Krc, Kp, Kwc] per subject |
| `predictions.csv` | Model-predicted P(shock) per trial |
| `decisions.csv` | Actual binary shock decisions (ground truth) |
| `fit_metrics.csv` | R2, accuracy, balanced accuracy, log evidence per subject |
| `free_energy_matrix.csv` | Free energy matrix (subjects x models) |
| `subject_ids.csv` | Subject ID list |
| `mc_coefs.npz` | Monte Carlo-sampled coefficients (1000 draws x 126 subjects x 4 params) for consensus clustering |

#### Simulation recovery

| File | Description |
| ---- | ----------- |
| `corr_preds.mat` | Per-subject parameter correlation matrices and covariance predictions |
| `cov_stats.mat` | Condition numbers and determinants of parameter covariance matrices |

### `cohort_b/`

Cohort B (N=44, 37 after exclusion). Same structure as `cohort_a/`. Beliefs on 0–5 scale (rescaled to 0–10 in pipeline).

Does not contain `mc_coefs.npz`, `corr_preds.mat`, or `cov_stats.mat` (Cohort B is projected onto Cohort A centroids, not independently clustered).

## VBA model outputs

Each cohort's `vba_models/` directory contains 256 `.mat` files from a grid search over prior standard deviations for the four model parameters.

**Naming convention:** `sd{Kp}sd{Kr1}sd{Krc}sd{Kwc}.mat`, where each value is in {1, 2, 4, 8}.

**Produced by:** `pipeline/vba/grid_search.m`

**BMA outputs** in `vba_bma/`:

- `free_energy.mat`: Free energy per model (input to BMA)
- `bma_results.mat`: BMA group and summary results (`groupResult`, `summaryResult`)

**Produced by:** `pipeline/vba/bma.m`

## Pipeline outputs

### `processed/`

Intermediate CSVs consumed by brms scripts. Each file is produced by one pipeline script and consumed by one or more downstream scripts.

#### Clustering

Produced by `pixi run cluster_behavior`.

| File | Description |
| ---- | ----------- |
| `behav_Xa.csv` | Cohort A behavioural features with cluster labels and PCA coordinates |
| `behav_Xb.csv` | Cohort B behavioural features projected onto Cohort A centroids |
| `mc_consensus_k-means_3.npz` | Consensus clustering label counts (1000 MC draws) |

#### Clustering sensitivity

Produced by `pixi run cluster_behavior`. Saved to `processed/sensitivity/`.

| File | Description |
| ---- | ----------- |
| `grid_k_solver.csv` | k x solver silhouette/gap grid |
| `best_solver_ablate_k.csv` | Best solver per k with silhouette, gap, gap_diff |
| `ablate_metric.csv` | Silhouette when ablating individual features |
| `ari_metric.csv` | ARI stability across VBA fit metrics |

#### Behavioural

Produced by `pixi run prepare_behavior`.

| File | Consumed by |
| ---- | ----------- |
| `shock_long.csv` | `brms_shocks`, `brms_shocks_overview` |
| `psap_ilr.csv` | `brms_psap` |
| `shock_latency_long.csv` | `brms_latency` |

#### Physiological

Produced by `pixi run prepare_physio`.

| File | Consumed by |
| ---- | ----------- |
| `baseline_hr.csv` | `brms_baseline_hr` |
| `delta_hr_long.csv` | `brms_delta_hr` |
| `delta_hr_long_b.csv` | `brms_delta_hr_rep` |
| `physio_cardiac_long.csv` | `brms_physio_cardiac` |
| `hormones.csv` | `brms_hormones` |

## Bayesian model outputs

### `brms/`

Each subdirectory corresponds to a brms script in `pipeline/brms/` and stores cached model fits and derived outputs.

| Directory | Script | Model |
| --------- | ------ | ----- |
| `shocks/` | `shocks.R` | Binomial mixed model (shock ~ cluster x opponent) |
| `shocks_overview/` | `shocks_overview.R` | Binomial model (cluster main effect) |
| `psap/` | `psap.R` | Dirichlet regression on PSAP compositions |
| `delta_hr/` | `delta_hr.R` | Student-t RI model (delta-HR ~ cluster x block) |
| `baseline_hr/` | `baseline_hr.R` | Baseline HR by cluster |
| `beliefs_overview/` | `beliefs_overview.R` | Belief ratings by cluster |
| `latency/` | `latency.R` | Log-normal shock latency model |
| `physio_cardiac/` | `physio_cardiac.R` | Multivariate cardiac model (HR + HRV RCs) |
| `trial_duration/` | `trial_duration.R` | Trial duration by cluster |
| `physio_hormones/` | `hormones.R` | Cortisol/testosterone models |

#### Standard outputs per model

- `fit_*.rds`: Fitted brms model object(s)
- `fit_prior*.rds`: Prior-only model for prior predictive checks
- `summary.txt`: Model summary (fixed effects, random effects, family, formula)
- `bayes_factors.csv`: Savage-Dickey or pairwise Bayes factors
- `predicted_means.csv`: Posterior predicted means per condition
- `posterior_epred.csv`: Posterior expected predictions (required; used by notebooks for plotting)
- `prior_predictive_check.png`, `posterior_predictive_check.png`: Predictive check plots
- `trace_plots.png`: MCMC trace and density plots

Some models also save `fixed_effects.csv`, `random_effects.csv`, `model_comparison.txt`, or `posterior_draws.csv`.

#### `physio_hormones/` subdirectories

| Subdirectory | Outcome |
| ------------ | ------- |
| `TotalCort/` | Total cortisol by cluster |
| `Testo_mean/` | Mean testosterone by cluster |
| `StressChange_corrected/` | Stress-corrected cortisol change by cluster |
| `TC_ratio/` | Testosterone/cortisol ratio by cluster |
