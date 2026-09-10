# Data

All data for the Bravura analysis pipeline. Not tracked in git.

Paths are configured in `src/_config.py` (`DEFAULT_DATA_DIR = "data_v2"`).

## Raw data

### `raw/`

Original data.

| Path | Description |
| ---- | ----------- |
| `MatlabEvents/` | Per-subject BioPac event files (`.mat`, one per subject, both cohorts) |
| `Sync_phys/` | Per-subject synchronised physiology files (`.mat`, one per subject, both cohorts) |
| `psap/` | PSAP task data (`.mat`, one per Cohort A subject) |
| `additional.xlsx` | Additional participant metadata (PSAP button presses, questionnaires) |
| `additionalAll.xlsx` | Extended participant metadata (all variables) |
| `Markers legend.xlsx` | BioPac event marker definitions |
| `CortisolData.xlsx` | Salivary cortisol concentrations |
| `VR main-testosterone-september2019-longxlsx.xlsx` | Salivary testosterone concentrations |

### `shared/`

Files shared across both cohorts.

| File | Description |
| ---- | ----------- |
| `AggressionScript.xlsx` | Preprogrammed opponent behaviour (shocks, wins per trial) |
| `trial_events.csv` | Parsed per-trial data from BioPac events (duration, choice, outcome) |

## Per-cohort data

### `cohort_a/`

Cohort A.

#### Task and questionnaire data

| File | Description |
| ---- | ----------- |
| `aggroPerformance.xlsx` | Per-trial shock decisions, latencies, choices (126 subjects x 30 trials) |
| `latPerformance.xlsx` | Clean latency data (124 subjects) |
| `beliefs.xlsx` | Belief ratings per opponent (0-10 scale) |
| `subject_ids.csv` | Subject ID list |
| `outliers.mat` | 12 excluded subject IDs (MATLAB format) |
| `outliers.txt` | 12 excluded subject IDs (text format) |

#### Pipeline-derived behavioural CSVs

Produced by `pixi run prepare_behavior`.

| File | Consumed by |
| ---- | ----------- |
| `shock_long.csv` | `brms_shocks`, `brms_shocks_overview` |
| `psap_ilr.csv` | `brms_psap` |
| `shock_latency_long.csv` | `brms_latency` |

#### VBA model fitting (`vba/`)

Produced by `pipeline/vba/` (MATLAB). See [VBA model outputs](#vba-model-outputs) for details.

| Path | Description |
| ---- | ----------- |
| `vba_input.xlsx` | Transposed input matrix for MATLAB VBA |
| `models/` | Per-grid-point model fits (256 `.mat` files) |
| `bma/bma_results.mat` | BMA group and summary results (`groupResult`, `summaryResult`) |
| `bma/free_energy.mat` | Free energy per model (input to BMA) |
| `vba_posteriors.mat` | Posterior means (muPhi) and covariances (SigmaPhi) per subject |

#### VBA-derived summaries (`vba/`)

| File | Description |
| ---- | ----------- |
| `coefficients.csv` | BMA-averaged model coefficients [Kr1, Krc, Kp, Kwc] per subject |
| `predictions.csv` | Model-predicted P(shock) per trial |
| `decisions.csv` | Actual binary shock decisions (ground truth) |
| `fit_metrics.csv` | R2, accuracy, balanced accuracy, log evidence per subject |
| `free_energy_matrix.csv` | Free energy matrix (subjects x models) |

#### Simulation recovery (`vba/`)

| File | Description |
| ---- | ----------- |
| `corr_preds.mat` | Per-subject parameter correlation matrices and covariance predictions |
| `cov_stats.mat` | Condition numbers and determinants of parameter covariance matrices |

#### Clustering (`clustering/`)

Produced by `pixi run cluster_behavior`.

| File | Description |
| ---- | ----------- |
| `clusters.csv` | Subject cluster assignments |
| `mc_k-means_3.npz` | Consensus clustering label counts (1000 MC draws) |
| `mc_k-means_3_stats.txt` | Consensus clustering summary statistics |
| `grid_k_solver.csv` | k x solver silhouette/gap grid |
| `ablate_k.csv` | Best solver per k with silhouette, gap, gap_diff |
| `ablate_metric.csv` | Silhouette when ablating individual features |

#### Physiology (`physio/`)

| File | Description |
| ---- | ----------- |
| `physPerformance.xlsx` | Per-subject physiological measures (raw) |
| `physPerformanceAll.xlsx` | Per-subject physiological measures (all variables) |
| `hrv_pca_loadings.csv` | Varimax-rotated PCA loadings on 17 HRV features |
| `hormones.csv` | Salivary hormone concentrations per subject |

Pipeline-derived physio CSVs, produced by `pixi run prepare_physio`:

| File | Consumed by |
| ---- | ----------- |
| `baseline_hr.csv` | `brms_baseline_hr` |
| `delta_hr_long.csv` | `brms_delta_hr`, `brms_delta_hr_joint` |
| `physio_cardiac_long.csv` | `brms_physio_cardiac` |

### `cohort_b/`

Cohort B. Same structure as `cohort_a/`, with these differences:

- Beliefs on 0-5 scale (rescaled to 0-10 in pipeline).
- No `psap_ilr.csv` or `shock_latency_long.csv` (Cohort A only).
- No simulation recovery files (`corr_preds.mat`, `cov_stats.mat`).
- No clustering sensitivity files (`grid_k_solver.csv`, `ablate_k.csv`, `ablate_metric.csv`, `mc_k-means_3.npz`). Cohort B is projected onto Cohort A centroids, not independently clustered.
- No `hormones.csv` or `hrv_pca_loadings.csv` (Cohort A only).
- `physio/delta_hr_long.csv` consumed by `brms_delta_hr_joint` for replication.

## VBA model outputs

Each cohort's `vba/models/` directory contains 256 `.mat` files from a grid search over prior standard deviations for the four model parameters.

**Naming convention:** `sd{Kp}sd{Kr1}sd{Krc}sd{Kwc}.mat`, where each value is in {1, 2, 4, 8}.

**Produced by:** `pipeline/vba/grid_search.m`

**BMA outputs** in `vba/bma/`:

- `free_energy.mat`: Free energy per model (input to BMA)
- `bma_results.mat`: BMA group and summary results (`groupResult`, `summaryResult`)

**Produced by:** `pipeline/vba/bma.m`

## Bayesian model outputs

### `brms/`

Each subdirectory corresponds to a brms script in `pipeline/brms/` and stores cached model fits and derived outputs.

| Directory | Script | Model |
| --------- | ------ | ----- |
| `shocks/` | `shocks.R` | Binomial mixed model (shock ~ cluster x opponent) |
| `shocks_overview/` | `shocks_overview.R` | Binomial model (cluster main effect) |
| `psap/` | `psap.R` | Dirichlet regression on PSAP compositions |
| `delta_hr/` | `delta_hr.R` | Student-t RI model (delta-HR ~ cluster x block) |
| `delta_hr_joint/` | `delta_hr_joint.R` | Joint Cohort A + B delta-HR model, per block |
| `baseline_hr/` | `baseline_hr.R` | Baseline HR by cluster |
| `beliefs_overview/` | `beliefs_overview.R` | Belief ratings by cluster |
| `latency/` | `latency.R` | Log-normal shock latency model |
| `physio_cardiac/` | `physio_cardiac.R` | Multivariate cardiac model (HR + HRV RCs) |
| `trial_duration/` | `trial_duration.R` | Trial duration by cluster |

#### `delta_hr_joint/` substructure

Contains per-block subdirectories:

| Subdirectory | Description |
| ------------ | ----------- |
| `block_1_1/` | Joint model for block Op1T1 (initial response) |
| `block_2_1/` | Joint model for block Op2T1 |

Each block subdirectory contains the standard outputs plus `bayes_factors_cluster.csv`, `bayes_factors_per_cohort.csv`, and `interaction_contrasts.csv`.

#### Standard outputs per model

- `fit_*.rds`: Fitted brms model object(s)
- `fit_prior*.rds` / `fit_*_prior.rds`: Prior-only model for prior predictive checks
- `summary.txt`: Model summary (fixed effects, random effects, family, formula)
- `bayes_factors.csv`: Savage-Dickey or pairwise Bayes factors
- `predicted_means.csv`: Posterior predicted means per condition
- `posterior_epred.csv`: Posterior expected predictions (required; used by notebooks for plotting)
- `prior_predictive_check.{png,pdf}`, `posterior_predictive_check.{png,pdf}`: Predictive check plots
- `trace_plots.png`: MCMC trace and density plots

Some models also save `fixed_effects.csv`, `random_effects.csv`, `model_comparison.txt`, `posterior_draws.csv`, or `posterior_predictive_check_grouped.png`.
