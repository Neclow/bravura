# Clustering

Clusters participants into aggression subtypes based on behavioural features
derived from VBA model coefficients and task performance.

- **Command:** `pixi run cluster_behavior`
- **Script:** `pipeline/cluster_behavior.py`
- **Requires:** VBA export outputs for both cohorts ([1_vba.md](1_vba.md))
  and outlier lists (see below).
- **Inputs:** Per-cohort files in `data_v2/cohort_{a,b}/`:
  `coefficients.csv`, `fit_metrics.csv`, `predictions.csv`, `decisions.csv`,
  `subject_ids.csv`, `vba_posteriors.mat`, `aggroPerformance.xlsx`,
  `beliefs.xlsx`, `outliers.txt`

## Prerequisite: outlier detection

Identifies participants with extreme shock counts AND low belief scores.
Must run before clustering.

- **Command:** `pixi run prepare_behavior_pre`
- **Script:** `pipeline/prepare_behavior_pre.py`
- **Inputs:** Per-cohort `aggroPerformance.xlsx`, `beliefs.xlsx`
- **Outputs:** Per-cohort `outliers.mat` (legacy), `outliers.txt`
- **Thresholds:** `MIN_SHOCKS`, `MAX_SHOCKS`, `MIN_BELIEF` from `src/_config.py`
- **Outputs:** See per-phase tables below.

## CLI arguments

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `-kbest` | `auto` | Override number of clusters (auto = gap statistic) |
| `-sbest` | `auto` | Override solver (auto = highest mean silhouette) |
| `--overwrite` | off | Re-run consensus clustering even if outputs exist |

## Features

8 features per subject, standardised with `sklearn.StandardScaler` (ddof=0):

| Feature | Source |
| ------- | ------ |
| `Kp` | VBA: baseline preference |
| `Kr1` | VBA: immediate retaliation |
| `Krc` | VBA: cumulative provocation |
| `Kwc` | VBA: win-loss response |
| `R2` | VBA: model fit |
| `shock_opp1` | Task: shocks given to opponent 1 |
| `shock_opp2` | Task: shocks given to opponent 2 |
| `first_shock` | Task: first trial with a shock (30 if never) |

Belief scores are excluded from clustering (measurement noise, 15 imputed
participants, Cohort B rescaling) but retained in the output dataframe for
downstream analyses.

## Phase 1: Grid search

Sweeps k in [2, 10] across four solvers (k-means, k-medoids, GMM, HAC).
Picks best solver by highest mean silhouette, optimal k by first gap_diff > 0.

- **Outputs:** `data_v2/processed/sensitivity/grid_k_solver.csv`,
  `best_solver_ablate_k.csv`

## Phase 2: Consensus clustering

Runs `fuzzy_fit_predict` with 1000 Monte Carlo draws (resampling VBA
posterior coefficients from multivariate normals). Consensus labels = argmax of
per-subject label counts. Cohort B is projected onto Cohort A centroids.

Cluster labels are assigned by mean shocks: lowest = Non-aggressive,
highest = Proactive, remaining = Reactive.

PCA (2 components) is fit on Cohort A and applied to both cohorts for
visualisation.

- **Outputs:**
  - `data_v2/processed/behav_Xa.csv`: Cohort A features + cluster labels + PCA
  - `data_v2/processed/behav_Xb.csv`: Cohort B features + cluster labels + PCA
  - `data_v2/processed/mc_consensus_k-means_3.npz`: MC label counts and stability

## Phase 3: Metric ablation

Tests which VBA fit metric (R2, Brier, AUC, etc.) to include as a clustering
feature. Starts with the base 7 features (no metric), adds one metric at a
time, compares silhouette scores.

- **Outputs:** `data_v2/processed/sensitivity/ablate_metric.csv`,
  `ari_metric.csv`
