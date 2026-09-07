# Computational modelling (VBA)

Variational Bayesian Analysis of aggression decisions. Fits a sigmoid
observation model to per-trial shock/no-shock choices, then averages across a
grid of prior specifications via Bayesian Model Averaging.

The observation function (`src/vba/g_Aggression_short.m`) predicts P(shock) as:

```
sigmoid(Kp + Kr1 * ShockedTm1 + Krc * ShockedSum - Kwc * WinSum)
```

Four parameters: Kp (baseline preference), Kr1 (immediate retaliation),
Krc (cumulative provocation), Kwc (win-loss response).

## 1. Grid search

- **Command:** Run in MATLAB: `grid_search('a')` / `grid_search('b')`
- **Script:** `pipeline/vba/grid_search.m`
- **Requires:** VBA toolbox on MATLAB path, `src/vba/` on path.
- **Inputs:** `data_v2/cohort_{a,b}/vba_input.xlsx`
- **Outputs:** `data_v2/cohort_{a,b}/vba_models/sd{Kp}sd{Kr1}sd{Krc}sd{Kwc}.mat`

Sweeps all combinations of prior standard deviations {1, 2, 4, 8} across four
parameters (prior means fixed at 0). Produces 256 `.mat` files per cohort, one
per grid point.

The model-fitting function (`src/vba/modelFit_short.m`) loops over subjects,
calling `VBA_NLStateSpaceModel` with a static observation model (no hidden-state
dynamics).

## 2. Bayesian Model Averaging

- **Command:** Run in MATLAB: `bma('a')` / `bma('b')`
- **Script:** `pipeline/vba/bma.m`
- **Requires:** All grid search outputs.
- **Inputs:** `data_v2/cohort_{a,b}/vba_models/*.mat`
- **Outputs:**
  - `data_v2/cohort_{a,b}/vba_bma/bma_results.mat` — BMA group and summary results
  - `data_v2/cohort_{a,b}/vba_bma/free_energy.mat` — Free energy matrix (subjects x models)

Collects posteriors and free energies from all 256 models, passes them to
`VBA_BMA` for evidence-weighted averaging per subject.

## 3. Export

- **Command:** Run in MATLAB: `export('a')` / `export('b')`
- **Script:** `src/vba/export.m`
- **Requires:** BMA results.
- **Inputs:** `data_v2/cohort_{a,b}/vba_bma/bma_results.mat`
- **Outputs:**
  - `coefficients.csv` — BMA-averaged [Kr1, Krc, Kp, Kwc] per subject
  - `predictions.csv` — Predicted P(shock) per trial
  - `decisions.csv` — Actual binary decisions
  - `fit_metrics.csv` — R2, accuracy, balanced accuracy, log evidence
  - `subject_ids.csv` — Subject IDs
  - `free_energy_matrix.csv` — Free energies (subjects x models)
  - `vba_posteriors.mat` — Posterior means (muPhi) and covariances (SigmaPhi)

All outputs written to `data_v2/cohort_{a,b}/`.

## 4. Simulation recovery

- **Command:** Run in MATLAB: `simulation_recovery('a')` / `simulation_recovery('b')`
- **Script:** `pipeline/vba/simulation_recovery.m`
- **Requires:** `data_v2/cohort_{a,b}/vba_input.xlsx`
- **Outputs:**
  - `data_v2/cohort_{a,b}/cov_stats.mat` — Determinant and condition number per subject
  - `data_v2/cohort_{a,b}/corr_preds.mat` — Recovered parameter correlation and covariance matrices

For each subject, simulates 30 synthetic datasets with known parameters
(drawn from N(0, 3.75^2)), re-fits the model, and checks whether parameters
are recovered. Uses fixed seed (42). Covariance condition numbers and
correlation matrices diagnose parameter identifiability.
