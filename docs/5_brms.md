# Bayesian models (brms)

All Bayesian regression models are fit with
[brms](https://paul-buerkner.github.io/brms/) via R scripts in
`pipeline/brms/`. Shared utilities live in `src/brms/utils.R`.

## Shared settings

Defined in `src/brms/utils.R`:

| Parameter | Value |
| --------- | ----- |
| Seed | 42 |
| Chains | 4 |
| Cores | 4 |
| Iterations | 4000 (unless noted) |
| Warmup | 2000 (unless noted) |

`fit_or_load()` caches model fits as `.rds` files and reloads from cache on
subsequent runs. `save_diagnostics()` writes `summary.txt`, predictive checks,
and trace plots.

## Standard outputs

Every brms script saves at least:

- `posterior_epred.csv`: posterior expected predictions (used by plot scripts)
- `predicted_means.csv`: posterior predicted means per condition
- `bayes_factors.csv`: Savage-Dickey or pairwise BFs
- `summary.txt`: model summary

See the [data README](../data_v2/README.md#brms) for the full output listing.

## Model table

| Command | Script | Input | Formula | Family |
| ------- | ------ | ----- | ------- | ------ |
| `pixi run brms_shocks` | `shocks.R` | `shock_long.csv` | `shocks \| trials(15) ~ Cluster * opponent + (1 \| subject)` | binomial |
| `pixi run brms_shocks_overview` | `shocks_overview.R` | `behav_Xa/Xb.csv` | `shocks \| trials(15) ~ opponent * cohort + (1 \| subject)` | binomial |
| `pixi run brms_psap` | `psap.R` | `psap_ilr.csv` | `cbind(Earn, Steal, Protect) ~ Cluster * phase + (1 \|p\| Subject)` | dirichlet |
| `pixi run brms_baseline_hr` | `baseline_hr.R` | `baseline_hr.csv` | `HR_Pre ~ Cluster` | student |
| `pixi run brms_beliefs_overview` | `beliefs_overview.R` | `behav_Xa/Xb.csv` | `belief ~ opponent * cohort + (1 \| subject)` | student |
| `pixi run brms_delta_hr` | `delta_hr.R` | `delta_hr_long.csv` | `delta_hr ~ Cluster * block + (1 \| subject)` | student |
| `pixi run brms_delta_hr_rep` | `delta_hr_replication.R` | `delta_hr_long.csv`, `delta_hr_long_b.csv` | N/A (BayesRep) | N/A |
| `pixi run brms_latency` | `latency.R` | `shock_latency_long.csv` | `latency ~ Cluster * opponent + (1 \| subject)` | lognormal |
| `pixi run brms_physio_cardiac` | `physio_cardiac.R` | `physio_cardiac_long.csv` | `mvbind(HR, HRV_RC1, HRV_RC2, HRV_RC3) ~ Cluster * block + (1 \|p\| subject)` | student |
| `pixi run brms_trial_duration` | `trial_duration.R` | `trial_events.csv` | `duration ~ decision * cohort + (1 \| subject)` | student |
| `pixi run brms_hormones` | `hormones.R` | `hormones.csv` | `{var} ~ Cluster + Condition` | student |

All outputs are written to `data_v2/brms/{model_name}/`.

## Model details

### shocks.R

Compares four candidate models: binomial RI, binomial RS, beta-binomial RI
(custom priors), and beta-binomial RI (default priors). Trials fixed at 15.
Priors: Intercept ~ N(0, 2), b ~ N(0, 1), sd ~ N(0, 1.5).

### delta_hr.R

Compares Gaussian RI vs Student-t RI (Student-t selected: better LOO, robust
to outliers). Uses 8000 iterations / 4000 warmup with `adapt_delta = 0.99`,
`max_treedepth = 15`. Cluster levels: Non-aggressive, Proactive, Reactive;
block levels: 1.1, 1.2, 2.1, 2.2.

### delta_hr_replication.R

Not a brms model. Uses the BayesRep package (Pawel & Held, 2022) to compute
replication Bayes factors (BFr) and sceptical Bayes factors (BFs) from
Cohort A and B effect sizes. Focuses on blocks 1.1 and 2.1 where Cohort A
showed strong effects. BFr < 1 = replication success.

### physio_cardiac.R

Multivariate model (4 DVs) with `set_rescor(TRUE)` to estimate residual
correlations. Shared correlation structure across random effects via `|p|`
syntax. Uses 8000 iterations / 4000 warmup.

### psap.R

Dirichlet regression on three-part compositions (Earn, Steal, Protect).
Contrasts computed via `pairwise_bf()`, conditioned on phase and button type.
Uses 95% CrIs (not BFs) for contrasts.

### trial_duration.R

Both cohorts combined. Removes extreme durations (>30s). Decisions coded as
Shock, Enlarge, or Nothing. Uses 16000 iterations / 8000 warmup with
`adapt_delta = 0.99`, `max_treedepth = 15`. Explicitly models degrees of
freedom (nu ~ Gamma(2, 0.1)).

### hormones.R

Exploratory models (output to `physio_hormones/`). Fits one Student-t model
per hormone variable (TotalCort, StressChange_corrected, Testo_mean,
TC_ratio). Condition (stress/control) included to marginalise over cortisol
sampling differences. Each variable has tailored prior scales.
