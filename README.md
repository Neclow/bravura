# Bravura

Analysis code for *"Bravura, a virtual reality-based paradigm for the study of physical aggression"*. Under prep.

## Installation

### Python + R

```bash
pixi install
pixi run post_install
```

### MATLAB (VBA toolbox)

```bash
git submodule add https://github.com/MBB-team/VBA-toolbox.git extern/VBA-toolbox
git submodule update --init
```

```matlab
addpath(genpath('src/vba'))
addpath(genpath('extern/VBA-toolbox'))
```

## VBA model fitting

Requires MATLAB with the VBA toolbox. Run from the repo root:

```matlab
grid_search('a')    % Fit all prior combinations for Cohort A
bma('a')            % Bayesian Model Averaging
export('a')         % Export to data/cohort_a/
grid_search('b')    % Repeat for Cohort B
bma('b')
export('b')
```

## Analysis

### Notebooks

Run in order: `fig1` → `fig2` → `fig3`. Each notebook generates figures to `img/`.

| Notebook | Description |
|----------|-------------|
| `fig1.ipynb` | Paradigm, model fit, provocation effect, identifiability |
| `fig2.ipynb` | Clustering, PSAP validation, Cohort B replication, shock latency |
| `fig3.ipynb` | HR time course, delta-HR by cluster × block, baseline HR, Cohort B replication, cardiac multivariate model |

### Bayesian modelling

All models are fit with [brms](https://paul-buerkner.github.io/brms/) and cached under `data/brms/`. Each script saves fixed effects, Bayes factors, posterior predictions, and diagnostics.

```bash
# Behavioural
pixi run brms_shocks            # Shocks ~ Cluster * opponent (binomial)
pixi run brms_shocks_overview   # Opponent provocation effect (binomial)
pixi run brms_psap              # PSAP validation (Dirichlet)
pixi run brms_latency           # Shock latency ~ Cluster * opponent (lognormal)
pixi run brms_trial_duration    # Trial duration by decision type (Student-t)

# Physiological
pixi run brms_delta_hr          # Delta-HR ~ Cluster * block (Student-t RI)
pixi run brms_baseline_hr       # Baseline HR by cluster (expected null)
pixi run brms_physio_cardiac    # Multivariate cardiac: HR + HRV RC1-3 (Student-t)
pixi run brms_physio_mv         # 6-DV model: HR, HRV, respiration, EDA (Student-t)

# Replication
pixi run brms_delta_hr_rep      # Sceptical Bayes factors via BayesRep
```
