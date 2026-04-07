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
| `fig1.ipynb` | Paradigm, model fit, provocation effect |
| `fig2.ipynb` | Clustering, PSAP validation, Cohort B replication |
| `fig3.ipynb` | Physiological markers (HR, HRV), replication |

### Bayesian modelling

```bash
pixi run brms_shocks          # Shocks ~ Cluster * opponent
pixi run brms_psap            # PSAP validation (Dirichlet)
pixi run brms_delta_hr        # Delta-HR ~ Cluster * block
pixi run brms_baseline_hr     # Baseline HR by cluster
pixi run brms_physio_cardiac  # Multivariate cardiac (HR + HRV)
pixi run brms_delta_hr_rep    # Replication BFs (BayesRep)
pixi run brms_latency         # Shock latency (lognormal)
```
