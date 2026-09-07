# Bravura

Repository for _Bravura, a virtual reality-based paradigm for the study of
physical aggression_, under review, 2026.

Bravura is a VR-based buzz-wire competitive task where participants
choose to shock or not shock fictitious opponents across 30 trials. A
variational Bayesian model estimates aggression coefficients from trial-by-trial
decisions. k-means clustering on behavioural features identifies three subtypes
(non-aggressive, reactive, proactive), validated against the PSAP, heart rate
physiology, and an independent replication cohort.

## Installation

### Prerequisites

- [pixi](https://pixi.sh) (conda-based package manager)
- MATLAB R2025b (for computational modelling and physiology extraction only)

### Dependencies

```bash
pixi install              # Python + R dependencies
pixi run post_install     # brms (installed from CRAN)
git submodule update --init --recursive  # MATLAB toolboxes
```

### MATLAB toolboxes

Four external toolboxes are tracked as git submodules in `extern/`:

| Submodule | Purpose |
| --------- | ------- |
| [VBA-toolbox](https://github.com/MBB-team/VBA-toolbox) | Variational Bayesian Analysis (model fitting, BMA) |
| [PhysioNet-Cardiovascular-Signal-Toolbox](https://github.com/Neclow/PhysioNet-Cardiovascular-Signal-Toolbox) | HRV analysis (`bravura` branch) |
| [MarcusVollmer-HRV](https://github.com/MarcusVollmer/HRV) | HRV toolbox |
| [ledalab](https://github.com/ledalab/ledalab) | EDA decomposition (Ledalab) |

Add them to the MATLAB path before running any pipeline scripts:

```matlab
addpath(genpath('src/vba'))
addpath(genpath('extern/VBA-toolbox-master'))
addpath(genpath('extern/PhysioNet-Cardiovascular-Signal-Toolbox'))
addpath(genpath('extern/MarcusVollmer-HRV'))
addpath(genpath('extern/ledalab'))
```

## Usage

See [docs/](docs) for detailed instructions for each pipeline stage. The
pipeline runs in six stages:

1. **Computational modelling** — VBA grid search, Bayesian model averaging,
   simulation recovery (MATLAB)
2. **Clustering** — k-means consensus clustering on behavioural features,
   sensitivity analyses
3. **Behavioural preparation** — reshape shock, PSAP, and latency data for
   modelling
4. **Physiology preparation** — baseline HR, delta HR, HRV components,
   multivariate physio, hormones
5. **Bayesian models** — brms regression models (shocks, PSAP, delta HR,
   latency, cardiac, replication)
6. **Plots** — publication figures

The `data_v2/` directory has its own [README](data_v2/README.md) describing all
data files and their provenance.

## Citation

If you use this code, please cite:

```bibtex
@article{scheidwasser2026bravura,
  title   = {Bravura, a virtual reality-based paradigm for the study of physical aggression},
  author  = {Scheidwasser, Neil and Rodrigues, Jo{\~a}o and Sandi, Carmen},
  year    = {2026},
  note    = {Under review}
}
```
