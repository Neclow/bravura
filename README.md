# Bravura

Repository for _Bravura: an open immersive virtual reality paradigm for human reactive and proactive aggression_, under review, 2026.

## Installation

### Prerequisites

- pixi (conda-based package manager)
- MATLAB (used version: R2025b; for computational modelling and physiology extraction only)

### Dependencies

```bash
pixi install                             # Python + R dependencies
pixi run post_install                    # brms (installed from CRAN)
git submodule update --init --recursive  # MATLAB toolboxes
```

### MATLAB toolboxes

Four external toolboxes are tracked as git submodules in `extern/`:

| Submodule | Purpose |
| --------- | ------- |
| [VBA-toolbox](https://github.com/MBB-team/VBA-toolbox) | Variational Bayesian Analysis (model fitting, BMA) |
| [PhysioNet-Cardiovascular-Signal-Toolbox](https://github.com/Neclow/PhysioNet-Cardiovascular-Signal-Toolbox) | HRV analysis (`bravura` branch) |
| [MarcusVollmer-HRV](https://github.com/MarcusVollmer/HRV) | HRV toolbox |

Add them to the MATLAB path before running any pipeline scripts:

```matlab
addpath(genpath('src/vba'))
addpath(genpath('extern/VBA-toolbox-master'))
addpath(genpath('extern/PhysioNet-Cardiovascular-Signal-Toolbox'))
addpath(genpath('extern/MarcusVollmer-HRV'))
```

## Usage

See [docs/](docs) for detailed instructions for each pipeline stage. The
pipeline runs in six stages:

1. **Computational modelling:** VBA grid search, Bayesian model averaging,
   simulation recovery (MATLAB)
2. **Clustering:** k-means consensus clustering on behavioural features,
   sensitivity analyses
3. **Behavioural data preparation:** reshape shock, PSAP, and latency data for
   modelling
4. **Physiological data preparation:** baseline HR, delta HR, HRV components,
   multivariate physio
5. **Bayesian models:** brms regression models (shocks, PSAP, delta HR,
   latency, cardiac)
6. **Plots:** publication figures

The `data_v2/` directory has its own [README](data_v2/README.md) describing all
data files and their provenance.

## Citation

If you use this code, please cite:

```bibtex
@article{scheidwasser2026bravura,
  title   = {Bravura: an open immersive virtual reality paradigm for human reactive and proactive aggression},
  author  = {Scheidwasser, Neil and Rodrigues, Jo{\~a}o and Streuber, Stephan and Sandi, Carmen},
  year    = {2026},
  note    = {Under review}
}
```
