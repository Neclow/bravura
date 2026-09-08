#!/usr/bin/env bash
set -euo pipefail

tar -czf data_v2.tar.gz \
    --exclude='data_v2/raw/Sync_phys' \
    --exclude='data_v2/processed/physio_multivariate_long.csv' \
    --exclude='data_v2/brms/physio_multivariate' \
    data_v2/
