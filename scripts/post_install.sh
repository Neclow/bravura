#!/bin/bash

# Install missing R packages
Rscript -e 'install.packages("brms", repos="https://cloud.r-project.org")'
