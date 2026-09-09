source("src/brms/utils.R")
library(bayesplot)
library(gridExtra)

brms_dir <- file.path(DEFAULT_BRMS_DIR, "physio_cardiac")
img_dir <- "img_v2/fig4"
dir.create(img_dir, showWarnings = FALSE, recursive = TRUE)

fit <- readRDS(file.path(brms_dir, "fit_cardiac_varimax.rds"))

# Student-t heavy tails produce extreme yrep draws that blow out KDE
# bandwidth. Clip yrep to 3x the display range before computing density.
resp_config <- list(
  HR = list(label = expression(Delta * "HR (bpm)"), xlim = c(-20, 50)),
  HRVRC1 = list(label = "HRV RC1 (power)", xlim = c(-4, 4)),
  HRVRC2 = list(label = "HRV RC2 (vagal)", xlim = c(-4, 4)),
  HRVRC3 = list(label = "HRV RC3 (entropy)", xlim = c(-4, 4))
)
resp_to_col <- c(
  HR = "HR",
  HRVRC1 = "HRV_RC1",
  HRVRC2 = "HRV_RC2",
  HRVRC3 = "HRV_RC3"
)

plots <- list()
for (resp in names(resp_config)) {
  cfg <- resp_config[[resp]]
  y <- fit$data[[resp_to_col[[resp]]]]
  yrep <- posterior_predict(fit, resp = resp, ndraws = 100)
  lo <- cfg$xlim[1] * 3
  hi <- cfg$xlim[2] * 3
  yrep[yrep < lo] <- lo
  yrep[yrep > hi] <- hi
  plots[[resp]] <- ppc_dens_overlay(y, yrep) +
    PAPER_THEME +
    labs(x = cfg$label, y = "Density") +
    coord_cartesian(xlim = cfg$xlim)
}

p <- arrangeGrob(
  plots$HR,
  plots$HRVRC1,
  plots$HRVRC2,
  plots$HRVRC3,
  ncol = 2
)
ggsave(
  file.path(brms_dir, "posterior_predictive_check.png"),
  p,
  width = 8,
  height = 6,
  dpi = 300
)
ggsave(
  file.path(brms_dir, "posterior_predictive_check.pdf"),
  p,
  width = 8,
  height = 6,
  device = cairo_pdf
)
ggsave(
  file.path(img_dir, "figS18_mv_cardiac_ppc.png"),
  p,
  width = 8,
  height = 6,
  dpi = 300
)
ggsave(
  file.path(img_dir, "figS18_mv_cardiac_ppc.pdf"),
  p,
  width = 8,
  height = 6,
  device = cairo_pdf
)
cat("Saved figS18_mv_cardiac_ppc\n")
