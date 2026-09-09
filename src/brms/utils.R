library(brms)
library(tidybayes)
library(dplyr)
library(tidyr)
library(emmeans)
library(bayestestR)
library(ggplot2)

# ── Theme (matches .matplotlib/paper.mplstyle) ─────────────────────────────

PAPER_THEME <- theme_minimal(base_size = 9, base_family = "Arial") +
  theme(
    text = element_text(face = "bold"),
    axis.text = element_text(face = "bold"),
    legend.text = element_text(face = "bold"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(linewidth = 0.4),
    legend.background = element_blank()
  )

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR <- "data_v2"
DEFAULT_BRMS_DIR <- file.path(DEFAULT_DATA_DIR, "brms")
DEFAULT_PROCESSED_DIR <- file.path(DEFAULT_DATA_DIR, "processed")

SEED <- 42
CHAINS <- 4
CORES <- 4
options(mc.cores = CORES)
ITER <- 4000
WARMUP <- 2000

# ── Helpers ──────────────────────────────────────────────────────────────────

fit_or_load <- function(name, out_dir, ..., overwrite = FALSE) {
  path <- file.path(out_dir, paste0(name, ".rds"))
  if (file.exists(path) && !overwrite) {
    cat("Loading cached model:", name, "\n")
    readRDS(path)
  } else {
    cat("Fitting model:", name, "\n")
    fit <- brm(...)
    saveRDS(fit, path)
    fit
  }
}

kfold_or_load <- function(fit, name, out_dir, K = 10, overwrite = FALSE) {
  path <- file.path(out_dir, paste0(name, "_kfold.rds"))
  if (file.exists(path) && !overwrite) {
    cat("Loading cached kfold:", name, "\n")
    readRDS(path)
  } else {
    cat("Computing kfold:", name, "\n")
    kf <- kfold(fit, K = K)
    saveRDS(kf, path)
    kf
  }
}

save_diagnostics <- function(best, best_label, out_dir, prior_fit = NULL,
                             ppc_labs = NULL, ppc_xlim = NULL,
                             ppc_group = NULL, ppc_width = 6) {
  sink(file.path(out_dir, "summary.txt"))
  cat(paste0("Model: ", best_label, "\n\n"))
  print(summary(best))
  cat("\n\nPrior summary:\n")
  print(prior_summary(best))
  sink()

  ppc_extras <- list(PAPER_THEME)
  if (!is.null(ppc_labs))  ppc_extras <- c(ppc_extras, list(ppc_labs))
  if (!is.null(ppc_xlim))  ppc_extras <- c(ppc_extras, list(coord_cartesian(xlim = ppc_xlim)))

  ppc_type <- if (!is.null(ppc_group)) "dens_overlay_grouped" else "dens_overlay"
  ppc_args <- list(object = best, ndraws = 100, type = ppc_type)
  if (!is.null(ppc_group)) ppc_args$group <- ppc_group
  if (!is.null(ppc_group)) {
    ppc_extras <- c(ppc_extras, list(theme(panel.spacing = unit(1, "lines"))))
  }

  if (!is.null(prior_fit)) {
    tryCatch({
      p <- pp_check(prior_fit, ndraws = 100) + PAPER_THEME
      ggsave(file.path(out_dir, "prior_predictive_check.png"),
             p, width = 6, height = 4, dpi = 300)
      ggsave(file.path(out_dir, "prior_predictive_check.pdf"),
             p, width = 6, height = 4, device = cairo_pdf)
    }, error = function(e) {
      cat("Skipping prior pp_check:", conditionMessage(e), "\n")
    })
  }

  tryCatch({
    p <- do.call(pp_check, ppc_args)
    for (layer in ppc_extras) p <- p + layer
    ggsave(file.path(out_dir, "posterior_predictive_check.png"),
           p, width = ppc_width, height = 4, dpi = 300)
    ggsave(file.path(out_dir, "posterior_predictive_check.pdf"),
           p, width = ppc_width, height = 4, device = cairo_pdf)
  }, error = function(e) {
    cat("Skipping posterior pp_check:", conditionMessage(e), "\n")
  })

  png(
    file.path(out_dir, "trace_plots.png"),
    width = 10, height = 8, units = "in", res = 300
  )
  print(plot(best))
  dev.off()
}

contrasts_eti <- function(em_contrasts) {
  # Return emmeans contrasts with equal-tailed 95% CrI instead of HPD
  df <- as.data.frame(summary(em_contrasts, ci.method = "quantile"))
  names(df)[names(df) == "lower.QL"] <- "Q2.5"
  names(df)[names(df) == "upper.QL"] <- "Q97.5"
  df
}

bf_table <- function(em_posterior, em_prior) {
  # Build a contrast table with equal-tailed 95% CrI and Savage-Dickey BFs
  contrasts <- contrasts_eti(em_posterior$contrasts)
  bf_df <- as.data.frame(
    bayesfactor_parameters(em_posterior$contrasts, prior = em_prior$contrasts)
  )
  contrasts %>%
    mutate(
      BF10 = exp(bf_df$log_BF),
      excl_zero = Q2.5 > 0 | Q97.5 < 0
    )
}

# ── Pairwise BF table from posterior draws ──────────────────────────────────
# epred: data.frame with columns for grouping vars, a value column, and .draw
# group_col: name of the column with group labels (e.g., "Cluster")
# value_col: name of the column with posterior draws (e.g., "proportion")
# by_cols: character vector of conditioning variables (e.g., c("phase", "button"))

pairwise_bf <- function(epred, group_col, value_col, by_cols, pairs = NULL) {
  groups <- sort(unique(epred[[group_col]]))
  if (is.null(pairs)) {
    pairs <- combn(groups, 2, simplify = FALSE)
  }

  # All unique combinations of the conditioning variables
  by_grid <- epred %>%
    distinct(across(all_of(by_cols)))

  bf_rows <- list()
  for (i in seq_len(nrow(by_grid))) {
    by_vals <- by_grid[i, , drop = FALSE]
    for (pair in pairs) {
      mask1 <- epred[[group_col]] == pair[1]
      mask2 <- epred[[group_col]] == pair[2]
      for (col in by_cols) {
        mask1 <- mask1 & epred[[col]] == by_vals[[col]]
        mask2 <- mask2 & epred[[col]] == by_vals[[col]]
      }

      draws1 <- epred[[value_col]][mask1]
      draws2 <- epred[[value_col]][mask2]

      # Align on draw index (assumes same number of draws per cell)
      diff_draws <- draws1 - draws2
      p_pos <- mean(diff_draws > 0)
      p_neg <- mean(diff_draws < 0)
      bf10 <- if (min(p_pos, p_neg) > 0) max(p_pos, p_neg) / min(p_pos, p_neg) else Inf

      row <- by_vals
      row$contrast <- paste0(pair[1], " - ", pair[2])
      row$mean_diff <- mean(diff_draws)
      row$Q2.5 <- as.numeric(quantile(diff_draws, 0.025))
      row$Q97.5 <- as.numeric(quantile(diff_draws, 0.975))
      row$BF10 <- round(bf10, 1)
      row$excl_zero <- row$Q2.5 > 0 | row$Q97.5 < 0

      bf_rows <- c(bf_rows, list(row))
    }
  }

  do.call(rbind, bf_rows)
}
