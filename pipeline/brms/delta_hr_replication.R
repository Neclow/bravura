source("src/brms/utils.R")

# Joint replication models for delta HR: Cohort A (discovery) and Cohort B
# (replication) in a single Student-t model, one per block (1.1 and 2.1).
#
# Model: delta_hr ~ Cluster * cohort, family = student()
#
# The Cluster:cohort interaction tests whether cluster effects differ between
# cohorts. A null interaction = consistent pattern across cohorts.

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

base_dir <- file.path(DEFAULT_BRMS_DIR, "delta_hr_replication")
dir.create(base_dir, showWarnings = FALSE, recursive = TRUE)

CLUSTER_LEVELS <- c("Non-aggressive", "Proactive", "Reactive")
BLOCKS <- c(1.1, 2.1)

# Load data

df_a <- read.csv(file.path(DEFAULT_PROCESSED_DIR, "delta_hr_long.csv"))
df_b <- read.csv(file.path(DEFAULT_PROCESSED_DIR, "delta_hr_long_b.csv"))

# Matching delta_hr.R priors (scaled for single-block delta HR in bpm)

formula_joint <- delta_hr ~ Cluster * cohort

priors <- c(
  prior(normal(5, 5), class = "Intercept"),
  prior(normal(0, 5), class = "b"),
  prior(student_t(3, 0, 10), class = "sigma")
)

fit_block <- function(blk) {
  blk_label <- gsub("\\.", "_", as.character(blk))
  out_dir <- file.path(base_dir, paste0("block_", blk_label))
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  cat("\n\n##########  Block", blk, " ##########\n\n")

  d_a <- df_a[df_a$block == blk, c("subject", "Cluster", "delta_hr")]
  d_a$cohort <- "A"
  d_b <- df_b[df_b$block == blk, c("subject", "Cluster", "delta_hr")]
  d_b$cohort <- "B"

  df <- rbind(d_a, d_b)
  df$Cluster <- factor(df$Cluster, levels = CLUSTER_LEVELS)
  df$cohort <- factor(df$cohort, levels = c("A", "B"))

  cat("Design:\n")
  cat("  Cohort A:", sum(df$cohort == "A"), "subjects\n")
  cat("  Cohort B:", sum(df$cohort == "B"), "subjects\n")
  cat("  Per cell:\n")
  print(table(df$Cluster, df$cohort))
  cat("\n")

  # Fit model

  fit <- fit_or_load(
    paste0("fit_joint_", blk_label),
    out_dir,
    formula = formula_joint,
    data = df,
    family = student(),
    prior = priors,
    chains = CHAINS,
    cores = CORES,
    iter = ITER,
    warmup = WARMUP,
    seed = SEED,
    overwrite = OVERWRITE
  )

  cat("Fixed effects:\n")
  print(round(fixef(fit), 3))

  # Prior-only fit for Savage-Dickey BFs

  fit_prior <- fit_or_load(
    paste0("fit_joint_", blk_label, "_prior"),
    out_dir,
    formula = formula_joint,
    data = df,
    family = student(),
    prior = priors,
    sample_prior = "only",
    chains = CHAINS,
    cores = CORES,
    iter = 12000,
    warmup = 2000,
    seed = SEED,
    overwrite = OVERWRITE
  )

  # Pairwise cluster contrasts (pooled across cohorts)

  em_post <- emmeans(fit, pairwise ~ Cluster)
  em_prior <- emmeans(fit_prior, pairwise ~ Cluster)
  bf_cluster <- bf_table(em_post, em_prior)

  cat("\nCluster contrasts (pooled across cohorts):\n")
  print(
    bf_cluster[, c("contrast", "estimate", "Q2.5", "Q97.5", "BF10",
                    "excl_zero")],
    digits = 3
  )

  write.csv(
    bf_cluster,
    file.path(out_dir, "bayes_factors_cluster.csv"),
    row.names = FALSE
  )

  # Cluster × cohort interaction contrasts

  em_post_int <- emmeans(fit, pairwise ~ Cluster | cohort)
  em_prior_int <- emmeans(fit_prior, pairwise ~ Cluster | cohort)
  bf_per_cohort <- bf_table(em_post_int, em_prior_int)

  cat("\nCluster contrasts per cohort:\n")
  print(
    bf_per_cohort[, c("contrast", "cohort", "estimate", "Q2.5", "Q97.5",
                       "excl_zero")],
    digits = 3
  )

  write.csv(
    bf_per_cohort,
    file.path(out_dir, "bayes_factors_per_cohort.csv"),
    row.names = FALSE
  )

  # Interaction: difference of cluster differences across cohorts

  em_int <- emmeans(fit, pairwise ~ Cluster * cohort)
  interaction_draws <- contrast(
    em_int[[1]],
    interaction = c(Cluster = "pairwise", cohort = "pairwise")
  )
  int_summary <- summary(interaction_draws, point.est = mean)
  int_df <- as.data.frame(int_summary)

  cat("\nCluster x cohort interactions:\n")
  print(int_df, digits = 3)

  write.csv(
    int_df,
    file.path(out_dir, "interaction_contrasts.csv"),
    row.names = FALSE
  )

  # Posterior predicted draws

  newdata <- expand.grid(
    Cluster = levels(df$Cluster),
    cohort = levels(df$cohort)
  )

  epred_long <- newdata |>
    add_epred_draws(fit, re_formula = NA) |>
    dplyr::rename(delta_hr = .epred) |>
    dplyr::select(Cluster, cohort, delta_hr, .draw)

  write.csv(
    epred_long,
    file.path(out_dir, "posterior_epred.csv"),
    row.names = FALSE
  )

  pred_summary <- epred_long |>
    dplyr::group_by(Cluster, cohort) |>
    dplyr::summarise(
      mean = mean(delta_hr),
      Q2.5 = quantile(delta_hr, 0.025),
      Q97.5 = quantile(delta_hr, 0.975),
      .groups = "drop"
    )
  write.csv(
    pred_summary,
    file.path(out_dir, "predicted_means.csv"),
    row.names = FALSE
  )

  cat("\nPredicted means:\n")
  print(pred_summary)

  # Diagnostics

  blk_expr <- if (blk == 1.1) {
    expression(Delta[1] * "HR (bpm)")
  } else {
    expression(Delta * "HR block 2.1 (bpm)")
  }

  save_diagnostics(
    fit,
    paste("student-t delta_hr ~ Cluster * cohort | block", blk),
    out_dir,
    prior_fit = fit_prior,
    ppc_labs = labs(x = blk_expr, y = "Density"),
    ppc_xlim = c(-20, 40),
    ppc_group = "Cluster",
    ppc_width = 8
  )

  # Effect sizes (Cohen's d) for reference

  cat("\nDescriptive effect sizes (Cohen's d):\n")
  for (coh in c("A", "B")) {
    cat(sprintf("\n  Cohort %s:\n", coh))
    dsub <- df[df$cohort == coh, ]
    clusters <- levels(df$Cluster)
    pairs <- combn(clusters, 2)
    for (i in seq_len(ncol(pairs))) {
      c1 <- pairs[1, i]
      c2 <- pairs[2, i]
      x1 <- dsub$delta_hr[dsub$Cluster == c1]
      x2 <- dsub$delta_hr[dsub$Cluster == c2]
      pooled_sd <- sqrt(
        ((length(x1) - 1) * var(x1) + (length(x2) - 1) * var(x2)) /
          (length(x1) + length(x2) - 2)
      )
      d <- (mean(x2) - mean(x1)) / pooled_sd
      cat(sprintf(
        "    %s - %s: d = %.2f (n = %d vs %d)\n",
        c2, c1, d, length(x2), length(x1)
      ))
    }
  }

  cat("\nDone block", blk, "-> ", out_dir, "\n")
}

for (blk in BLOCKS) {
  fit_block(blk)
}

cat("\nAll replication models complete.\n")
