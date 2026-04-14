source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

out_dir <- "data/brms/delta_hr"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

df <- read.csv("data/processed/delta_hr_long.csv")

df$Cluster <- factor(df$Cluster, levels = c("Non-aggressive", "Proactive", "Reactive"))
df$block <- factor(df$block, levels = c("1.1", "1.2", "2.1", "2.2"))
df$subject <- factor(df$subject)

cat("Design:\n")
cat("  Subjects:", nlevels(df$subject), "\n")
cat("  Clusters:", levels(df$Cluster), "\n")
cat("  Blocks:", levels(df$block), "\n")
cat("  Observations:", nrow(df), "\n\n")

# ── Formulas ─────────────────────────────────────────────────────────────────
# Mixed ANOVA equivalent: Cluster (between) × block (within) + random intercepts

formula_ri <- delta_hr ~ Cluster * block + (1 | subject)
formula_rs <- delta_hr ~ Cluster * block + (1 + block | subject)

# ── Priors ───────────────────────────────────────────────────────────────────
# Delta HR is in bpm; typical changes are ~5-10 bpm from baseline

informed_priors <- c(
  prior(normal(5, 5), class = "Intercept"),
  prior(normal(0, 5), class = "b"),
  prior(student_t(3, 0, 5), class = "sd"),
  prior(student_t(3, 0, 10), class = "sigma")
)

informed_priors_rs <- c(
  prior(normal(5, 5), class = "Intercept"),
  prior(normal(0, 5), class = "b"),
  prior(student_t(3, 0, 5), class = "sd"),
  prior(student_t(3, 0, 10), class = "sigma"),
  prior(lkj(2), class = "cor")
)

# ── Model registry ──────────────────────────────────────────────────────────

common <- list(
  data = df,
  chains = CHAINS,
  cores = CORES,
  iter = 8000,
  warmup = 4000,
  seed = SEED,
  control = list(adapt_delta = 0.99, max_treedepth = 15)
)

models <- list(
  list(
    name = "fit_gaussian_ri",
    label = "gaussian RI (informed)",
    formula = formula_ri,
    family = gaussian(),
    prior = informed_priors
  ),
  # RS model commented out: overparameterized (4 obs/subject for 4 random
  # effects + correlation matrix), fails to converge. Student-t RI selected.
  # list(
  #   name = "fit_gaussian_rs",
  #   label = "gaussian RS (informed)",
  #   formula = formula_rs,
  #   family = gaussian(),
  #   prior = informed_priors_rs
  # ),
  list(
    name = "fit_student_ri",
    label = "student-t RI (informed)",
    formula = formula_ri,
    family = student(),
    prior = informed_priors
  ),
  list(
    name = "fit_gaussian_ri_default",
    label = "gaussian RI (default)",
    formula = formula_ri,
    family = gaussian(),
    prior = NULL
  )
)

# ── Fit models ───────────────────────────────────────────────────────────────

fits <- list()
for (m in models) {
  model_args <- c(
    list(
      name = m$name,
      out_dir = out_dir,
      formula = m$formula,
      family = m$family
    ),
    common
  )
  if (!is.null(m$prior)) {
    model_args$prior <- m$prior
  }
  fits[[m$name]] <- do.call(fit_or_load, c(model_args, overwrite = OVERWRITE))
}

# ── Model comparison (LOO) ──────────────────────────────────────────────────

loos <- lapply(fits, loo)
comp_loo <- loo_compare(x = loos)

label_map <- setNames(
  sapply(models, `[[`, "label"),
  sapply(models, `[[`, "name")
)

sink(file.path(out_dir, "model_comparison.txt"))
cat("LOO-CV comparison (all models):\n\n")
print(comp_loo)
cat("\n\nModel formulas:\n")
for (m in models) {
  cat(sprintf("  %-30s %s\n", m$name, deparse(formula(fits[[m$name]]))))
}
sink()

# ── Select best model ───────────────────────────────────────────────────────
# RS wins LOO but fails to converge (divergences, Rhat > 1.05) because
# 4 obs/subject cannot support 4 random effects + correlation matrix.
# Student-t RI is second-best by LOO (~70 elpd ahead of Gaussian RI),
# converges cleanly, and is robust to HR outliers while preserving the
# compound-symmetry structure (direct Bayesian mixed ANOVA analogue).

best_name <- "fit_student_ri"
best <- fits[[best_name]]
best_label <- label_map[best_name]
cat("Selected model:", best_label, "\n")

# ── Diagnostics ──────────────────────────────────────────────────────────────

fit_prior <- fit_or_load(
  "fit_prior_best",
  out_dir,
  formula = formula(best),
  data = df,
  family = family(best),
  prior = informed_priors,
  sample_prior = "only",
  chains = CHAINS,
  iter = 12000,
  warmup = 2000,
  seed = SEED,
  overwrite = OVERWRITE
)

save_diagnostics(best, best_label, out_dir, prior_fit = fit_prior)

png(
  file.path(out_dir, "posterior_predictive_check_grouped.png"),
  width = 10,
  height = 4,
  units = "in",
  res = 300
)
pp_check(best, ndraws = 100, type = "dens_overlay_grouped", group = "Cluster")
dev.off()

# ── Save CSV outputs ────────────────────────────────────────────────────────

write.csv(as.data.frame(fixef(best)), file.path(out_dir, "fixed_effects.csv"))
write.csv(
  as.data.frame(ranef(best)$subject),
  file.path(out_dir, "random_effects.csv")
)

# Posterior predicted draws per cell (tidybayes, response scale)
newdata <- expand.grid(
  Cluster = levels(df$Cluster),
  block = levels(df$block)
)

epred_long <- newdata %>%
  add_epred_draws(best, re_formula = NA) %>%
  rename(delta_hr = .epred) %>%
  select(Cluster, block, delta_hr, .draw)

write.csv(
  epred_long,
  file.path(out_dir, "posterior_epred.csv"),
  row.names = FALSE
)

pred_summary <- epred_long %>%
  group_by(Cluster, block) %>%
  summarise(
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

# ── Pairwise contrasts with Bayes Factors (Savage-Dickey) ────────────────────

em_posterior <- emmeans(best, pairwise ~ Cluster | block)
em_prior <- emmeans(fit_prior, pairwise ~ Cluster | block)

bf_results <- bf_table(em_posterior, em_prior)

cat("\nPairwise contrasts (Savage-Dickey BF):\n")
print(bf_results, digits = 3)

write.csv(bf_results, file.path(out_dir, "bayes_factors.csv"), row.names = FALSE)

cat("Done. Outputs saved to", out_dir, "\n")
