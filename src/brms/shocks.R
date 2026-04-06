source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

out_dir <- "data/brms/shocks"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

df <- read.csv("data/processed/shock_long.csv")

formula_ri <- shocks | trials(15) ~ Cluster * opponent + (1 | subject)
formula_rs <- shocks | trials(15) ~ Cluster *
  opponent +
  (1 + opponent | subject)

# ── Priors ───────────────────────────────────────────────────────────────────

informed_priors_v1 <- c(
  prior(normal(-1, 2), class = "Intercept"),
  prior(normal(0, 2), class = "b"),
  prior(normal(0, 1.5), class = "sd")
)

# Tighter intercept based on reference group (Non-aggressive, Opp1): logit(0.067) ≈ -2.6
informed_priors_v2 <- c(
  prior(normal(-2.5, 1), class = "Intercept"),
  prior(normal(0, 2), class = "b"),
  prior(normal(0, 1.5), class = "sd")
)

informed_priors_betabin <- c(
  prior(normal(-2.5, 1), class = "Intercept"),
  prior(normal(0, 2), class = "b"),
  prior(normal(0, 1.5), class = "sd"),
  prior(gamma(1, 0.1), class = "phi")
)

# ── Model registry ──────────────────────────────────────────────────────────

common <- list(
  data = df,
  chains = CHAINS,
  iter = ITER,
  warmup = WARMUP,
  seed = SEED
)

models <- list(
  list(
    name = "fit_binomial",
    label = "binomial (informed v1)",
    formula = formula_ri,
    family = binomial(),
    prior = informed_priors_v1
  ),
  list(
    name = "fit_binomial_v2",
    label = "binomial (informed v2)",
    formula = formula_ri,
    family = binomial(),
    prior = informed_priors_v2
  ),
  list(
    name = "fit_binomial_rs",
    label = "binomial random slopes (informed v2)",
    formula = formula_rs,
    family = binomial(),
    prior = informed_priors_v2
  ),
  list(
    name = "fit_betabinomial",
    label = "beta-binomial (informed v2)",
    formula = formula_ri,
    family = beta_binomial(),
    prior = informed_priors_betabin
  ),
  list(
    name = "fit_betabinomial_default",
    label = "beta-binomial (default)",
    formula = formula_ri,
    family = beta_binomial(),
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

# # ── K-fold CV (optional, ~1h) ──────────────────────────────────────────────
# cat("Running K-fold CV (this may take a while)...\n")
# kfolds <- list()
# for (m in models) {
#   kfolds[[m$name]] <- kfold_or_load(
#     fits[[m$name]], m$name, out_dir, overwrite = OVERWRITE
#   )
# }
# comp_kfold <- loo_compare(x = kfolds)

# Build label lookup from registry
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
cat("\n\nBinomial (informed v2) priors:\n")
prior_summary(fits[["fit_binomial_v2"]])
cat("\n\nBinomial random slopes (informed v2) priors:\n")
prior_summary(fits[["fit_binomial_rs"]])
sink()

# ── Select best model ──────────────────────────────────────────────────────

best_name <- rownames(comp_loo)[1]
best <- fits[[best_name]]
best_label <- label_map[best_name]
cat("Best model (LOO):", best_label, "\n")

# ── Diagnostics ──────────────────────────────────────────────────────────────

fit_prior <- fit_or_load(
  "fit_prior_best",
  out_dir,
  formula = formula(best),
  data = df,
  family = family(best),
  prior = informed_priors_v2,
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
write.csv(
  as.data.frame(as_draws_df(best)),
  file.path(out_dir, "posterior_draws.csv")
)

# Posterior predicted draws per cell (tidybayes, response scale)
newdata <- expand.grid(
  Cluster = unique(df$Cluster),
  opponent = unique(df$opponent)
)

ppe_long <- newdata %>%
  add_epred_draws(best, re_formula = NA) %>%
  rename(shocks = .epred) %>%
  select(Cluster, opponent, shocks, .draw)

write.csv(
  ppe_long,
  file.path(out_dir, "posterior_epred.csv"),
  row.names = FALSE
)

pred_summary <- ppe_long %>%
  group_by(Cluster, opponent) %>%
  summarise(
    mean = mean(shocks),
    Q2.5 = quantile(shocks, 0.025),
    Q97.5 = quantile(shocks, 0.975),
    .groups = "drop"
  )
write.csv(
  pred_summary,
  file.path(out_dir, "predicted_means.csv"),
  row.names = FALSE
)

# ── Pairwise contrasts with Bayes Factors (Savage-Dickey) ────────────────────

em_posterior <- emmeans(best, pairwise ~ Cluster | opponent)
em_prior <- emmeans(fit_prior, pairwise ~ Cluster | opponent)

bf_obj <- bayesfactor_parameters(em_posterior$contrasts, prior = em_prior$contrasts)

# Merge emmeans summary (estimate + CrI) with BFs into one table
contrasts_summary <- as.data.frame(em_posterior$contrasts)
bf_df <- as.data.frame(bf_obj)

bf_table <- contrasts_summary %>%
  mutate(
    BF10 = exp(bf_df$log_BF),
    excl_zero = lower.HPD > 0 | upper.HPD < 0
  )

cat("\nPairwise contrasts (Savage-Dickey BF):\n")
print(bf_table, digits = 3)

write.csv(bf_table, file.path(out_dir, "bayes_factors.csv"), row.names = FALSE)

cat("Done. Outputs saved to", out_dir, "\n")
