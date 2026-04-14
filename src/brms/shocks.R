source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

out_dir <- "data/brms/shocks"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

df <- read.csv("data/processed/shock_long.csv")

# ── Formulas ─────────────────────────────────────────────────────────────────

formula_ri <- shocks | trials(15) ~ Cluster * opponent + (1 | subject)
formula_rs <- shocks | trials(15) ~ Cluster * opponent + (1 + opponent | subject)

# ── Priors ───────────────────────────────────────────────────────────────────
# Intercept: logit scale, N(0, 2) centres on 50% with broad coverage
# Slopes: N(0, 1) weakly informative on logit scale
# SD: half-normal, weakly informative

priors_binomial <- c(
  prior(normal(0, 2), class = "Intercept"),
  prior(normal(0, 1), class = "b"),
  prior(normal(0, 1.5), class = "sd")
)

priors_binomial_rs <- c(
  prior(normal(0, 2), class = "Intercept"),
  prior(normal(0, 1), class = "b"),
  prior(normal(0, 1.5), class = "sd"),
  prior(lkj(2), class = "cor")
)

priors_betabinomial <- c(
  prior(normal(0, 2), class = "Intercept"),
  prior(normal(0, 1), class = "b"),
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
    label = "binomial RI",
    formula = formula_ri,
    family = binomial(),
    prior = priors_binomial
  ),
  list(
    name = "fit_binomial_rs",
    label = "binomial RS",
    formula = formula_rs,
    family = binomial(),
    prior = priors_binomial_rs
  ),
  list(
    name = "fit_betabinomial",
    label = "beta-binomial RI",
    formula = formula_ri,
    family = beta_binomial(),
    prior = priors_betabinomial
  ),
  list(
    name = "fit_betabinomial_default",
    label = "beta-binomial RI (default priors)",
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
  prior = priors_binomial,
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

bf_results <- bf_table(em_posterior, em_prior)

cat("\nPairwise contrasts (Savage-Dickey BF):\n")
print(bf_results, digits = 3)

write.csv(bf_results, file.path(out_dir, "bayes_factors.csv"), row.names = FALSE)

cat("Done. Outputs saved to", out_dir, "\n")
