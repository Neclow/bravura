source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────
# Population-level opponent effect on opponent belief ratings (0-10).
# Mirrors shocks_overview.R but with belief as outcome.
# Cohort B beliefs assumed already rescaled from 0-5 to 0-10 in behav_Xb.csv.

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

out_dir <- "data/brms/beliefs_overview"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── Data preparation ─────────────────────────────────────────────────────────

load_beliefs <- function(path, cohort_label) {
  raw <- read.csv(path, row.names = 1)
  data.frame(
    subject = rep(rownames(raw), 2),
    cohort = cohort_label,
    opponent = factor(rep(c("Opponent 1", "Opponent 2"), each = nrow(raw))),
    belief = c(raw$belief_opp1, raw$belief_opp2)
  )
}

df <- rbind(
  load_beliefs("data/processed/behav_Xa.csv", "A"),
  load_beliefs("data/processed/behav_Xb.csv", "B")
)
df$cohort <- factor(df$cohort)

cat("N subjects per cohort:\n")
print(df %>% distinct(subject, cohort) %>% count(cohort))
cat("\nMean belief per opponent x cohort:\n")
print(df %>% group_by(cohort, opponent) %>% summarise(
  mean = mean(belief), sd = sd(belief), .groups = "drop"
))

# ── Formula ──────────────────────────────────────────────────────────────────
# Population-level opponent effect, random intercept by subject.

formula <- belief ~ opponent * cohort + (1 | subject)

# ── Priors ───────────────────────────────────────────────────────────────────
# Belief is bounded [0, 10]; centre prior at midpoint with broad scale.
# Student-t family for robustness to boundary clumping.

priors <- c(
  prior(normal(5, 2), class = "Intercept"),
  prior(normal(0, 1), class = "b"),
  prior(student_t(3, 0, 3), class = "sigma"),
  prior(normal(0, 1.5), class = "sd")
)

# ── Fit model ────────────────────────────────────────────────────────────────

fit <- fit_or_load(
  "fit_student",
  out_dir,
  formula = formula,
  family = student(),
  prior = priors,
  data = df,
  chains = CHAINS,
  cores = CORES,
  iter = ITER,
  warmup = WARMUP,
  seed = SEED,
  overwrite = OVERWRITE
)

# ── Prior-only model for Savage-Dickey BF ────────────────────────────────────

fit_prior <- fit_or_load(
  "fit_prior",
  out_dir,
  formula = formula,
  family = student(),
  prior = priors,
  data = df,
  sample_prior = "only",
  chains = CHAINS,
  cores = CORES,
  iter = 12000,
  warmup = 2000,
  seed = SEED,
  overwrite = OVERWRITE
)

# ── Diagnostics ──────────────────────────────────────────────────────────────

save_diagnostics(fit, "student-t RI (opponent * cohort)", out_dir, prior_fit = fit_prior)

# ── Predicted means per opponent ─────────────────────────────────────────────

newdata <- expand.grid(
  opponent = levels(df$opponent),
  cohort = levels(df$cohort)
)

ppe_long <- newdata %>%
  add_epred_draws(fit, re_formula = NA) %>%
  rename(belief = .epred) %>%
  select(opponent, cohort, belief, .draw)

pred_summary <- ppe_long %>%
  group_by(cohort, opponent) %>%
  summarise(
    mean = mean(belief),
    Q2.5 = quantile(belief, 0.025),
    Q97.5 = quantile(belief, 0.975),
    .groups = "drop"
  )

write.csv(pred_summary, file.path(out_dir, "predicted_means.csv"), row.names = FALSE)
write.csv(ppe_long, file.path(out_dir, "posterior_epred.csv"), row.names = FALSE)

# ── Opponent effect BF (Savage-Dickey) ───────────────────────────────────────

em_posterior <- emmeans(fit, pairwise ~ opponent | cohort)
em_prior <- emmeans(fit_prior, pairwise ~ opponent | cohort)

bf_results <- bf_table(em_posterior, em_prior)

cat("\nOpponent contrast (Savage-Dickey BF):\n")
print(bf_results, digits = 3)

write.csv(bf_results, file.path(out_dir, "bayes_factors.csv"), row.names = FALSE)

cat("Done. Outputs saved to", out_dir, "\n")
