source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────

out_dir <- "data/brms/psap"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Load PSAP data merged with Bravura clusters (exported from fig2.ipynb)
df <- read.csv("data/processed/psap_long.csv", check.names = FALSE)
names(df)[names(df) == "B-press proportion"] <- "B_press"

# ── Model registry ──────────────────────────────────────────────────────────

common <- list(data = df, chains = CHAINS, iter = ITER, warmup = WARMUP, seed = SEED)

models <- list(
  list(name = "fit_gaussian", label = "Gaussian",
       formula = bf(B_press ~ Cluster * phase),
       family = gaussian(), prior = NULL),
  list(name = "fit_zero_inflated_beta", label = "Zero-inflated Beta",
       formula = bf(B_press ~ Cluster * phase, phi ~ 1, zi ~ 1),
       family = zero_inflated_beta(), prior = NULL)
)

# ── Fit models ───────────────────────────────────────────────────────────────

fits <- list()
for (m in models) {
  args <- c(list(name = m$name, out_dir = out_dir, formula = m$formula, family = m$family), common)
  if (!is.null(m$prior)) args$prior <- m$prior
  fits[[m$name]] <- do.call(fit_or_load, args)
}

# ── Model comparison (LOO) ──────────────────────────────────────────────────

loos <- lapply(fits, loo)
comp_loo <- loo_compare(x = loos)

label_map <- setNames(
  sapply(models, `[[`, "label"),
  sapply(models, `[[`, "name")
)

sink(file.path(out_dir, "model_comparison.txt"))
cat("LOO-CV comparison:\n\n")
print(comp_loo)
cat("\n\nModel formulas:\n")
for (m in models) cat(sprintf("  %-20s %s\n", m$label, deparse(m$formula)))
sink()

# ── Select best model ──────────────────────────────────────────────────────

best_name <- rownames(comp_loo)[1]
best <- fits[[best_name]]
best_label <- label_map[best_name]
cat("Best model (LOO):", best_label, "\n")

# ── Diagnostics ──────────────────────────────────────────────────────────────

save_diagnostics(best, best_label, out_dir)

png(file.path(out_dir, "posterior_predictive_check_grouped.png"),
    width = 10, height = 4, units = "in", res = 300)
pp_check(best, ndraws = 100, type = "dens_overlay_grouped", group = "Cluster")
dev.off()

# ── Save CSV outputs ────────────────────────────────────────────────────────

write.csv(as.data.frame(fixef(best)), file.path(out_dir, "fixed_effects.csv"))
write.csv(as.data.frame(as_draws_df(best)), file.path(out_dir, "posterior_draws.csv"))

# Posterior predicted draws per cell
newdata <- expand.grid(
  Cluster = unique(df$Cluster),
  phase = unique(df$phase)
)

ppe_long <- newdata %>%
  add_epred_draws(best) %>%
  rename(B_press = .epred) %>%
  select(Cluster, phase, B_press, .draw)

write.csv(ppe_long, file.path(out_dir, "posterior_epred.csv"), row.names = FALSE)

pred_summary <- ppe_long %>%
  group_by(Cluster, phase) %>%
  summarise(
    mean = mean(B_press),
    Q2.5 = quantile(B_press, 0.025),
    Q97.5 = quantile(B_press, 0.975),
    .groups = "drop"
  )
write.csv(pred_summary, file.path(out_dir, "predicted_means.csv"), row.names = FALSE)

cat("Done. Outputs saved to", out_dir, "\n")
