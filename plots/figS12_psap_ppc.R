source("src/brms/utils.R")
library(gridExtra)

pdf(NULL)
out_dir <- "img_v2/fig3"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

fit <- readRDS(file.path(DEFAULT_BRMS_DIR, "psap", "fit_dirichlet.rds"))
df <- read.csv(file.path(DEFAULT_PROCESSED_DIR, "psap_ilr.csv"))

pp <- posterior_predict(fit, ndraws = 100)

pp_long <- list()
for (d in seq_len(nrow(pp))) {
  tmp <- data.frame(
    Cluster = df$Cluster,
    phase = df$phase,
    draw = d
  )
  for (j in seq_len(dim(pp)[3])) {
    tmp[[dimnames(pp)[[3]][j]]] <- pp[d, , j]
  }
  pp_long[[d]] <- tmp
}
pp_df <- do.call(rbind, pp_long) %>%
  pivot_longer(
    cols = c(Earn, Steal, Protect),
    names_to = "button",
    values_to = "value"
  )

obs_long <- df %>%
  pivot_longer(
    cols = c(Earn, Steal, Protect),
    names_to = "button",
    values_to = "value"
  )

cluster_order <- c("Non-aggressive", "Reactive", "Proactive")
button_order <- c("Earn", "Steal", "Protect")

make_panel <- function(cl, btn, show_title, show_ylabel, show_xlabel) {
  pp_sub <- pp_df[pp_df$Cluster == cl & pp_df$button == btn, ]
  obs_sub <- obs_long[obs_long$Cluster == cl & obs_long$button == btn, ]

  p <- ggplot() +
    stat_density(
      data = pp_sub,
      aes(x = value, group = draw),
      geom = "line", position = "identity",
      alpha = 0.05, color = "lightblue", linewidth = 0.3
    ) +
    geom_density(
      data = obs_sub,
      aes(x = value),
      color = "darkblue", linewidth = 0.8
    ) +
    PAPER_THEME

  if (show_title) {
    p <- p + ggtitle(btn)
  } else {
    p <- p + ggtitle(NULL)
  }

  if (show_ylabel) {
    p <- p + labs(y = cl)
  } else {
    p <- p + labs(y = NULL)
  }

  if (show_xlabel) {
    p <- p + labs(x = "Proportion")
  } else {
    p <- p + labs(x = NULL) + theme(axis.text.x = element_blank())
  }

  p
}

plots <- list()
for (i in seq_along(cluster_order)) {
  for (j in seq_along(button_order)) {
    plots[[(i - 1) * 3 + j]] <- make_panel(
      cluster_order[i], button_order[j],
      show_title = (i == 1),
      show_ylabel = (j == 1),
      show_xlabel = (i == 3)
    )
  }
}

grobs <- lapply(plots, ggplotGrob)
max_widths <- do.call(grid::unit.pmax, lapply(grobs, function(g) g$widths))
grobs <- lapply(grobs, function(g) { g$widths <- max_widths; g })
g <- arrangeGrob(grobs = grobs, ncol = 3)
ggsave(file.path(out_dir, "figS12_psap_ppc.png"),
       g, width = 8, height = 6, dpi = 300)
ggsave(file.path(out_dir, "figS12_psap_ppc.pdf"),
       g, width = 8, height = 6, device = cairo_pdf)
cat("Saved figS12_psap_ppc\n")
