#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(ggplot2)
  library(broom)
})

in_file <- file.path("src", "statistical_validation", "outputs", "final", "consolidated_for_external_analysis.csv")
out_dir <- file.path("src", "statistical_validation", "outputs", "final")
plot_dir <- file.path(out_dir, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(in_file)) {
  stop("Input not found: ", in_file, ". Run final_build_consolidated.R first.")
}

df <- read_csv(in_file, show_col_types = FALSE)

# Group label from auto_subject_name
df <- df %>% mutate(
  set = case_when(
    str_detect(auto_subject_name, regex("retake", ignore_case = TRUE)) ~ "Retake",
    str_detect(auto_subject_name, regex("originals", ignore_case = TRUE)) ~ "originals",
    TRUE ~ NA_character_
  )
)

df <- df %>% filter(!is.na(set)) %>%
  mutate(
    measured_average_fluorescence = as.numeric(measured_average_fluorescence),
    auto_rfp_sum_mean = as.numeric(auto_rfp_sum_mean)
  ) %>%
  filter(is.finite(measured_average_fluorescence), is.finite(auto_rfp_sum_mean))

analyze <- function(data, label) {
  if (nrow(data) < 3) return(NULL)
  pt <- suppressWarnings(cor.test(data$measured_average_fluorescence, data$auto_rfp_sum_mean, method = "pearson"))
  st <- suppressWarnings(cor.test(data$measured_average_fluorescence, data$auto_rfp_sum_mean, method = "spearman"))
  fit <- lm(auto_rfp_sum_mean ~ measured_average_fluorescence, data = data)
  td <- tidy(fit)
  gl <- glance(fit)

  # Plot
  p <- ggplot(data, aes(x = measured_average_fluorescence, y = auto_rfp_sum_mean)) +
    geom_point(alpha = 0.75) +
    geom_smooth(method = "lm", se = FALSE, color = "steelblue") +
    theme_bw(base_size = 12) +
    labs(title = paste0("Auto rfp_sum_mean vs Manual measured avg fluorescence (", label, ")"),
         x = "Manual measured avg fluorescence (μm²-intensity)", y = "Auto rfp_sum_mean")
  fn <- file.path(plot_dir, paste0("scatter_correlation_", gsub("[^A-Za-z0-9_]+", "_", label), ".png"))
  ggsave(fn, p, width = 6, height = 5, dpi = 150)

  tibble(
    set = label,
    n = nrow(data),
    pearson_r = unname(pt$estimate),
    pearson_p = pt$p.value,
    spearman_rho = unname(st$estimate),
    spearman_p = st$p.value,
    lm_intercept = td$estimate[td$term == "(Intercept)"],
    lm_slope = td$estimate[td$term == "measured_average_fluorescence"],
    lm_intercept_p = td$p.value[td$term == "(Intercept)"],
    lm_slope_p = td$p.value[td$term == "measured_average_fluorescence"],
    r_squared = gl$r.squared
  )
}

stats <- df %>% group_by(set) %>% group_modify(~ analyze(.x, unique(.x$set)[1])) %>% bind_rows()

out_csv <- file.path(out_dir, "correlation_by_set_originals_retakes.csv")
write_csv(stats, out_csv)
print(stats)
cat("\nSaved stats to ", out_csv, " and plots to ", plot_dir, "\n")

<<<<<<< HEAD
# Eva subset (auto subject names containing 'Eva') -------------------------
eva_df <- df %>% filter(str_detect(auto_subject_name, regex("Eva", ignore_case = TRUE)))
eva_stats <- analyze(eva_df, label = "Eva")
if (!is.null(eva_stats) && nrow(eva_stats)) {
  eva_csv <- file.path(out_dir, "correlation_eva.csv")
  write_csv(eva_stats, eva_csv)
  print(eva_stats)
  cat("Saved Eva subset stats to ", eva_csv, "\n")
} else {
  cat("Eva subset: not enough rows for correlation (n < 3).\n")
}
=======
>>>>>>> 29b622c (stats start)
