#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(ggplot2)
  library(broom)
})

base_dir <- "."
out_dir <- file.path(base_dir, "outputs", "stats")
plot_dir <- file.path(base_dir, "outputs", "plots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# Inputs
exact_path <- file.path(base_dir, "outputs", "manual_auto_exact_slice_join.csv")
auto_detailed_path <- file.path("..", "..", "results", "simple_dask", "simple_dask_image_summary_detailed.csv")

if (!file.exists(exact_path)) stop("Missing exact join: ", exact_path)
if (!file.exists(auto_detailed_path)) stop("Missing auto detailed CSV: ", auto_detailed_path)

df <- read_csv(exact_path, show_col_types = FALSE)
auto_det <- read_csv(auto_detailed_path, show_col_types = FALSE)

# Prepare rfp_sum_mean column source: prefer existing in df; otherwise join from detailed
has_rfp_sum_in_df <- "rfp_sum_mean" %in% names(df)
if (!has_rfp_sum_in_df) {
  auto_det_small <- auto_det %>% transmute(`subject name` = subject_name, rfp_sum_mean = rfp_sum_mean)
  df <- df %>% left_join(auto_det_small, by = "subject name")
}

# Build set label and collapse duplicates per manual slice by precedence
dat_all <- df %>%
  filter(rating_bool) %>%
  mutate(
    set_raw = subject_meta,
    set = case_when(
      str_detect(set_raw, "^Jej LYZ") ~ "Jej LYZ",
      str_detect(set_raw, "^originals/G2-ABX/Eva Image J - lyz") ~ "originals/G2-ABX/Eva Image J - lyz",
      str_detect(set_raw, "^originals") ~ "originals",
      str_detect(set_raw, "^Retake") ~ "Retake",
      TRUE ~ set_raw
    )
  ) %>%
  filter(set %in% c("originals", "originals/G2-ABX/Eva Image J - lyz", "Retake")) %>%
  mutate(
    manual_integrated = mean_intensity_mean * average_size_mean,
    auto_integrated = rfp_sum_mean
  ) %>%
  filter(is.finite(manual_integrated), is.finite(auto_integrated))

# Precedence: originals > Eva > Retake
dat <- dat_all %>%
  mutate(set_rank = factor(set, levels = c("originals", "originals/G2-ABX/Eva Image J - lyz", "Retake"))) %>%
  arrange(mouse_id, slice_name, set_rank) %>%
  group_by(mouse_id, slice_name) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  select(mouse_id, slice_name, `subject name`, set, manual_integrated, auto_integrated)

analyze_set <- function(df_set, set_name) {
  if (nrow(df_set) < 3) return(NULL)
  pear <- suppressWarnings(cor.test(df_set$auto_integrated, df_set$manual_integrated, method = "pearson"))
  spea <- suppressWarnings(cor.test(df_set$auto_integrated, df_set$manual_integrated, method = "spearman"))
  fit <- lm(auto_integrated ~ manual_integrated, data = df_set)
  td <- tidy(fit)
  gl <- glance(fit)
  bias <- mean(df_set$auto_integrated - df_set$manual_integrated)
  sd_diff <- sd(df_set$auto_integrated - df_set$manual_integrated)
  loa_low <- bias - 1.96 * sd_diff
  loa_high <- bias + 1.96 * sd_diff
  rmse <- sqrt(mean((df_set$auto_integrated - df_set$manual_integrated)^2))
  mae <- mean(abs(df_set$auto_integrated - df_set$manual_integrated))

  # plots
  safe <- function(x) { x %>% gsub("[^A-Za-z0-9_]+", "_", .) %>% gsub("_+", "_", .) }
  scatter <- ggplot(df_set, aes(x = manual_integrated, y = auto_integrated)) +
    geom_point(alpha = 0.75) + geom_smooth(method = "lm", se = FALSE, color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color = "gray40") +
    theme_bw(base_size = 12) + labs(title = paste0("Integrated intensity: ", set_name), x = "Manual: Mean * AvgSize", y = "Auto: rfp_sum_mean")
  ba <- ggplot(df_set, aes(x = (manual_integrated+auto_integrated)/2, y = auto_integrated - manual_integrated)) +
    geom_point(alpha = 0.75) +
    geom_hline(yintercept = bias, color = "steelblue") +
    geom_hline(yintercept = loa_low, linetype = 2, color = "gray40") +
    geom_hline(yintercept = loa_high, linetype = 2, color = "gray40") +
    theme_bw(base_size = 12) + labs(title = paste0("Bland-Altman: ", set_name), x = "Mean of methods", y = "Auto - Manual")
  ggsave(file.path(plot_dir, paste0("scatter_integrated_", safe(set_name), ".png")), scatter, width = 6, height = 5, dpi = 150)
  ggsave(file.path(plot_dir, paste0("bland_altman_integrated_", safe(set_name), ".png")), ba, width = 6, height = 5, dpi = 150)

  tibble(
    set = set_name,
    n = nrow(df_set),
    pearson_r = unname(pear$estimate), pearson_p = pear$p.value,
    spearman_rho = unname(spea$estimate), spearman_p = spea$p.value,
    lm_intercept = td$estimate[td$term == "(Intercept)"],
    lm_slope = td$estimate[td$term == "manual_integrated"],
    lm_intercept_p = td$p.value[td$term == "(Intercept)"],
    lm_slope_p = td$p.value[td$term == "manual_integrated"],
    r_squared = gl$r.squared,
    rmse = rmse, mae = mae, bias = bias, loa_low = loa_low, loa_high = loa_high
  )
}

sets <- sort(unique(dat$set))
stats <- purrr::map_dfr(sets, ~ analyze_set(dat %>% filter(set == .x), .x))
write_csv(dat, file.path(out_dir, "integrated_by_set_records.csv"))
write_csv(stats, file.path(out_dir, "integrated_by_set_stats.csv"))

print(stats)
cat("\nSaved: ", file.path(out_dir, "integrated_by_set_stats.csv"), "\n")
