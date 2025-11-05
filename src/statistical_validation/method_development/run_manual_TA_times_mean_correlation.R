#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(ggplot2)
  library(broom)
  library(tidyr)
})

base_dir <- "."
out_dir <- file.path(base_dir, "outputs", "stats")
plot_dir <- file.path(base_dir, "outputs", "plots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# Inputs produced by execute_stats_plan.R
manual_records_path <- file.path(base_dir, "outputs", "manual_slice_records.csv")
exact_join_path <- file.path(base_dir, "outputs", "manual_auto_exact_slice_join.csv")

if (!file.exists(manual_records_path) || !file.exists(exact_join_path)) {
  stop("Missing required inputs. Run execute_stats_plan.R first.")
}

manual_rec <- read_csv(manual_records_path, show_col_types = FALSE)
exact <- read_csv(exact_join_path, show_col_types = FALSE)

# Compute per-replicate (crypt) integrated intensity = Total Area * Mean
# Then collapse to slice-level by averaging across replicates for that slice
manual_slice_integrated <- manual_rec %>%
  mutate(
    total_area = suppressWarnings(as.numeric(total_area)),
    mean_intensity = suppressWarnings(as.numeric(mean_intensity)),
    integrated = total_area * mean_intensity
  ) %>%
  group_by(mouse_id, slice) %>%
  summarise(
    manual_integrated_mean = mean(integrated, na.rm = TRUE),
    manual_integrated_sd = sd(integrated, na.rm = TRUE),
    manual_integrated_n = sum(is.finite(integrated)),
    .groups = "drop"
  )

# Join to exact slice pairs, use auto rfp_sum_mean for comparison; keep rating flag
dat <- exact %>%
  select(mouse_id, slice_name, `subject name`, image_source_type, subject_meta, rating_bool, rfp_sum_mean) %>%
  rename(slice = slice_name, auto_integrated = rfp_sum_mean) %>%
  left_join(manual_slice_integrated, by = c("mouse_id", "slice")) %>%
  filter(rating_bool) %>%
  filter(is.finite(manual_integrated_mean), is.finite(auto_integrated))

# Overall correlation ------------------------------------------------------
analyze <- function(df, label) {
  if (nrow(df) < 3) return(NULL)
  pear <- suppressWarnings(cor.test(df$auto_integrated, df$manual_integrated_mean, method = "pearson"))
  spea <- suppressWarnings(cor.test(df$auto_integrated, df$manual_integrated_mean, method = "spearman"))
  fit <- lm(auto_integrated ~ manual_integrated_mean, data = df)
  td <- tidy(fit)
  gl <- glance(fit)
  diff <- df$auto_integrated - df$manual_integrated_mean
  bias <- mean(diff)
  sd_diff <- sd(diff)
  loa_low <- bias - 1.96 * sd_diff
  loa_high <- bias + 1.96 * sd_diff

  sc <- ggplot(df, aes(x = manual_integrated_mean, y = auto_integrated)) +
    geom_point(alpha = 0.75) +
    geom_smooth(method = "lm", se = FALSE, color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color = "gray40") +
    theme_bw(base_size = 12) +
    labs(title = paste0("Manual (TA*Mean) vs Auto rfp_sum_mean: ", label), x = "Manual TA*Mean (slice mean)", y = "Auto rfp_sum_mean")
  ba <- ggplot(df, aes(x = (manual_integrated_mean + auto_integrated)/2, y = diff)) +
    geom_point(alpha = 0.75) +
    geom_hline(yintercept = bias, color = "steelblue") +
    geom_hline(yintercept = loa_low, linetype = 2, color = "gray40") +
    geom_hline(yintercept = loa_high, linetype = 2, color = "gray40") +
    theme_bw(base_size = 12) +
    labs(title = paste0("Bland-Altman: ", label), x = "Mean of methods", y = "Auto - Manual")

  fname_lab <- gsub("[^A-Za-z0-9_]+", "_", label)
  ggsave(file.path(plot_dir, paste0("scatter_manual_TAmean_vs_auto_", fname_lab, ".png")), sc, width = 6, height = 5, dpi = 150)
  ggsave(file.path(plot_dir, paste0("bland_altman_manual_TAmean_vs_auto_", fname_lab, ".png")), ba, width = 6, height = 5, dpi = 150)

  tibble(
    label = label,
    n = nrow(df),
    pearson_r = unname(pear$estimate), pearson_p = pear$p.value,
    spearman_rho = unname(spea$estimate), spearman_p = spea$p.value,
    lm_intercept = td$estimate[td$term == "(Intercept)"],
    lm_slope = td$estimate[td$term == "manual_integrated_mean"],
    lm_intercept_p = td$p.value[td$term == "(Intercept)"],
    lm_slope_p = td$p.value[td$term == "manual_integrated_mean"],
    r_squared = gl$r.squared,
    bias = bias,
    loa_low = loa_low,
    loa_high = loa_high
  )
}

overall <- analyze(dat, "overall")

# By set label (subject_meta
dat_sets <- dat %>% mutate(set = case_when(
  str_detect(subject_meta, "^Jej LYZ") ~ "Jej LYZ",
  str_detect(subject_meta, "^originals/G2-ABX/Eva Image J - lyz") ~ "originals/G2-ABX/Eva Image J - lyz",
  str_detect(subject_meta, "^originals") ~ "originals",
  str_detect(subject_meta, "^Retake") ~ "Retake",
  TRUE ~ subject_meta
))
by_set <- dat_sets %>% group_by(set) %>% group_modify(~ analyze(.x, unique(.x$set)[1])) %>% bind_rows()

stats <- bind_rows(overall, by_set)
readr::write_csv(dat, file.path(out_dir, "manual_TAmean_vs_auto_records.csv"))
readr::write_csv(stats, file.path(out_dir, "manual_TAmean_vs_auto_stats.csv"))
print(stats)
cat("\nSaved: ", file.path(out_dir, "manual_TAmean_vs_auto_stats.csv"), "\n")

