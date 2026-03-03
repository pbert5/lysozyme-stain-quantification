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
df_base <- df %>% mutate(
  set = case_when(
    str_detect(auto_subject_name, regex("retake", ignore_case = TRUE)) ~ "Retake",
    str_detect(auto_subject_name, regex("originals", ignore_case = TRUE)) ~ "originals",
    TRUE ~ NA_character_
  )
)

analyze_metric <- function(data, label, y_col, plot_prefix, y_axis_label) {
  if (nrow(data) < 3) return(NULL)
  x_vals <- data$measured_average_fluorescence
  y_vals <- data[[y_col]]
  pt <- suppressWarnings(cor.test(x_vals, y_vals, method = "pearson"))
  st <- suppressWarnings(cor.test(x_vals, y_vals, method = "spearman"))
  fit <- lm(reformulate("measured_average_fluorescence", response = y_col), data = data)
  td <- tidy(fit)
  gl <- glance(fit)

  # Plot
  p <- ggplot(data, aes(x = measured_average_fluorescence, y = .data[[y_col]])) +
    geom_point(alpha = 0.75) +
    geom_smooth(method = "lm", se = FALSE, color = "steelblue") +
    theme_bw(base_size = 12) +
    labs(
      title = paste0("Auto ", y_col, " vs Manual measured avg fluorescence (", label, ")"),
      x = "Manual measured avg fluorescence (μm²-intensity)",
      y = y_axis_label
    )
  fn <- file.path(plot_dir, paste0(plot_prefix, "_", gsub("[^A-Za-z0-9_]+", "_", label), ".png"))
  ggsave(fn, p, width = 6, height = 5, dpi = 150)

  tibble(
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

build_stats <- function(df_input, y_col, plot_prefix, y_label) {
  if (!nrow(df_input) || !("set" %in% names(df_input))) return(tibble())
  groups <- split(df_input, df_input$set)
  rows <- lapply(groups, function(grp) {
    label <- unique(grp$set)[1]
    res <- analyze_metric(grp, label, y_col, plot_prefix, y_label)
    if (is.null(res) || !nrow(res)) return(NULL)
    dplyr::mutate(res, set = label, .before = 1)
  })
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (!length(rows)) return(tibble())
  bind_rows(rows)
}

brown_forsythe_test <- function(values, groups, label) {
  df <- tibble(
    values = as.numeric(values),
    group = as.character(groups)
  ) %>%
    filter(is.finite(values), !is.na(group))
  if (!nrow(df) || dplyr::n_distinct(df$group) < 2) return(tibble())

  med_lookup <- df %>% group_by(group) %>% summarise(med = median(values, na.rm = TRUE), .groups = "drop")
  med_map <- setNames(med_lookup$med, med_lookup$group)
  abs_dev <- abs(df$values - med_map[df$group])
  test <- oneway.test(abs_dev ~ df$group)
  tibble(
    label = label,
    method = "Brown-Forsythe (median-centered)",
    groups = paste(sort(unique(df$group)), collapse = " vs "),
    statistic = unname(test$statistic),
    df1 = unname(test$parameter[1]),
    df2 = unname(test$parameter[2]),
    p_value = test$p.value
  )
}

# Normalized RFP correlations (existing behavior) -------------------------
df_norm <- df_base %>%
  filter(!is.na(set)) %>%
  mutate(
    measured_average_fluorescence = as.numeric(measured_average_fluorescence),
    auto_rfp_sum_mean = as.numeric(auto_rfp_sum_mean)
  ) %>%
  filter(is.finite(measured_average_fluorescence), is.finite(auto_rfp_sum_mean))

stats <- build_stats(df_norm, "auto_rfp_sum_mean", "scatter_correlation", "Auto rfp_sum_mean (normalized)")

out_csv <- file.path(out_dir, "correlation_by_set_originals_retakes.csv")
write_csv(stats, out_csv)
print(stats)
cat("\nSaved normalized-value stats to ", out_csv, " and plots to ", plot_dir, "\n")

# Eva subset (auto subject names containing 'Eva') -------------------------
eva_df <- df_norm %>% filter(str_detect(auto_subject_name, regex("Eva", ignore_case = TRUE)))
eva_stats <- analyze_metric(eva_df, label = "Eva", y_col = "auto_rfp_sum_mean", plot_prefix = "scatter_correlation_eva", y_axis_label = "Auto rfp_sum_mean (normalized)")
if (!is.null(eva_stats) && nrow(eva_stats)) {
  eva_stats <- eva_stats %>% mutate(set = "Eva", .before = 1)
  eva_csv <- file.path(out_dir, "correlation_eva.csv")
  write_csv(eva_stats, eva_csv)
  print(eva_stats)
  cat("Saved Eva subset stats (normalized) to ", eva_csv, "\n")
} else {
  cat("Eva subset (normalized): not enough rows for correlation (n < 3).\n")
}

# Raw (non-normalized) RFP correlations ------------------------------------
if ("auto_rfp_sum_mean_raw" %in% names(df_base)) {
  df_raw <- df_base %>%
    filter(!is.na(set)) %>%
    mutate(
      measured_average_fluorescence = as.numeric(measured_average_fluorescence),
      auto_rfp_sum_mean_raw = as.numeric(auto_rfp_sum_mean_raw)
    ) %>%
    filter(is.finite(measured_average_fluorescence), is.finite(auto_rfp_sum_mean_raw))

  stats_raw <- build_stats(df_raw, "auto_rfp_sum_mean_raw", "scatter_correlation_raw", "Auto rfp_sum_mean_raw (unnormalized)")

  out_csv_raw <- file.path(out_dir, "correlation_by_set_originals_retakes_raw.csv")
  write_csv(stats_raw, out_csv_raw)
  print(stats_raw)
  cat("Saved raw-value stats to ", out_csv_raw, " and plots to ", plot_dir, "\n")

  eva_df_raw <- df_raw %>% filter(str_detect(auto_subject_name, regex("Eva", ignore_case = TRUE)))
  eva_stats_raw <- analyze_metric(
    eva_df_raw,
    label = "Eva",
    y_col = "auto_rfp_sum_mean_raw",
    plot_prefix = "scatter_correlation_raw_eva",
    y_axis_label = "Auto rfp_sum_mean_raw (unnormalized)"
  )
  if (!is.null(eva_stats_raw) && nrow(eva_stats_raw)) {
    eva_stats_raw <- eva_stats_raw %>% mutate(set = "Eva", .before = 1)
    eva_csv_raw <- file.path(out_dir, "correlation_eva_raw.csv")
    write_csv(eva_stats_raw, eva_csv_raw)
    print(eva_stats_raw)
    cat("Saved Eva subset stats (raw) to ", eva_csv_raw, "\n")
  } else {
    cat("Eva subset (raw): not enough rows for correlation (n < 3).\n")
  }

  # Brown-Forsythe variance test: normalized vs raw auto values (Eva subset)
  eva_auto <- df_base %>%
    filter(str_detect(auto_subject_name, regex("Eva", ignore_case = TRUE))) %>%
    transmute(
      auto_rfp_sum_mean = as.numeric(auto_rfp_sum_mean),
      auto_rfp_sum_mean_raw = as.numeric(auto_rfp_sum_mean_raw)
    ) %>%
    filter(is.finite(auto_rfp_sum_mean), is.finite(auto_rfp_sum_mean_raw))

  if (nrow(eva_auto) >= 2) {
    eva_bf <- brown_forsythe_test(
      values = c(eva_auto$auto_rfp_sum_mean, eva_auto$auto_rfp_sum_mean_raw),
      groups = c(rep("normalized", nrow(eva_auto)), rep("raw", nrow(eva_auto))),
      label = "Eva auto: normalized vs raw"
    )
    if (!is.null(eva_bf) && nrow(eva_bf)) {
      bf_csv <- file.path(out_dir, "brown_forsythe_eva_auto_normalized_vs_raw.csv")
      write_csv(eva_bf, bf_csv)
      print(eva_bf)
      cat("Saved Brown-Forsythe variance test (Eva) to ", bf_csv, "\n")
    } else {
      cat("Eva Brown-Forsythe: insufficient data after filtering.\n")
    }
  } else {
    cat("Eva Brown-Forsythe: not enough rows (n < 2).\n")
  }
} else {
  cat("Column auto_rfp_sum_mean_raw missing; skipping raw-value correlations.\n")
}
