#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(stringr)
})

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

ensure_pkg("ggplot2")
ensure_pkg("broom")
suppressPackageStartupMessages({
  library(ggplot2)
  library(broom)
})

safe_name <- function(x) {
  x <- gsub(" ", "_", x)
  x <- gsub("[^A-Za-z0-9_]+", "_", x)
  x <- gsub("_+", "_", x)
  x
}

out_dir <- file.path("outputs", "stats")
plot_dir <- file.path("outputs", "plots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# Load prepared datasets --------------------------------------------------
base_dir <- "."
exact_path <- file.path(base_dir, "outputs", "manual_auto_exact_slice_join.csv")
mouse_path <- file.path(base_dir, "outputs", "manual_auto_by_mouse_join.csv")

if (!file.exists(exact_path) || !file.exists(mouse_path)) {
  stop("Expected inputs not found. Run execute_stats_plan.R first.")
}

df_exact <- read_csv(exact_path, show_col_types = FALSE)
df_mouse <- read_csv(mouse_path, show_col_types = FALSE)

# Utilities ---------------------------------------------------------------

metric_specs_exact <- tibble::tribble(
  ~metric,            ~manual_col,                  ~auto_col,
  "Total Area",       "total_area_mean",            "Total Area",
  "Average Size",     "average_size_mean",          "Average Size",
  "Mean Fluorescence", "mean_intensity_mean",        "Mean",
  # New comparisons per request
  "Manual %Area -> μm² vs Auto Total Area", "manual_area_um2_from_percent", "Total Area",
  "Manual %Area -> μm² vs Auto EffFull Sum", "manual_area_um2_from_percent", "effective_full_intensity_um2_sum",
  "Average Size vs Auto EffFull Mean",        "average_size_mean",          "effective_full_intensity_um2_mean"
)

metric_specs_mouse <- tibble::tribble(
  ~metric,            ~manual_col,                     ~auto_col,
  "Total Area",       "manual_total_area",             "auto_Total Area",
  "Average Size",     "manual_average_size",           "auto_Average Size",
  "Mean Fluorescence", "manual_mean_intensity",         "auto_Mean",
  # New comparisons per request (aggregated by mouse)
  "Manual %Area -> μm² vs Auto Total Area", "manual_area_um2_from_percent", "auto_Total Area",
  "Manual %Area -> μm² vs Auto EffFull Sum", "manual_area_um2_from_percent", "auto_effective_full_intensity_um2_sum",
  "Average Size vs Auto EffFull Mean",        "manual_average_size",         "auto_effective_full_intensity_um2_mean"
)

analyze_pairs <- function(df, metric_specs, id_cols, label) {
  res_list <- list()
  record_list <- list()
  plot_list <- list()

  for (i in seq_len(nrow(metric_specs))) {
    spec <- metric_specs[i, ]
    mcol <- spec$manual_col
    acol <- spec$auto_col
    metric <- spec$metric

    if (!(mcol %in% names(df)) || !(acol %in% names(df))) next

    dat <- df |>
      select(any_of(id_cols), manual = all_of(mcol), auto = all_of(acol)) |>
      filter(is.finite(as.numeric(manual)), is.finite(as.numeric(auto)))

    if (nrow(dat) < 3) next

    dat <- dat |>
      mutate(
        manual = as.numeric(manual),
        auto = as.numeric(auto),
        diff = auto - manual,
        mean_pair = (auto + manual) / 2,
        pct_error = ifelse(manual == 0, NA_real_, 100 * (auto - manual) / manual)
      )

    # Correlations
    pearson <- suppressWarnings(cor.test(dat$auto, dat$manual, method = "pearson"))
    spearman <- suppressWarnings(cor.test(dat$auto, dat$manual, method = "spearman"))

    # Linear regression auto ~ manual
    fit <- lm(auto ~ manual, data = dat)
    fit_tidy <- broom::tidy(fit)
    fit_glance <- broom::glance(fit)

    bias <- mean(dat$diff)
    sd_diff <- sd(dat$diff)
    loa_low <- bias - 1.96 * sd_diff
    loa_high <- bias + 1.96 * sd_diff
    rmse <- sqrt(mean((dat$diff)^2))
    mae <- mean(abs(dat$diff))
    mape <- mean(abs(dat$pct_error), na.rm = TRUE)

    res_list[[metric]] <- tibble::tibble(
      dataset = label,
      metric = metric,
      n = nrow(dat),
      pearson_r = unname(pearson$estimate),
      pearson_p = pearson$p.value,
      spearman_rho = unname(spearman$estimate),
      spearman_p = spearman$p.value,
      lm_intercept = fit_tidy$estimate[fit_tidy$term == "(Intercept)"],
      lm_slope = fit_tidy$estimate[fit_tidy$term == "manual"],
      lm_intercept_p = fit_tidy$p.value[fit_tidy$term == "(Intercept)"],
      lm_slope_p = fit_tidy$p.value[fit_tidy$term == "manual"],
      r_squared = fit_glance$r.squared,
      rmse = rmse,
      mae = mae,
      bias = bias,
      loa_low = loa_low,
      loa_high = loa_high,
      mape = mape
    )

    # Save records per metric
    record_list[[metric]] <- dat |>
      mutate(metric = metric, dataset = label) |>
      relocate(metric, dataset)

    # Plots
    scatter <- ggplot(dat, aes(x = manual, y = auto)) +
      geom_point(alpha = 0.7) +
      geom_abline(slope = 1, intercept = 0, linetype = 2, color = "gray40") +
      geom_smooth(method = "lm", se = FALSE, color = "steelblue") +
      labs(title = paste0(metric, " (", label, ")"), x = "Manual", y = "Automated") +
      theme_bw(base_size = 12)

    ba <- ggplot(dat, aes(x = mean_pair, y = diff)) +
      geom_point(alpha = 0.7) +
      geom_hline(yintercept = bias, color = "steelblue") +
      geom_hline(yintercept = loa_low, linetype = 2, color = "gray40") +
      geom_hline(yintercept = loa_high, linetype = 2, color = "gray40") +
      labs(title = paste0("Bland-Altman: ", metric, " (", label, ")"), x = "Mean of methods", y = "Auto - Manual") +
      theme_bw(base_size = 12)

    mname <- safe_name(metric)
    ggplot2::ggsave(filename = file.path(plot_dir, paste0("scatter_", label, "_", mname, ".png")), plot = scatter, width = 6, height = 5, dpi = 150)
    ggplot2::ggsave(filename = file.path(plot_dir, paste0("bland_altman_", label, "_", mname, ".png")), plot = ba, width = 6, height = 5, dpi = 150)

    # Scaled analysis: align units by robust ratio (median auto/manual)
    ratio <- dat$auto / dat$manual
    ratio <- ratio[is.finite(ratio) & dat$manual != 0]
    if (length(ratio) >= 3) {
      k <- median(ratio, na.rm = TRUE)
      dat_s <- dat |>
        mutate(manual_scaled = manual * k,
               diff = auto - manual_scaled,
               mean_pair = (auto + manual_scaled) / 2,
               pct_error = ifelse(manual_scaled == 0, NA_real_, 100 * (auto - manual_scaled) / manual_scaled))

      pearson_s <- suppressWarnings(cor.test(dat_s$auto, dat_s$manual_scaled, method = "pearson"))
      spearman_s <- suppressWarnings(cor.test(dat_s$auto, dat_s$manual_scaled, method = "spearman"))
      fit_s <- lm(auto ~ manual_scaled, data = dat_s)
      fit_tidy_s <- broom::tidy(fit_s)
      fit_glance_s <- broom::glance(fit_s)

      bias_s <- mean(dat_s$diff)
      sd_diff_s <- sd(dat_s$diff)
      loa_low_s <- bias_s - 1.96 * sd_diff_s
      loa_high_s <- bias_s + 1.96 * sd_diff_s
      rmse_s <- sqrt(mean((dat_s$diff)^2))
      mae_s <- mean(abs(dat_s$diff))
      mape_s <- mean(abs(dat_s$pct_error), na.rm = TRUE)

      res_list[[paste0(metric, "__scaled")]] <- tibble::tibble(
        dataset = paste0(label, "_scaled"),
        metric = metric,
        n = nrow(dat_s),
        scale_k = k,
        pearson_r = unname(pearson_s$estimate),
        pearson_p = pearson_s$p.value,
        spearman_rho = unname(spearman_s$estimate),
        spearman_p = spearman_s$p.value,
        lm_intercept = fit_tidy_s$estimate[fit_tidy_s$term == "(Intercept)"],
        lm_slope = fit_tidy_s$estimate[fit_tidy_s$term == "manual_scaled"],
        lm_intercept_p = fit_tidy_s$p.value[fit_tidy_s$term == "(Intercept)"],
        lm_slope_p = fit_tidy_s$p.value[fit_tidy_s$term == "manual_scaled"],
        r_squared = fit_glance_s$r.squared,
        rmse = rmse_s,
        mae = mae_s,
        bias = bias_s,
        loa_low = loa_low_s,
        loa_high = loa_high_s,
        mape = mape_s
      )

      scatter_s <- ggplot(dat_s, aes(x = manual_scaled, y = auto)) +
        geom_point(alpha = 0.7) +
        geom_abline(slope = 1, intercept = 0, linetype = 2, color = "gray40") +
        geom_smooth(method = "lm", se = FALSE, color = "tomato") +
        labs(title = paste0(metric, " (", label, ": scaled)"), x = "Manual (scaled)", y = "Automated") +
        theme_bw(base_size = 12)
      ba_s <- ggplot(dat_s, aes(x = mean_pair, y = diff)) +
        geom_point(alpha = 0.7) +
        geom_hline(yintercept = bias_s, color = "tomato") +
        geom_hline(yintercept = loa_low_s, linetype = 2, color = "gray40") +
        geom_hline(yintercept = loa_high_s, linetype = 2, color = "gray40") +
        labs(title = paste0("Bland-Altman (scaled): ", metric, " (", label, ")"), x = "Mean of methods", y = "Auto - Manual_scaled") +
        theme_bw(base_size = 12)
      ggplot2::ggsave(filename = file.path(plot_dir, paste0("scatter_", label, "_scaled_", mname, ".png")), plot = scatter_s, width = 6, height = 5, dpi = 150)
      ggplot2::ggsave(filename = file.path(plot_dir, paste0("bland_altman_", label, "_scaled_", mname, ".png")), plot = ba_s, width = 6, height = 5, dpi = 150)
    }
  }

  res <- dplyr::bind_rows(res_list)
  records <- dplyr::bind_rows(record_list)
  list(summary = res, records = records)
}

# Run analyses ------------------------------------------------------------
exact <- analyze_pairs(
  df_exact,
  metric_specs_exact,
  id_cols = c("mouse_id", "slice_name"),
  label = "exact_slice"
)

by_mouse <- analyze_pairs(
  df_mouse,
  metric_specs_mouse,
  id_cols = c("mouse_id"),
  label = "by_mouse"
)

# Write outputs -----------------------------------------------------------

if (nrow(exact$summary) > 0) write_csv(exact$summary, file.path(out_dir, "exact_slice_stats_summary.csv"))
if (nrow(by_mouse$summary) > 0) write_csv(by_mouse$summary, file.path(out_dir, "by_mouse_stats_summary.csv"))

if (!is.null(exact$records) && nrow(exact$records) > 0) write_csv(exact$records, file.path(out_dir, "exact_slice_records.csv"))
if (!is.null(by_mouse$records) && nrow(by_mouse$records) > 0) write_csv(by_mouse$records, file.path(out_dir, "by_mouse_records.csv"))

cat("Comparative statistics computed and saved.\n")
