#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(ggplot2)
  library(broom)
  library(purrr)
})

in_file <- file.path("src", "statistical_validation", "outputs", "final", "consolidated_for_external_analysis.csv")
out_dir <- file.path("src", "statistical_validation", "outputs", "final")
plot_dir <- file.path(out_dir, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

stopifnot(file.exists(in_file))
base <- read_csv(in_file, show_col_types = FALSE)

# Label sets from auto subject name
add_set <- function(df) {
  df %>% mutate(
    set = case_when(
      str_detect(auto_subject_name, regex("retake", ignore_case = TRUE)) ~ "Retake",
      str_detect(auto_subject_name, regex("originals", ignore_case = TRUE)) ~ "originals",
      TRUE ~ NA_character_
    )
  )
}

df0 <- base %>% add_set() %>% filter(!is.na(set)) %>%
  mutate(
    measured_average_fluorescence = as.numeric(measured_average_fluorescence),
    auto_rfp_sum_mean = as.numeric(auto_rfp_sum_mean)
  ) %>%
  filter(is.finite(measured_average_fluorescence), is.finite(auto_rfp_sum_mean))

analyze <- function(data, label, x = "measured_average_fluorescence", y = "auto_rfp_sum_mean") {
  if (nrow(data) < 3) return(NULL)
  xv <- data[[x]]; yv <- data[[y]]
  pt <- suppressWarnings(cor.test(xv, yv, method = "pearson"))
  st <- suppressWarnings(cor.test(xv, yv, method = "spearman"))
  fit <- lm(reformulate(x, response = y), data = data)
  td <- tidy(fit); gl <- glance(fit)
  tibble(
    label = label, n = nrow(data),
    pearson_r = unname(pt$estimate), pearson_p = pt$p.value,
    spearman_rho = unname(st$estimate), spearman_p = st$p.value,
    lm_intercept = td$estimate[td$term == "(Intercept)"],
    lm_slope = td$estimate[td$term == x],
    lm_intercept_p = td$p.value[td$term == "(Intercept)"],
    lm_slope_p = td$p.value[td$term == x],
    r_squared = gl$r.squared
  )
}

save_stats <- function(tbl, name) {
  if (!is.null(tbl) && nrow(tbl)) write_csv(tbl, file.path(out_dir, name))
}

# 1) Element-level: combined + per-set -----------------------------------
elem_combined <- analyze(df0, "element_combined")
elem_by_set <- df0 %>% group_by(set) %>% group_split() %>%
  purrr::map(~ analyze(.x, paste0("element_", unique(.x$set)[1]))) %>%
  purrr::compact() %>% bind_rows()
save_stats(bind_rows(elem_combined, elem_by_set), "correlation_element_sets.csv")

# 2) Element-level by auto_source_type -----------------------------------
elem_src <- df0 %>% group_by(set, auto_source_type) %>% group_split() %>%
  purrr::map(~ analyze(.x, paste0("element_", unique(.x$set)[1], "__", unique(.x$auto_source_type)[1]))) %>%
  purrr::compact() %>% bind_rows()
save_stats(elem_src, "correlation_element_by_source.csv")

# 3) Per-mouse averages (combine all rows per mouse) ----------------------
mouse_avg <- df0 %>%
  group_by(mouse_id) %>%
  summarise(
    measured_average_fluorescence = mean(measured_average_fluorescence, na.rm = TRUE),
    auto_rfp_sum_mean = mean(auto_rfp_sum_mean, na.rm = TRUE),
    .groups = "drop"
  )
mouse_combined <- analyze(mouse_avg, "mouseavg_combined")

mouse_by_set <- df0 %>% group_by(set, mouse_id) %>% summarise(
  measured_average_fluorescence = mean(measured_average_fluorescence, na.rm = TRUE),
  auto_rfp_sum_mean = mean(auto_rfp_sum_mean, na.rm = TRUE), .groups = "drop") %>%
  group_by(set) %>% group_split() %>%
  purrr::map(~ analyze(.x, paste0("mouseavg_", unique(.x$set)[1]))) %>%
  purrr::compact() %>% bind_rows()

save_stats(bind_rows(mouse_combined, mouse_by_set), "correlation_mouseavg_sets.csv")

# 4) Per-mouse averages split by source type ------------------------------
mouse_by_src <- df0 %>% group_by(set, auto_source_type, mouse_id) %>% summarise(
  measured_average_fluorescence = mean(measured_average_fluorescence, na.rm = TRUE),
  auto_rfp_sum_mean = mean(auto_rfp_sum_mean, na.rm = TRUE), .groups = "drop") %>%
  group_by(set, auto_source_type) %>% group_split() %>%
  purrr::map(~ analyze(.x, paste0("mouseavg_", unique(.x$set)[1], "__", unique(.x$auto_source_type)[1]))) %>%
  purrr::compact() %>% bind_rows()
save_stats(mouse_by_src, "correlation_mouseavg_by_source.csv")

# 5) Per-mouse averages averaged across source types ----------------------
mouse_src_avg <- df0 %>% group_by(set, mouse_id, auto_source_type) %>% summarise(
  measured_average_fluorescence = mean(measured_average_fluorescence, na.rm = TRUE),
  auto_rfp_sum_mean = mean(auto_rfp_sum_mean, na.rm = TRUE), .groups = "drop") %>%
  group_by(set, mouse_id) %>% summarise(
  measured_average_fluorescence = mean(measured_average_fluorescence, na.rm = TRUE),
  auto_rfp_sum_mean = mean(auto_rfp_sum_mean, na.rm = TRUE), .groups = "drop")

mouse_src_avg_comb <- analyze(mouse_src_avg %>% select(mouse_id, measured_average_fluorescence, auto_rfp_sum_mean), "mouseavg_src_averaged_combined")

mouse_src_avg_by_set <- mouse_src_avg %>% group_by(set) %>% group_split() %>%
  purrr::map(~ analyze(.x, paste0("mouseavg_src_averaged_", unique(.x$set)[1]))) %>%
  purrr::compact() %>% bind_rows()

save_stats(bind_rows(mouse_src_avg_comb, mouse_src_avg_by_set), "correlation_mouseavg_source_averaged.csv")

# 6) Eva-filled priority selection (one row per mouse) --------------------

df_sel <- base %>%
  mutate(
    measured_average_fluorescence = as.numeric(measured_average_fluorescence),
    auto_rfp_sum_mean = as.numeric(auto_rfp_sum_mean)
  ) %>%
  filter(is.finite(measured_average_fluorescence), is.finite(auto_rfp_sum_mean)) %>%
  mutate(
    extracted_meta = stringr::str_extract(auto_subject_name, "\\[[^]]+\\]"),
    subject_meta = ifelse(is.na(extracted_meta), NA_character_, stringr::str_replace_all(extracted_meta, "\\[|\\]", "")),
    base_name = trimws(sub("\\s*\\[[^]]+\\]$", "", auto_subject_name, perl = TRUE)),
    is_eva = stringr::str_detect(auto_subject_name, regex("Eva", ignore_case = TRUE)),
    is_orig_40x = stringr::str_detect(subject_meta %||% "", regex("^originals", ignore_case = TRUE)) &
                  stringr::str_detect(base_name, regex("40x", ignore_case = TRUE)),
    is_retake = stringr::str_detect(subject_meta %||% "", regex("^Retake", ignore_case = TRUE)),
    is_retakes20 = stringr::str_detect(subject_meta %||% "", regex("lysozyme retakes2.0", ignore_case = TRUE)),
    pri_meta = dplyr::case_when(
      is_eva ~ 0L,
      is_orig_40x ~ 1L,
      is_retake ~ 2L,
      is_retakes20 ~ 3L,
      TRUE ~ 4L
    ),
    src_rank = ifelse(auto_source_type == "combined_channels", 0L, 1L)
  ) %>%
  arrange(mouse_id, pri_meta, src_rank, dplyr::desc(auto_rfp_sum_mean), auto_subject_name) %>%
  group_by(mouse_id) %>%
  slice_head(n = 1) %>%
  ungroup()

readr::write_csv(df_sel, file.path(out_dir, "selection_eva_priority_records.csv"))

sel_stats <- analyze(df_sel %>% select(measured_average_fluorescence, auto_rfp_sum_mean), "selection_eva_priority_overall")
sel_by_src <- df_sel %>% group_by(auto_source_type) %>% group_split() %>%
  purrr::map(~ analyze(.x, paste0("selection_eva_priority__", unique(.x$auto_source_type)[1]))) %>%
  purrr::compact() %>% bind_rows()
save_stats(bind_rows(sel_stats, sel_by_src), "correlation_selection_eva_priority.csv")

# 7) Element-level collapsed across source types (average combined+separate)
elem_src_averaged <- base %>%
  add_set() %>%
  filter(!is.na(set)) %>%
  mutate(
    measured_average_fluorescence = as.numeric(measured_average_fluorescence),
    auto_rfp_sum_mean = as.numeric(auto_rfp_sum_mean)
  ) %>%
  filter(is.finite(measured_average_fluorescence), is.finite(auto_rfp_sum_mean)) %>%
  mutate(base_name = trimws(sub("\\s*\\[[^]]+\\]$", "", auto_subject_name, perl = TRUE))) %>%
  group_by(set, mouse_id, manual_slice_name, base_name) %>%
  summarise(
    measured_average_fluorescence = mean(measured_average_fluorescence, na.rm = TRUE),
    auto_rfp_sum_mean = mean(auto_rfp_sum_mean, na.rm = TRUE),
    .groups = "drop"
  )

elem_src_avg_combined <- analyze(elem_src_averaged, "element_src_averaged_combined")
elem_src_avg_by_set <- elem_src_averaged %>% group_by(set) %>% group_split() %>%
  purrr::map(~ analyze(.x, paste0("element_src_averaged_", unique(.x$set)[1]))) %>%
  purrr::compact() %>% bind_rows()
save_stats(bind_rows(elem_src_avg_combined, elem_src_avg_by_set), "correlation_element_source_averaged.csv")

# 8) Correlations by manual source (element-level and mouse-avg) ---------
by_manualsrc_elem <- base %>% add_set() %>% filter(!is.na(set)) %>%
  mutate(
    measured_average_fluorescence = as.numeric(measured_average_fluorescence),
    auto_rfp_sum_mean = as.numeric(auto_rfp_sum_mean)
  ) %>%
  filter(is.finite(measured_average_fluorescence), is.finite(auto_rfp_sum_mean)) %>%
  group_by(manual_slice_source) %>% group_split() %>%
  purrr::map(~ analyze(.x, paste0("manualsrc_elem_", unique(.x$manual_slice_source)[1]))) %>%
  purrr::compact() %>% bind_rows()
save_stats(by_manualsrc_elem, "correlation_element_by_manual_source.csv")

by_manualsrc_mouse <- base %>% add_set() %>% filter(!is.na(set)) %>%
  mutate(
    measured_average_fluorescence = as.numeric(measured_average_fluorescence),
    auto_rfp_sum_mean = as.numeric(auto_rfp_sum_mean)
  ) %>%
  filter(is.finite(measured_average_fluorescence), is.finite(auto_rfp_sum_mean)) %>%
  group_by(manual_slice_source, mouse_id) %>% summarise(
    measured_average_fluorescence = mean(measured_average_fluorescence, na.rm = TRUE),
    auto_rfp_sum_mean = mean(auto_rfp_sum_mean, na.rm = TRUE), .groups = "drop"
  ) %>%
  group_by(manual_slice_source) %>% group_split() %>%
  purrr::map(~ analyze(.x, paste0("manualsrc_mouseavg_", unique(.x$manual_slice_source)[1]))) %>%
  purrr::compact() %>% bind_rows()
save_stats(by_manualsrc_mouse, "correlation_mouseavg_by_manual_source.csv")

# 9) Manual sources averaged together by animal (across sources) ---------
manual_sources_avg_mouse <- base %>% add_set() %>% filter(!is.na(set)) %>%
  mutate(
    measured_average_fluorescence = as.numeric(measured_average_fluorescence),
    auto_rfp_sum_mean = as.numeric(auto_rfp_sum_mean)
  ) %>%
  filter(is.finite(measured_average_fluorescence), is.finite(auto_rfp_sum_mean)) %>%
  group_by(mouse_id) %>% summarise(
    measured_average_fluorescence = mean(measured_average_fluorescence, na.rm = TRUE),
    auto_rfp_sum_mean = mean(auto_rfp_sum_mean, na.rm = TRUE), .groups = "drop"
  )
save_stats(analyze(manual_sources_avg_mouse, "manualsrcs_avg_by_mouse"), "correlation_manualsources_avg_by_mouse.csv")

cat("Wrote correlation variants CSVs into ", out_dir, "\n")
