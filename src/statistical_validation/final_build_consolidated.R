#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(purrr)
})

# =========================
# Configuration (edit here)
# =========================

MANUAL_DIR <- file.path("src", "statistical_validation", "manual_stats_results")
AUTO_SUMMARY_PATH <- file.path("results", "simple_dask", "simple_dask_image_summary.csv")
AUTO_DETAILED_PATH <- file.path("results", "simple_dask", "simple_dask_image_summary_detailed.csv")
RATINGS_PATH <- file.path("results", "simple_dask", "manual_verification_ratings.csv")

# Microns-per-pixel mapping from naming patterns found in slice names
DEFAULT_UM_PER_PX <- 0.4476
SCALE_KEYS <- c("40x")
SCALE_VALUES <- c(0.2253)

OUT_DIR <- file.path("src", "statistical_validation", "outputs", "final")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# =========================
# Helpers
# =========================

`%||%` <- function(x, y) if (is.null(x)) y else x

coerce_num <- function(x) {
  suppressWarnings(as.numeric(x))
}

resolve_um_per_px <- function(slice_name) {
  if (is.na(slice_name) || slice_name == "") return(DEFAULT_UM_PER_PX)
  nn <- tolower(slice_name)
  for (i in seq_along(SCALE_KEYS)) {
    if (str_detect(nn, tolower(SCALE_KEYS[i]))) return(SCALE_VALUES[i])
  }
  DEFAULT_UM_PER_PX
}

read_manual_file <- function(path) {
  raw <- readr::read_csv(path, col_names = FALSE, show_col_types = FALSE)
  if (nrow(raw) <= 1) return(tibble())

  header <- raw |> slice(1) |> unlist(use.names = FALSE) |> as.character()
  header <- header |> str_replace_all("\\s+", " ") |> str_trim(side = "both")
  if (length(header) >= 1 && (is.na(header[1]) || header[1] == "")) header[1] <- "Mouse ID"
  data <- raw |> slice(-1)

  keep <- !is.na(header) & header != ""
  header <- header[keep]
  data <- data[, keep, drop = FALSE]

  rep_idx <- ave(seq_along(header), header, FUN = seq_along)
  safe_names <- paste0(header, "_rep", rep_idx)
  colnames(data) <- safe_names

  df <- as_tibble(data, .name_repair = "minimal") |>
    mutate(across(everything(), ~ na_if(str_trim(as.character(.x)), ""))) |>
    mutate(
      manual_source = basename(path),
      mouse_id = str_trim(.data[["Mouse ID_rep1"]] %||% NA_character_)
    ) |>
    select(manual_source, mouse_id, everything(), -starts_with("Mouse ID_rep"))

  long <- df |>
    mutate(row_index = row_number()) |>
    pivot_longer(
      cols = matches("_rep\\d+$"),
      names_to = c("metric", "rep"),
      names_pattern = "(.+?)_rep(\\d+)$",
      values_to = "value"
    ) |>
    mutate(metric = str_trim(metric), rep = as.integer(rep)) |>
    pivot_wider(id_cols = c(manual_source, mouse_id, row_index, rep), names_from = metric, values_from = value)

  if (!"Slice" %in% names(long)) long$Slice <- NA_character_
  # Ensure expected columns exist
  for (nm in c("Count", "Total Area", "Average Size", "% Area", "Mean")) {
    if (!nm %in% names(long)) long[[nm]] <- NA_character_
  }

  long |>
    transmute(
      manual_source,
      mouse_id = str_trim(mouse_id),
      slice = str_trim(tools::file_path_sans_ext(Slice)),
      count = coerce_num(`Count`),
      total_area_px = coerce_num(`Total Area`),
      avg_size_px = coerce_num(`Average Size`),
      percent_area = coerce_num(`% Area`),
      mean_intensity_px = coerce_num(`Mean`)
    ) |>
    filter(!is.na(slice) & slice != "")
}

build_manual_uncollapsed <- function(manual_dir) {
  files <- list.files(manual_dir, pattern = "[.]csv$", full.names = TRUE)
  if (length(files) == 0) stop("No manual CSVs found in ", manual_dir)
  purrr::map_dfr(files, read_manual_file)
}

# =========================
# Load manual (uncollapsed) and compute per-crypt metrics
# =========================

manual_unc <- build_manual_uncollapsed(MANUAL_DIR) |>
  mutate(
    um_per_px = vapply(slice, resolve_um_per_px, numeric(1)),
    # per-crypt integrated intensity in pixel units
    manual_integrated_px = total_area_px * mean_intensity_px,
    # standardized to μm² by scaling pixel area
    manual_integrated_um2 = manual_integrated_px * (um_per_px^2)
  )

readr::write_csv(manual_unc, file.path(OUT_DIR, "manual_uncollapsed_with_metrics.csv"))

# =========================
# Collapse manual by (manual_source, slice): mean, sd, n
# =========================

manual_collapsed <- manual_unc |>
  group_by(manual_source, mouse_id, slice) |>
  summarise(
    measured_avg_fluorescence_um2 = mean(manual_integrated_um2, na.rm = TRUE),
    measured_avg_fluorescence_um2_sd = sd(manual_integrated_um2, na.rm = TRUE),
    measured_avg_fluorescence_px = mean(manual_integrated_px, na.rm = TRUE),
    measured_avg_fluorescence_px_sd = sd(manual_integrated_px, na.rm = TRUE),
    number_of_manual_crypts = sum(is.finite(manual_integrated_px)),
    um_per_px = dplyr::first(um_per_px),
    .groups = "drop"
  )

readr::write_csv(manual_collapsed, file.path(OUT_DIR, "manual_collapsed_by_source_slice.csv"))

# =========================
# Load automated data and build subject_base for joining
# =========================

auto_summary <- readr::read_csv(AUTO_SUMMARY_PATH, show_col_types = FALSE)
auto_detailed <- readr::read_csv(AUTO_DETAILED_PATH, show_col_types = FALSE)
ratings <- if (file.exists(RATINGS_PATH)) readr::read_csv(RATINGS_PATH, show_col_types = FALSE) else tibble()

auto <- auto_summary |>
  mutate(
    subject_base = str_trim(str_replace(`subject name`, "\\s*\\[.*$", ""))
  ) |>
  select(`subject name`, subject_base, image_source_type)

# Keep relevant auto metrics from detailed
auto_det_small <- auto_detailed |>
  transmute(
    `subject name` = subject_name,
    auto_rfp_sum_mean = rfp_sum_mean,
    auto_crypt_count = crypt_count
  )

auto <- auto |>
  left_join(auto_det_small, by = "subject name")

if (nrow(ratings)) {
  auto <- auto |>
    mutate(
      subject_meta = str_match(`subject name`, "\\[(.+?)\\]")[,2],
      dup_suffix = str_match(`subject name`, "\\]\\s*\\((\\d+)\\)")[,2],
      dup_part = ifelse(is.na(dup_suffix), "", paste0("_(", dup_suffix, ")")),
      subject_key = paste0(
        str_replace_all(subject_base, "[\\s]+", "_"), "_",
        str_replace_all(subject_meta %||% "", "[\\s/]+", "_"),
        dup_part, "_", image_source_type
      )
    ) |>
    left_join(ratings |> select(metadata_key, rating_bool, rating_text), by = c("subject_key" = "metadata_key"))
}

# =========================
# Build many-to-many mapping: manual slice name ↔ auto subject_base
# =========================

mapping <- manual_collapsed |>
  rename(manual_slice_name = slice, manual_slice_source = manual_source) |>
  inner_join(auto, by = c("manual_slice_name" = "subject_base"))

# If ratings are available, restrict to Good quality subjects
if ("rating_bool" %in% names(mapping)) {
  mapping <- mapping |>
    filter(!is.na(rating_bool) & rating_bool)
}

readr::write_csv(mapping, file.path(OUT_DIR, "manual_to_auto_mapping.csv"))

# =========================
# Final consolidated output for external analysis
# One row per manual↔auto mapping (no collapse of auto subjects)
# =========================

consolidated <- mapping |>
  transmute(
    mouse_id,
    manual_slice_source,
    manual_slice_name,
    measured_average_fluorescence = measured_avg_fluorescence_um2,
    number_of_manual_crypts,
    auto_subject_name = `subject name`,
    auto_source_type = image_source_type,
    auto_rfp_sum_mean,
    auto_crypt_count,
    rating_bool = rating_bool %||% NA,
    rating_text = rating_text %||% NA
  )

readr::write_csv(consolidated, file.path(OUT_DIR, "consolidated_for_external_analysis.csv"))

cat("Wrote outputs in ", OUT_DIR, "\n")
cat("- manual_uncollapsed_with_metrics.csv\n")
cat("- manual_collapsed_by_source_slice.csv\n")
cat("- manual_to_auto_mapping.csv\n")
cat("- consolidated_for_external_analysis.csv\n")
