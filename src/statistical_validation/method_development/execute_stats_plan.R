#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(tibble)
})

# Helper -----------------------------------------------------------------

rename_if_exists <- function(data, from, to) {
  if (from %in% names(data)) {
    names(data)[names(data) == from] <- to
  }
  data
}

parse_numeric_cols <- function(data, cols) {
  present <- intersect(cols, names(data))
  if (length(present) == 0) {
    return(data)
  }
  data |> mutate(across(all_of(present), \(x) readr::parse_double(x, na = c("", "NA"))))
}

read_manual_file <- function(path) {
  raw <- readr::read_csv(path, col_names = FALSE, show_col_types = FALSE)
  if (nrow(raw) <= 1) {
    return(tibble())
  }

  header <- raw |> slice(1) |> unlist(use.names = FALSE) |> as.character()
  header <- header |> str_replace_all("\\s+", " ") |> str_trim(side = "both")
  if (length(header) >= 1 && (is.na(header[1]) || header[1] == "")) {
    header[1] <- "Mouse ID"
  }
  data <- raw |> slice(-1)

  keep <- !is.na(header) & header != ""
  header <- header[keep]
  data <- data[, keep, drop = FALSE]

  rep_idx <- ave(seq_along(header), header, FUN = seq_along)
  safe_names <- paste0(header, "_rep", rep_idx)
  colnames(data) <- safe_names

  data_tbl <- as_tibble(data, .name_repair = "minimal") |>
    mutate(across(everything(), ~ na_if(str_trim(as.character(.x)), ""))) |>
    mutate(
      rater = str_extract(basename(path), "(?<=\\().+(?=\\))"),
      mouse_id = str_trim(.data[["Mouse ID_rep1"]] %||% NA_character_)
    )

  data_tbl <- data_tbl |> select(rater, mouse_id, everything(), -starts_with("Mouse ID_rep"))

  tidy <- data_tbl |>
    mutate(row_index = row_number()) |>
    pivot_longer(
      cols = matches("_rep\\d+$"),
      names_to = c("metric", "rep"),
      names_pattern = "(.+?)_rep(\\d+)$",
      values_to = "value"
    ) |>
    mutate(
      metric = str_trim(metric),
      rep = as.integer(rep),
      value = na_if(value, "")
    ) |>
    drop_na(metric)

  wide <- tidy |>
    pivot_wider(
      id_cols = c(rater, mouse_id, row_index, rep),
      names_from = metric,
      values_from = value,
      values_fn = list(value = dplyr::first)
    )

  if (nrow(wide) == 0) {
    return(tibble())
  }

  wide <- wide |>
    rename_if_exists("Slice", "slice") |>
    rename_if_exists("Count", "count") |>
    rename_if_exists("Total Area", "total_area") |>
    rename_if_exists("Average Size", "average_size") |>
    rename_if_exists("% Area", "percent_area") |>
    rename_if_exists("Mean", "mean_intensity")

  wide <- wide |>
    mutate(
      slice = str_trim(slice),
      mouse_id = str_trim(mouse_id),
      rater = coalesce(rater, "unknown")
    ) |>
    parse_numeric_cols(c("count", "total_area", "average_size", "percent_area", "mean_intensity")) |>
    filter(
      !is.na(slice),
      slice != "",
      !(slice == "Average" & is.na(count) & is.na(total_area))
    ) |>
    select(rater, mouse_id, rep, slice, count, total_area, average_size, percent_area, mean_intensity)

  wide |>
    mutate(
      slice = str_remove(slice, "\\.tif$"),
      slice = str_remove(slice, "\\.TIF$"),
      slice = str_trim(slice)
    )
}

`%||%` <- function(x, y) if (is.null(x)) y else x

manual_dir <- "manual_stats_results"

manual_files <- list.files(manual_dir, pattern = "\\.csv$", full.names = TRUE)

manual_tidy <- manual_files |>
  set_names(basename) |>
  map_dfr(read_manual_file, .id = "source_file")

if (nrow(manual_tidy) == 0) {
  stop("No manual quantification data available after parsing.")
}

manual_slice_summary <- manual_tidy |>
  group_by(source_file, rater, mouse_id, slice) |>
  summarise(
    across(c(count, total_area, average_size, percent_area, mean_intensity),
           list(
             mean = ~ mean(.x, na.rm = TRUE),
             sd = ~ sd(.x, na.rm = TRUE),
             n = ~ sum(!is.na(.x))
           ),
           .names = "{.col}_{.fn}"),
    .groups = "drop"
  )

manual_combined_by_slice <- manual_slice_summary |>
  group_by(mouse_id, slice) |>
  summarise(
    across(ends_with("_mean"), ~ mean(.x, na.rm = TRUE), .names = "{.col}"),
    across(ends_with("_sd"), ~ mean(.x, na.rm = TRUE), .names = "{.col}"),
    across(ends_with("_n"), ~ sum(.x), .names = "{.col}"),
    raters = n_distinct(rater),
    sources = n_distinct(source_file),
    .groups = "drop"
  )

manual_by_mouse <- manual_tidy |>
  group_by(mouse_id) |>
  summarise(
    across(c(count, total_area, average_size, percent_area, mean_intensity),
           list(mean = ~ mean(.x, na.rm = TRUE), sd = ~ sd(.x, na.rm = TRUE)),
           .names = "{.col}_{.fn}"),
    n_slices = n_distinct(slice),
    .groups = "drop"
  )

# Automated data ----------------------------------------------------------

automated_summary_path <- file.path("..", "..", "results", "simple_dask", "simple_dask_image_summary.csv")
automated_detailed_path <- file.path("..", "..", "results", "simple_dask", "simple_dask_image_summary_detailed.csv")
ratings_path <- file.path("..", "..", "results", "simple_dask", "manual_verification_ratings.csv")

if (!file.exists(automated_summary_path)) {
  stop("Automated summary file not found: ", automated_summary_path)
}

auto_summary <- read_csv(automated_summary_path, show_col_types = FALSE)
auto_detailed <- read_csv(automated_detailed_path, show_col_types = FALSE)

# Override auto 'Mean' using detailed rfp_sum_mean scaled by μm/px
auto_d_mean <- auto_detailed |>
  rename(`subject name` = subject_name) |>
  select(`subject name`, microns_per_px, rfp_sum_mean) |>
  mutate(Mean_override = rfp_sum_mean * microns_per_px)

auto_summary <- auto_summary |>
  left_join(auto_d_mean, by = "subject name") |>
  mutate(Mean = dplyr::coalesce(Mean_override, Mean)) |>
  select(-Mean_override)

# Bring effective-full metrics into auto summary for downstream comparisons
auto_eff <- auto_detailed |>
  rename(`subject name` = subject_name) |>
  select(`subject name`, effective_full_intensity_um2_mean, effective_full_intensity_um2_sum)

auto_summary <- auto_summary |>
  left_join(auto_eff, by = "subject name")

# Approximate auto image area (μm^2) from Total Area and % Area when available
auto_summary <- auto_summary |>
  mutate(auto_image_area_um2 = ifelse(`% Area` > 0, `Total Area` / (`% Area`/100), NA_real_))
ratings <- read_csv(ratings_path, show_col_types = FALSE)

sanitize_subject_key <- function(subject_name, source_type) {
  map2_chr(subject_name, source_type, function(sn, st) {
    base <- str_trim(str_replace(sn, "\\s*\\[.*$", ""))
    meta <- str_match(sn, "\\[(.+?)\\]")[, 2]
    meta <- str_replace(meta, "\\s*\\(2\\)$", "")
    parts <- c(base, meta, st)
    parts <- parts[!is.na(parts) & parts != ""]
    if (length(parts) == 0) {
      return(NA_character_)
    }
    sanitized <- parts |>
      str_replace_all("\\s+", "_") |>
      str_replace_all("/", "_") |>
      str_replace_all("__+", "_") |>
      str_replace_all("[^-A-Za-z0-9_()+]+", "")
    key <- paste(sanitized, collapse = "_")
    str_replace_all(key, "__+", "_")
  })
}

derive_image_group <- function(meta_vec) {
  map_chr(meta_vec, function(meta) {
    if (is.na(meta) || meta == "") {
      return(NA_character_)
    }
    meta_clean <- str_trim(meta)
    meta_clean <- str_replace(meta_clean, "\\s*\\(2\\)$", "")
    if (str_detect(meta_clean, "/")) {
      return(str_replace(meta_clean, "/[^/]+$", ""))
    }
    if (str_detect(str_to_lower(meta_clean), "lysozyme retakes2.0")) {
      return("lysozyme retakes2.0")
    }
    if (str_detect(meta_clean, " ")) {
      return(str_extract(meta_clean, "^[^ ]+"))
    }
    meta_clean
  })
}

auto_summary <- auto_summary |>
  mutate(
    subject_key = sanitize_subject_key(`subject name`, image_source_type),
    subject_base = str_trim(str_replace(`subject name`, "\\s*\\[.*$", "")),
    subject_meta = str_match(`subject name`, "\\[(.+?)\\]")[, 2],
    subject_meta = str_replace(subject_meta, "\\s*\\(2\\)$", ""),
    mouse_id = str_extract(subject_base, "^[A-Za-z0-9]{4,5}"),
    slice_name = str_trim(subject_base),
    slice_name = str_replace(slice_name, "\\.tif$", ""),
    slice_name = str_replace(slice_name, "\\.TIF$", ""),
    image_group = derive_image_group(subject_meta),
    image_group_detail = subject_meta
  )

ratings_pass <- ratings |>
  filter(rating_bool) |>
  mutate(subject_key = metadata_key)

auto_pass <- auto_summary |>
  inner_join(ratings_pass, by = "subject_key")

auto_fields <- intersect(c("Count", "Total Area", "Average Size", "% Area", "Mean",
                           "effective_full_intensity_um2_mean", "rfp_intensity_um2_sum"),
                         names(auto_pass))
auto_pass_grouped <- auto_pass |>
  group_by(mouse_id, image_group) |>
  summarise(
    across(all_of(auto_fields),
           list(mean = ~ mean(.x, na.rm = TRUE), sd = ~ sd(.x, na.rm = TRUE)),
           .names = "{.col}_{.fn}"),
    n_images = n(),
    .groups = "drop"
  )

# Mapping -----------------------------------------------------------------

manual_auto_exact <- manual_combined_by_slice |>
  rename(slice_name = slice) |>
  inner_join(auto_pass, by = c("mouse_id", "slice_name" = "subject_base")) |>
  mutate(manual_area_um2_from_percent = (percent_area_mean/100) * auto_image_area_um2)

manual_by_mouse_combined <- manual_tidy |>
  group_by(mouse_id) |>
  summarise(
    across(c(count, total_area, average_size, percent_area, mean_intensity), ~ mean(.x, na.rm = TRUE), .names = "manual_{.col}"),
    .groups = "drop"
  )

auto_by_mouse_group <- auto_pass |>
  group_by(mouse_id) |>
  summarise(
    across(c(Count, `Total Area`, `Average Size`, `% Area`, Mean), ~ mean(.x, na.rm = TRUE), .names = "auto_{.col}"),
    n_images = n(),
    .groups = "drop"
  )

manual_auto_by_mouse <- manual_by_mouse_combined |>
  inner_join(auto_by_mouse_group, by = "mouse_id")

# Add per-mouse aggregates for manual_area_from_percent and effective-full metrics
by_mouse_extra_manual <- manual_auto_exact |>
  group_by(mouse_id) |>
  summarise(manual_area_um2_from_percent = mean(manual_area_um2_from_percent, na.rm = TRUE), .groups = "drop")

if (all(c("effective_full_intensity_um2_mean", "effective_full_intensity_um2_sum") %in% names(auto_pass))) {
  by_mouse_extra_auto <- auto_pass |>
    group_by(mouse_id) |>
    summarise(
      auto_effective_full_intensity_um2_mean = mean(effective_full_intensity_um2_mean, na.rm = TRUE),
      auto_effective_full_intensity_um2_sum = mean(effective_full_intensity_um2_sum, na.rm = TRUE),
      .groups = "drop"
    )
} else {
  by_mouse_extra_auto <- distinct(auto_pass, mouse_id) |>
    mutate(
      auto_effective_full_intensity_um2_mean = NA_real_,
      auto_effective_full_intensity_um2_sum = NA_real_
    )
}

manual_auto_by_mouse <- manual_auto_by_mouse |>
  left_join(by_mouse_extra_manual, by = "mouse_id") |>
  left_join(by_mouse_extra_auto, by = "mouse_id")

# Output ------------------------------------------------------------------

output_dir <- file.path("outputs")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write_csv(manual_tidy, file.path(output_dir, "manual_slice_records.csv"))
write_csv(manual_slice_summary, file.path(output_dir, "manual_slice_summary_by_rater.csv"))
write_csv(manual_combined_by_slice, file.path(output_dir, "manual_slice_summary_combined.csv"))
write_csv(manual_by_mouse, file.path(output_dir, "manual_by_mouse.csv"))
write_csv(auto_pass, file.path(output_dir, "auto_pass_image_summary.csv"))
write_csv(auto_pass_grouped, file.path(output_dir, "auto_pass_grouped_by_mouse_image_group.csv"))
write_csv(manual_auto_exact, file.path(output_dir, "manual_auto_exact_slice_join.csv"))
write_csv(manual_auto_by_mouse, file.path(output_dir, "manual_auto_by_mouse_join.csv"))

cat("Statistical validation data preparation complete.\n")
