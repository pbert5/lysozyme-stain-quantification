#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(purrr)
})

in_file <- file.path("src", "statistical_validation", "outputs", "final", "consolidated_for_external_analysis.csv")
out_dir <- file.path("src", "statistical_validation", "outputs", "final")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

stopifnot(file.exists(in_file))
base <- read_csv(in_file, show_col_types = FALSE) %>%
  mutate(
    measured_average_fluorescence = as.numeric(measured_average_fluorescence),
    auto_rfp_sum_mean = as.numeric(auto_rfp_sum_mean)
  ) %>%
  filter(is.finite(measured_average_fluorescence), is.finite(auto_rfp_sum_mean))

# Derive set label
derive_set <- function(s) {
  s <- as.character(s)
  ifelse(str_detect(s, regex("Eva", ignore_case = TRUE)), "Eva",
  ifelse(str_detect(s, regex("lysozyme retakes2.0", ignore_case = TRUE)), "retakes2.0",
  ifelse(str_detect(s, regex("Retake", ignore_case = TRUE)), "Retake",
  ifelse(str_detect(s, regex("originals", ignore_case = TRUE)), "originals", "other"))))
}

base <- base %>% mutate(set = derive_set(auto_subject_name))

# Simplify manual sources to Eva/Adam based on parentheses in filename
extract_person <- function(src) {
  grp <- stringr::str_match(src, "\\(([^)]+)\\)")
  who <- if (!is.null(grp)) grp[,2] else NA_character_
  if (is.na(who)) return(NA_character_)
  who_up <- tolower(who)
  if (grepl("eva", who_up)) return("Eva")
  if (grepl("adam", who_up)) return("Adam")
  return(NA_character_)
}
base <- base %>% mutate(manual_source_group = vapply(manual_slice_source, extract_person, character(1)))

sets <- sort(unique(base$set))
sets <- sets[!is.na(sets)]

set_subsets <- purrr::map(seq_along(sets), ~ sets[.x]) # singleton
# add combined subsets up to size 3 to keep search manageable
if (length(sets) >= 2) set_subsets <- c(set_subsets, combn(sets, 2, simplify = FALSE))
if (length(sets) >= 3) set_subsets <- c(set_subsets, combn(sets, 3, simplify = FALSE))
if (length(sets) >= 4) set_subsets <- c(set_subsets, list(sets))

source_opts <- c("both", "combined_channels", "separate_channels")

analyze_df <- function(df) {
  res <- try(suppressWarnings(cor.test(df$measured_average_fluorescence, df$auto_rfp_sum_mean, method = "pearson")), silent = TRUE)
  if (inherits(res, "try-error")) return(NA_real_)
  unname(res$estimate)
}

base <- base %>% mutate(base_name = trimws(vapply(strsplit(auto_subject_name, " [", fixed = TRUE), function(x) x[1], character(1))))

summarizers <- list(
  element = function(df) df,
  mouse_avg = function(df) df %>% group_by(mouse_id) %>% summarise(measured_average_fluorescence = mean(measured_average_fluorescence, na.rm = TRUE), auto_rfp_sum_mean = mean(auto_rfp_sum_mean, na.rm = TRUE), .groups = "drop"),
  mouse_pick_max_auto = function(df) df %>% group_by(mouse_id) %>% arrange(desc(auto_rfp_sum_mean)) %>% slice_head(n = 1) %>% ungroup() %>% select(mouse_id, measured_average_fluorescence, auto_rfp_sum_mean),
  mouse_pick_pref_combined = function(df) df %>% group_by(mouse_id) %>% arrange(auto_source_type != "combined_channels", desc(auto_rfp_sum_mean)) %>% slice_head(n = 1) %>% ungroup() %>% select(mouse_id, measured_average_fluorescence, auto_rfp_sum_mean),
  element_src_avg = function(df) df %>% group_by(mouse_id, manual_slice_name, base_name) %>% summarise(measured_average_fluorescence = mean(measured_average_fluorescence, na.rm = TRUE), auto_rfp_sum_mean = mean(auto_rfp_sum_mean, na.rm = TRUE), .groups = "drop")
)

results <- list()
present_groups <- sort(unique(na.omit(base$manual_source_group)))
manual_sources <- c("ALL", present_groups)
for (ms in manual_sources) {
  base_ms <- if (ms == "ALL") base else base %>% filter(manual_source_group == ms)
  for (ss in set_subsets) {
    for (src in source_opts) {
      df <- base_ms %>% filter(set %in% ss)
      if (src != "both") df <- df %>% filter(auto_source_type == src)
      for (name in names(summarizers)) {
        df_s <- summarizers[[name]](df)
        n <- nrow(df_s)
        if (!is.finite(n) || n < 6) next
        r <- analyze_df(df_s)
        if (is.na(r)) next
        results[[length(results)+1]] <- tibble(
          manual_source = ms,
          sets = paste(ss, collapse = "+"),
          source = src,
          summarizer = name,
          n = n,
          pearson_r = r
        )
      }
    }
  }
}

res_tbl <- bind_rows(results) %>% arrange(desc(pearson_r))
top_file <- file.path(out_dir, "bruteforce_top_by_pearson.csv")
write_csv(res_tbl, top_file)
print(head(res_tbl, 20))
cat("\nSaved: ", top_file, "\n")
