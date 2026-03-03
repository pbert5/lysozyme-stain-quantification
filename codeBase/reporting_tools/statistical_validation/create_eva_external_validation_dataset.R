# Create clean dataset for external validation
# Filter to Eva set (both sources), keep manual reviewers separate
# Focus on interpretability with clear column naming

library(dplyr)
library(tidyr)
library(readr)

# Read the manual data with metrics (has all the fluorescence data we need)
manual_data <- read_csv("outputs/final/manual_uncollapsed_with_metrics.csv")

# Read the mapping file to get subject keys
mapping_data <- read_csv("outputs/final/manual_to_auto_mapping.csv")

# Extract just what we need from mapping: subject_key relationship
subject_lookup <- mapping_data %>%
  select(manual_slice_source, mouse_id, manual_slice_name, subject_key) %>%
  distinct()

# Join to get subject keys
combined_data <- manual_data %>%
  left_join(subject_lookup, 
            by = c("manual_source" = "manual_slice_source", 
                   "mouse_id" = "mouse_id", 
                   "slice" = "manual_slice_name"))

# Filter to only Eva sources (both Adam and Eva's manual ratings)
# Eva sources are the two CSV files
eva_data <- combined_data %>%
  filter(manual_source %in% c("image_j_quantification_eva_04222025(Adam).csv",
                              "image_j_quantification_eva_04222025(Eva).csv"))

# Create clean column names and select relevant data
clean_data <- eva_data %>%
  mutate(
    manual_reviewer = case_when(
      manual_source == "image_j_quantification_eva_04222025(Adam).csv" ~ "Adam",
      manual_source == "image_j_quantification_eva_04222025(Eva).csv" ~ "Eva",
      TRUE ~ NA_character_
    )
  ) %>%
  select(
    manual_reviewer,
    mouse_id,
    slice_name = slice,
    crypt_count = count,
    total_area_px,
    avg_crypt_size_px = avg_size_px,
    percent_tissue_area = percent_area,
    mean_fluorescence_px = mean_intensity_px,
    um_per_px,
    integrated_fluorescence_px = manual_integrated_px,
    integrated_fluorescence_um2 = manual_integrated_um2,
    subject_key
  ) %>%
  # Remove rows with no data (marked as "NO" in original)
  filter(!is.na(crypt_count))

# Reorder columns for readability
final_data <- clean_data %>%
  select(
    manual_reviewer,
    mouse_id,
    slice_name,
    subject_key,
    crypt_count,
    # Pixel-based measurements
    total_area_px,
    avg_crypt_size_px,
    mean_fluorescence_px,
    integrated_fluorescence_px,
    # Micrometers-based measurements
    um_per_px,
    integrated_fluorescence_um2,
    # Other
    percent_tissue_area
  ) %>%
  arrange(manual_reviewer, mouse_id, slice_name)

# Write output
write_csv(final_data, "outputs/final/eva_set_external_validation.csv")

# Print summary
cat("\n=== Dataset Summary ===\n")
cat("Total rows:", nrow(final_data), "\n")
cat("Rows per reviewer:\n")
print(table(final_data$manual_reviewer))
cat("\nUnique mice:", length(unique(final_data$mouse_id)), "\n")
cat("\nColumn names:\n")
print(names(final_data))
cat("\nFirst few rows:\n")
print(head(final_data, 10))
cat("\nOutput saved to: outputs/final/eva_set_external_validation.csv\n")
