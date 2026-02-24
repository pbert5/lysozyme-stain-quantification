# Figure Map (Updated N-series)

## Core source assets (C01-C11)
- `C01` -> `assets/C01_ileum_ch2_rfp_input.png`: RFP channel input.
- `C02` -> `assets/C02_ileum_ch2_dapi_input.png`: DAPI channel input.
- `C03` -> `assets/C03_ileum_ch2_crypt_preprocessed.png`: RFP standardized/preprocessed.
- `C04` -> `assets/C04_ileum_ch2_tissue_preprocessed.png`: DAPI standardized/preprocessed.
- `C05` -> `assets/C05_ileum_ch2_tissue_caps_troughs.png`: DAPI morphology (caps/troughs).
- `C06` -> `assets/C06_ileum_ch2_good_crypts.png`: RFP morphology-derived candidate regions.
- `C07` -> `assets/C07_ileum_ch2_distance_image.png`: Combined likelihood/distance map.
- `C08` -> `assets/C08_ileum_ch2_seed_labels.png`: Seed labels.
- `C09` -> `assets/C09_ileum_ch2_base_labels.png`: Base labels after seeded growth.
- `C10` -> `assets/C10_ileum_ch2_final_crypt_labels.png`: Final selected labels.
- `C11` -> `assets/C11_roi_mt_quality_fixed_hue.png`: Quality hue reference.

## Generated poster boards (N1-N4)
- `N1` -> `assets/N1_pipeline_flowchart.png`: High-level pipeline flowchart used to guide figure order.
- `N2` -> `assets/N2_channel_split_standardization.png`: Original field split into standardized DAPI/RFP channels.
- `N3` -> `assets/N3_morphology_seed_flowchart.png`: Morphology-based likelihood and seed-to-label flow.
- `N4` -> `assets/N4_quality_scoring_breakdown.png`: Scoring criteria, weights, and quality-saturation interpretation.

## Figure text files
- `figure_text/N1_pipeline_flowchart.txt`
- `figure_text/N2_channel_split_standardization.txt`
- `figure_text/N3_morphology_seed_flowchart.txt`
- `figure_text/N4_quality_scoring_breakdown.txt`
