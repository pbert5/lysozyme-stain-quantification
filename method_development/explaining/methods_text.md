# Methods Text (Poster-Ready, Plain Language)

## One-line goal
Detect lysozyme-producing intestinal crypt regions from paired DAPI and RFP images, then rank and select the most representative crypts per field.

## 1) Pipeline framing
The pipeline starts from paired fluorescence channels, standardizes each channel independently, and then builds morphology-informed maps that encode where crypt-like structures are most likely to exist.

## 2) Morphology-informed likelihood
In DAPI, we identify tissue borders and cavity-like structures; in RFP, we identify strong lysozyme-positive regions in expected size/shape ranges. These maps are combined into a distance/likelihood image where high values better match the target crypt profile.

## 3) Seed to region progression
For figure communication, we select a zoom window around the highest-quality candidate and show the morphology flow locally: channel overlap, seed labels over grayscale distance image, and base labels over grayscale zoomed context with explicit label boundaries.

## 4) Weighted quality scoring
Candidate regions are scored using circularity, area, line-fit alignment, and red-intensity features. Each metric row uses a tissue overlay saturation map for that single metric, while a separate cumulative saturation map shows the weighted total quality.
