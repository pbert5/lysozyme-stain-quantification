# Methods Text (Poster-Ready, Plain Language)

## One-line goal
Detect intestinal crypt regions and quantify crypt-associated lysozyme from paired DAPI and RFP anti-LYZ fluorescence images.

## 0) What you're looking at
Each field contains two channels. DAPI (blue) marks nuclei and provides tissue architecture; RFP anti-LYZ (red) reports lysozyme signal concentrated around crypt regions. Intestinal crypts are pocket-like invaginations between villi; their lumens appear as DAPI-low cavities, adjacent to lysozyme signal. Crypt-level lysozyme is a commonly used readout in gut inflammation studies.

## 1) Pipeline framing
Starting from the paired channels, we standardize intensity per channel and build morphology maps that capture (a) DAPI-defined tissue borders/cavities and (b) RFP-defined lysozyme-positive regions in expected size/shape ranges.

## 2) Morphology-informed likelihood
We combine the DAPI and RFP morphology maps into a crypt-likelihood (distance) image where higher values better match the target crypt profile, even when staining is diffuse.

## 3) Seed to region progression
High-likelihood peaks define non-overlapping seed regions. Seeds are expanded into full crypt labels, visualized as seed overlays on the likelihood map and final label boundaries on a zoomed analysis window.

## 4) Weighted quality scoring and selection
Candidate crypt labels are scored using circularity, area consistency, spatial alignment (line fit), and red-intensity features. A weighted sum ranks detections, and the top-scoring crypts are selected for downstream quantification.
