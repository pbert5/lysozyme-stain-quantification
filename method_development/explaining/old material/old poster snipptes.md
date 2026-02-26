# Poster Flavor Text (Draft, Minimal)

## Before the pipeline (show example images)
Each sample is a paired fluorescence field:
- **DAPI (blue):** nuclei / tissue architecture (helps locate epithelial borders and luminal cavities).
- **RFP anti-LYZ (red):** lysozyme signal concentrated around crypt regions.

**Target:** identify individual intestinal crypt regions and quantify crypt-associated lysozyme (a commonly used readout in gut inflammation studies).

---

## N1 — Pipeline overview (glanceable)
**Subtitle:** Pipeline overview: from paired channels to ranked crypt detections.

**Box text:** split + standardize channels -> build morphology maps (DAPI, RFP) -> combine maps to predict crypt locations -> label individual crypt regions -> score + select representative crypts for quantification.

---

## N2 — Channel split + standardization
**Subtitle:** Example inputs and channel meaning, after per-channel intensity standardization.

**Box text:** DAPI provides tissue context; RFP anti-LYZ reports lysozyme signal. We normalize each channel independently to align contrast across images, making morphology and thresholding less sensitive to exposure/staining variability.

---

## N3 — Crypt definition + morphology overlap + labeling
**Subtitle:** What is a crypt, and how dual-channel morphology drives seed-to-label segmentation.

**Box text:** Intestinal crypts are pocket-like invaginations between villi. Their lumens appear as DAPI-low cavities bordered by epithelial cells, and lysozyme signal is concentrated near these pockets. Lysozyme levels can increase with inflammation. We combine DAPI-derived tissue borders/cavities with RFP-derived lysozyme-positive regions into a crypt-likelihood map, then convert high-likelihood peaks into seeds and expand them into full crypt labels.

---

## N4 — Scoring and selection
**Subtitle:** Weighted quality scoring to prioritize well-formed, high-signal crypt detections.

**Box text:** Not every candidate region is a true, well-isolated crypt (noise, partial crypts, merged neighbors). We compute a weighted quality score from shape and signal metrics, then keep the top-scoring detections per field for quantification. Saturation maps show per-metric quality (higher saturation = higher quality), and a cumulative map shows the weighted total.

**Metrics (higher is better):**
- circularity: shape compactness
- area: size consistency
- line_fit: spatial alignment
- red_intensity: lysozyme signal strength (RFP)
