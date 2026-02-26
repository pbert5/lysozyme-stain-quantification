Materials and methods 

Overview/workflow diagram 

Overview 

The Lysozyme Quantification Pipeline (LQP) automates the identification of intestinal crypts in histological images and quantifies lysozyme expression within each crypt. The pipeline integrates preprocessing, segmentation, and intensity quantification modules, implemented in Python using scikit-image, OpenCV, and NumPy. It produces per-image and per-crypt summary statistics suitable for downstream statistical analysis. 

Explanation of crypt morphology 

DAPI: assume 3 non overlapping nested/layerd regions 

Inner tissue: ideally homogenous light solid blue  

Outer layer of cells:  

10% complexity stages 

Input image 

Preprocessing 

Crypt segmentation 

Lysozyme quantification 

Output metrics 

20% complexity Stages 

Initialization 

Specify parameters 

Scoring weights 

Appx crypt size 

Rfp and dapi keys 

Scale keys to scale ppm mapping 

Input data preparation and metadata parsing 

Split or combined channel recognition 

Split channel paring 

Path to metadata parsing 

Identifying info 

Rfp or dapi if split 

Scale based on keys in name 

Segmenting crypts 

Identify potential crypts and return selections as rois 

Identify seeds - Test 2 methods to identify “seeds” and returns product of better 

“new”: uses a modified morphological gradient ( big dilation – small dilation) + more for a very precise and adaptive segmentation of “bright” crypt regions 

Clean up 

Remove rois touching the image border 

Score and select the best potential crypts 

Data sources + samples 

- Type of data

- Acquisition, preprocessing, and any normalization or calibration steps.

- This would be a quick section with demo images on each step, i.e. if i can illustrate the resuts of the rfp normalization

- Extent of algo training

- None regarding ai weights, but a bunch of manual inspection and itter

- Samples: Describe the biological context (e.g., paraffin-embedded pig or mouse jejunum sections).

- What exact “training data” did i use to develop the algorithm ( i.e. which tissue did it come from, what was the “insult” and how did we expect it to impact the tissue

- Illeum

- Staining protocol: Lysozyme immunohistochemistry (or RFP-tagged reporter).

- Microscopy: e.g., brightfield or fluorescence, 40× magnification, TIFF format.

- Mix of 40x and 20x fluorescent images, primarily 20x

- Image acquisition parameters: pixel size (µm/pixel), illumination, exposure settings, etc.

- 

- Dataset size: total number of slides, replicates, conditions.

Key Morphological techniches 

Standards 

Obv there are the standard: 

- Dilation

- Erosion

- Opening

- Closing

Can prob just link over or embed skimage documentation 

Some slighltly less straightforward: 

- Local maxima

- Subtracting a white tophat to remove salt and pepper noise

- Label

- Distance transformation

- Watershed

Custom ops/variations 

The whole cap sequence 

Hats = (dilation with big r disk) – (dilation with small r disk)   
effectively cuts out the peak value regions of the image while surrounding their previous location with a high value gradient   
(technicaly it’s a modificaltion of a morphological gradient)   
( must include demo image here) 

Clean = image\_eq - np.minimum(image\_eq, hats) 

Method details 

- Preparation

- Images are collected,

- metadata embedded in the paths is ingested

- Scale (microns per pixel) are assigned based on the keys in image name

- Same with groups

- If images represent one of two required channels, they are paired together according to assigned keys: ["\_RFP", "\_DAPI"]

- Then single multichanel and split channel image pairs are collapsed together, to each subject\_name index ( unique to each image instance( or pair) the rfp and dapi channels are stored in seperate python variables for ease of analysis

- Per each subject

- Segment crypts

- Uses the dapi and rfp channels

- + an estimate of the appx crypt diameter to tune expansion

- High level:

- Manipulates the dapi and rfp data in order to isolate/exaserbate key features of the tissue which when combined are used to establish the locations and perimeters of the crypts

- Then using a set of (currently 5) manualy established weights it scores the identified potential crypt rois to find the best 5

- Segmented regions were ranked according to a weighted scoring system prioritizing morphological and intensity features. Features and weights were: circularity (0.35), area consistency (0.25), linear alignment along the gut wall (0.15), average RFP intensity (0.15), and center-of-mass consistency (0.10). The top five scoring regions were selected as putative crypts. Additionally, average background tissue intensity and crypt-specific RFP intensity were recorded for downstream normalization.

- The algorithm starts off by identifying crypt seeds

- Crypt seeds represent the non overlapping seperate regions of the image that most likely correspond with the center position of crypts in the tissue sample

- We look for a center approximate region to avoid burdening ourselves with the more complex task of separating adjacent crypts ( as regions are treated as homogenous by literal adgjacency for the sake of simplicity, attempting to classify the full region of a single crypt would loose the advantage that we have where borders between adjacent crypts are effectively implicit:

- i.e. if we said take all pixels that could be crypts it would result in one elongated blob as the regions between adjacent crypts that contains bleed signal would be picked up as well -- while when we look for only the most likely regions we get a “blob” located in the center of each crypt( when working optimaly) that is usually smaller in area then the crypt itself, so it wont include the borders between adjacent crypts ( which despite signal bleed will be low prob) -&gt; borders are much more implicit

- We test each of two methods designated “new” and “old” in order to identify these seeds

- Then the algorithm expands the crypt seeds to fully encompass the estimated area of the crypts

- It achives this by first identifieying  boundaries where we are certain that the crypts are not present, followed by a basic watershed expansion of a labeld array containing both the crypts and known non crypt regions

- Known non crypt regions are identified by:

- Scoring

- So far the resultant detections are composed of a combination of high quality detections and noise from random lysozyme+ “splotches”

- Identifying those “good” detections requires a quantification of qualitlly aka a scoring system

- Each detection is scored on:

- Line fit

- Distance of detection center from a line fit to the wighted centers of the lysozyme+ image

- Circularity

- Red value intensity

- estimate the true detected crypt count - Simpson-guided template aggregation and peak-preserving β-detection

- Although the algorithm is designed to identify the five best individual crypts per image, detection errors can arise from local variations in staining or signal quality. These errors manifest primarily as oversegmentation or undersegmentation, each introducing systematic biases in downstream statistics if left uncorrected.

- Oversegmentation occurs when a single biological crypt is divided into two or more separate detections.

- This typically arises in regions where the lysozyme signal within a crypt is fragmented, producing multiple apparent intensity peaks separated by distances similar to those between true neighboring crypts.

- In such cases, part of a single crypt may be counted more than once, leading either to (a) the exclusion of one fragment—causing an underestimation of signal intensity for that crypt—or (b) the inclusion of multiple fragments—causing an overcount that dilutes the overall average.

- Undersegmentation represents the opposite failure mode: multiple real crypts are merged into a single detection.

- This typically occurs when fluorescence from adjacent crypts bleeds together or when crypt boundaries are poorly defined.

- In practice, such merged detections tend to be localized, forming a small subset of the total detections.

- While they may be few in number, each can represent multiple true crypts—sometimes six or more within one detection.

- These merged detections also tend to exhibit lower circularity or shape regularity, which penalizes them in the scoring system, even though their summed fluorescence signal can be comparatively high.

- Impact on Quantitative Metrics

- Both error types—fragmentation and merging—distort the calculated averages derived from the five “best” detections.

- Since these statistics assume that each detection corresponds to exactly one true crypt, any deviation from that assumption directly skews the results, either over- or under-representing the true biological values.

- Rationale for a Morphology-Based Crypt Count Estimator

- To mitigate this, we introduce a morphological estimate of the number of true crypts detected within the final labeled image.

- The key observation is that, within a given image, the morphology of true crypts (area, width, aspect ratio, circularity) varies little compared to the variability introduced by segmentation errors.

- Thus, we can treat the average correctly segmented crypt—derived from the top-scoring detections—as a representative “template” crypt, denoted 𝛼

- To obtain an accurate representation of how many biologically distinct crypts are truly present, we quantify an effective number of crypts rather than relying on the raw detection count. This approach uses a Simpson-weighted sum as the mathematical backbone, combined with intensity-based template matching to integrate morphological and fluorescence information.

- Simpson-guided template aggregation and peak-preserving β-detection

- Overview (α → similarity → β → final count)

- Goal: robustly estimate the true crypt count despite α-over/under-segmentation.

- Key idea: use a Simpson-weighted selection of α detections to build a broad, score-weighted similarity map (less sensitive to one bad α), then convert that map to β peaks via cap/clean/troughs (+ Otsu), and finally compute an area-weighted Simpson effective count restricted to regions supported by the best α’s.

- Step 1 — Simpson-guided α selection (pre-template)

- Inputs: labeled α detections 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-1"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-2"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-3">L</span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><mi mathvariant="normal" class="SCXW20802579 BCX0">L</mi></math>
, RFP image 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-4"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-5"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-6">F</span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><mi mathvariant="normal" class="SCXW20802579 BCX0">F</mi></math>
.

- Per-label mass: 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-7"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-8"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-9"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-10">m</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-11">i</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-12">=</span><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-13"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-14">∑</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-15"><span class="mfenced SCXW20802579 BCX0" id="MathJax-Span-16"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-27"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0">(</span></span></span><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-18"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-19">x</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-20">,</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-21">y<span class="SCXW20802579 BCX0"></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-28"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0">)</span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-23">∈</span><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-24">label<span class="SCXW20802579 BCX0"> </span></span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-29">i</span></span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-30"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-31">m</span></span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-32">ax</span><span class="mfenced SCXW20802579 BCX0" id="MathJax-Span-33"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-34"><span class="SCXW20802579 BCX0">(</span></span><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-35"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-36">F<span class="SCXW20802579 BCX0"></span></span><span class="mfenced SCXW20802579 BCX0" id="MathJax-Span-37"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-38"><span class="SCXW20802579 BCX0">(</span></span><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-39"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-40">x</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-41">,</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-42">y<span class="SCXW20802579 BCX0"></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-43"><span class="SCXW20802579 BCX0">)</span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-44">,</span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-45">0</span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-46"><span class="SCXW20802579 BCX0">)</span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">m</mi><mi class="SCXW20802579 BCX0">i</mi></msub><mo class="SCXW20802579 BCX0">=</mo><msubsup class="SCXW20802579 BCX0"><mo stretchy="false" class="SCXW20802579 BCX0">∑</mo><mrow class="SCXW20802579 BCX0"><mfenced class="SCXW20802579 BCX0"><mrow class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">x</mi><mo class="SCXW20802579 BCX0">,</mo><mi class="SCXW20802579 BCX0">y</mi></mrow></mfenced><mo class="SCXW20802579 BCX0">∈</mo><mtext class="SCXW20802579 BCX0">label </mtext><mi class="SCXW20802579 BCX0">i</mi></mrow><mrow class="SCXW20802579 BCX0"></mrow></msubsup><mrow class="SCXW20802579 BCX0"><mi mathvariant="normal" class="SCXW20802579 BCX0">m</mi></mrow><mi class="SCXW20802579 BCX0">ax</mi><mfenced class="SCXW20802579 BCX0"><mrow class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">F</mi><mfenced class="SCXW20802579 BCX0"><mrow class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">x</mi><mo class="SCXW20802579 BCX0">,</mo><mi class="SCXW20802579 BCX0">y</mi></mrow></mfenced><mo class="SCXW20802579 BCX0">,</mo><mn class="SCXW20802579 BCX0">0</mn></mrow></mfenced></math>
.

- Proportions: 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-47"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-48"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-49"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-50">p</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-51">i</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-52">=</span><span class="msub SCXW20802579 BCX0" id="MathJax-Span-53"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-54">m</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-55">i</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-56">/</span><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-57"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-58">∑</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-59">j</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-60"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-61"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-62">m</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-63">j</span><span class="SCXW20802579 BCX0"></span></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">p</mi><mi class="SCXW20802579 BCX0">i</mi></msub><mo class="SCXW20802579 BCX0">=</mo><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">m</mi><mi class="SCXW20802579 BCX0">i</mi></msub><mo class="SCXW20802579 BCX0">/</mo><msubsup class="SCXW20802579 BCX0"><mo stretchy="false" class="SCXW20802579 BCX0">∑</mo><mi class="SCXW20802579 BCX0">j</mi><mrow class="SCXW20802579 BCX0"></mrow></msubsup><mrow class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">m</mi><mi class="SCXW20802579 BCX0">j</mi></msub></mrow></math>
.

- Simpson effective count (initial): 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-64"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-65"><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-66"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-67">N<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-68"><span class="mfenced SCXW20802579 BCX0" id="MathJax-Span-69"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-70"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0">(</span></span></span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-71">0</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-72"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0">)</span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-73"><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-74">eff</span></span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-75">=</span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-76">1</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-77">/</span><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-78"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-79">∑</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-80">i</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-81"><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-82"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-83">p</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mn SCXW20802579 BCX0" id="MathJax-Span-84">2</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-85">i</span><span class="SCXW20802579 BCX0"></span></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msubsup class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">N</mi><mrow class="SCXW20802579 BCX0"><mtext class="SCXW20802579 BCX0">eff</mtext></mrow><mrow class="SCXW20802579 BCX0"><mfenced class="SCXW20802579 BCX0"><mn class="SCXW20802579 BCX0">0</mn></mfenced></mrow></msubsup><mo class="SCXW20802579 BCX0">=</mo><mn class="SCXW20802579 BCX0">1</mn><mo class="SCXW20802579 BCX0">/</mo><msubsup class="SCXW20802579 BCX0"><mo stretchy="false" class="SCXW20802579 BCX0">∑</mo><mi class="SCXW20802579 BCX0">i</mi><mrow class="SCXW20802579 BCX0"></mrow></msubsup><mrow class="SCXW20802579 BCX0"><msubsup class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">p</mi><mi class="SCXW20802579 BCX0">i</mi><mn class="SCXW20802579 BCX0">2</mn></msubsup></mrow></math>
.

- Use: 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-86"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-87"><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-88"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-89">N<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-90"><span class="mfenced SCXW20802579 BCX0" id="MathJax-Span-91"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-92"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0">(</span></span></span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-93">0</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-94"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0">)</span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-95"><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-96">eff</span></span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msubsup class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">N</mi><mrow class="SCXW20802579 BCX0"><mtext class="SCXW20802579 BCX0">eff</mtext></mrow><mrow class="SCXW20802579 BCX0"><mfenced class="SCXW20802579 BCX0"><mn class="SCXW20802579 BCX0">0</mn></mfenced></mrow></msubsup></math>
is the target number of α templates to include in building the first similarity map (large enough to avoid bias from the 5 “best” only, small enough to exclude obvious speckle/tiny junk).

- Step 2 — Score-bounded α pool & per-α similarity maps

- Score and select: run the scorer to pick at most 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-97"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-98"><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-99"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-100">N<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-101"><span class="mfenced SCXW20802579 BCX0" id="MathJax-Span-102"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-103"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0">(</span></span></span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-104">0</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-105"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0">)</span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-106"><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-107">eff</span></span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msubsup class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">N</mi><mrow class="SCXW20802579 BCX0"><mtext class="SCXW20802579 BCX0">eff</mtext></mrow><mrow class="SCXW20802579 BCX0"><mfenced class="SCXW20802579 BCX0"><mn class="SCXW20802579 BCX0">0</mn></mfenced></mrow></msubsup></math>
α’s (quality-ordered); this caps how many α templates can influence the similarity map.

- Per-α template matching: for each selected 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-108"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-109"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-110"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-111">α</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-112">i</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">α</mi><mi class="SCXW20802579 BCX0">i</mi></msub></math>
, compute 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-113"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-114"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-115"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-116">S<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-117">i</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-118">=</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-119">match</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-120">_</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-121">template</span><span class="mfenced SCXW20802579 BCX0" id="MathJax-Span-122"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-123"><span class="SCXW20802579 BCX0">(</span></span><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-124"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-125">F</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-126">,</span><span class="msub SCXW20802579 BCX0" id="MathJax-Span-127"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-128">α</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-129">i</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-130"><span class="SCXW20802579 BCX0">)</span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">S</mi><mi class="SCXW20802579 BCX0">i</mi></msub><mo class="SCXW20802579 BCX0">=</mo><mi class="SCXW20802579 BCX0">match</mi><mo class="SCXW20802579 BCX0">_</mo><mi class="SCXW20802579 BCX0">template</mi><mfenced class="SCXW20802579 BCX0"><mrow class="SCXW20802579 BCX0"><mi mathvariant="normal" class="SCXW20802579 BCX0">F</mi><mo class="SCXW20802579 BCX0">,</mo><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">α</mi><mi class="SCXW20802579 BCX0">i</mi></msub></mrow></mfenced></math>

- Self-bias removal: in each 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-131"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-132"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-133"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-134">S<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-135">i</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">S</mi><mi class="SCXW20802579 BCX0">i</mi></msub></math>
, blank the template’s own support in label 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-136"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-137"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-138">i</span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><mi mathvariant="normal" class="SCXW20802579 BCX0">i</mi></math>
 and biharmonically inpaint to the local background.

- Rationale: prevents 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-139"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-140"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-141"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-142">α</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-143">i</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">α</mi><mi class="SCXW20802579 BCX0">i</mi></msub></math>
 from producing an artificial dominant peak at its origin, which would shift maxima scales and suppress legitimate non-home peaks.

- Step 3 — Score-weighted geometric collapse of similarity

- Quality weights: 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-144"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-145"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-146"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-147">w</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-148">i</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-149">∝</span><span class="msub SCXW20802579 BCX0" id="MathJax-Span-150"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-151">q<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-152">i</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-153"> </span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">𝒘</mi><mi class="SCXW20802579 BCX0">𝒊</mi></msub><mo class="SCXW20802579 BCX0">∝</mo><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">𝒒</mi><mi class="SCXW20802579 BCX0">𝒊</mi></msub><mo class="SCXW20802579 BCX0"> </mo></math>
from the scorer (normalize 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-154"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-155"><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-156"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-157">∑</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-158">i</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-159"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-160"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-161">w</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-162">i</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-163">=</span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-164">1</span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msubsup class="SCXW20802579 BCX0"><mo stretchy="false" class="SCXW20802579 BCX0">∑</mo><mi class="SCXW20802579 BCX0">i</mi><mrow class="SCXW20802579 BCX0"></mrow></msubsup><mrow class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">w</mi><mi class="SCXW20802579 BCX0">i</mi></msub></mrow><mo class="SCXW20802579 BCX0">=</mo><mn class="SCXW20802579 BCX0">1</mn></math>
).

- Collapse: 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-165"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-166"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-167"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-168">S<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-169"><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-170">coll</span></span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-171"><span class="SCXW20802579 BCX0">(</span></span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-172">x</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-173">,</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-174">y</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-175"><span class="SCXW20802579 BCX0">)</span></span><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-176"> </span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-177">=</span><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-178"> </span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-179">e</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-180">x</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-181">p</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-182"></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-183"><span class="SCXW20802579 BCX0">(</span></span><span class="munderover SCXW20802579 BCX0" id="MathJax-Span-184"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-185">∑</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-186">i</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-187"></span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-188"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-189"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-190">w</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-191">i</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-192"> </span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-193">l</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-194">o</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-195">g</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-196"></span><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-197"> </span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-198"><span class="SCXW20802579 BCX0">(</span></span><span class="msub SCXW20802579 BCX0" id="MathJax-Span-199"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-200">S<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-201">i</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-202"><span class="SCXW20802579 BCX0">(</span></span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-203">x</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-204">,</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-205">y</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-206"><span class="SCXW20802579 BCX0">)</span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-207">+</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-208">ϵ</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-209"><span class="SCXW20802579 BCX0">)</span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-210"><span class="SCXW20802579 BCX0">)</span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-211">.</span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">𝑺</mi><mrow class="SCXW20802579 BCX0"><mtext class="SCXW20802579 BCX0">coll</mtext></mrow></msub><mo fence="false" class="SCXW20802579 BCX0">(</mo><mi class="SCXW20802579 BCX0">𝒙</mi><mo class="SCXW20802579 BCX0">,</mo><mi class="SCXW20802579 BCX0">𝒚</mi><mo fence="false" class="SCXW20802579 BCX0">)</mo><mtext class="SCXW20802579 BCX0"> </mtext><mo class="SCXW20802579 BCX0">=</mo><mtext class="SCXW20802579 BCX0"> </mtext><mi class="SCXW20802579 BCX0">𝐞</mi><mi class="SCXW20802579 BCX0">𝐱</mi><mi class="SCXW20802579 BCX0">𝐩</mi><mo class="SCXW20802579 BCX0">⁡</mo><mo fence="false" class="SCXW20802579 BCX0">(</mo><munderover class="SCXW20802579 BCX0"><mo stretchy="false" class="SCXW20802579 BCX0">∑</mo><mi class="SCXW20802579 BCX0">𝒊</mi><mrow class="SCXW20802579 BCX0"></mrow></munderover><mrow class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">𝒘</mi><mi class="SCXW20802579 BCX0">𝒊</mi></msub></mrow><mtext class="SCXW20802579 BCX0"> </mtext><mi class="SCXW20802579 BCX0">𝐥</mi><mi class="SCXW20802579 BCX0">𝐨</mi><mi class="SCXW20802579 BCX0">𝐠</mi><mo class="SCXW20802579 BCX0">⁡</mo><mtext class="SCXW20802579 BCX0"> </mtext><mo fence="false" class="SCXW20802579 BCX0">(</mo><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">𝑺</mi><mi class="SCXW20802579 BCX0">𝒊</mi></msub><mo fence="false" class="SCXW20802579 BCX0">(</mo><mi class="SCXW20802579 BCX0">𝒙</mi><mo class="SCXW20802579 BCX0">,</mo><mi class="SCXW20802579 BCX0">𝒚</mi><mo fence="false" class="SCXW20802579 BCX0">)</mo><mo class="SCXW20802579 BCX0">+</mo><mi class="SCXW20802579 BCX0">𝝐</mi><mo fence="false" class="SCXW20802579 BCX0">)</mo><mo fence="false" class="SCXW20802579 BCX0">)</mo><mo class="SCXW20802579 BCX0">.</mo></math>

- Why geometric? similarities combine multiplicatively; this dampens outliers and lets many decent α’s outweigh a few shoddy ones (which receive low 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-212"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-213"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-214"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-215">w</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-216">i</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">w</mi><mi class="SCXW20802579 BCX0">i</mi></msub></math>
).

- Step 4 — Peak preservation: cap/clean/troughs → Otsu → β labels

- Contrast lift: apply the cap operator (two-scale dilation difference) to 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-217"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-218"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-219"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-220">S<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-221"><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-222">coll</span></span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-223"> </span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">S</mi><mrow class="SCXW20802579 BCX0"><mtext class="SCXW20802579 BCX0">coll</mtext></mrow></msub><mo class="SCXW20802579 BCX0"> </mo></math>
to get clean; this exaggerates inter-peak valleys across a range of intensities.

- Parameters: small radius 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-224"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-225"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-226"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-227">r</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-228">s</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-229">≈</span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-230">0</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-231">.</span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-232">1</span><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-233"> </span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-234">d<span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">r</mi><mi class="SCXW20802579 BCX0">s</mi></msub><mo class="SCXW20802579 BCX0">≈</mo><mn class="SCXW20802579 BCX0">0</mn><mo class="SCXW20802579 BCX0">.</mo><mn class="SCXW20802579 BCX0">1</mn><mtext class="SCXW20802579 BCX0"> </mtext><mi class="SCXW20802579 BCX0">d</mi></math>
, big radius 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-235"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-236"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-237"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-238">r</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-239">b</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-240">≈</span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-241">0</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-242">.</span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-243">5</span><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-244"> </span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-245">d<span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">r</mi><mi class="SCXW20802579 BCX0">b</mi></msub><mo class="SCXW20802579 BCX0">≈</mo><mn class="SCXW20802579 BCX0">0</mn><mo class="SCXW20802579 BCX0">.</mo><mn class="SCXW20802579 BCX0">5</mn><mtext class="SCXW20802579 BCX0"> </mtext><mi class="SCXW20802579 BCX0">d</mi></math>
, where 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-246"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-247"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-248">d<span class="SCXW20802579 BCX0"></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-249"> </span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">d</mi><mo class="SCXW20802579 BCX0"> </mo></math>
is the estimated crypt diameter; set 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-250"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-251"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-252">d<span class="SCXW20802579 BCX0"></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-253"> </span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">d</mi><mo class="SCXW20802579 BCX0"> </mo></math>
from microns/px or median α size.

- Threshold: Otsu on clean → binary peaks.

- Label: connected components → β detections (each β region ≈ one true crypt).

- Design intent: avoid watershed over-split of bridged peaks; preserve peak areas (critical for later Simpson weighting).

- Step 5 — α-supported β filtering (reduce stray peaks)

- Best-5 α gate: keep only β regions that overlap (even partially) with any of the top-5 α detections (by score).

- Rationale: (1) removes far-field speckle; (2) tolerates α’s clipped by edge-touch removal; (3) makes counting focus on tissue zones we trust.

- Step 6 — Final area-weighted Simpson crypt count on β

- Mass by area: for each retained 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-254"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-255"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-256"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-257">β<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-258">k</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">β</mi><mi class="SCXW20802579 BCX0">k</mi></msub></math>
, take 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-259"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-260"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-261"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-262">a</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-263">k</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-264">=</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-265"> </span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-266">∣</span><span class="msub SCXW20802579 BCX0" id="MathJax-Span-267"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-268">β<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-269">k</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-270">∣</span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">a</mi><mi class="SCXW20802579 BCX0">k</mi></msub><mo class="SCXW20802579 BCX0">=</mo><mo class="SCXW20802579 BCX0"> </mo><mo class="SCXW20802579 BCX0">∣</mo><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">𝛽</mi><mi class="SCXW20802579 BCX0">k</mi></msub><mo class="SCXW20802579 BCX0">∣</mo></math>
 (pixel area), optionally restricted to overlap with the union of chosen α’s (so edge-truncation doesn’t depress the context).

- Proportions: 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-271"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-272"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-273"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-274">p</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-275">k</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-276">=</span><span class="msub SCXW20802579 BCX0" id="MathJax-Span-277"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-278">a</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-279">k</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-280">/</span><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-281"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-282">∑</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-283">j</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-284"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-285"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-286">a</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-287">j</span><span class="SCXW20802579 BCX0"></span></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">p</mi><mi class="SCXW20802579 BCX0">k</mi></msub><mo class="SCXW20802579 BCX0">=</mo><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">a</mi><mi class="SCXW20802579 BCX0">k</mi></msub><mo class="SCXW20802579 BCX0">/</mo><msubsup class="SCXW20802579 BCX0"><mo stretchy="false" class="SCXW20802579 BCX0">∑</mo><mi class="SCXW20802579 BCX0">j</mi><mrow class="SCXW20802579 BCX0"></mrow></msubsup><mrow class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">a</mi><mi class="SCXW20802579 BCX0">j</mi></msub></mrow></math>
.

- Report:

- Primary: 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-288"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-289"><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-290"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-291">N<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-292"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-293">(</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-294">Simpson</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-295">)</span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-296"><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-297">eff</span></span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-298">=</span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-299">1</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-300">/</span><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-301"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-302">∑</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-303">k</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-304"><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-305"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-306">p</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mn SCXW20802579 BCX0" id="MathJax-Span-307">2</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-308">k</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-309"> </span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msubsup class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">N</mi><mrow class="SCXW20802579 BCX0"><mtext class="SCXW20802579 BCX0">eff</mtext></mrow><mrow class="SCXW20802579 BCX0"><mo fence="false" class="SCXW20802579 BCX0">(</mo><mi class="SCXW20802579 BCX0">Simpson</mi><mo fence="false" class="SCXW20802579 BCX0">)</mo></mrow></msubsup><mo class="SCXW20802579 BCX0">=</mo><mn class="SCXW20802579 BCX0">1</mn><mo class="SCXW20802579 BCX0">/</mo><msubsup class="SCXW20802579 BCX0"><mo stretchy="false" class="SCXW20802579 BCX0">∑</mo><mi class="SCXW20802579 BCX0">k</mi><mrow class="SCXW20802579 BCX0"></mrow></msubsup><mrow class="SCXW20802579 BCX0"><msubsup class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">p</mi><mi class="SCXW20802579 BCX0">k</mi><mn class="SCXW20802579 BCX0">2</mn></msubsup></mrow><mo class="SCXW20802579 BCX0"> </mo></math>
(effective number of similarly-sized β peaks).

- Also: 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-310"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-311"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-312"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-313">K<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-314">β<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">K</mi><mi class="SCXW20802579 BCX0">β</mi></msub></math>
 (raw β count), 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-315"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-316"><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-317"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-318">N<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-319"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-320">(</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-321">Shannon</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-322">)</span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-323"><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-324">eff</span></span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-325">=</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-326">exp</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-327"></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-328"><span class="SCXW20802579 BCX0">(</span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-329">−</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-330">∑</span><span class="msub SCXW20802579 BCX0" id="MathJax-Span-331"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-332">p</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-333">k</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-334">log</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-335"></span><span class="msub SCXW20802579 BCX0" id="MathJax-Span-336"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-337">p</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-338">k</span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-339"><span class="SCXW20802579 BCX0">)</span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msubsup class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">N</mi><mrow class="SCXW20802579 BCX0"><mtext class="SCXW20802579 BCX0">eff</mtext></mrow><mrow class="SCXW20802579 BCX0"><mo fence="false" class="SCXW20802579 BCX0">(</mo><mi class="SCXW20802579 BCX0">Shannon</mi><mo fence="false" class="SCXW20802579 BCX0">)</mo></mrow></msubsup><mo class="SCXW20802579 BCX0">=</mo><mi class="SCXW20802579 BCX0">exp</mi><mo class="SCXW20802579 BCX0">⁡</mo><mo fence="false" class="SCXW20802579 BCX0">(</mo><mo class="SCXW20802579 BCX0">−</mo><mo largeop="false" class="SCXW20802579 BCX0">∑</mo><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">p</mi><mi class="SCXW20802579 BCX0">k</mi></msub><mi class="SCXW20802579 BCX0">log</mi><mo class="SCXW20802579 BCX0">⁡</mo><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">p</mi><mi class="SCXW20802579 BCX0">k</mi></msub><mo fence="false" class="SCXW20802579 BCX0">)</mo></math>
, and evenness 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-340"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-341"><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-342"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-343">N<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-344"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-345">(</span><span class="mi SCXW20802579 BCX0" id="MathJax-Span-346">Simpson</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-347">)</span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-348"><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-349">eff</span></span><span class="SCXW20802579 BCX0"></span></span></span></span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-350">/</span><span class="msub SCXW20802579 BCX0" id="MathJax-Span-351"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-352">K<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-353">β<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msubsup class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">N</mi><mrow class="SCXW20802579 BCX0"><mtext class="SCXW20802579 BCX0">eff</mtext></mrow><mrow class="SCXW20802579 BCX0"><mo fence="false" class="SCXW20802579 BCX0">(</mo><mi class="SCXW20802579 BCX0">Simpson</mi><mo fence="false" class="SCXW20802579 BCX0">)</mo></mrow></msubsup><mo class="SCXW20802579 BCX0">/</mo><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">K</mi><mi class="SCXW20802579 BCX0">β</mi></msub></math>
.

- Behavior on α failure modes (why this works)

- α oversegmentation (one true crypt → multiple α’s):

- After Step 2–3, bridged similarity around the true center forms one dominant β peak (self-bias removed; cap enhances valley between fragments).

- With sufficiently large 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-354"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-355"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-356"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-357">r</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-358">b</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">r</mi><mi class="SCXW20802579 BCX0">b</mi></msub></math>
, a single β emerges between the α fragments; contributes ≈ one crypt to Simpson.

- α undersegmentation (many true crypts → one α):

- Multiple β peaks of similar area appear within the merged α ROI; Simpson area-weighting counts them correctly.

- Small speckles carry tiny area → minimal Simpson impact.

- Practical parameter notes

- Choosing 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-359"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-360"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-361"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-362">r</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-363">b</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">r</mi><mi class="SCXW20802579 BCX0">b</mi></msub></math>
: set from crypt diameter (pixels) or median α size; too small → bridged peaks don’t separate; large enough → valleys deepen and Otsu succeeds.

- How many α into similarity: the initial Simpson estimate 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-374"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-375"><span class="msubsup SCXW20802579 BCX0" id="MathJax-Span-376"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-377">N<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-378"><span class="mo SCXW20802579 BCX0" id="MathJax-Span-379">(</span><span class="mn SCXW20802579 BCX0" id="MathJax-Span-380">0</span><span class="mo SCXW20802579 BCX0" id="MathJax-Span-381">)</span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-382"><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-383">eff</span></span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msubsup class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">N</mi><mrow class="SCXW20802579 BCX0"><mtext class="SCXW20802579 BCX0">eff</mtext></mrow><mrow class="SCXW20802579 BCX0"><mo fence="false" class="SCXW20802579 BCX0">(</mo><mn class="SCXW20802579 BCX0">0</mn><mo fence="false" class="SCXW20802579 BCX0">)</mo></mrow></msubsup></math>
avoids the “top-5 bias” that would over-imprint α fragments and make the whole field look oversegmented.

- Weights 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-364"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-365"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-366"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-367">w</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-368">i</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">w</mi><mi class="SCXW20802579 BCX0">i</mi></msub></math>
: low-quality α’s are punished (small 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-369"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-370"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-371"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-372">w</span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-373">i</span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">w</mi><mi class="SCXW20802579 BCX0">i</mi></msub></math>
), so they don’t dominate the collapsed map but still help populate valid similarities in crowded fields.

- Known limitations and current mitigations

- Missed α fragments not in the “best-5” gate: a β peak might represent a partial crypt if its supporting α fragments weren’t selected.

- Mitigation: (planned) allow fractional β credit when overlap with α support is incomplete, but still include the full β area in Simpson to maintain a consistent crypt-size context.

- Sparse/low-signal images: if only a few α succeed, similarity variability rises and some α may yield no β.

- Often coincides with genuinely abysmal fluorescence in that crypt; we report 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-384"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-385"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-386"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-387">K<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-388">β<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">K</mi><mi class="SCXW20802579 BCX0">𝛽</mi></msub></math>
and 
<nobr aria-hidden="true" class="SCXW20802579 BCX0"><span class="math SCXW20802579 BCX0" id="MathJax-Span-389"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-390"><span class="msub SCXW20802579 BCX0" id="MathJax-Span-391"><span class="SCXW20802579 BCX0"><span class="SCXW20802579 BCX0"><span class="mi SCXW20802579 BCX0" id="MathJax-Span-392">N<span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"></span></span><span class="SCXW20802579 BCX0"><span class="mrow SCXW20802579 BCX0" id="MathJax-Span-393"><span class="mtext SCXW20802579 BCX0" id="MathJax-Span-394">eff</span></span><span class="SCXW20802579 BCX0"></span></span></span></span></span><span class="SCXW20802579 BCX0"></span></span></span><span class="SCXW20802579 BCX0"></span></span></nobr><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="SCXW20802579 BCX0"><msub class="SCXW20802579 BCX0"><mi class="SCXW20802579 BCX0">N</mi><mrow class="SCXW20802579 BCX0"><mtext class="SCXW20802579 BCX0">eff</mtext></mrow></msub></math>
alongside quality diagnostics so users can flag such fields.

- One-paragraph “what to cite” summary

- We compute an initial Simpson effective count over α detections (mass = RFP signal) to set a cap on how many α templates contribute to similarity. We then build per-α normalized cross-correlation maps, remove self-bias via inpainting at α origins, and merge maps with a quality-weighted geometric mean. A cap/clean/troughs operator enhances inter-peak valleys, after which Otsu yields β peak regions whose areas drive a final Simpson effective count, restricted to β’s overlapping the best-5 α. This pipeline converts noisy α segmentation into a peak-preserving, area-weighted estimate of true crypt number that is robust to both α over- and under-segmentation.

- normalize crypt flouresesnce

- Since the contrast between crypt and non-crypt staining can vary significantly between images due to differences in staining efficiency, imaging conditions, and tissue autofluorescence, we implement a sophisticated normalization approach to enable meaningful cross-image comparison.

- To account for slide-to-slide variability in staining and imaging conditions, RFP signal was standardized relative to DAPI. Specifically, the ratio of RFP to DAPI intensities was computed separately for crypt and non-crypt tissue regions, and the ratio between these two values was used to apply a gain-like scaling factor. This ensured that crypt RFP intensities were comparable across images, independent of global signal variation.

- Output data:

- Qc renderings + csv’s