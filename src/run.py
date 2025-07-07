print("Running BulkBlobProcessor...")
from pathlib import Path

from primary import BulkBlobProcessor 

# 1) Define your inputs:
img_dir    = Path("/home/user/nfs/analysisdata/Stt4 Lysozyme stain quantification/Lysozome images")       # ← change this
img_glob   = "**/*.tif"                            # ← or "*.png", etc.
img_paths  = sorted(img_dir.glob(img_glob))

# 2) Define output + options:
out_root   = Path("/home/user/nfs/analysisdata/Stt4 Lysozyme stain quantification/results")                    # where to save
results_dir = None                              # not used for now
expand_by  = 1.0                                # how much to expand each ROI
debug      = True                               # dump debug artifacts?
singleton_penalty = 4                                # proportion of how much more perimeter needs to be in contact then not for merge to happen

# 3) Instantiate & run:
# Ensure img_paths is a list of strings
max_images = 200  # Set the maximum number of images to process for testing
img_paths = [
    Path(p) for p in img_paths[:max_images]
    if not (str(p).endswith("_DAPI.tif") or str(p).endswith("_RFP.tif"))
]

# Ensure results_dir is a string if it's not None
results_dir = Path(results_dir) if results_dir else None

# Convert expand_by to an integer
ROI_expand_by = int(expand_by)

processor = BulkBlobProcessor(
    img_paths=img_paths,
    out_root=out_root,
    results_dir=results_dir,
    ROI_expand_by=ROI_expand_by,
    debug=debug,
)
print(f"Processing {len(processor.paths)} images...")
processor.load_images()     # read all images into memory
print("Loaded images, starting processing...")
processor.process_all(low_thresh=30, high_thresh=150, singleton_penalty=singleton_penalty)     # run your BlobDetector on each
print("Processing complete, saving results...")
processor.save_results()    # emit .geojson, .npy, and prints
print("Results saved, generating visuals...")
processor.save_visuals(out_root / "quick_check")
print("Visuals saved, all done!")

# Call the primary function from lysozyme-stain-quantification/src/primary.py


