from __future__ import annotations

# # ─────────────────────────────────── stdlib ────────────────────────────────────
# import os, gc, json
from pathlib import Path
from typing import List, Dict, Any
from ThingDetector import BulkBlobProcessor


if __name__ == "__main__":
    #from tools.expand import BulkBlobProcessor
    import glob

    imgs = glob.glob("QupathProj-Stt4 Lysozyme stain quantification/Stt4 Lysozyme stain quantification/*/*.tif")

    summaries = BulkBlobProcessor(
        img_paths=imgs,
        out_root="lysozyme-stain-quantification/results",
        debug=True           # prints progress
    ).process_all()

    print("Master summary JSON:", summaries[0].keys())
    