#!/usr/bin/env python3
"""
Standalone launcher for the visual inspector tool.
Run this to review processed images without reprocessing.
"""

import sys
from pathlib import Path

# Add the src directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

from visual_inspector import VisualInspector

def main():
    """Main entry point."""
    # Default results directory
    default_results = Path(r"/home/phillip/documents/lysozyme/results/simple_dask")
    default_metadata = default_results / "karen_detect_crypts.csv"
    
    print("=== Lysozyme Stain Visual Inspector ===")
    print()
    
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        print(f"Using default results directory: {default_results}")
        print("(Pass a different path as argument if needed)")
        print()
        results_dir = default_results
    
    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        print("Make sure you have run the main processing pipeline first.")
        sys.exit(1)
    
    if len(sys.argv) > 2:
        metadata_csv = Path(sys.argv[2])
    else:
        metadata_csv = default_metadata if default_metadata.exists() else None
    
    print(f"Loading results from: {results_dir}")
    if metadata_csv and metadata_csv.exists():
        print(f"Using metadata CSV: {metadata_csv}")
    elif metadata_csv:
        print(f"Warning: Metadata CSV not found at {metadata_csv}. Continuing without metadata file.")
        metadata_csv = None
    print()
    
    try:
        inspector = VisualInspector(results_dir, metadata_csv=metadata_csv)
        inspector.run()
    except KeyboardInterrupt:
        print("\nInspection interrupted by user.")
    except Exception as e:
        print(f"Error running visual inspector: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
