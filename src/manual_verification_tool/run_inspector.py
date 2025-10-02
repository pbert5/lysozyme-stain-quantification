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
    default_results = Path(r"C:\Users\admin\Documents\Pierre lab\projects\Colustrum-ABX\lysozyme stain quantification\results\All")
    
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
    
    # Check for required files
    summaries_dir = results_dir / 'summaries'
    if not (summaries_dir / 'consolidated_summary.csv').exists():
        print(f"Error: No consolidated_summary.csv found in {summaries_dir}")
        print("Make sure you have run the main processing pipeline first.")
        sys.exit(1)
    
    print(f"Loading results from: {results_dir}")
    print()
    
    try:
        inspector = VisualInspector(results_dir)
        inspector.run()
    except KeyboardInterrupt:
        print("\nInspection interrupted by user.")
    except Exception as e:
        print(f"Error running visual inspector: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
