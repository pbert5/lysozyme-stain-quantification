from tifffile import TiffFile
import sys
from pprint import pprint

def inspect_tiff(path):
    print(f"Opening file: {path}\n")

    with TiffFile(path) as tif:
        print(f"Number of pages: {len(tif.pages)}")
        print(f"Is BigTIFF: {tif.is_bigtiff}")
        print(f"Is OME-TIFF: {tif.is_ome}")
        print(f"Is Tiled: {any(p.is_tiled for p in tif.pages)}\n")

        for i, page in enumerate(tif.pages):
            print(f"--- Page {i} ---")
            print(f"  Shape: {page.shape}")
            print(f"  Dtype: {page.dtype}")
            print(f"  Axes: {page.axes}")
            print(f"  Tiled: {page.is_tiled}")
            print(f"  Compression: {page.compression}")
            print(f"  Resolution (X, Y): {page.tags.get('XResolution', 'N/A')}, {page.tags.get('YResolution', 'N/A')}")
            print(f"  Description: {page.description!r}")
            print("  Tags:")
            for tag in page.tags.values():
                name, value = tag.name, tag.value
                print(f"    {name}: {repr(value)[:200]}")  # Trim long entries
            print()

        if tif.is_ome:
            ome_metadata = tif.ome_metadata
            print("--- OME Metadata ---")
            print(ome_metadata[:2000])  # Trimmed for readability
            print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_tif.py <path_to_file.tif>")
    else:
        inspect_tiff(sys.argv[1])
