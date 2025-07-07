import os
import sys
from collections import defaultdict

def find_all_files(root_dir):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root_dir)
            files.append((rel_path, full_path))
    return files

def get_new_name(parts, levels_up):
    # parts: ['sub1', 'sub2', 'file.txt']
    if levels_up == 0:
        return parts[-1]
    name, ext = os.path.splitext(parts[-1])
    folders = parts[-(levels_up+1):-1]
    suffix = "_".join(folders)
    if suffix:
        return f"{name}_{suffix}{ext}"
    else:
        return parts[-1]

def resolve_duplicates(files, root_dir):
    # files: list of (rel_path, full_path)
    name_to_paths = defaultdict(list)
    for rel_path, full_path in files:
        name_to_paths[os.path.basename(rel_path)].append((rel_path, full_path))

    # Find duplicates
    duplicates = {k: v for k, v in name_to_paths.items() if len(v) > 1}
    renamed = {}

    for name, paths in duplicates.items():
        levels_up = 1
        still_duplicates = True
        rel_parts = [p[0].split(os.sep) for p in paths]
        while still_duplicates:
            new_names = [get_new_name(parts, levels_up) for parts in rel_parts]
            if len(set(new_names)) == len(new_names):
                still_duplicates = False
            else:
                levels_up += 1
                # If we reach the root, just append a unique number
                if levels_up > max(len(parts)-1 for parts in rel_parts):
                    new_names = [f"{os.path.splitext(name)[0]}_{i}{os.path.splitext(name)[1]}" for i in range(len(paths))]
                    still_duplicates = False
        # Rename files
        for (rel_path, full_path), new_name in zip(paths, new_names):
            dir_path = os.path.dirname(full_path)
            new_full_path = os.path.join(dir_path, new_name)
            if full_path != new_full_path:
                os.rename(full_path, new_full_path)
                renamed[full_path] = new_full_path
    return renamed

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python anti-repeat-fixer.py <directory>")
        sys.exit(1)
    root_dir = sys.argv[1]
    files = find_all_files(root_dir)
    renamed = resolve_duplicates(files, root_dir)
    for old, new in renamed.items():
        print(f"Renamed: {old} -> {new}")