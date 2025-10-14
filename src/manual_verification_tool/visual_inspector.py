#!/usr/bin/env python3
"""
Visual inspection tool for reviewing processed lysozyme stain images.
Loads pre-generated visualization images and allows manual quality assessment.
"""

import os
import sys
import json
import atexit
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from datetime import datetime

# Configure matplotlib backend with headless-friendly fallbacks.
_WEB_BACKEND_ACTIVE = False
if os.environ.get("MPLBACKEND"):
    matplotlib.use(os.environ["MPLBACKEND"])
else:
    backend_set = False
    display_available = (
        sys.platform.startswith("win")
        or any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET"))
    )
    if display_available:
        for candidate in ("Qt5Agg", "TkAgg"):
            try:
                matplotlib.use(candidate)
                backend_set = True
                break
            except Exception:
                continue
    if not backend_set:
        try:
            matplotlib.use("WebAgg")
            _WEB_BACKEND_ACTIVE = True
        except Exception:
            matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class VisualInspector:
    """Tool for visually inspecting processed images using pre-generated visualizations."""
    
    def __init__(self, results_dir, load_existing=True, metadata_csv=None, output_dir=None, image_patterns=None):
        """
        Initialize the visual inspector.
        
        Args:
            results_dir: Path to results directory or renderings folder containing visualizations.
            load_existing: Whether to load existing inspection results if present.
            metadata_csv: Optional explicit path to metadata CSV for extra context.
            output_dir: Optional directory where rating progress files should be written.
            image_patterns: Optional iterable of glob patterns for locating visualization images.
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")
        
        self.image_patterns = list(image_patterns or [
            "*_detected_regions.png",
            "*_RFP_detected_regions.png",
            "*_crypt_overlay.png",
            "*_overlay.png"
        ])
        
        self.image_dir = self._resolve_image_dir()
        self.output_dir = Path(output_dir) if output_dir else self._resolve_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_csv = Path(metadata_csv) if metadata_csv else self._resolve_metadata_csv()
        self.summary_df = self._load_summary()
        self.metadata_mode = 'none'
        self.metadata_lookup = {}
        self.metadata_columns = []
        self._build_metadata_lookup()
        
        self.image_files = self._find_visualization_images()
        
        # Initialize inspection state
        self.current_index = 0
        self.ratings = {}  # image_name -> True/False/None
        self.session_start = datetime.now()
        self.ratings_since_save = 0  # Track ratings since last save for auto-save
        self.auto_save_interval = 5  # Auto-save every 5 ratings
        
        # Progress file configuration
        self.ratings_stem = 'manual_verification_ratings'
        self.ratings_file = self.output_dir / f"{self.ratings_stem}.json"
        self.ratings_csv_file = self.output_dir / f"{self.ratings_stem}.csv"
        
        if load_existing:
            self._load_existing_ratings()
        
        # Set up matplotlib
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('close_event', self._on_close)  # Handle window close
        
        # Register emergency save on program exit
        atexit.register(self._emergency_exit_save)
        
        print("Visual Inspector initialized")
        print(f"Image directory: {self.image_dir}")
        if self.metadata_csv:
            print(f"Metadata CSV: {self.metadata_csv}")
        print(f"Progress files will be saved to: {self.output_dir}")
        print(f"Found {len(self.image_files)} visualization images")
        print()
        print(f"IMPORTANT: Your work will auto-save every {self.auto_save_interval} ratings!")
        print()
        print("Controls:")
        print("  ↑ Arrow = Mark as GOOD and move to next")
        print("  ↓ Arrow = Mark as BAD and move to next")
        print("  → Arrow = Skip to next (no rating)")
        print("  ← Arrow = Go back to previous")
        print("  'r' key = Reset current image rating")
        print("  's' key = Save progress manually")
        print("  'q' key = Quit and save all ratings")
        print()
        backend_name = matplotlib.get_backend()
        print(f"Matplotlib backend: {backend_name}")
        if _WEB_BACKEND_ACTIVE:
            print("Web backend active. After launching, forward the listed port (default 8988) "
                  "through VS Code's Ports view and open the URL in your local browser.")
        elif backend_name.lower() == "agg":
            print("Warning: Agg backend is non-interactive. Set MPLBACKEND or enable X11/Wayland "
                  "forwarding to use the visual inspector interactively.")
        print()
        print("🔥 PRESS 'q' TO QUIT AND SAVE WHEN DONE! 🔥")
        print("=" * 60)
    
    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------
    
    def _resolve_image_dir(self):
        """Determine which directory contains the visualization images."""
        visualizations_dir = self.results_dir / 'visualizations'
        if visualizations_dir.exists():
            return visualizations_dir
        
        renderings_dir = self.results_dir / 'renderings'
        if renderings_dir.exists():
            return renderings_dir
        
        if self.results_dir.name.lower() in {'renderings', 'visualizations'}:
            return self.results_dir
        
        return self.results_dir
    
    def _resolve_output_dir(self):
        """Pick the directory where progress files should be written."""
        summaries_dir = self.results_dir / 'summaries'
        if summaries_dir.exists():
            return summaries_dir
        
        if self.results_dir.name.lower() in {'renderings', 'visualizations'}:
            return self.results_dir.parent
        
        return self.results_dir
    
    def _resolve_metadata_csv(self):
        """Find a metadata CSV if available."""
        legacy_summary = self.results_dir / 'summaries' / 'consolidated_summary.csv'
        if legacy_summary.exists():
            return legacy_summary
        
        candidate_names = ['karen_detect_crypts.csv']
        for name in candidate_names:
            candidate = self.results_dir / name
            if candidate.exists():
                return candidate
            
            if self.results_dir.name.lower() in {'renderings', 'visualizations'}:
                parent_candidate = self.results_dir.parent / name
                if parent_candidate.exists():
                    return parent_candidate
        
        if self.results_dir.name.lower() in {'renderings', 'visualizations'}:
            for candidate in self.results_dir.parent.glob('*.csv'):
                if 'detect_crypts' in candidate.name:
                    return candidate
        
        return None
    
    # ------------------------------------------------------------------
    # Metadata loading and lookup
    # ------------------------------------------------------------------
    
    def _load_summary(self):
        """Load metadata for display if available."""
        if self.metadata_csv and self.metadata_csv.exists():
            try:
                df = pd.read_csv(self.metadata_csv)
                print(f"Loaded metadata from {self.metadata_csv}")
                return df
            except Exception as exc:
                print(f"Warning: Could not load metadata CSV ({self.metadata_csv}): {exc}")
        
        print("Warning: No metadata CSV found. Metadata will be unavailable.")
        return pd.DataFrame()
    
    def _build_metadata_lookup(self):
        """Build lookup tables for metadata display."""
        self.metadata_lookup = {}
        self.metadata_columns = list(self.summary_df.columns)
        
        if self.summary_df.empty:
            self.metadata_mode = 'none'
            return
        
        if 'subject_name' in self.summary_df.columns:
            records = self.summary_df.to_dict(orient='records')
            for record in records:
                key = self._sanitize_subject_name(str(record['subject_name']))
                self.metadata_lookup[key] = record
            self.metadata_mode = 'karen'
            return
        
        if 'image_name' in self.summary_df.columns:
            self.metadata_mode = 'legacy'
            # For legacy exports we aggregate on demand, but still keep expected column order.
            self.metadata_columns = ['image_name', 'n_regions', 'subdir', 'avg_intensity', 'total_area']
            return
        
        self.metadata_mode = 'unknown'
    
    # ------------------------------------------------------------------
    # Image discovery and ratings persistence
    # ------------------------------------------------------------------
    
    def _find_visualization_images(self):
        """Find visualization images."""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Visualization directory not found: {self.image_dir}")
        
        image_files = []
        for pattern in self.image_patterns:
            image_files.extend(self.image_dir.glob(pattern))
        
        if not image_files:
            image_files = list(self.image_dir.glob("*.png"))
        
        if not image_files:
            raise FileNotFoundError(f"No visualization images found in {self.image_dir}")
        
        return sorted(set(image_files))
    
    def _load_existing_ratings(self):
        """Load existing ratings from previous session if available."""
        if self.ratings_file.exists():
            try:
                with open(self.ratings_file, 'r') as fh:
                    data = json.load(fh)
                    self.ratings = data.get('ratings', {})
                    print(f"Loaded {len(self.ratings)} existing ratings from previous session")
            except Exception as exc:
                print(f"Warning: Could not load existing ratings: {exc}")
    
    def _save_ratings(self, quiet_mode=False):
        """Save current ratings to file."""
        try:
            rated_values = [r for r in self.ratings.values() if r is not None]
            data = {
                'session_start': self.session_start.isoformat(),
                'last_updated': datetime.now().isoformat(),
                'total_images': len(self.image_files),
                'rated_images': len(rated_values),
                'ratings': self.ratings,
                'rating_summary': {
                    'good': len([r for r in rated_values if r is True]),
                    'bad': len([r for r in rated_values if r is False]),
                    'unrated': len(self.image_files) - len(rated_values)
                }
            }
            
            with open(self.ratings_file, 'w') as fh:
                json.dump(data, fh, indent=2)
            
            if not quiet_mode:
                print(f"Ratings saved to {self.ratings_file}")
            
            self.ratings_since_save = 0
            
            records = []
            for img_file in self.image_files:
                rating_value = self.ratings.get(img_file.name)
                metadata = self._get_image_metadata(img_file.name)
                record = {
                    'image_file': img_file.name,
                    'display_name': metadata.get('display_name', ''),
                    'metadata_key': metadata.get('metadata_key', ''),
                    'rating_bool': rating_value,
                    'rating_text': 'Good' if rating_value is True else
                                   'Bad' if rating_value is False else 'Unrated'
                }
                
                raw_metadata = metadata.get('record') or {}
                for column in self.metadata_columns:
                    if column in raw_metadata:
                        record[column] = raw_metadata[column]
                records.append(record)
            
            ratings_df = pd.DataFrame(records)
            ratings_df.to_csv(self.ratings_csv_file, index=False)
            if not quiet_mode:
                print(f"Ratings also saved to {self.ratings_csv_file}")
        
        except Exception as exc:
            print(f"Error saving ratings: {exc}")
    
    def _auto_save_check(self):
        """Check if we should auto-save and do it if needed."""
        if self.ratings_since_save >= self.auto_save_interval:
            rated_count = len([r for r in self.ratings.values() if r is not None])
            print(f"💾 Auto-saving progress... ({rated_count} images rated)")
            self._save_ratings(quiet_mode=True)
    
    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    
    def _extract_image_name(self, filename):
        """Extract clean image name from visualization filename."""
        cleaned = filename
        suffixes = [
            '_RFP_detected_regions.png',
            '_detected_regions.png',
            '_crypt_overlay.png',
            '_overlay.png',
            '.png'
        ]
        for suffix in suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)]
                break
        return cleaned
    
    def _sanitize_subject_name(self, subject_name):
        """Normalize subject names to match rendered image file prefixes."""
        s = subject_name.strip()
        s = s.replace(' - ', '_-_')
        s = s.replace(' ', '_')
        s = s.replace('/', '_')
        for ch in '[]+':
            s = s.replace(ch, '')
        while '__' in s:
            s = s.replace('__', '_')
        return s
    
    def _get_metadata_key_from_filename(self, filename):
        """Derive the lookup key for metadata from the image filename."""
        name = self._extract_image_name(filename)
        if self.metadata_mode == 'karen' and '-' in name:
            name = name.rsplit('-', 1)[0]
        return name
    
    def _format_float(self, value, decimals=1):
        """Format numeric values while handling missing data."""
        if value is None:
            return "Unknown"
        try:
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            if isinstance(value, str):
                value = float(value)
            if isinstance(value, float) and np.isnan(value):
                return "Unknown"
            return f"{float(value):.{decimals}f}"
        except Exception:
            return str(value)
    
    def _format_int(self, value):
        """Format integer-like values."""
        if value is None:
            return "Unknown"
        try:
            if isinstance(value, str) and value.strip() == '':
                return "Unknown"
            numeric = float(value)
            if np.isnan(numeric):
                return "Unknown"
            return str(int(round(numeric)))
        except Exception:
            return str(value)
    
    def _get_image_metadata(self, filename):
        """Get metadata for current image based on available summary data."""
        metadata_key = self._get_metadata_key_from_filename(filename)
        
        if self.metadata_mode == 'karen':
            record = self.metadata_lookup.get(metadata_key)
            if record:
                subject_name = record.get('subject_name', metadata_key)
                info_lines = [
                    f"Crypts: {self._format_int(record.get('crypt_count'))} | "
                    f"Area μm² mean±std: {self._format_float(record.get('crypt_area_um2_mean'), 1)} ± "
                    f"{self._format_float(record.get('crypt_area_um2_std'), 1)}",
                    f"RFP mean: {self._format_float(record.get('rfp_intensity_mean'), 3)} | "
                    f"RFP max mean: {self._format_float(record.get('rfp_max_intensity_mean'), 3)} | "
                    f"μm/px: {self._format_float(record.get('microns_per_px'), 4)}"
                ]
                return {
                    'display_name': subject_name,
                    'retake': 'RETAKE' in subject_name.upper(),
                    'info_lines': info_lines,
                    'record': record,
                    'metadata_key': metadata_key
                }
        
        if self.metadata_mode == 'legacy' and not self.summary_df.empty:
            rows = self.summary_df[self.summary_df['image_name'] == metadata_key]
            if not rows.empty:
                n_regions = len(rows)
                first_row = rows.iloc[0]
                is_retake = bool(first_row.get('is_retake', False))
                subdir = first_row.get('subdir', 'Unknown')
                avg_intensity = rows['red_intensity'].mean() if 'red_intensity' in rows else np.nan
                total_area = rows['area_um2'].sum() if 'area_um2' in rows else np.nan
                
                info_lines = [
                    f"Regions: {n_regions} | Dir: {subdir}",
                    f"Avg Intensity: {self._format_float(avg_intensity, 1)} | "
                    f"Total Area: {self._format_float(total_area, 1)} μm²"
                ]
                record = {
                    'image_name': metadata_key,
                    'n_regions': n_regions,
                    'subdir': subdir,
                    'avg_intensity': avg_intensity,
                    'total_area': total_area
                }
                return {
                    'display_name': metadata_key,
                    'retake': is_retake,
                    'info_lines': info_lines,
                    'record': record,
                    'metadata_key': metadata_key
                }
        
        return {
            'display_name': metadata_key,
            'retake': 'RETAKE' in metadata_key.upper(),
            'info_lines': ["Metadata unavailable"],
            'record': None,
            'metadata_key': metadata_key
        }
    
    # ------------------------------------------------------------------
    # Display and interaction
    # ------------------------------------------------------------------
    
    def display_image(self):
        """Display the current image with metadata and rating status."""
        if self.current_index >= len(self.image_files):
            self.current_index = len(self.image_files) - 1
        if self.current_index < 0:
            self.current_index = 0
        
        current_file = self.image_files[self.current_index]
        
        try:
            img = mpimg.imread(current_file)
            self.ax.clear()
            self.ax.imshow(img)
            self.ax.axis('off')
            
            metadata = self._get_image_metadata(current_file.name)
            display_name = metadata.get('display_name', current_file.name)
            retake_text = " [RETAKE]" if metadata.get('retake', False) else ""
            info_lines = metadata.get('info_lines', [])
            
            current_rating = self.ratings.get(current_file.name)
            rating_text = "GOOD" if current_rating is True else "BAD" if current_rating is False else "UNRATED"
            rating_color = "green" if current_rating is True else "red" if current_rating is False else "orange"
            
            title_lines = [
                f"Image {self.current_index + 1}/{len(self.image_files)}: {display_name}{retake_text}"
            ]
            for line in info_lines:
                if line:
                    title_lines.append(line)
            title_lines.append(f"Rating: {rating_text}")
            
            self.ax.set_title('\n'.join(title_lines), fontsize=12, pad=20)
            
            for spine in self.ax.spines.values():
                spine.set_linewidth(4)
                spine.set_color(rating_color)
            
            plt.tight_layout()
            plt.draw()
        
        except Exception as exc:
            print(f"Error loading image {current_file}: {exc}")
    
    def _on_key_press(self, event):
        """Handle keyboard input."""
        current_file = self.image_files[self.current_index]
        
        if event.key == 'up':
            self.ratings[current_file.name] = True
            self.ratings_since_save += 1
            print(f"Marked {self._extract_image_name(current_file.name)} as GOOD")
            self._auto_save_check()
            self._move_to_next()
        
        elif event.key == 'down':
            self.ratings[current_file.name] = False
            self.ratings_since_save += 1
            print(f"Marked {self._extract_image_name(current_file.name)} as BAD")
            self._auto_save_check()
            self._move_to_next()
        
        elif event.key == 'right':
            self._move_to_next()
        
        elif event.key == 'left':
            self._move_to_previous()
        
        elif event.key == 'r':
            if current_file.name in self.ratings:
                del self.ratings[current_file.name]
            print(f"Reset rating for {self._extract_image_name(current_file.name)}")
            self.display_image()
        
        elif event.key == 's':
            self._save_ratings()
        
        elif event.key in ['q', 'escape']:
            self._quit_and_save()
    
    def _on_close(self, event):
        """Handle window close event - emergency save!"""
        rated_count = len([r for r in self.ratings.values() if r is not None])
        if rated_count > 0:
            print(f"🚨 WINDOW CLOSING! Emergency saving {rated_count} ratings...")
            self._save_ratings()
            print("✅ Emergency save complete!")
    
    def _emergency_exit_save(self):
        """Final emergency save when program exits."""
        try:
            rated_count = len([r for r in self.ratings.values() if r is not None])
            if rated_count > 0 and self.ratings_since_save > 0:
                self._save_ratings(quiet_mode=True)
        except Exception:
            pass  # Don't let exit saves crash the program
    
    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------
    
    def _move_to_next(self):
        """Move to next image."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_image()
        else:
            rated_count = len([r for r in self.ratings.values() if r is not None])
            if rated_count > 0:
                print("🔥🔥🔥 REACHED END OF IMAGES! 🔥🔥🔥")
                print(f"💾 OH CRUD YOU DIDN'T SAVE! Auto-saving {rated_count} ratings now...")
                self._save_ratings()
                print("📁 Your work has been saved! Press 'q' to quit properly.")
            else:
                print("Reached end of images. Press 'q' to quit and save.")
    
    def _move_to_previous(self):
        """Move to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()
        else:
            print("At first image.")
    
    def _quit_and_save(self):
        """Quit the inspector and save ratings."""
        self._save_ratings()
        self._print_summary()
        plt.close(self.fig)
        print("Visual inspection complete!")
    
    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    
    def _print_summary(self):
        """Print summary of inspection session."""
        total = len(self.image_files)
        rated = len([r for r in self.ratings.values() if r is not None])
        good = len([r for r in self.ratings.values() if r is True])
        bad = len([r for r in self.ratings.values() if r is False])
        unrated = total - rated
        
        print("\n" + "=" * 60)
        print("VISUAL INSPECTION SUMMARY")
        print("=" * 60)
        print(f"Total images: {total}")
        print(f"Rated images: {rated}")
        print(f"Good ratings: {good}")
        print(f"Bad ratings: {bad}")
        print(f"Unrated: {unrated}")
        if rated > 0:
            print(f"Quality rate: {good / rated * 100:.1f}% good")
        print("=" * 60)
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def run(self):
        """Run the visual inspector."""
        if not self.image_files:
            print("No images to inspect!")
            return
        
        print("Starting visual inspection...")
        print("Use arrow keys to navigate and rate images.")
        
        self.display_image()
        
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self._quit_and_save()
        except Exception as exc:
            print(f"\nUnexpected error: {exc}")
            rated_count = len([r for r in self.ratings.values() if r is not None])
            if rated_count > 0:
                print(f"🚨 Emergency saving {rated_count} ratings...")
                self._save_ratings()
                print("✅ Emergency save complete!")
        finally:
            rated_count = len([r for r in self.ratings.values() if r is not None])
            if rated_count > 0 and self.ratings_since_save > 0:
                print("🔒 Final safety save...")
                self._save_ratings(quiet_mode=True)


def main():
    """Main entry point for the visual inspector."""
    default_results_dir = Path(r"C:\Users\admin\Documents\Pierre lab\projects\Colustrum-ABX\lysozyme stain quantification\results\All")
    
    print("Lysozyme Stain Quantification - Visual Inspector")
    print("=" * 60)
    
    if not default_results_dir.exists():
        print(f"Default results directory not found: {default_results_dir}")
        return
    
    try:
        inspector = VisualInspector(default_results_dir)
        inspector.run()
    except Exception as exc:
        print(f"Error running visual inspector: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
