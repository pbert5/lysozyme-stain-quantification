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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from datetime import datetime

# Try to use different backends for compatibility
try:
    import matplotlib
    matplotlib.use('Qt5Agg')  # Try Qt first
except ImportError:
    try:
        matplotlib.use('TkAgg')  # Fallback to Tk
    except ImportError:
        matplotlib.use('Agg')  # Last resort


class VisualInspector:
    """Tool for visually inspecting processed images using pre-generated visualizations."""
    
    def __init__(self, results_dir, load_existing=True):
        """
        Initialize the visual inspector.
        
        Args:
            results_dir: Path to results directory containing visualizations
            load_existing: Whether to load existing inspection results
        """
        self.results_dir = Path(results_dir)
        self.visualizations_dir = self.results_dir / 'visualizations'
        self.summaries_dir = self.results_dir / 'summaries'
        
        # Load consolidated summary for metadata
        self.summary_df = self._load_summary()
        
        # Find all visualization images
        self.image_files = self._find_visualization_images()
        
        # Initialize inspection state
        self.current_index = 0
        self.ratings = {}  # image_name -> True/False/None
        self.session_start = datetime.now()
        self.ratings_since_save = 0  # Track ratings since last save for auto-save
        self.auto_save_interval = 5  # Auto-save every 5 ratings
        
        # Load existing ratings if available
        self.ratings_file = self.summaries_dir / 'visual_inspection_ratings.json'
        if load_existing:
            self._load_existing_ratings()
        
        # Set up matplotlib
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('close_event', self._on_close)  # Handle window close
        
        # Register emergency save on program exit
        atexit.register(self._emergency_exit_save)
        
        print(f"Visual Inspector initialized")
        print(f"Found {len(self.image_files)} visualization images")
        print(f"")
        print(f"IMPORTANT: Your work will auto-save every {self.auto_save_interval} ratings!")
        print(f"")
        print(f"Controls:")
        print(f"  ↑ Arrow = Mark as GOOD and move to next")
        print(f"  ↓ Arrow = Mark as BAD and move to next") 
        print(f"  → Arrow = Skip to next (no rating)")
        print(f"  ← Arrow = Go back to previous")
        print(f"  'r' key = Reset current image rating")
        print(f"  's' key = Save progress manually")
        print(f"  'q' key = Quit and save all ratings")
        print(f"")
        print(f"🔥 PRESS 'q' TO QUIT AND SAVE WHEN DONE! 🔥")
        print("=" * 60)
    
    def _load_summary(self):
        """Load the consolidated summary for metadata."""
        summary_path = self.summaries_dir / 'consolidated_summary.csv'
        if summary_path.exists():
            return pd.read_csv(summary_path)
        else:
            print(f"Warning: No consolidated summary found at {summary_path}")
            return pd.DataFrame()
    
    def _find_visualization_images(self):
        """Find all visualization images in the visualizations directory."""
        if not self.visualizations_dir.exists():
            raise FileNotFoundError(f"Visualizations directory not found: {self.visualizations_dir}")
        
        # Find all _detected_regions.png files
        pattern = "*_detected_regions.png"
        image_files = list(self.visualizations_dir.glob(pattern))
        
        if not image_files:
            raise FileNotFoundError(f"No visualization images found in {self.visualizations_dir}")
        
        # Sort by filename for consistent ordering
        return sorted(image_files)
    
    def _load_existing_ratings(self):
        """Load existing ratings from previous session if available."""
        if self.ratings_file.exists():
            try:
                with open(self.ratings_file, 'r') as f:
                    data = json.load(f)
                    self.ratings = data.get('ratings', {})
                    print(f"Loaded {len(self.ratings)} existing ratings from previous session")
            except Exception as e:
                print(f"Warning: Could not load existing ratings: {e}")
    
    def _save_ratings(self, quiet_mode=False):
        """Save current ratings to file."""
        try:
            # Prepare data to save
            data = {
                'session_start': self.session_start.isoformat(),
                'last_updated': datetime.now().isoformat(),
                'total_images': len(self.image_files),
                'rated_images': len([r for r in self.ratings.values() if r is not None]),
                'ratings': self.ratings,
                'rating_summary': {
                    'good': len([r for r in self.ratings.values() if r is True]),
                    'bad': len([r for r in self.ratings.values() if r is False]),
                    'unrated': len([r for r in self.ratings.values() if r is None]) + 
                              (len(self.image_files) - len(self.ratings))
                }
            }
            
            # Save to JSON file
            with open(self.ratings_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            if not quiet_mode:
                print(f"Ratings saved to {self.ratings_file}")
            
            # Reset the counter
            self.ratings_since_save = 0
            
            # Also save a CSV version for easy analysis
            if not quiet_mode:
                csv_file = self.summaries_dir / 'visual_inspection_ratings.csv'
                ratings_df = pd.DataFrame([
                    {
                        'image_file': img_file.name,
                        'image_name': self._extract_image_name(img_file.name),
                        'rating': self.ratings.get(img_file.name),
                        'rating_text': 'Good' if self.ratings.get(img_file.name) is True else 
                                      'Bad' if self.ratings.get(img_file.name) is False else 'Unrated'
                    }
                    for img_file in self.image_files
                ])
                ratings_df.to_csv(csv_file, index=False)
                print(f"Ratings also saved to {csv_file}")
            
        except Exception as e:
            print(f"Error saving ratings: {e}")
    
    def _auto_save_check(self):
        """Check if we should auto-save and do it if needed."""
        if self.ratings_since_save >= self.auto_save_interval:
            rated_count = len([r for r in self.ratings.values() if r is not None])
            print(f"💾 Auto-saving progress... ({rated_count} images rated)")
            self._save_ratings(quiet_mode=True)
    
    def _extract_image_name(self, filename):
        """Extract clean image name from visualization filename."""
        # Remove _RFP_detected_regions.png suffix
        return filename.replace('_RFP_detected_regions.png', '')
    
    def _get_image_metadata(self, filename):
        """Get metadata for current image from summary."""
        image_name = self._extract_image_name(filename)
        
        if not self.summary_df.empty:
            # Find rows for this image
            matching_rows = self.summary_df[self.summary_df['image_name'] == image_name]
            if not matching_rows.empty:
                n_regions = len(matching_rows)
                is_retake = matching_rows.iloc[0]['is_retake']
                subdir = matching_rows.iloc[0]['subdir']
                avg_intensity = matching_rows['red_intensity'].mean()
                total_area = matching_rows['area_um2'].sum()
                
                return {
                    'n_regions': n_regions,
                    'is_retake': is_retake,
                    'subdir': subdir,
                    'avg_intensity': avg_intensity,
                    'total_area': total_area
                }
        
        return {'n_regions': 'Unknown', 'is_retake': False, 'subdir': 'Unknown', 
                'avg_intensity': 'Unknown', 'total_area': 'Unknown'}
    
    def display_image(self):
        """Display the current image with metadata and rating status."""
        if self.current_index >= len(self.image_files):
            self.current_index = len(self.image_files) - 1
        if self.current_index < 0:
            self.current_index = 0
        
        # Get current image file
        current_file = self.image_files[self.current_index]
        image_name = self._extract_image_name(current_file.name)
        
        # Load and display image
        try:
            img = mpimg.imread(current_file)
            self.ax.clear()
            self.ax.imshow(img)
            self.ax.axis('off')
            
            # Get metadata
            metadata = self._get_image_metadata(current_file.name)
            
            # Get current rating
            current_rating = self.ratings.get(current_file.name)
            rating_text = "GOOD" if current_rating is True else "BAD" if current_rating is False else "UNRATED"
            rating_color = "green" if current_rating is True else "red" if current_rating is False else "orange"
            
            # Prepare title with metadata
            retake_text = " [RETAKE]" if metadata['is_retake'] else ""
            title_lines = [
                f"Image {self.current_index + 1}/{len(self.image_files)}: {image_name}{retake_text}",
                f"Regions: {metadata['n_regions']} | Dir: {metadata['subdir']}",
                f"Avg Intensity: {metadata['avg_intensity']:.1f} | Total Area: {metadata['total_area']:.1f} μm²" if isinstance(metadata['avg_intensity'], (int, float)) else f"Avg Intensity: {metadata['avg_intensity']} | Total Area: {metadata['total_area']}",
                f"Rating: {rating_text}"
            ]
            
            self.ax.set_title('\n'.join(title_lines), fontsize=12, pad=20)
            
            # Add rating indicator as colored border
            for spine in self.ax.spines.values():
                spine.set_linewidth(4)
                spine.set_color(rating_color)
            
            plt.tight_layout()
            plt.draw()
            
        except Exception as e:
            print(f"Error loading image {current_file}: {e}")
    
    def _on_key_press(self, event):
        """Handle keyboard input."""
        current_file = self.image_files[self.current_index]
        
        if event.key == 'up':
            # Mark as good
            self.ratings[current_file.name] = True
            self.ratings_since_save += 1
            print(f"Marked {self._extract_image_name(current_file.name)} as GOOD")
            self._auto_save_check()
            self._move_to_next()
            
        elif event.key == 'down':
            # Mark as bad
            self.ratings[current_file.name] = False
            self.ratings_since_save += 1
            print(f"Marked {self._extract_image_name(current_file.name)} as BAD")
            self._auto_save_check()
            self._move_to_next()
            
        elif event.key == 'right':
            # Skip to next
            self._move_to_next()
            
        elif event.key == 'left':
            # Go back
            self._move_to_previous()
            
        elif event.key == 'r':
            # Reset current rating
            if current_file.name in self.ratings:
                del self.ratings[current_file.name]
            print(f"Reset rating for {self._extract_image_name(current_file.name)}")
            self.display_image()
            
        elif event.key == 's':
            # Save progress
            self._save_ratings()
            
        elif event.key in ['q', 'escape']:
            # Quit and save
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
                # Only save if we have unsaved work
                self._save_ratings(quiet_mode=True)
        except:
            pass  # Don't let exit saves crash the program
    
    def _move_to_next(self):
        """Move to next image."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_image()
        else:
            # At the end - trigger "oh crud" save protection
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
            print(f"Quality rate: {good/rated*100:.1f}% good")
        print("=" * 60)
    
    def run(self):
        """Run the visual inspector."""
        if not self.image_files:
            print("No images to inspect!")
            return
        
        print("Starting visual inspection...")
        print("Use arrow keys to navigate and rate images.")
        
        # Display first image
        self.display_image()
        
        # Keep the plot open
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self._quit_and_save()
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            rated_count = len([r for r in self.ratings.values() if r is not None])
            if rated_count > 0:
                print(f"🚨 Emergency saving {rated_count} ratings...")
                self._save_ratings()
                print("✅ Emergency save complete!")
        finally:
            # Final safety net
            rated_count = len([r for r in self.ratings.values() if r is not None])
            if rated_count > 0 and self.ratings_since_save > 0:
                print("🔒 Final safety save...")
                self._save_ratings(quiet_mode=True)


def main():
    """Main entry point for the visual inspector."""
    # Default results directory
    default_results_dir = Path(r"C:\Users\admin\Documents\Pierre lab\projects\Colustrum-ABX\lysozyme stain quantification\results\All")
    
    print("Lysozyme Stain Quantification - Visual Inspector")
    print("=" * 60)
    
    # Check if default directory exists
    if not default_results_dir.exists():
        print(f"Default results directory not found: {default_results_dir}")
        return
    
    # Initialize and run inspector
    try:
        inspector = VisualInspector(default_results_dir)
        inspector.run()
    except Exception as e:
        print(f"Error running visual inspector: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
