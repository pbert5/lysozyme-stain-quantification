"""
Lysozyme stain quantification pipeline.

A complete pipeline for detecting and quantifying lysozyme stains in microscopy images.
"""

__version__ = "1.0.0"

from .pipeline.bulk_processor import BulkProcessor
from .processing.individual_processor import IndividualProcessor
from .processing.extractor_pipeline import ExtractorPipeline
from .processing.merge_pipeline import MergePipeline

__all__ = [
    'BulkProcessor',
    'IndividualProcessor', 
    'ExtractorPipeline',
    'MergePipeline'
]
