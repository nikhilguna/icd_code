"""Data extraction, preprocessing, and dataset utilities for ICD code prediction."""

# Import core modules (always available)
from .preprocessing import ClinicalTextPreprocessor
from .dataset import ICDDataset, create_dataloaders
from .label_encoder import ICDLabelEncoder

# New preprocessing improvements
from .clinical_tokenizer import ClinicalSentenceTokenizer
from .hierarchical_encoder import HierarchicalICDEncoder
from .enhanced_preprocessing import EnhancedClinicalPreprocessor

# Optional imports (may require additional dependencies)
try:
    from .athena_extraction import MIMICExtractor
    _ATHENA_AVAILABLE = True
except ImportError:
    _ATHENA_AVAILABLE = False
    MIMICExtractor = None

__all__ = [
    # Core
    "ClinicalTextPreprocessor", 
    "ICDDataset",
    "create_dataloaders",
    "ICDLabelEncoder",
    # New improvements
    "ClinicalSentenceTokenizer",
    "HierarchicalICDEncoder",
    "EnhancedClinicalPreprocessor",
    # Optional
    "MIMICExtractor",
]
