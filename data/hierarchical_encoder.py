"""
Hierarchical ICD-9 code encoder.

Maps detailed ICD-9 codes to 19 parent categories to address the long-tail problem.
Enables multi-task learning and coarse-grained prediction.

Based on the ICD-9-CM structure:
- 001-139: Infectious and Parasitic Diseases
- 140-239: Neoplasms
- 240-279: Endocrine, Nutritional and Metabolic Diseases
- 280-289: Blood and Blood-Forming Organs
- 290-319: Mental Disorders
- 320-389: Nervous System
- 390-459: Circulatory System
- 460-519: Respiratory System
- 520-579: Digestive System
- 580-629: Genitourinary System
- 630-679: Pregnancy, Childbirth, and Puerperium
- 680-709: Skin and Subcutaneous Tissue
- 710-739: Musculoskeletal and Connective Tissue
- 740-759: Congenital Anomalies
- 760-779: Perinatal Period
- 780-799: Symptoms, Signs, and Ill-Defined Conditions
- 800-999: Injury and Poisoning
- V01-V91: Supplementary Classification
- E000-E999: External Causes of Injury
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


# ICD-9-CM chapter structure
ICD9_HIERARCHY = {
    'infectious': {
        'range': (1, 139),
        'description': 'Infectious and Parasitic Diseases',
    },
    'neoplasms': {
        'range': (140, 239),
        'description': 'Neoplasms',
    },
    'endocrine': {
        'range': (240, 279),
        'description': 'Endocrine, Nutritional and Metabolic Diseases',
    },
    'blood': {
        'range': (280, 289),
        'description': 'Diseases of Blood and Blood-Forming Organs',
    },
    'mental': {
        'range': (290, 319),
        'description': 'Mental Disorders',
    },
    'nervous': {
        'range': (320, 389),
        'description': 'Diseases of the Nervous System',
    },
    'circulatory': {
        'range': (390, 459),
        'description': 'Diseases of the Circulatory System',
    },
    'respiratory': {
        'range': (460, 519),
        'description': 'Diseases of the Respiratory System',
    },
    'digestive': {
        'range': (520, 579),
        'description': 'Diseases of the Digestive System',
    },
    'genitourinary': {
        'range': (580, 629),
        'description': 'Diseases of the Genitourinary System',
    },
    'pregnancy': {
        'range': (630, 679),
        'description': 'Complications of Pregnancy, Childbirth, and Puerperium',
    },
    'skin': {
        'range': (680, 709),
        'description': 'Diseases of the Skin and Subcutaneous Tissue',
    },
    'musculoskeletal': {
        'range': (710, 739),
        'description': 'Diseases of the Musculoskeletal System',
    },
    'congenital': {
        'range': (740, 759),
        'description': 'Congenital Anomalies',
    },
    'perinatal': {
        'range': (760, 779),
        'description': 'Certain Conditions Originating in Perinatal Period',
    },
    'symptoms': {
        'range': (780, 799),
        'description': 'Symptoms, Signs, and Ill-Defined Conditions',
    },
    'injury': {
        'range': (800, 999),
        'description': 'Injury and Poisoning',
    },
    'v_codes': {
        'range': ('V', 'V'),
        'description': 'Supplementary Classification of Factors Influencing Health Status',
    },
    'e_codes': {
        'range': ('E', 'E'),
        'description': 'Supplementary Classification of External Causes',
    },
}


class HierarchicalICDEncoder:
    """
    Encoder for hierarchical ICD-9 codes.
    
    Provides both fine-grained (full ICD codes) and coarse-grained (19 categories)
    label representations.
    
    Usage:
        encoder = HierarchicalICDEncoder()
        encoder.fit(icd_codes_list)
        
        # Get both levels
        fine_labels, coarse_labels = encoder.transform_hierarchical(icd_codes_list)
        
        # Get just parent categories
        categories = encoder.get_parent_categories(["410.71", "401.9"])
        # Returns: ["circulatory", "circulatory"]
    """
    
    def __init__(
        self,
        top_k_fine: Optional[int] = 50,
        min_frequency: int = 10,
        include_all_coarse: bool = True,
    ):
        """
        Initialize hierarchical encoder.
        
        Args:
            top_k_fine: Number of fine-grained codes to keep (None = all)
            min_frequency: Minimum frequency for fine-grained codes
            include_all_coarse: Include all 19 coarse categories regardless of frequency
        """
        self.top_k_fine = top_k_fine
        self.min_frequency = min_frequency
        self.include_all_coarse = include_all_coarse
        
        # Fine-grained encoding
        self.code_to_idx: Dict[str, int] = {}
        self.idx_to_code: Dict[int, str] = {}
        self.code_frequencies: Dict[str, int] = defaultdict(int)
        
        # Coarse-grained encoding (19 categories)
        self.category_to_idx: Dict[str, int] = {
            cat: idx for idx, cat in enumerate(sorted(ICD9_HIERARCHY.keys()))
        }
        self.idx_to_category: Dict[int, str] = {
            idx: cat for cat, idx in self.category_to_idx.items()
        }
        
        self.is_fitted = False
        
        logger.info(f"Initialized hierarchical encoder with {len(self.category_to_idx)} parent categories")
    
    def get_three_digit_prefix(self, icd_code: str) -> str:
        """
        Get 3-digit prefix of ICD code.
        
        Args:
            icd_code: Full ICD code (e.g., "410.71" or "V45.81")
            
        Returns:
            3-digit prefix (e.g., "410" or "V45")
        """
        # Remove decimal point and any non-alphanumeric chars
        clean_code = icd_code.replace('.', '').replace('-', '').strip().upper()
        
        # Get first 3 characters
        if len(clean_code) >= 3:
            return clean_code[:3]
        else:
            return clean_code
    
    def get_parent_category(self, icd_code: str) -> Optional[str]:
        """
        Map ICD-9 code to parent category.
        
        Args:
            icd_code: ICD-9 code (e.g., "410.71")
            
        Returns:
            Parent category name or None if not found
        """
        prefix = self.get_three_digit_prefix(icd_code)
        
        # Check for V codes
        if prefix.startswith('V'):
            return 'v_codes'
        
        # Check for E codes
        if prefix.startswith('E'):
            return 'e_codes'
        
        # Try to parse as numeric
        try:
            code_num = int(prefix)
        except ValueError:
            logger.warning(f"Could not parse ICD code: {icd_code}")
            return None
        
        # Find matching category
        for category, info in ICD9_HIERARCHY.items():
            start, end = info['range']
            if isinstance(start, int) and start <= code_num <= end:
                return category
        
        logger.warning(f"No category found for code: {icd_code} (prefix: {prefix})")
        return None
    
    def fit(self, icd_codes: List[List[str]]) -> 'HierarchicalICDEncoder':
        """
        Fit encoder on list of ICD code sets.
        
        Args:
            icd_codes: List of ICD code lists, one per sample
            
        Returns:
            self
        """
        # Count code frequencies
        for codes in icd_codes:
            for code in codes:
                self.code_frequencies[code] += 1
        
        # Sort by frequency
        sorted_codes = sorted(
            self.code_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter by frequency and top-k
        filtered_codes = [
            code for code, freq in sorted_codes
            if freq >= self.min_frequency
        ]
        
        if self.top_k_fine is not None:
            filtered_codes = filtered_codes[:self.top_k_fine]
        
        # Build encoding
        self.code_to_idx = {code: idx for idx, code in enumerate(filtered_codes)}
        self.idx_to_code = {idx: code for code, idx in self.code_to_idx.items()}
        
        self.is_fitted = True
        
        logger.info(f"Fitted encoder on {len(icd_codes)} samples")
        logger.info(f"Fine-grained: {len(self.code_to_idx)} codes (from {len(self.code_frequencies)} total)")
        logger.info(f"Coarse-grained: {len(self.category_to_idx)} categories")
        
        # Log category distribution
        category_counts = defaultdict(int)
        for codes in icd_codes:
            categories = set()
            for code in codes:
                cat = self.get_parent_category(code)
                if cat:
                    categories.add(cat)
            for cat in categories:
                category_counts[cat] += 1
        
        logger.info("Category distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / len(icd_codes)
            logger.info(f"  {cat:20s}: {count:5d} samples ({pct:5.1f}%)")
        
        return self
    
    def transform_fine(self, icd_codes: List[List[str]]) -> np.ndarray:
        """
        Transform to fine-grained multi-label binary vectors.
        
        Args:
            icd_codes: List of ICD code lists
            
        Returns:
            Binary matrix of shape (n_samples, n_codes)
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        n_samples = len(icd_codes)
        n_codes = len(self.code_to_idx)
        
        labels = np.zeros((n_samples, n_codes), dtype=np.float32)
        
        for i, codes in enumerate(icd_codes):
            for code in codes:
                if code in self.code_to_idx:
                    labels[i, self.code_to_idx[code]] = 1
        
        return labels
    
    def transform_coarse(self, icd_codes: List[List[str]]) -> np.ndarray:
        """
        Transform to coarse-grained (19 categories) multi-label binary vectors.
        
        Args:
            icd_codes: List of ICD code lists
            
        Returns:
            Binary matrix of shape (n_samples, 19)
        """
        n_samples = len(icd_codes)
        n_categories = len(self.category_to_idx)
        
        labels = np.zeros((n_samples, n_categories), dtype=np.float32)
        
        for i, codes in enumerate(icd_codes):
            # Get unique categories for this sample
            categories = set()
            for code in codes:
                cat = self.get_parent_category(code)
                if cat and cat in self.category_to_idx:
                    categories.add(cat)
            
            # Set binary labels
            for cat in categories:
                labels[i, self.category_to_idx[cat]] = 1
        
        return labels
    
    def transform_hierarchical(
        self,
        icd_codes: List[List[str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform to both fine and coarse labels.
        
        Args:
            icd_codes: List of ICD code lists
            
        Returns:
            Tuple of (fine_labels, coarse_labels)
        """
        fine_labels = self.transform_fine(icd_codes)
        coarse_labels = self.transform_coarse(icd_codes)
        
        return fine_labels, coarse_labels
    
    def inverse_transform_fine(self, labels: np.ndarray, threshold: float = 0.5) -> List[List[str]]:
        """
        Convert fine-grained binary predictions back to ICD codes.
        
        Args:
            labels: Binary matrix of shape (n_samples, n_codes)
            threshold: Threshold for binary prediction
            
        Returns:
            List of ICD code lists
        """
        results = []
        for label_vec in labels:
            codes = []
            for idx, val in enumerate(label_vec):
                if val >= threshold and idx in self.idx_to_code:
                    codes.append(self.idx_to_code[idx])
            results.append(codes)
        
        return results
    
    def inverse_transform_coarse(self, labels: np.ndarray, threshold: float = 0.5) -> List[List[str]]:
        """
        Convert coarse-grained binary predictions back to category names.
        
        Args:
            labels: Binary matrix of shape (n_samples, n_categories)
            threshold: Threshold for binary prediction
            
        Returns:
            List of category name lists
        """
        results = []
        for label_vec in labels:
            categories = []
            for idx, val in enumerate(label_vec):
                if val >= threshold and idx in self.idx_to_category:
                    categories.append(self.idx_to_category[idx])
            results.append(categories)
        
        return results
    
    @property
    def num_fine_labels(self) -> int:
        """Number of fine-grained labels."""
        return len(self.code_to_idx)
    
    @property
    def num_coarse_labels(self) -> int:
        """Number of coarse-grained labels."""
        return len(self.category_to_idx)
    
    def get_category_description(self, category: str) -> str:
        """Get human-readable description of category."""
        if category in ICD9_HIERARCHY:
            return ICD9_HIERARCHY[category]['description']
        return "Unknown category"
    
    def get_statistics(self) -> Dict:
        """Get encoder statistics."""
        return {
            'num_fine_labels': self.num_fine_labels,
            'num_coarse_labels': self.num_coarse_labels,
            'total_codes_seen': len(self.code_frequencies),
            'min_frequency': self.min_frequency,
            'top_k_fine': self.top_k_fine,
            'is_fitted': self.is_fitted,
        }


def test_hierarchical_encoder():
    """Test the hierarchical encoder."""
    
    # Sample ICD codes
    sample_codes = [
        ["410.71", "401.9", "250.00"],  # Circulatory + Endocrine
        ["480.9", "496", "491.21"],     # Respiratory
        ["585.9", "403.90"],            # Genitourinary + Circulatory
        ["V45.81", "E849.0"],           # V-codes + E-codes
    ]
    
    encoder = HierarchicalICDEncoder(top_k_fine=None, min_frequency=1)
    encoder.fit(sample_codes)
    
    print("Hierarchical ICD Encoder Test")
    print("=" * 60)
    print(f"Fine-grained codes: {encoder.num_fine_labels}")
    print(f"Coarse-grained categories: {encoder.num_coarse_labels}")
    print()
    
    # Test parent category mapping
    print("Parent category mapping:")
    test_codes = ["410.71", "250.00", "480.9", "V45.81", "E849.0"]
    for code in test_codes:
        cat = encoder.get_parent_category(code)
        if cat:
            desc = encoder.get_category_description(cat)
            print(f"  {code:10s} -> {cat:20s} ({desc})")
    print()
    
    # Test transformation
    fine_labels, coarse_labels = encoder.transform_hierarchical(sample_codes)
    
    print(f"Fine-grained labels shape: {fine_labels.shape}")
    print(f"Coarse-grained labels shape: {coarse_labels.shape}")
    print()
    
    # Show first sample
    print("Sample 1 labels:")
    print(f"  Codes: {sample_codes[0]}")
    print(f"  Fine-grained active: {fine_labels[0].sum():.0f} labels")
    print(f"  Coarse-grained active: {coarse_labels[0].sum():.0f} categories")
    
    # Inverse transform
    pred_categories = encoder.inverse_transform_coarse(coarse_labels[:1])
    print(f"  Predicted categories: {pred_categories[0]}")
    
    print("=" * 60)


if __name__ == "__main__":
    test_hierarchical_encoder()

