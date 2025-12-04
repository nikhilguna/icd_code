"""
Multi-label encoding for ICD codes.

Handles:
- Frequency-based filtering of ICD codes
- Multi-label binarization
- Stratified label analysis (head/medium/tail)
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Union
from collections import Counter
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


class ICDLabelEncoder:
    """
    Encoder for multi-label ICD codes with frequency-based filtering.
    
    Usage:
        encoder = ICDLabelEncoder(top_k=50, min_frequency=10)
        encoder.fit(icd_code_lists)
        
        # Encode labels
        binary_labels = encoder.transform(icd_code_lists)
        
        # Decode predictions
        predicted_codes = encoder.inverse_transform(binary_predictions)
    """
    
    def __init__(
        self,
        top_k: Optional[int] = 50,
        min_frequency: int = 10,
        frequency_bins: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize label encoder.
        
        Args:
            top_k: Only keep top-k most frequent codes. If None, keep all above min_frequency.
            min_frequency: Minimum occurrences for a code to be included.
            frequency_bins: Dict with frequency thresholds for stratification.
                           Default: {"head": 1000, "medium": 100, "tail": 0}
        """
        self.top_k = top_k
        self.min_frequency = min_frequency
        self.frequency_bins = frequency_bins or {
            "head": 1000,
            "medium": 100,
            "tail": 0
        }
        
        self.code_frequencies: Dict[str, int] = {}
        self.selected_codes: List[str] = []
        self.code_to_idx: Dict[str, int] = {}
        self.idx_to_code: Dict[int, str] = {}
        self.mlb: Optional[MultiLabelBinarizer] = None
        self.is_fitted = False
        
        # Stratification
        self.code_strata: Dict[str, str] = {}  # code -> stratum (head/medium/tail)
        self.stratum_indices: Dict[str, List[int]] = {}  # stratum -> list of label indices
    
    @property
    def num_labels(self) -> int:
        """Number of labels in the encoder."""
        return len(self.selected_codes)
    
    @property
    def classes_(self) -> List[str]:
        """List of encoded classes (ICD codes)."""
        return self.selected_codes
    
    def fit(
        self,
        icd_lists: List[List[str]],
        code_descriptions: Optional[Dict[str, str]] = None,
    ) -> "ICDLabelEncoder":
        """
        Fit encoder on training data.
        
        Args:
            icd_lists: List of ICD code lists (one list per document)
            code_descriptions: Optional dict mapping codes to descriptions
            
        Returns:
            self
        """
        # Count code frequencies
        all_codes = [code for codes in icd_lists for code in codes]
        self.code_frequencies = Counter(all_codes)
        
        logger.info(f"Total unique codes: {len(self.code_frequencies)}")
        logger.info(f"Total code occurrences: {sum(self.code_frequencies.values())}")
        
        # Filter by minimum frequency
        filtered_codes = {
            code: freq for code, freq in self.code_frequencies.items()
            if freq >= self.min_frequency
        }
        logger.info(f"Codes with frequency >= {self.min_frequency}: {len(filtered_codes)}")
        
        # Sort by frequency and take top-k
        sorted_codes = sorted(filtered_codes.items(), key=lambda x: x[1], reverse=True)
        
        if self.top_k is not None and len(sorted_codes) > self.top_k:
            sorted_codes = sorted_codes[:self.top_k]
            logger.info(f"Selected top {self.top_k} codes")
        
        self.selected_codes = [code for code, _ in sorted_codes]
        
        # Build mappings
        self.code_to_idx = {code: idx for idx, code in enumerate(self.selected_codes)}
        self.idx_to_code = {idx: code for code, idx in self.code_to_idx.items()}
        
        # Fit sklearn MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer(classes=self.selected_codes)
        self.mlb.fit([self.selected_codes])  # Fit with all possible codes
        
        # Compute stratification
        self._compute_stratification()
        
        self.is_fitted = True
        
        logger.info(f"Encoder fitted with {self.num_labels} labels")
        self._log_stratification_stats()
        
        return self
    
    def _compute_stratification(self) -> None:
        """Assign each code to a frequency stratum."""
        sorted_bins = sorted(self.frequency_bins.items(), key=lambda x: x[1], reverse=True)
        
        self.code_strata = {}
        self.stratum_indices = {stratum: [] for stratum in self.frequency_bins}
        
        for code in self.selected_codes:
            freq = self.code_frequencies[code]
            
            # Find appropriate stratum
            assigned_stratum = sorted_bins[-1][0]  # Default to lowest
            for stratum, threshold in sorted_bins:
                if freq >= threshold:
                    assigned_stratum = stratum
                    break
            
            self.code_strata[code] = assigned_stratum
            self.stratum_indices[assigned_stratum].append(self.code_to_idx[code])
    
    def _log_stratification_stats(self) -> None:
        """Log statistics about label stratification."""
        for stratum, indices in self.stratum_indices.items():
            if indices:
                codes = [self.idx_to_code[i] for i in indices]
                freqs = [self.code_frequencies[c] for c in codes]
                logger.info(
                    f"  {stratum}: {len(indices)} codes, "
                    f"freq range [{min(freqs)}, {max(freqs)}]"
                )
    
    def transform(
        self,
        icd_lists: List[List[str]],
        return_tensor: bool = False,
    ) -> np.ndarray:
        """
        Transform ICD code lists to binary label matrix.
        
        Args:
            icd_lists: List of ICD code lists
            return_tensor: If True, return torch tensor instead of numpy array
            
        Returns:
            Binary label matrix of shape (n_samples, n_labels)
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before transform")
        
        # Filter to only include known codes
        filtered_lists = [
            [code for code in codes if code in self.code_to_idx]
            for codes in icd_lists
        ]
        
        # Transform using sklearn
        labels = self.mlb.transform(filtered_lists)
        
        if return_tensor:
            import torch
            return torch.tensor(labels, dtype=torch.float32)
        
        return labels.astype(np.float32)
    
    def fit_transform(
        self,
        icd_lists: List[List[str]],
        return_tensor: bool = False,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(icd_lists)
        return self.transform(icd_lists, return_tensor=return_tensor)
    
    def inverse_transform(
        self,
        binary_labels: np.ndarray,
        threshold: float = 0.5,
    ) -> List[List[str]]:
        """
        Convert binary labels or probabilities back to ICD codes.
        
        Args:
            binary_labels: Binary label matrix or probability matrix
            threshold: Threshold for converting probabilities to binary
            
        Returns:
            List of ICD code lists
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before inverse_transform")
        
        # Apply threshold if probabilities
        if binary_labels.dtype in [np.float32, np.float64]:
            binary_labels = (binary_labels >= threshold).astype(int)
        
        return self.mlb.inverse_transform(binary_labels)
    
    def get_code_info(self, code: str) -> Dict:
        """Get information about a specific ICD code."""
        if code not in self.code_to_idx:
            return {"error": "Code not in encoder"}
        
        return {
            "code": code,
            "index": self.code_to_idx[code],
            "frequency": self.code_frequencies[code],
            "stratum": self.code_strata[code],
        }
    
    def get_stratum_mask(self, stratum: str) -> np.ndarray:
        """
        Get binary mask for a specific stratum.
        
        Args:
            stratum: One of "head", "medium", "tail"
            
        Returns:
            Binary mask of shape (n_labels,)
        """
        mask = np.zeros(self.num_labels, dtype=bool)
        mask[self.stratum_indices[stratum]] = True
        return mask
    
    def get_label_weights(
        self,
        method: str = "inverse_freq",
        smoothing: float = 0.1,
    ) -> np.ndarray:
        """
        Compute class weights for handling imbalance.
        
        Args:
            method: Weighting method ("inverse_freq", "log_inverse", "effective_num")
            smoothing: Smoothing factor
            
        Returns:
            Weight array of shape (n_labels,)
        """
        freqs = np.array([self.code_frequencies[code] for code in self.selected_codes])
        total = freqs.sum()
        
        if method == "inverse_freq":
            # Inverse frequency weighting
            weights = total / (freqs + smoothing)
            weights = weights / weights.mean()  # Normalize
            
        elif method == "log_inverse":
            # Log inverse frequency (less extreme)
            weights = np.log(total / (freqs + 1) + 1)
            weights = weights / weights.mean()
            
        elif method == "effective_num":
            # Effective number of samples (from Class-Balanced Loss paper)
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, freqs)
            weights = (1.0 - beta) / (effective_num + 1e-8)
            weights = weights / weights.mean()
            
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        return weights.astype(np.float32)
    
    def save(self, path: str) -> None:
        """Save encoder to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "top_k": self.top_k,
            "min_frequency": self.min_frequency,
            "frequency_bins": self.frequency_bins,
            "code_frequencies": self.code_frequencies,
            "selected_codes": self.selected_codes,
            "code_to_idx": self.code_to_idx,
            "idx_to_code": self.idx_to_code,
            "code_strata": self.code_strata,
            "stratum_indices": self.stratum_indices,
            "is_fitted": self.is_fitted,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved encoder to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ICDLabelEncoder":
        """Load encoder from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        encoder = cls(
            top_k=state["top_k"],
            min_frequency=state["min_frequency"],
            frequency_bins=state["frequency_bins"],
        )
        
        encoder.code_frequencies = state["code_frequencies"]
        encoder.selected_codes = state["selected_codes"]
        encoder.code_to_idx = state["code_to_idx"]
        encoder.idx_to_code = state["idx_to_code"]
        encoder.code_strata = state["code_strata"]
        encoder.stratum_indices = state["stratum_indices"]
        encoder.is_fitted = state["is_fitted"]
        
        # Recreate MLB
        encoder.mlb = MultiLabelBinarizer(classes=encoder.selected_codes)
        encoder.mlb.fit([encoder.selected_codes])
        
        logger.info(f"Loaded encoder from {path} with {encoder.num_labels} labels")
        
        return encoder


def analyze_label_distribution(
    icd_lists: List[List[str]],
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Analyze the distribution of ICD codes.
    
    Args:
        icd_lists: List of ICD code lists
        top_n: Number of top codes to show in detail
        
    Returns:
        DataFrame with distribution statistics
    """
    all_codes = [code for codes in icd_lists for code in codes]
    code_counts = Counter(all_codes)
    
    # Basic statistics
    total_codes = len(all_codes)
    unique_codes = len(code_counts)
    labels_per_doc = [len(codes) for codes in icd_lists]
    
    print(f"\n{'='*50}")
    print("ICD Code Distribution Analysis")
    print(f"{'='*50}")
    print(f"Total documents: {len(icd_lists)}")
    print(f"Total code occurrences: {total_codes}")
    print(f"Unique codes: {unique_codes}")
    print(f"\nLabels per document:")
    print(f"  Mean: {np.mean(labels_per_doc):.2f}")
    print(f"  Median: {np.median(labels_per_doc):.0f}")
    print(f"  Min: {np.min(labels_per_doc)}")
    print(f"  Max: {np.max(labels_per_doc)}")
    
    # Frequency distribution
    freqs = list(code_counts.values())
    print(f"\nCode frequency distribution:")
    print(f"  Mean: {np.mean(freqs):.2f}")
    print(f"  Median: {np.median(freqs):.0f}")
    print(f"  Min: {np.min(freqs)}")
    print(f"  Max: {np.max(freqs)}")
    
    # Codes by frequency threshold
    for threshold in [1000, 500, 100, 50, 10, 1]:
        count = sum(1 for f in freqs if f >= threshold)
        print(f"  Codes with freq >= {threshold}: {count}")
    
    # Top codes
    df = pd.DataFrame([
        {"code": code, "frequency": freq, "percentage": freq / total_codes * 100}
        for code, freq in code_counts.most_common(top_n)
    ])
    
    print(f"\nTop {top_n} codes:")
    print(df.to_string(index=False))
    
    return df
