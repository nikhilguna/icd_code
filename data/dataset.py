"""
PyTorch Dataset and DataLoader for ICD code prediction.

Provides efficient data loading for both CAML and LED models with:
- Multi-label binary targets
- Variable length handling
- Stratified train/val/test splitting
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from .preprocessing import ClinicalTextPreprocessor, ProcessedDocument
from .label_encoder import ICDLabelEncoder

logger = logging.getLogger(__name__)


class ICDDataset(Dataset):
    """
    PyTorch Dataset for ICD code prediction.
    
    Handles tokenized clinical text and multi-label ICD code targets.
    
    Usage:
        dataset = ICDDataset(
            texts=["patient presented with..."],
            icd_codes=[["410.71", "401.9"]],
            tokenizer_name="allenai/longformer-base-4096",
            max_length=4096,
            label_encoder=encoder,
        )
        
        # Get single sample
        sample = dataset[0]
        # sample = {
        #     "input_ids": tensor,
        #     "attention_mask": tensor,
        #     "labels": tensor,
        #     "hadm_id": int,
        # }
    """
    
    def __init__(
        self,
        texts: List[str],
        icd_codes: List[List[str]],
        tokenizer_name: str = "allenai/longformer-base-4096",
        max_length: int = 4096,
        label_encoder: Optional[ICDLabelEncoder] = None,
        hadm_ids: Optional[List[int]] = None,
        preprocessor: Optional[ClinicalTextPreprocessor] = None,
        cache_tokenization: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of clinical document texts
            icd_codes: List of ICD code lists (parallel to texts)
            tokenizer_name: HuggingFace tokenizer to use
            max_length: Maximum sequence length
            label_encoder: Fitted ICDLabelEncoder (if None, will fit on provided codes)
            hadm_ids: Optional hospital admission IDs
            preprocessor: Optional preprocessor (creates new one if None)
            cache_tokenization: Whether to cache tokenized sequences
        """
        assert len(texts) == len(icd_codes), "texts and icd_codes must have same length"
        
        self.texts = texts
        self.icd_codes = icd_codes
        self.max_length = max_length
        self.hadm_ids = hadm_ids or list(range(len(texts)))
        self.cache_tokenization = cache_tokenization
        
        # Initialize or use provided preprocessor
        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = ClinicalTextPreprocessor(
                tokenizer_name=tokenizer_name,
                max_length=max_length,
            )
        
        self.tokenizer = self.preprocessor.tokenizer
        
        # Initialize or use provided label encoder
        if label_encoder is not None:
            self.label_encoder = label_encoder
        else:
            self.label_encoder = ICDLabelEncoder()
            self.label_encoder.fit(icd_codes)
        
        # Encode labels
        self.labels = self.label_encoder.transform(icd_codes)
        
        # Cache for tokenized sequences
        self._token_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        
        logger.info(f"Created dataset with {len(self)} samples, {self.label_encoder.num_labels} labels")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Check cache
        if self.cache_tokenization and idx in self._token_cache:
            cached = self._token_cache[idx]
            return {
                "input_ids": cached["input_ids"],
                "attention_mask": cached["attention_mask"],
                "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
                "hadm_id": self.hadm_ids[idx],
            }
        
        # Process document
        doc = self.preprocessor.process_document(
            text=self.texts[idx],
            hadm_id=self.hadm_ids[idx],
            tokenize=True,
        )
        
        input_ids = torch.tensor(doc.token_ids, dtype=torch.long)
        attention_mask = torch.tensor(doc.attention_mask, dtype=torch.long)
        
        # Cache if enabled
        if self.cache_tokenization:
            self._token_cache[idx] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "hadm_id": self.hadm_ids[idx],
        }
    
    @property
    def num_labels(self) -> int:
        return self.label_encoder.num_labels
    
    def get_label_weights(self, method: str = "inverse_freq") -> torch.Tensor:
        """Get class weights for loss function."""
        weights = self.label_encoder.get_label_weights(method=method)
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> np.ndarray:
        """
        Compute sample weights based on label rarity.
        
        Samples with rare labels get higher weights.
        """
        label_freqs = self.labels.sum(axis=0)  # Frequency of each label
        
        # Weight each sample by inverse of its labels' frequencies
        sample_weights = np.zeros(len(self))
        for i, label_vec in enumerate(self.labels):
            active_labels = np.where(label_vec > 0)[0]
            if len(active_labels) > 0:
                # Average inverse frequency of active labels
                sample_weights[i] = np.mean(1.0 / (label_freqs[active_labels] + 1))
            else:
                sample_weights[i] = 1.0
        
        # Normalize
        sample_weights = sample_weights / sample_weights.mean()
        
        return sample_weights


class ICDDatasetPreTokenized(Dataset):
    """
    Dataset for pre-tokenized data (faster loading).
    
    Use this when you've already tokenized and saved your data.
    """
    
    def __init__(
        self,
        token_ids: np.ndarray,
        attention_masks: np.ndarray,
        labels: np.ndarray,
        hadm_ids: Optional[np.ndarray] = None,
    ):
        """
        Initialize pre-tokenized dataset.
        
        Args:
            token_ids: Array of shape (n_samples, max_length)
            attention_masks: Array of shape (n_samples, max_length)
            labels: Array of shape (n_samples, n_labels)
            hadm_ids: Optional array of admission IDs
        """
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.hadm_ids = hadm_ids if hadm_ids is not None else np.arange(len(token_ids))
    
    def __len__(self) -> int:
        return len(self.token_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.token_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "hadm_id": int(self.hadm_ids[idx]),
        }
    
    @property
    def num_labels(self) -> int:
        return self.labels.shape[1]


def create_dataloaders(
    df: pd.DataFrame,
    label_encoder: ICDLabelEncoder,
    tokenizer_name: str = "allenai/longformer-base-4096",
    max_length: int = 4096,
    batch_size: int = 16,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    random_seed: int = 42,
    use_weighted_sampling: bool = False,
    text_column: str = "discharge_text",
    icd_column: str = "icd_codes",
    hadm_column: str = "hadm_id",
) -> Tuple[DataLoader, DataLoader, DataLoader, ICDLabelEncoder]:
    """
    Create train/val/test DataLoaders from a DataFrame.
    
    Args:
        df: DataFrame with text and ICD codes
        label_encoder: Fitted ICDLabelEncoder
        tokenizer_name: HuggingFace tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer
        random_seed: Random seed for splits
        use_weighted_sampling: Use weighted sampling for rare labels
        text_column: Column name for text
        icd_column: Column name for ICD codes
        hadm_column: Column name for admission IDs
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_encoder)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Convert icd_codes from string representation to list if needed
    if df[icd_column].dtype == object and isinstance(df[icd_column].iloc[0], str):
        import ast
        df[icd_column] = df[icd_column].apply(ast.literal_eval)
    
    # Split data
    indices = np.arange(len(df))
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=random_seed,
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        random_state=random_seed,
    )
    
    logger.info(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create datasets
    def make_dataset(idx_array: np.ndarray) -> ICDDataset:
        subset_df = df.iloc[idx_array]
        return ICDDataset(
            texts=subset_df[text_column].tolist(),
            icd_codes=subset_df[icd_column].tolist(),
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            label_encoder=label_encoder,
            hadm_ids=subset_df[hadm_column].tolist() if hadm_column in subset_df else None,
        )
    
    train_dataset = make_dataset(train_idx)
    val_dataset = make_dataset(val_idx)
    test_dataset = make_dataset(test_idx)
    
    # Create samplers
    train_sampler = None
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader, label_encoder


def load_mimic_data(
    data_path: str,
    label_encoder: Optional[ICDLabelEncoder] = None,
    top_k_codes: int = 50,
    min_frequency: int = 10,
) -> Tuple[pd.DataFrame, ICDLabelEncoder]:
    """
    Load MIMIC data from parquet file and prepare label encoder.
    
    Args:
        data_path: Path to parquet file
        label_encoder: Optional pre-fitted encoder
        top_k_codes: Number of top codes to use
        min_frequency: Minimum code frequency
        
    Returns:
        Tuple of (DataFrame, fitted ICDLabelEncoder)
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Parse ICD codes if stored as string
    if df["icd_codes"].dtype == object and isinstance(df["icd_codes"].iloc[0], str):
        import ast
        df["icd_codes"] = df["icd_codes"].apply(ast.literal_eval)
    
    # Fit label encoder if not provided
    if label_encoder is None:
        label_encoder = ICDLabelEncoder(
            top_k=top_k_codes,
            min_frequency=min_frequency,
        )
        label_encoder.fit(df["icd_codes"].tolist())
    
    return df, label_encoder


def collate_fn_variable_length(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable length sequences.
    
    Pads sequences to the maximum length in the batch rather than a fixed length.
    More memory efficient for batches with shorter documents.
    """
    # Find max length in batch
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    batch_size = len(batch)
    
    # Prepare output tensors
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.stack([item["labels"] for item in batch])
    hadm_ids = [item["hadm_id"] for item in batch]
    
    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "hadm_ids": hadm_ids,
    }

