#!/usr/bin/env python3
"""
Test data loading pipeline: label encoding, datasets, and dataloaders.

Tests:
- ICD label encoding
- PyTorch Dataset creation
- DataLoader batching
- Label distribution analysis
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.label_encoder import ICDLabelEncoder, analyze_label_distribution
from data.dataset import ICDDataset
from data.preprocessing import ClinicalTextPreprocessor


def load_mock_data(data_dir):
    """Load mock MIMIC data."""
    data_dir = Path(data_dir)
    
    # Load notes
    notes_df = pd.read_csv(data_dir / "NOTEEVENTS.csv")
    
    # Load diagnoses
    diagnoses_df = pd.read_csv(data_dir / "DIAGNOSES_ICD.csv")
    
    # Merge to get ICD codes per admission
    # Group diagnoses by HADM_ID
    icd_by_admission = diagnoses_df.groupby("HADM_ID")["ICD9_CODE"].apply(list).reset_index()
    icd_by_admission.columns = ["HADM_ID", "icd_codes"]
    
    # Merge with notes
    merged = notes_df.merge(icd_by_admission, on="HADM_ID", how="left")
    
    # Filter out any without codes
    merged = merged.dropna(subset=["icd_codes"])
    
    return merged


def test_label_encoder(icd_lists, top_k=50):
    """Test ICDLabelEncoder."""
    print("\n" + "="*60)
    print("TEST: Label Encoder")
    print("="*60)
    
    # Initialize encoder
    encoder = ICDLabelEncoder(top_k=top_k, min_frequency=2)
    
    # Fit
    print(f"\nFitting encoder on {len(icd_lists)} samples...")
    encoder.fit(icd_lists)
    
    print(f"✓ Encoder fitted successfully")
    print(f"  Number of labels: {encoder.num_labels}")
    print(f"  Top-k requested: {top_k}")
    
    # Check code frequencies
    print(f"\nCode frequency distribution:")
    freqs = [encoder.code_frequencies[code] for code in encoder.selected_codes]
    print(f"  Min frequency: {min(freqs)}")
    print(f"  Max frequency: {max(freqs)}")
    print(f"  Mean frequency: {np.mean(freqs):.1f}")
    
    # Check stratification
    print(f"\nLabel stratification:")
    for stratum, indices in encoder.stratum_indices.items():
        if indices:
            codes = [encoder.idx_to_code[i] for i in indices]
            freqs = [encoder.code_frequencies[c] for c in codes]
            print(f"  {stratum:8s}: {len(indices):3d} codes, freq range [{min(freqs)}, {max(freqs)}]")
    
    # Test transform
    print(f"\nTesting transform...")
    labels = encoder.transform(icd_lists)
    
    print(f"✓ Transform successful")
    print(f"  Label matrix shape: {labels.shape}")
    print(f"  Label matrix dtype: {labels.dtype}")
    print(f"  Sparsity: {(labels == 0).mean()*100:.1f}% zeros")
    
    # Check labels per sample
    labels_per_sample = labels.sum(axis=1)
    print(f"\nLabels per sample:")
    print(f"  Mean: {labels_per_sample.mean():.2f}")
    print(f"  Median: {np.median(labels_per_sample):.0f}")
    print(f"  Min: {labels_per_sample.min():.0f}")
    print(f"  Max: {labels_per_sample.max():.0f}")
    
    # Test inverse transform
    print(f"\nTesting inverse transform...")
    reconstructed = encoder.inverse_transform(labels)
    
    # Check reconstruction (should match or be subset due to top-k filtering)
    matches = 0
    for orig, recon in zip(icd_lists, reconstructed):
        orig_in_encoder = [c for c in orig if c in encoder.code_to_idx]
        if set(orig_in_encoder) == set(recon):
            matches += 1
    
    print(f"✓ Inverse transform successful")
    print(f"  Perfect reconstructions: {matches}/{len(icd_lists)} ({matches/len(icd_lists)*100:.1f}%)")
    
    return encoder


def test_dataset(merged_df, encoder, tokenizer_name="bert-base-uncased", max_length=512):
    """Test ICDDataset."""
    print("\n" + "="*60)
    print("TEST: PyTorch Dataset")
    print("="*60)
    
    # Create dataset
    print(f"\nCreating dataset...")
    dataset = ICDDataset(
        texts=merged_df["TEXT"].tolist(),
        icd_codes=merged_df["icd_codes"].tolist(),
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        label_encoder=encoder,
        hadm_ids=merged_df["HADM_ID"].tolist(),
        cache_tokenization=True,
    )
    
    print(f"✓ Dataset created")
    print(f"  Length: {len(dataset)}")
    print(f"  Number of labels: {dataset.num_labels}")
    
    # Test __getitem__
    print(f"\nTesting __getitem__...")
    sample = dataset[0]
    
    print(f"✓ __getitem__ successful")
    print(f"  Keys: {list(sample.keys())}")
    print(f"  Shapes:")
    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f"    {key:20s}: {value.shape} ({value.dtype})")
        else:
            print(f"    {key:20s}: {type(value).__name__}")
    
    # Validate sample
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert "hadm_id" in sample
    
    assert sample["input_ids"].shape[0] == max_length
    assert sample["attention_mask"].shape[0] == max_length
    assert sample["labels"].shape[0] == encoder.num_labels
    
    # Check data types
    assert sample["input_ids"].dtype == torch.long
    assert sample["attention_mask"].dtype == torch.long
    assert sample["labels"].dtype == torch.float32
    
    # Check value ranges
    assert sample["input_ids"].min() >= 0
    assert sample["attention_mask"].min() >= 0
    assert sample["attention_mask"].max() <= 1
    assert sample["labels"].min() >= 0
    assert sample["labels"].max() <= 1
    
    print(f"\n✓ All validations passed")
    
    # Test multiple samples
    print(f"\nTesting multiple samples...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        assert sample["input_ids"].shape[0] == max_length
    
    print(f"✓ Successfully retrieved {min(5, len(dataset))} samples")
    
    return dataset


def test_dataloader(dataset, batch_size=4):
    """Test PyTorch DataLoader."""
    print("\n" + "="*60)
    print("TEST: PyTorch DataLoader")
    print("="*60)
    
    # Create dataloader
    print(f"\nCreating dataloader...")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
        drop_last=False,
    )
    
    print(f"✓ DataLoader created")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(loader)}")
    
    # Test iteration
    print(f"\nTesting batch retrieval...")
    batch = next(iter(loader))
    
    print(f"✓ Batch retrieved successfully")
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Batch shapes:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"    {key:20s}: {value.shape} ({value.dtype})")
        else:
            print(f"    {key:20s}: {type(value).__name__} (length {len(value)})")
    
    # Validate batch
    actual_batch_size = batch["input_ids"].shape[0]
    assert actual_batch_size <= batch_size
    
    assert batch["input_ids"].shape[1] == dataset[0]["input_ids"].shape[0]
    assert batch["attention_mask"].shape[1] == dataset[0]["attention_mask"].shape[0]
    assert batch["labels"].shape[1] == dataset[0]["labels"].shape[0]
    
    print(f"\n✓ All batch validations passed")
    
    # Test full iteration
    print(f"\nTesting full iteration...")
    total_samples = 0
    for i, batch in enumerate(loader):
        total_samples += batch["input_ids"].shape[0]
        if i == 0:
            print(f"  Batch 0 size: {batch['input_ids'].shape[0]}")
    
    print(f"✓ Full iteration successful")
    print(f"  Total batches: {len(loader)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Expected samples: {len(dataset)}")
    assert total_samples == len(dataset), "Sample count mismatch!"
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test data loading pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="mock_data/raw/",
        help="Directory with mock data",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top codes to keep",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer to use",
    )
    args = parser.parse_args()
    
    print("="*60)
    print("DATA PIPELINE TEST")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Top-k codes: {args.top_k}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Tokenizer: {args.tokenizer}")
    
    # Load data
    print("\nLoading mock data...")
    try:
        merged_df = load_mock_data(args.data_dir)
        print(f"✓ Loaded {len(merged_df)} discharge summaries with ICD codes")
        
        # Basic statistics
        codes_per_admission = merged_df["icd_codes"].apply(len)
        print(f"  Codes per admission: mean={codes_per_admission.mean():.1f}, "
              f"median={codes_per_admission.median():.0f}")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        print(f"\nPlease run first:")
        print(f"  python scripts/generate_mock_data.py --output {args.data_dir}")
        return 1
    
    # Run tests
    try:
        # Test 1: Label encoder
        encoder = test_label_encoder(merged_df["icd_codes"].tolist(), top_k=args.top_k)
        
        # Test 2: Dataset
        dataset = test_dataset(
            merged_df, 
            encoder, 
            tokenizer_name=args.tokenizer,
            max_length=args.max_length
        )
        
        # Test 3: DataLoader
        test_dataloader(dataset, batch_size=args.batch_size)
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✅ ALL TESTS PASSED")
    print("\nData pipeline is working correctly!")
    print("\nNext steps:")
    print("  python scripts/test_caml.py --num-labels", encoder.num_labels)
    print("  python scripts/test_led.py --num-labels", encoder.num_labels)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

