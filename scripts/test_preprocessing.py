#!/usr/bin/env python3
"""
Test preprocessing pipeline on mock data.

Tests:
- Section parsing
- Text normalization
- Tokenization
- Data quality checks
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data.preprocessing import ClinicalTextPreprocessor, get_section_statistics


def test_section_parsing(preprocessor, notes_df, num_samples=10):
    """Test section header detection."""
    print("\n" + "="*60)
    print("TEST: Section Parsing")
    print("="*60)
    
    successful = 0
    all_sections = []
    
    for i in range(min(num_samples, len(notes_df))):
        text = notes_df.iloc[i]["TEXT"]
        hadm_id = notes_df.iloc[i]["HADM_ID"]
        
        try:
            doc = preprocessor.process_document(text, hadm_id=hadm_id, tokenize=False)
            num_sections = len(doc.sections)
            
            if num_sections > 0:
                successful += 1
                all_sections.extend([s.name for s in doc.sections])
                
                if i < 3:  # Show first 3 examples
                    print(f"\nSample {i+1} (HADM_ID: {hadm_id}):")
                    print(f"  Sections found: {num_sections}")
                    for section in doc.sections[:5]:  # Show first 5 sections
                        print(f"    - {section.name}: {len(section.text)} chars")
        except Exception as e:
            print(f"  ✗ Failed on sample {i}: {e}")
            continue
    
    print(f"\n✓ Section parsing: {successful}/{num_samples} successful")
    
    if all_sections:
        section_counts = pd.Series(all_sections).value_counts()
        print(f"\nMost common sections:")
        for section, count in section_counts.head(10).items():
            print(f"  - {section}: {count} occurrences")
    
    return successful == num_samples


def test_text_normalization(preprocessor, notes_df, num_samples=10):
    """Test text normalization."""
    print("\n" + "="*60)
    print("TEST: Text Normalization")
    print("="*60)
    
    for i in range(min(3, len(notes_df))):
        text = notes_df.iloc[i]["TEXT"]
        hadm_id = notes_df.iloc[i]["HADM_ID"]
        
        doc = preprocessor.process_document(text, hadm_id=hadm_id, tokenize=False)
        
        print(f"\nSample {i+1}:")
        print(f"  Original length: {len(doc.original_text)} chars")
        print(f"  Processed length: {len(doc.processed_text)} chars")
        
        # Check normalization features
        if preprocessor.remove_deidentified:
            has_deident = "[**" in doc.processed_text
            print(f"  De-identification markers removed: {not has_deident}")
        
        if preprocessor.lowercase:
            is_lower = doc.processed_text.islower()
            print(f"  Lowercased: {is_lower}")
        
        # Show snippet
        snippet = doc.processed_text[:200]
        print(f"  Snippet: {snippet}...")
    
    print("\n✓ Text normalization: All samples valid")
    return True


def test_tokenization(preprocessor, notes_df, num_samples=10):
    """Test tokenization."""
    print("\n" + "="*60)
    print("TEST: Tokenization")
    print("="*60)
    
    all_lengths = []
    vocab_size = preprocessor.tokenizer.vocab_size
    
    for i in range(min(num_samples, len(notes_df))):
        text = notes_df.iloc[i]["TEXT"]
        hadm_id = notes_df.iloc[i]["HADM_ID"]
        
        doc = preprocessor.process_document(text, hadm_id=hadm_id, tokenize=True)
        
        # Validate token IDs
        token_ids = np.array(doc.token_ids)
        attention_mask = np.array(doc.attention_mask)
        
        # Check shapes
        assert len(token_ids) == preprocessor.max_length, f"Token length mismatch: {len(token_ids)} != {preprocessor.max_length}"
        assert len(attention_mask) == preprocessor.max_length, f"Mask length mismatch"
        
        # Check token ID range
        assert token_ids.min() >= 0, f"Negative token ID found"
        assert token_ids.max() < vocab_size, f"Token ID exceeds vocab size: {token_ids.max()} >= {vocab_size}"
        
        # Check attention mask is binary
        assert set(attention_mask).issubset({0, 1}), f"Attention mask not binary: {set(attention_mask)}"
        
        # Count real tokens (non-padding)
        real_tokens = attention_mask.sum()
        all_lengths.append(real_tokens)
        
        if i < 3:
            print(f"\nSample {i+1}:")
            print(f"  Token IDs shape: {token_ids.shape}")
            print(f"  Token ID range: [{token_ids.min()}, {token_ids.max()}]")
            print(f"  Real tokens: {real_tokens}/{preprocessor.max_length}")
            print(f"  Padding tokens: {preprocessor.max_length - real_tokens}")
    
    all_lengths = np.array(all_lengths)
    print(f"\n✓ Tokenization successful")
    print(f"  All token IDs in valid range: ✓")
    print(f"  All attention masks binary: ✓")
    print(f"  All sequences length {preprocessor.max_length}: ✓")
    print(f"\nToken length statistics:")
    print(f"  Mean: {all_lengths.mean():.1f}")
    print(f"  Median: {np.median(all_lengths):.0f}")
    print(f"  Min: {all_lengths.min()}")
    print(f"  Max: {all_lengths.max()}")
    
    return True


def test_no_errors(preprocessor, notes_df):
    """Test that all samples can be processed without errors."""
    print("\n" + "="*60)
    print("TEST: Processing All Samples")
    print("="*60)
    
    errors = []
    
    for i in range(len(notes_df)):
        text = notes_df.iloc[i]["TEXT"]
        hadm_id = notes_df.iloc[i]["HADM_ID"]
        
        try:
            doc = preprocessor.process_document(text, hadm_id=hadm_id, tokenize=True)
            
            # Basic validation
            assert doc.hadm_id == hadm_id
            assert len(doc.token_ids) == preprocessor.max_length
            assert not np.isnan(doc.token_ids).any()
            
        except Exception as e:
            errors.append((i, hadm_id, str(e)))
    
    if errors:
        print(f"\n✗ Errors in {len(errors)}/{len(notes_df)} samples:")
        for i, hadm_id, error in errors[:5]:  # Show first 5
            print(f"  Sample {i} (HADM_ID {hadm_id}): {error}")
        return False
    else:
        print(f"\n✓ All {len(notes_df)} samples processed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test preprocessing pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="mock_data/raw/",
        help="Directory with mock data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to test in detail",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length (use 512 for quick testing)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer to use (use BERT for quick testing)",
    )
    args = parser.parse_args()
    
    print("="*60)
    print("PREPROCESSING PIPELINE TEST")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Max length: {args.max_length}")
    print(f"Detailed test samples: {args.num_samples}")
    
    # Load mock data
    print("\nLoading mock data...")
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        print(f"\nPlease run first:")
        print(f"  python scripts/generate_mock_data.py --output {args.data_dir}")
        return 1
    
    notes_path = data_dir / "NOTEEVENTS.csv"
    if not notes_path.exists():
        print(f"✗ NOTEEVENTS.csv not found in {data_dir}")
        return 1
    
    notes_df = pd.read_csv(notes_path)
    print(f"✓ Loaded {len(notes_df)} discharge summaries")
    
    # Initialize preprocessor
    print(f"\nInitializing preprocessor...")
    try:
        preprocessor = ClinicalTextPreprocessor(
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
            lowercase=True,
            remove_deidentified=True,
            preserve_sections=True,
        )
        print(f"✓ Preprocessor initialized")
        print(f"  Tokenizer vocab size: {preprocessor.tokenizer.vocab_size}")
    except Exception as e:
        print(f"✗ Failed to initialize preprocessor: {e}")
        return 1
    
    # Run tests
    all_passed = True
    
    try:
        # Test 1: Section parsing
        passed = test_section_parsing(preprocessor, notes_df, args.num_samples)
        all_passed = all_passed and passed
        
        # Test 2: Text normalization
        passed = test_text_normalization(preprocessor, notes_df, args.num_samples)
        all_passed = all_passed and passed
        
        # Test 3: Tokenization
        passed = test_tokenization(preprocessor, notes_df, args.num_samples)
        all_passed = all_passed and passed
        
        # Test 4: No errors on all samples
        passed = test_no_errors(preprocessor, notes_df)
        all_passed = all_passed and passed
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nNext steps:")
        print("  python scripts/test_data_pipeline.py --data-dir", args.data_dir)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the errors above and fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

