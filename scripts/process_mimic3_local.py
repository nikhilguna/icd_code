#!/usr/bin/env python3
"""
Process local MIMIC-III data into parquet format for training.

This script reads MIMIC-III CSV files from MIMIC_DATA/MIMIC-III/ directory
and creates a processed parquet file ready for model training.

Usage:
    python scripts/process_mimic3_local.py --output data/processed/mimic3_full.parquet
    
    # Process with limit for testing
    python scripts/process_mimic3_local.py --output data/processed/mimic3_test.parquet --limit 1000
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.athena_extraction import extract_mimic_data_local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Process local MIMIC-III data into parquet format"
    )
    
    parser.add_argument(
        "--notes-csv",
        type=str,
        default="MIMIC_DATA/MIMIC-III/NOTEEVENTS.csv.gz",
        help="Path to NOTEEVENTS.csv.gz (default: MIMIC_DATA/MIMIC-III/NOTEEVENTS.csv.gz)"
    )
    parser.add_argument(
        "--diagnoses-csv",
        type=str,
        default="MIMIC_DATA/MIMIC-III/DIAGNOSES_ICD.csv.gz",
        help="Path to DIAGNOSES_ICD.csv.gz (default: MIMIC_DATA/MIMIC-III/DIAGNOSES_ICD.csv.gz)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/mimic3_full.parquet",
        help="Output parquet file path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--min-codes",
        type=int,
        default=1,
        help="Minimum number of ICD codes per admission (default: 1)"
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=100,
        help="Minimum text length in characters (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Check if input files exist
    notes_path = Path(args.notes_csv)
    diagnoses_path = Path(args.diagnoses_csv)
    
    if not notes_path.exists():
        logger.error(f"Notes file not found: {notes_path}")
        logger.error("Please ensure MIMIC-III data is in MIMIC_DATA/MIMIC-III/")
        sys.exit(1)
    
    if not diagnoses_path.exists():
        logger.error(f"Diagnoses file not found: {diagnoses_path}")
        logger.error("Please ensure MIMIC-III data is in MIMIC_DATA/MIMIC-III/")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("MIMIC-III Data Processing")
    logger.info("=" * 70)
    logger.info(f"Notes file: {notes_path}")
    logger.info(f"Diagnoses file: {diagnoses_path}")
    logger.info(f"Output: {args.output}")
    if args.limit:
        logger.info(f"Sample limit: {args.limit}")
    logger.info("=" * 70)
    
    # Extract and process data
    logger.info("\nStep 1: Loading and merging data...")
    df = extract_mimic_data_local(
        notes_csv=str(notes_path),
        diagnoses_csv=str(diagnoses_path),
        output_path=args.output,
        mimic_version=3,
    )
    
    # Apply filters
    logger.info(f"\nStep 2: Applying filters...")
    logger.info(f"  Initial samples: {len(df)}")
    
    # Filter by minimum codes
    if args.min_codes > 0:
        df = df[df['num_codes'] >= args.min_codes]
        logger.info(f"  After min_codes filter ({args.min_codes}): {len(df)}")
    
    # Filter by text length
    if args.min_text_length > 0:
        df = df[df['discharge_text'].str.len() >= args.min_text_length]
        logger.info(f"  After text length filter ({args.min_text_length}): {len(df)}")
    
    # Apply sample limit
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42)
        logger.info(f"  After sampling: {len(df)}")
    
    # Save
    logger.info(f"\nStep 3: Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    # Print summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("Processing Complete!")
    logger.info("=" * 70)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Unique admissions: {df['hadm_id'].nunique()}")
    logger.info(f"Unique patients: {df['subject_id'].nunique()}")
    logger.info(f"\nICD Codes Statistics:")
    logger.info(f"  Average codes per admission: {df['num_codes'].mean():.2f}")
    logger.info(f"  Median: {df['num_codes'].median():.0f}")
    logger.info(f"  Min: {df['num_codes'].min()}")
    logger.info(f"  Max: {df['num_codes'].max()}")
    
    # Get unique codes
    all_codes = []
    for codes in df['icd_codes']:
        if isinstance(codes, list):
            all_codes.extend(codes)
        elif isinstance(codes, str):
            all_codes.extend(codes.split(','))
    unique_codes = len(set(all_codes))
    logger.info(f"  Unique ICD codes: {unique_codes}")
    
    logger.info(f"\nText Length Statistics:")
    text_lengths = df['discharge_text'].str.len()
    logger.info(f"  Average length: {text_lengths.mean():.0f} chars")
    logger.info(f"  Median: {text_lengths.median():.0f} chars")
    logger.info(f"  Min: {text_lengths.min()} chars")
    logger.info(f"  Max: {text_lengths.max()} chars")
    
    logger.info(f"\nOutput saved to: {output_path.absolute()}")
    logger.info("=" * 70)
    
    logger.info("\n✅ Next steps:")
    logger.info("1. Train CAML model:")
    logger.info(f"   python scripts/train_caml.py --data {output_path} --epochs 50 --top-k-codes 50")
    logger.info("\n2. Train LED model:")
    logger.info(f"   python scripts/train_led.py --data {output_path} --epochs 10 --top-k-codes 50")
    logger.info("\n3. Evaluate model:")
    logger.info(f"   python scripts/evaluate.py --model caml --checkpoint checkpoints/caml/best_model.pt --data {output_path}")


if __name__ == "__main__":
    main()

