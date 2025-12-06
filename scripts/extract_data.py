#!/usr/bin/env python3
"""
Script to extract MIMIC data using Amazon Athena or local CSV files.

Usage:
    # Using Athena
    python scripts/extract_data.py --source athena --dataset mimic3 \
        --output data/raw/mimic3.parquet \
        --bucket s3://your-bucket/athena-results/
    
    # Using local CSV files
    python scripts/extract_data.py --source local --dataset mimic3 \
        --notes-csv /path/to/NOTEEVENTS.csv \
        --diagnoses-csv /path/to/DIAGNOSES_ICD.csv \
        --output data/raw/mimic3.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.athena_extraction import MIMICExtractor, extract_mimic_data_local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract MIMIC discharge summaries with ICD codes"
    )
    
    parser.add_argument(
        "--source",
        choices=["local", "athena"],
        default="local",
        help="Data source: 'local' for CSV files (default), 'athena' for AWS Athena"
    )
    parser.add_argument(
        "--dataset",
        choices=["mimic3", "mimic4"],
        required=True,
        help="Dataset to extract"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for parquet file"
    )
    
    # Athena-specific arguments
    parser.add_argument(
        "--bucket",
        type=str,
        help="S3 bucket for Athena results (required for --source athena)"
    )
    parser.add_argument(
        "--database",
        type=str,
        help="Athena database name (default: mimiciii or mimiciv)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region"
    )
    
    # Local-specific arguments
    parser.add_argument(
        "--notes-csv",
        type=str,
        help="Path to NOTEEVENTS.csv or discharge.csv (default: MIMIC_DATA/MIMIC-III/NOTEEVENTS.csv.gz)"
    )
    parser.add_argument(
        "--diagnoses-csv",
        type=str,
        help="Path to DIAGNOSES_ICD.csv (default: MIMIC_DATA/MIMIC-III/DIAGNOSES_ICD.csv.gz)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing MIMIC data files (alternative to specifying individual files)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--icd-version",
        type=int,
        choices=[9, 10],
        help="Filter MIMIC-IV by ICD version (9 or 10)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute ICD code statistics, don't extract full data"
    )
    
    args = parser.parse_args()
    
    # Set default paths for local data
    if args.source == "local":
        if args.data_dir:
            # Use data directory structure
            if args.dataset == "mimic3":
                notes_default = Path(args.data_dir) / "NOTEEVENTS.csv.gz"
                diagnoses_default = Path(args.data_dir) / "DIAGNOSES_ICD.csv.gz"
            else:
                notes_default = Path(args.data_dir) / "discharge.csv.gz"
                diagnoses_default = Path(args.data_dir) / "diagnoses_icd.csv.gz"
        else:
            # Use default MIMIC_DATA structure
            if args.dataset == "mimic3":
                notes_default = Path("MIMIC_DATA/MIMIC-III/NOTEEVENTS.csv.gz")
                diagnoses_default = Path("MIMIC_DATA/MIMIC-III/DIAGNOSES_ICD.csv.gz")
            else:
                notes_default = Path("MIMIC_DATA/MIMIC-IV/discharge.csv.gz")
                diagnoses_default = Path("MIMIC_DATA/MIMIC-IV/diagnoses_icd.csv.gz")
        
        # Use defaults if not specified
        if not args.notes_csv:
            args.notes_csv = str(notes_default)
        if not args.diagnoses_csv:
            args.diagnoses_csv = str(diagnoses_default)
    
    # Set default database names
    if args.database is None:
        args.database = "mimiciii" if args.dataset == "mimic3" else "mimiciv"
    
    mimic_version = 3 if args.dataset == "mimic3" else 4
    
    if args.source == "athena":
        if not args.bucket:
            parser.error("--bucket is required when using --source athena")
        
        extractor = MIMICExtractor(
            output_bucket=args.bucket,
            region=args.region,
        )
        
        if args.stats_only:
            logger.info("Computing ICD code statistics...")
            stats_df = extractor.get_icd_code_statistics(
                args.database, 
                mimic_version=mimic_version
            )
            stats_output = args.output.replace(".parquet", "_stats.parquet")
            stats_df.to_parquet(stats_output, index=False)
            logger.info(f"Saved statistics to {stats_output}")
            logger.info(f"\nTop 20 codes:\n{stats_df.head(20)}")
            return
        
        if args.dataset == "mimic3":
            df = extractor.extract_mimic3(
                database=args.database,
                output_path=args.output,
            )
        else:
            df = extractor.extract_mimic4(
                database=args.database,
                output_path=args.output,
                icd_version=args.icd_version,
            )
    
    else:  # local
        if not args.notes_csv or not args.diagnoses_csv:
            parser.error(
                "--notes-csv and --diagnoses-csv are required when using --source local\n"
                f"Defaults: notes={args.notes_csv}, diagnoses={args.diagnoses_csv}"
            )
        
        # Check if files exist
        if not Path(args.notes_csv).exists():
            logger.error(f"Notes file not found: {args.notes_csv}")
            parser.error(f"Notes file does not exist: {args.notes_csv}")
        if not Path(args.diagnoses_csv).exists():
            logger.error(f"Diagnoses file not found: {args.diagnoses_csv}")
            parser.error(f"Diagnoses file does not exist: {args.diagnoses_csv}")
        
        logger.info(f"Using local files:")
        logger.info(f"  Notes: {args.notes_csv}")
        logger.info(f"  Diagnoses: {args.diagnoses_csv}")
        
        df = extract_mimic_data_local(
            notes_csv=args.notes_csv,
            diagnoses_csv=args.diagnoses_csv,
            output_path=args.output,
            mimic_version=mimic_version,
        )
    
    # Print summary statistics
    logger.info("\n" + "=" * 50)
    logger.info("Extraction Summary")
    logger.info("=" * 50)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Unique admissions: {df['hadm_id'].nunique()}")
    logger.info(f"Average codes per sample: {df['num_codes'].mean():.2f}")
    logger.info(f"Min codes: {df['num_codes'].min()}")
    logger.info(f"Max codes: {df['num_codes'].max()}")
    logger.info(f"Text length (chars) - Mean: {df['discharge_text'].str.len().mean():.0f}")
    logger.info(f"Text length (chars) - Median: {df['discharge_text'].str.len().median():.0f}")
    logger.info(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
