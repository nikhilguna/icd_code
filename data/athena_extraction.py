"""
Amazon Athena data extraction for MIMIC-III and MIMIC-IV datasets.

MIMIC data is hosted on AWS (physionet-data S3 bucket) and can be queried
via Amazon Athena. This module provides utilities to extract discharge
summaries with their associated ICD codes.

Prerequisites:
1. AWS account with Athena access
2. PhysioNet credentialed access to MIMIC data
3. Configured AWS credentials (aws configure)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


# SQL Queries for MIMIC-III
MIMIC3_DISCHARGE_QUERY = """
SELECT 
    n.hadm_id,
    n.subject_id,
    n.chartdate,
    n.text as discharge_text,
    n.category
FROM {database}.noteevents n
WHERE n.category = 'Discharge summary'
    AND n.text IS NOT NULL
    AND LENGTH(n.text) > 100
"""

MIMIC3_ICD_QUERY = """
SELECT 
    hadm_id,
    icd9_code,
    seq_num
FROM {database}.diagnoses_icd
WHERE icd9_code IS NOT NULL
ORDER BY hadm_id, seq_num
"""

MIMIC3_COMBINED_QUERY = """
WITH discharge_notes AS (
    SELECT 
        n.hadm_id,
        n.subject_id,
        n.chartdate,
        n.text as discharge_text
    FROM {database}.noteevents n
    WHERE n.category = 'Discharge summary'
        AND n.text IS NOT NULL
        AND LENGTH(n.text) > 100
),
icd_codes AS (
    SELECT 
        hadm_id,
        ARRAY_AGG(icd9_code ORDER BY seq_num) as icd_codes,
        COUNT(*) as num_codes
    FROM {database}.diagnoses_icd
    WHERE icd9_code IS NOT NULL
    GROUP BY hadm_id
)
SELECT 
    d.hadm_id,
    d.subject_id,
    d.chartdate,
    d.discharge_text,
    i.icd_codes,
    i.num_codes
FROM discharge_notes d
INNER JOIN icd_codes i ON d.hadm_id = i.hadm_id
"""

# SQL Queries for MIMIC-IV
MIMIC4_DISCHARGE_QUERY = """
SELECT 
    hadm_id,
    subject_id,
    charttime,
    text as discharge_text
FROM {database}.discharge
WHERE text IS NOT NULL
    AND LENGTH(text) > 100
"""

MIMIC4_ICD_QUERY = """
SELECT 
    hadm_id,
    icd_code,
    icd_version,
    seq_num
FROM {database}.diagnoses_icd
WHERE icd_code IS NOT NULL
ORDER BY hadm_id, seq_num
"""

MIMIC4_COMBINED_QUERY = """
WITH discharge_notes AS (
    SELECT 
        hadm_id,
        subject_id,
        charttime,
        text as discharge_text
    FROM {database}.discharge
    WHERE text IS NOT NULL
        AND LENGTH(text) > 100
),
icd_codes AS (
    SELECT 
        hadm_id,
        ARRAY_AGG(icd_code ORDER BY seq_num) as icd_codes,
        ARRAY_AGG(icd_version ORDER BY seq_num) as icd_versions,
        COUNT(*) as num_codes
    FROM {database}.diagnoses_icd
    WHERE icd_code IS NOT NULL
    GROUP BY hadm_id
)
SELECT 
    d.hadm_id,
    d.subject_id,
    d.charttime,
    d.discharge_text,
    i.icd_codes,
    i.icd_versions,
    i.num_codes
FROM discharge_notes d
INNER JOIN icd_codes i ON d.hadm_id = i.hadm_id
"""


class MIMICExtractor:
    """
    Extract MIMIC data using Amazon Athena.
    
    Usage:
        extractor = MIMICExtractor(
            output_bucket="s3://your-bucket/athena-results/",
            region="us-east-1"
        )
        
        # Extract MIMIC-III data
        df = extractor.extract_mimic3(database="mimiciii")
        
        # Extract MIMIC-IV data
        df = extractor.extract_mimic4(database="mimiciv")
    """
    
    def __init__(
        self,
        output_bucket: str,
        region: str = "us-east-1",
        workgroup: str = "primary",
    ):
        """
        Initialize Athena extractor.
        
        Args:
            output_bucket: S3 bucket for Athena query results (e.g., "s3://bucket/path/")
            region: AWS region
            workgroup: Athena workgroup name
        """
        self.output_bucket = output_bucket
        self.region = region
        self.workgroup = workgroup
        
        self.athena_client = boto3.client("athena", region_name=region)
        self.s3_client = boto3.client("s3", region_name=region)
        
    def _execute_query(
        self,
        query: str,
        database: str,
        max_wait_seconds: int = 300,
    ) -> str:
        """
        Execute Athena query and wait for completion.
        
        Args:
            query: SQL query string
            database: Athena database name
            max_wait_seconds: Maximum time to wait for query completion
            
        Returns:
            Query execution ID
        """
        # Start query execution
        response = self.athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": database},
            ResultConfiguration={"OutputLocation": self.output_bucket},
            WorkGroup=self.workgroup,
        )
        
        query_execution_id = response["QueryExecutionId"]
        logger.info(f"Started query execution: {query_execution_id}")
        
        # Wait for query to complete
        start_time = time.time()
        while True:
            response = self.athena_client.get_query_execution(
                QueryExecutionId=query_execution_id
            )
            state = response["QueryExecution"]["Status"]["State"]
            
            if state == "SUCCEEDED":
                logger.info(f"Query completed successfully")
                return query_execution_id
            elif state in ["FAILED", "CANCELLED"]:
                reason = response["QueryExecution"]["Status"].get(
                    "StateChangeReason", "Unknown"
                )
                raise RuntimeError(f"Query {state}: {reason}")
            
            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                raise TimeoutError(
                    f"Query did not complete within {max_wait_seconds} seconds"
                )
            
            logger.debug(f"Query state: {state}, waiting...")
            time.sleep(2)
    
    def _get_query_results(self, query_execution_id: str) -> pd.DataFrame:
        """
        Retrieve query results as a pandas DataFrame.
        
        Args:
            query_execution_id: Athena query execution ID
            
        Returns:
            DataFrame with query results
        """
        # Get result location
        response = self.athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        result_location = response["QueryExecution"]["ResultConfiguration"][
            "OutputLocation"
        ]
        
        # Parse S3 path
        # Format: s3://bucket/path/query_id.csv
        s3_path = result_location.replace("s3://", "")
        bucket = s3_path.split("/")[0]
        key = "/".join(s3_path.split("/")[1:])
        
        # Download and read CSV
        logger.info(f"Downloading results from {result_location}")
        
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj["Body"])
        
        return df
    
    def extract_mimic3(
        self,
        database: str = "mimiciii",
        output_path: Optional[str] = None,
        use_combined_query: bool = True,
    ) -> pd.DataFrame:
        """
        Extract MIMIC-III discharge summaries with ICD-9 codes.
        
        Args:
            database: Athena database name for MIMIC-III
            output_path: Optional path to save results as parquet
            use_combined_query: If True, use single combined query (more efficient)
            
        Returns:
            DataFrame with columns: hadm_id, subject_id, discharge_text, icd_codes
        """
        logger.info("Extracting MIMIC-III data...")
        
        if use_combined_query:
            query = MIMIC3_COMBINED_QUERY.format(database=database)
            query_id = self._execute_query(query, database)
            df = self._get_query_results(query_id)
        else:
            # Execute separate queries and merge
            notes_query = MIMIC3_DISCHARGE_QUERY.format(database=database)
            notes_id = self._execute_query(notes_query, database)
            notes_df = self._get_query_results(notes_id)
            
            icd_query = MIMIC3_ICD_QUERY.format(database=database)
            icd_id = self._execute_query(icd_query, database)
            icd_df = self._get_query_results(icd_id)
            
            # Aggregate ICD codes per admission
            icd_agg = icd_df.groupby("hadm_id").agg({
                "icd9_code": list
            }).reset_index()
            icd_agg.columns = ["hadm_id", "icd_codes"]
            
            # Merge
            df = notes_df.merge(icd_agg, on="hadm_id", how="inner")
        
        logger.info(f"Extracted {len(df)} MIMIC-III discharge summaries")
        
        if output_path:
            self._save_results(df, output_path)
        
        return df
    
    def extract_mimic4(
        self,
        database: str = "mimiciv",
        output_path: Optional[str] = None,
        use_combined_query: bool = True,
        icd_version: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Extract MIMIC-IV discharge summaries with ICD codes.
        
        Args:
            database: Athena database name for MIMIC-IV
            output_path: Optional path to save results as parquet
            use_combined_query: If True, use single combined query
            icd_version: Filter by ICD version (9 or 10). If None, include all.
            
        Returns:
            DataFrame with columns: hadm_id, subject_id, discharge_text, icd_codes
        """
        logger.info("Extracting MIMIC-IV data...")
        
        if use_combined_query:
            query = MIMIC4_COMBINED_QUERY.format(database=database)
            query_id = self._execute_query(query, database)
            df = self._get_query_results(query_id)
        else:
            # Execute separate queries and merge
            notes_query = MIMIC4_DISCHARGE_QUERY.format(database=database)
            notes_id = self._execute_query(notes_query, database)
            notes_df = self._get_query_results(notes_id)
            
            icd_query = MIMIC4_ICD_QUERY.format(database=database)
            icd_id = self._execute_query(icd_query, database)
            icd_df = self._get_query_results(icd_id)
            
            # Filter by ICD version if specified
            if icd_version is not None:
                icd_df = icd_df[icd_df["icd_version"] == icd_version]
            
            # Aggregate ICD codes per admission
            icd_agg = icd_df.groupby("hadm_id").agg({
                "icd_code": list,
                "icd_version": lambda x: list(x)[0] if len(set(x)) == 1 else list(x)
            }).reset_index()
            icd_agg.columns = ["hadm_id", "icd_codes", "icd_version"]
            
            # Merge
            df = notes_df.merge(icd_agg, on="hadm_id", how="inner")
        
        logger.info(f"Extracted {len(df)} MIMIC-IV discharge summaries")
        
        if output_path:
            self._save_results(df, output_path)
        
        return df
    
    def _save_results(self, df: pd.DataFrame, output_path: str) -> None:
        """Save DataFrame to parquet file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
    
    def get_icd_code_statistics(
        self,
        database: str,
        mimic_version: int = 3,
    ) -> pd.DataFrame:
        """
        Get statistics about ICD code frequencies.
        
        Args:
            database: Athena database name
            mimic_version: 3 for MIMIC-III, 4 for MIMIC-IV
            
        Returns:
            DataFrame with code frequencies
        """
        if mimic_version == 3:
            query = f"""
            SELECT 
                icd9_code as icd_code,
                COUNT(*) as frequency,
                COUNT(DISTINCT hadm_id) as num_admissions
            FROM {database}.diagnoses_icd
            WHERE icd9_code IS NOT NULL
            GROUP BY icd9_code
            ORDER BY frequency DESC
            """
        else:
            query = f"""
            SELECT 
                icd_code,
                icd_version,
                COUNT(*) as frequency,
                COUNT(DISTINCT hadm_id) as num_admissions
            FROM {database}.diagnoses_icd
            WHERE icd_code IS NOT NULL
            GROUP BY icd_code, icd_version
            ORDER BY frequency DESC
            """
        
        query_id = self._execute_query(query, database)
        return self._get_query_results(query_id)


def extract_mimic_data_local(
    notes_csv: str,
    diagnoses_csv: str,
    output_path: str,
    mimic_version: int = 3,
) -> pd.DataFrame:
    """
    Extract MIMIC data from local CSV files (alternative to Athena).
    
    Use this if you have downloaded MIMIC data locally from PhysioNet.
    
    Args:
        notes_csv: Path to NOTEEVENTS.csv (MIMIC-III) or discharge.csv (MIMIC-IV)
        diagnoses_csv: Path to DIAGNOSES_ICD.csv
        output_path: Path to save output parquet
        mimic_version: 3 or 4
        
    Returns:
        DataFrame with discharge summaries and ICD codes
    """
    logger.info("Loading local MIMIC data...")
    
    # Load notes
    if mimic_version == 3:
        notes_df = pd.read_csv(
            notes_csv,
            usecols=["HADM_ID", "SUBJECT_ID", "CHARTDATE", "CATEGORY", "TEXT"],
            low_memory=False,
        )
        notes_df.columns = notes_df.columns.str.lower()
        notes_df = notes_df[notes_df["category"] == "Discharge summary"]
        notes_df = notes_df.dropna(subset=["text"])
        notes_df = notes_df[notes_df["text"].str.len() > 100]
        notes_df = notes_df.rename(columns={"text": "discharge_text"})
    else:
        notes_df = pd.read_csv(
            notes_csv,
            usecols=["hadm_id", "subject_id", "charttime", "text"],
            low_memory=False,
        )
        notes_df = notes_df.dropna(subset=["text"])
        notes_df = notes_df[notes_df["text"].str.len() > 100]
        notes_df = notes_df.rename(columns={"text": "discharge_text"})
    
    logger.info(f"Loaded {len(notes_df)} discharge summaries")
    
    # Load diagnoses
    if mimic_version == 3:
        icd_df = pd.read_csv(
            diagnoses_csv,
            usecols=["HADM_ID", "ICD9_CODE", "SEQ_NUM"],
            low_memory=False,
        )
        icd_df.columns = icd_df.columns.str.lower()
        icd_df = icd_df.dropna(subset=["icd9_code"])
        code_col = "icd9_code"
    else:
        icd_df = pd.read_csv(
            diagnoses_csv,
            usecols=["hadm_id", "icd_code", "icd_version", "seq_num"],
            low_memory=False,
        )
        icd_df = icd_df.dropna(subset=["icd_code"])
        code_col = "icd_code"
    
    logger.info(f"Loaded {len(icd_df)} diagnosis records")
    
    # Aggregate ICD codes per admission
    icd_agg = icd_df.sort_values(["hadm_id", "seq_num"]).groupby("hadm_id").agg({
        code_col: list
    }).reset_index()
    icd_agg.columns = ["hadm_id", "icd_codes"]
    icd_agg["num_codes"] = icd_agg["icd_codes"].apply(len)
    
    # Merge
    df = notes_df.merge(icd_agg, on="hadm_id", how="inner")
    
    logger.info(f"Final dataset: {len(df)} samples with ICD codes")
    logger.info(f"Average codes per sample: {df['num_codes'].mean():.1f}")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    return df
