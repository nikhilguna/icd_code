"""
Preprocessing pipeline for clinical discharge summaries.

Handles:
- Section segmentation using clinical headers
- Text normalization (lowercase, de-identification cleanup)
- Tokenization using HuggingFace tokenizers
- Document truncation for different model architectures
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import pandas as pd
import numpy as np
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# Common clinical section headers found in MIMIC discharge summaries
SECTION_HEADERS = [
    # Primary sections
    "HISTORY OF PRESENT ILLNESS",
    "CHIEF COMPLAINT",
    "PRESENT ILLNESS",
    "HOSPITAL COURSE",
    "BRIEF HOSPITAL COURSE",
    "PAST MEDICAL HISTORY",
    "PAST SURGICAL HISTORY",
    "MEDICATIONS ON ADMISSION",
    "ADMISSION MEDICATIONS",
    "DISCHARGE MEDICATIONS",
    "MEDICATIONS ON DISCHARGE",
    "ALLERGIES",
    "SOCIAL HISTORY",
    "FAMILY HISTORY",
    "PHYSICAL EXAM",
    "PHYSICAL EXAMINATION",
    "ADMISSION PHYSICAL EXAM",
    "DISCHARGE PHYSICAL EXAM",
    "PERTINENT RESULTS",
    "LABORATORY DATA",
    "LABS",
    "IMAGING",
    "STUDIES",
    "PROCEDURES",
    "OPERATIONS",
    "CONSULTATIONS",
    "DISCHARGE DIAGNOSIS",
    "DISCHARGE DIAGNOSES",
    "ADMISSION DIAGNOSIS",
    "ADMISSION DIAGNOSES",
    "PRINCIPAL DIAGNOSIS",
    "SECONDARY DIAGNOSES",
    "DISCHARGE CONDITION",
    "DISCHARGE DISPOSITION",
    "DISCHARGE INSTRUCTIONS",
    "FOLLOWUP",
    "FOLLOW UP",
    "FOLLOW-UP",
    "PLAN",
    "ASSESSMENT AND PLAN",
    "IMPRESSION",
    "CODE STATUS",
    "REVIEW OF SYSTEMS",
]

# Pattern to match section headers (case insensitive)
SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*[:.]?\s*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE
)

# Pattern to match de-identified placeholders in MIMIC
DEIDENTIFIED_PATTERN = re.compile(r"\[\*\*[^\]]*\*\*\]")

# Pattern to match dates
DATE_PATTERN = re.compile(
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
)

# Pattern to match times
TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b")

# Pattern to match numeric lab values with units
LAB_VALUE_PATTERN = re.compile(r"\b\d+\.?\d*\s*(?:mg|ml|mcg|mEq|mmol|g|L|%|IU|units?)\b", re.IGNORECASE)


@dataclass
class Section:
    """Represents a section of a clinical document."""
    name: str
    text: str
    start_pos: int
    end_pos: int


@dataclass
class ProcessedDocument:
    """Represents a fully processed clinical document."""
    hadm_id: int
    original_text: str
    processed_text: str
    sections: List[Section]
    token_ids: Optional[List[int]] = None
    attention_mask: Optional[List[int]] = None
    section_spans: Optional[Dict[str, Tuple[int, int]]] = None  # Section name -> (start_token, end_token)


class ClinicalTextPreprocessor:
    """
    Preprocessor for clinical discharge summaries.
    
    Handles section parsing, text normalization, and tokenization.
    
    Usage:
        preprocessor = ClinicalTextPreprocessor(
            tokenizer_name="allenai/longformer-base-4096",
            max_length=4096,
            lowercase=True,
        )
        
        processed = preprocessor.process_document(text, hadm_id=12345)
    """
    
    def __init__(
        self,
        tokenizer_name: str = "allenai/longformer-base-4096",
        max_length: int = 4096,
        lowercase: bool = True,
        remove_deidentified: bool = True,
        remove_dates: bool = False,
        remove_times: bool = False,
        preserve_sections: bool = True,
    ):
        """
        Initialize preprocessor.
        
        Args:
            tokenizer_name: HuggingFace tokenizer to use
            max_length: Maximum sequence length for tokenization
            lowercase: Whether to lowercase text
            remove_deidentified: Remove [**...**] placeholders
            remove_dates: Remove date patterns
            remove_times: Remove time patterns  
            preserve_sections: Keep section headers in output
        """
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.lowercase = lowercase
        self.remove_deidentified = remove_deidentified
        self.remove_dates = remove_dates
        self.remove_times = remove_times
        self.preserve_sections = preserve_sections
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def parse_sections(self, text: str) -> List[Section]:
        """
        Parse clinical document into sections.
        
        Args:
            text: Raw discharge summary text
            
        Returns:
            List of Section objects
        """
        sections = []
        
        # Find all section headers
        matches = list(SECTION_PATTERN.finditer(text))
        
        if not matches:
            # No sections found, treat entire document as one section
            sections.append(Section(
                name="FULL_DOCUMENT",
                text=text.strip(),
                start_pos=0,
                end_pos=len(text)
            ))
            return sections
        
        # Handle text before first section
        if matches[0].start() > 0:
            pre_text = text[:matches[0].start()].strip()
            if pre_text:
                sections.append(Section(
                    name="PREAMBLE",
                    text=pre_text,
                    start_pos=0,
                    end_pos=matches[0].start()
                ))
        
        # Extract each section
        for i, match in enumerate(matches):
            section_name = match.group(1).upper().strip()
            start_pos = match.end()
            
            # End position is start of next section or end of document
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            section_text = text[start_pos:end_pos].strip()
            
            if section_text:  # Only add non-empty sections
                sections.append(Section(
                    name=section_name,
                    text=section_text,
                    start_pos=start_pos,
                    end_pos=end_pos
                ))
        
        return sections
    
    def normalize_text(self, text: str) -> str:
        """
        Apply text normalization.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove de-identified placeholders
        if self.remove_deidentified:
            text = DEIDENTIFIED_PATTERN.sub(" ", text)
        
        # Remove dates
        if self.remove_dates:
            text = DATE_PATTERN.sub(" ", text)
        
        # Remove times
        if self.remove_times:
            text = TIME_PATTERN.sub(" ", text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        
        return text
    
    def process_document(
        self,
        text: str,
        hadm_id: int,
        tokenize: bool = True,
    ) -> ProcessedDocument:
        """
        Process a single clinical document.
        
        Args:
            text: Raw discharge summary text
            hadm_id: Hospital admission ID
            tokenize: Whether to tokenize the text
            
        Returns:
            ProcessedDocument with sections and optional tokens
        """
        # Parse sections
        sections = self.parse_sections(text)
        
        # Build processed text
        if self.preserve_sections:
            processed_parts = []
            for section in sections:
                normalized = self.normalize_text(section.text)
                if section.name != "FULL_DOCUMENT" and section.name != "PREAMBLE":
                    processed_parts.append(f"[{section.name}] {normalized}")
                else:
                    processed_parts.append(normalized)
            processed_text = " ".join(processed_parts)
        else:
            # Just concatenate all section text
            all_text = " ".join(s.text for s in sections)
            processed_text = self.normalize_text(all_text)
        
        result = ProcessedDocument(
            hadm_id=hadm_id,
            original_text=text,
            processed_text=processed_text,
            sections=sections,
        )
        
        # Tokenize
        if tokenize:
            encoding = self.tokenizer(
                processed_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            result.token_ids = encoding["input_ids"]
            result.attention_mask = encoding["attention_mask"]
            
            # Compute section spans in token space
            result.section_spans = self._compute_section_spans(processed_text, sections)
        
        return result
    
    def _compute_section_spans(
        self,
        processed_text: str,
        sections: List[Section],
    ) -> Dict[str, Tuple[int, int]]:
        """
        Map section boundaries to token positions.
        
        Args:
            processed_text: Processed document text
            sections: List of sections
            
        Returns:
            Dict mapping section names to (start_token, end_token) tuples
        """
        section_spans = {}
        
        for section in sections:
            # Find section marker in processed text
            if self.preserve_sections and section.name not in ["FULL_DOCUMENT", "PREAMBLE"]:
                marker = f"[{section.name}]"
                if self.lowercase:
                    marker = marker.lower()
            else:
                continue
            
            # Find position in processed text
            start_char = processed_text.find(marker)
            if start_char == -1:
                continue
            
            # Get the normalized section text
            normalized_section = self.normalize_text(section.text)
            end_char = start_char + len(marker) + 1 + len(normalized_section)
            
            # Convert char positions to token positions
            # This is approximate - tokenization may not align exactly
            prefix = processed_text[:start_char]
            prefix_tokens = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            start_token = len(prefix_tokens) + 1  # +1 for CLS token
            
            section_text = processed_text[start_char:end_char]
            section_tokens = self.tokenizer(section_text, add_special_tokens=False)["input_ids"]
            end_token = start_token + len(section_tokens)
            
            # Clamp to max length
            end_token = min(end_token, self.max_length - 1)
            
            section_spans[section.name] = (start_token, end_token)
        
        return section_spans
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "discharge_text",
        hadm_column: str = "hadm_id",
        tokenize: bool = True,
        show_progress: bool = True,
    ) -> List[ProcessedDocument]:
        """
        Process a DataFrame of discharge summaries.
        
        Args:
            df: DataFrame with discharge summaries
            text_column: Column name for text
            hadm_column: Column name for admission IDs
            tokenize: Whether to tokenize
            show_progress: Show progress bar
            
        Returns:
            List of ProcessedDocument objects
        """
        from tqdm import tqdm
        
        documents = []
        iterator = tqdm(df.iterrows(), total=len(df), disable=not show_progress)
        
        for _, row in iterator:
            doc = self.process_document(
                text=row[text_column],
                hadm_id=row[hadm_column],
                tokenize=tokenize,
            )
            documents.append(doc)
        
        return documents
    
    def save_processed_data(
        self,
        documents: List[ProcessedDocument],
        output_path: str,
        include_original: bool = False,
    ) -> None:
        """
        Save processed documents to parquet.
        
        Args:
            documents: List of ProcessedDocument objects
            output_path: Output file path
            include_original: Whether to include original text (increases file size)
        """
        records = []
        for doc in documents:
            record = {
                "hadm_id": doc.hadm_id,
                "processed_text": doc.processed_text,
                "token_ids": doc.token_ids,
                "attention_mask": doc.attention_mask,
                "num_sections": len(doc.sections),
                "section_names": [s.name for s in doc.sections],
            }
            if include_original:
                record["original_text"] = doc.original_text
            if doc.section_spans:
                record["section_spans"] = doc.section_spans
            records.append(record)
        
        df = pd.DataFrame(records)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(documents)} processed documents to {output_path}")


def get_section_statistics(documents: List[ProcessedDocument]) -> pd.DataFrame:
    """
    Compute statistics about sections across documents.
    
    Args:
        documents: List of ProcessedDocument objects
        
    Returns:
        DataFrame with section statistics
    """
    section_counts = {}
    section_lengths = {}
    
    for doc in documents:
        for section in doc.sections:
            name = section.name
            if name not in section_counts:
                section_counts[name] = 0
                section_lengths[name] = []
            section_counts[name] += 1
            section_lengths[name].append(len(section.text))
    
    stats = []
    for name in section_counts:
        lengths = section_lengths[name]
        stats.append({
            "section": name,
            "count": section_counts[name],
            "frequency": section_counts[name] / len(documents),
            "mean_length": np.mean(lengths),
            "median_length": np.median(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
        })
    
    return pd.DataFrame(stats).sort_values("count", ascending=False)
