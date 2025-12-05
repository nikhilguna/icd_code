"""
Enhanced preprocessing pipeline with scispacy and advanced clinical text handling.

Integrates improvements from comparative analysis:
- scispacy for medical NLP
- Clinical sentence tokenization
- Header extraction as separate features
- Better de-identification handling
- Stemming/lemmatization options
- Vocabulary filtering
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# Optional scispacy import (may not be installed yet)
try:
    import spacy
    import scispacy
    SCISPACY_AVAILABLE = True
except ImportError:
    SCISPACY_AVAILABLE = False
    logging.warning("scispacy not available - install with: pip install scispacy")

from .preprocessing import (
    Section,
    ProcessedDocument,
    ClinicalTextPreprocessor,
    SECTION_HEADERS,
)
from .clinical_tokenizer import ClinicalSentenceTokenizer

logger = logging.getLogger(__name__)


# Headers to discard (administrative/non-clinical)
DISCARD_HEADERS = {
    'ADMISSION DATE', 'DISCHARGE DATE', 'DATE OF BIRTH', 
    'PHONE', 'FAX', 'ATTENDING', 'DICTATED BY',
    'COMPLETED BY', 'PROVIDER', 'JOB#', 'MD PHONE',
    'M.D. PHONE', 'MD', 'PHD', 'SIGNED ELECTRONICALLY BY',
}

# Patterns for headers to discard
DISCARD_HEADER_PATTERNS = [
    r'^\s*[IVX]+\s*$',  # Roman numerals only
    r'^\s*[A-Z]{1,2}\s*$',  # Single/double letters
    r'.*[Pp]hone.*',
    r'.*[Dd]ate.*',
    r'.*[Tt]ime.*',
    r'^\s*\d+\s*$',  # Numbers only
]

# Improved de-identification pattern
DEID_PATTERN = re.compile(r'\[\*\*[^\]]*\*\*\]')
DEID_REPLACEMENT = '[DEID]'


@dataclass
class EnhancedProcessedDocument(ProcessedDocument):
    """Extended ProcessedDocument with additional clinical features."""
    headers: List[str] = field(default_factory=list)
    header_positions: List[Tuple[int, int]] = field(default_factory=list)
    clinical_sentences: List[str] = field(default_factory=list)
    vocabulary_tokens: Optional[List[str]] = None


class EnhancedClinicalPreprocessor(ClinicalTextPreprocessor):
    """
    Enhanced preprocessor with scispacy and advanced clinical text handling.
    
    Adds:
    - scispacy medical NLP
    - Clinical sentence tokenization
    - Header extraction
    - Stemming/lemmatization
    - Vocabulary filtering
    - Better de-identification
    
    Usage:
        preprocessor = EnhancedClinicalPreprocessor(
            tokenizer_name="allenai/led-base-16384",
            max_length=16384,
            use_scispacy=True,
            use_clinical_tokenizer=True,
            extract_headers=True,
        )
        
        doc = preprocessor.process_document(text, hadm_id=12345)
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
        use_scispacy: bool = True,
        scispacy_model: str = "en_core_sci_md",
        use_clinical_tokenizer: bool = True,
        extract_headers: bool = True,
        filter_headers: bool = True,
        use_stemming: bool = False,
        use_lemmatization: bool = False,
        vocabulary: Optional[Set[str]] = None,
    ):
        """
        Initialize enhanced preprocessor.
        
        Args:
            tokenizer_name: HuggingFace tokenizer
            max_length: Maximum sequence length
            lowercase: Lowercase text
            remove_deidentified: Replace de-id markers with [DEID]
            remove_dates: Remove date patterns
            remove_times: Remove time patterns
            preserve_sections: Keep section structure
            use_scispacy: Use scispacy for medical NLP
            scispacy_model: scispacy model name
            use_clinical_tokenizer: Use clinical sentence tokenizer
            extract_headers: Extract headers as separate features
            filter_headers: Filter non-clinical headers
            use_stemming: Apply Porter stemming
            use_lemmatization: Apply lemmatization
            vocabulary: Optional vocabulary filter
        """
        super().__init__(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            lowercase=lowercase,
            remove_deidentified=remove_deidentified,
            remove_dates=remove_dates,
            remove_times=remove_times,
            preserve_sections=preserve_sections,
        )
        
        self.use_scispacy = use_scispacy
        self.scispacy_model = scispacy_model
        self.use_clinical_tokenizer = use_clinical_tokenizer
        self.extract_headers = extract_headers
        self.filter_headers = filter_headers
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.vocabulary = vocabulary
        
        # Initialize scispacy
        self.nlp = None
        if use_scispacy:
            if not SCISPACY_AVAILABLE:
                logger.warning("scispacy not available, falling back to standard processing")
                self.use_scispacy = False
            else:
                try:
                    logger.info(f"Loading scispacy model: {scispacy_model}")
                    # Disable tagger and NER for speed (we only need tokenization)
                    self.nlp = spacy.load(scispacy_model, disable=['tagger', 'ner'])
                    logger.info("scispacy loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load scispacy model: {e}")
                    logger.warning("Falling back to standard processing")
                    self.use_scispacy = False
        
        # Initialize clinical tokenizer
        self.clinical_tokenizer = None
        if use_clinical_tokenizer:
            self.clinical_tokenizer = ClinicalSentenceTokenizer()
        
        # Initialize stemmer
        self.stemmer = None
        if use_stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
            except ImportError:
                logger.warning("NLTK not available for stemming")
                self.use_stemming = False
    
    def normalize_text(self, text: str) -> str:
        """
        Enhanced text normalization.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Replace de-identified markers with special token
        if self.remove_deidentified:
            text = DEID_PATTERN.sub(DEID_REPLACEMENT, text)
        
        # Remove dates
        if self.remove_dates:
            from .preprocessing import DATE_PATTERN
            text = DATE_PATTERN.sub(" ", text)
        
        # Remove times
        if self.remove_times:
            from .preprocessing import TIME_PATTERN
            text = TIME_PATTERN.sub(" ", text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Apply stemming if requested
        if self.use_stemming and self.stemmer:
            tokens = text.split()
            tokens = [self.stemmer.stem(t) for t in tokens]
            text = ' '.join(tokens)
        
        # Apply lemmatization if requested
        if self.use_lemmatization and self.nlp:
            doc = self.nlp(text)
            text = ' '.join([token.lemma_ for token in doc])
        
        # Filter by vocabulary if provided
        if self.vocabulary:
            tokens = text.split()
            tokens = [t if t in self.vocabulary else '[UNK]' for t in tokens]
            text = ' '.join(tokens)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def extract_header_list(self, sections: List[Section]) -> List[str]:
        """
        Extract headers from sections.
        
        Args:
            sections: List of Section objects
            
        Returns:
            List of header strings
        """
        headers = []
        
        for section in sections:
            if section.name not in ["FULL_DOCUMENT", "PREAMBLE"]:
                header = section.name
                
                # Filter if requested
                if self.filter_headers:
                    header_upper = header.upper().strip()
                    
                    # Check discard list
                    if header_upper in DISCARD_HEADERS:
                        continue
                    
                    # Check discard patterns
                    if any(re.match(pat, header, re.IGNORECASE) for pat in DISCARD_HEADER_PATTERNS):
                        continue
                
                headers.append(header)
        
        return headers
    
    def tokenize_with_scispacy(self, text: str) -> List[str]:
        """
        Tokenize using scispacy.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not self.nlp:
            return text.split()
        
        doc = self.nlp(text)
        return [token.text for token in doc]
    
    def process_document(
        self,
        text: str,
        hadm_id: int,
        tokenize: bool = True,
    ) -> EnhancedProcessedDocument:
        """
        Process a clinical document with enhanced features.
        
        Args:
            text: Raw discharge summary text
            hadm_id: Hospital admission ID
            tokenize: Whether to tokenize
            
        Returns:
            EnhancedProcessedDocument with additional features
        """
        # Parse sections (use parent method)
        sections = self.parse_sections(text)
        
        # Extract headers if requested
        headers = []
        if self.extract_headers:
            headers = self.extract_header_list(sections)
        
        # Get clinical sentences if requested
        clinical_sentences = []
        if self.use_clinical_tokenizer and self.clinical_tokenizer:
            clinical_sentences = self.clinical_tokenizer.tokenize(text)
        
        # Build processed text
        if self.preserve_sections:
            processed_parts = []
            for section in sections:
                normalized = self.normalize_text(section.text)
                if section.name != "FULL_DOCUMENT" and section.name != "PREAMBLE":
                    # Add section marker
                    if self.lowercase:
                        marker = f"[{section.name.lower()}]"
                    else:
                        marker = f"[{section.name}]"
                    processed_parts.append(f"{marker} {normalized}")
                else:
                    processed_parts.append(normalized)
            processed_text = " ".join(processed_parts)
        else:
            # Just concatenate
            all_text = " ".join(s.text for s in sections)
            processed_text = self.normalize_text(all_text)
        
        # Create result
        result = EnhancedProcessedDocument(
            hadm_id=hadm_id,
            original_text=text,
            processed_text=processed_text,
            sections=sections,
            headers=headers,
            clinical_sentences=clinical_sentences,
        )
        
        # Tokenize with transformer tokenizer
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
            
            # Compute section spans
            result.section_spans = self._compute_section_spans(processed_text, sections)
        
        # Get vocabulary tokens if using scispacy
        if self.use_scispacy and self.nlp:
            result.vocabulary_tokens = self.tokenize_with_scispacy(processed_text)
        
        return result


def build_vocabulary_from_documents(
    documents: List[EnhancedProcessedDocument],
    top_k: int = 10000,
    min_frequency: int = 2,
) -> Tuple[Set[str], Dict[str, int]]:
    """
    Build vocabulary from processed documents.
    
    Args:
        documents: List of processed documents
        top_k: Number of top words to keep
        min_frequency: Minimum word frequency
        
    Returns:
        Tuple of (vocabulary_set, word_frequencies)
    """
    from collections import Counter
    
    word_counts = Counter()
    
    for doc in documents:
        if doc.vocabulary_tokens:
            word_counts.update(doc.vocabulary_tokens)
        else:
            tokens = doc.processed_text.split()
            word_counts.update(tokens)
    
    # Filter by frequency
    filtered = {
        word: count for word, count in word_counts.items()
        if count >= min_frequency
    }
    
    # Get top-k
    top_words = dict(Counter(filtered).most_common(top_k))
    vocabulary = set(top_words.keys())
    
    logger.info(f"Built vocabulary: {len(vocabulary)} words from {len(word_counts)} total")
    logger.info(f"Coverage: {sum(top_words.values()) / sum(word_counts.values()):.2%}")
    
    return vocabulary, top_words


def get_header_statistics(documents: List[EnhancedProcessedDocument]) -> pd.DataFrame:
    """
    Compute statistics about extracted headers.
    
    Args:
        documents: List of processed documents
        
    Returns:
        DataFrame with header statistics
    """
    from collections import Counter
    
    all_headers = []
    for doc in documents:
        all_headers.extend(doc.headers)
    
    header_counts = Counter(all_headers)
    
    stats = []
    for header, count in header_counts.most_common():
        stats.append({
            'header': header,
            'count': count,
            'frequency': count / len(documents),
        })
    
    return pd.DataFrame(stats)


def test_enhanced_preprocessor():
    """Test the enhanced preprocessor."""
    
    sample_text = """Admission Date: 2023-01-15 Discharge Date: 2023-01-20
Date of Birth: [**1960-3-12**] Sex: M

CHIEF COMPLAINT: Chest pain

HISTORY OF PRESENT ILLNESS:
The patient is a 63-year-old male who presented to the emergency department
with acute onset chest pain radiating to the left arm. He has a history of
hypertension and type 2 diabetes mellitus.

PAST MEDICAL HISTORY:
1. Hypertension
2. Type 2 Diabetes Mellitus
3. Hyperlipidemia

DISCHARGE DIAGNOSIS: Acute myocardial infarction
"""
    
    print("Enhanced Clinical Preprocessor Test")
    print("=" * 70)
    
    # Test without scispacy
    preprocessor = EnhancedClinicalPreprocessor(
        max_length=512,
        use_scispacy=False,
        use_clinical_tokenizer=True,
        extract_headers=True,
    )
    
    doc = preprocessor.process_document(text=sample_text, hadm_id=12345, tokenize=True)
    
    print(f"Processed document ID: {doc.hadm_id}")
    print(f"Number of sections: {len(doc.sections)}")
    print(f"Number of headers: {len(doc.headers)}")
    print(f"Number of clinical sentences: {len(doc.clinical_sentences)}")
    print(f"Number of tokens: {len(doc.token_ids)}")
    print()
    
    print("Extracted headers:")
    for h in doc.headers:
        print(f"  - {h}")
    print()
    
    print("Processed text (first 200 chars):")
    print(f"  {doc.processed_text[:200]}...")
    print()
    
    print("=" * 70)
    
    # Test with scispacy if available
    if SCISPACY_AVAILABLE:
        print("\nTesting with scispacy...")
        try:
            preprocessor_sci = EnhancedClinicalPreprocessor(
                max_length=512,
                use_scispacy=True,
                use_clinical_tokenizer=True,
                extract_headers=True,
            )
            
            doc_sci = preprocessor_sci.process_document(text=sample_text, hadm_id=12345, tokenize=True)
            
            print(f"scispacy vocabulary tokens: {len(doc_sci.vocabulary_tokens) if doc_sci.vocabulary_tokens else 0}")
            if doc_sci.vocabulary_tokens:
                print(f"First 10 tokens: {doc_sci.vocabulary_tokens[:10]}")
        except Exception as e:
            print(f"scispacy test failed: {e}")


if __name__ == "__main__":
    test_enhanced_preprocessor()

