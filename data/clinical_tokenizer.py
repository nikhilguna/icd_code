"""
Clinical sentence tokenizer with heuristic rules for medical text.

Based on the approach from the reference project, adapted for our pipeline.
Handles clinical-specific formatting like:
- Section headers with colons
- Enumerated lists (1. 2. 3.)
- Separator lines (---, ___)
- Inline titles (BP:, HR:, etc.)
- De-identification markers
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ClinicalSentenceTokenizer:
    """
    Rule-based sentence tokenizer optimized for clinical discharge summaries.
    
    Handles the unique formatting patterns found in MIMIC clinical notes.
    
    Usage:
        tokenizer = ClinicalSentenceTokenizer()
        sentences = tokenizer.tokenize(discharge_text)
    """
    
    # Common stopwords that can appear in titles
    TITLE_STOPWORDS = {'of', 'on', 'or', 'and', 'the', 'a', 'an'}
    
    # Non-title patterns (common false positives)
    NON_TITLE_PATTERNS = [
        'Disp',  # Medication dispense line
        'Sig',   # Medication signature
    ]
    
    def __init__(self):
        """Initialize tokenizer with clinical patterns."""
        pass
    
    @staticmethod
    def strip(s: str) -> str:
        """Strip whitespace."""
        return s.strip()
    
    def is_title(self, text: str) -> bool:
        """
        Check if text is a section header (ends with colon, capitalized).
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a section title
        """
        if not text.endswith(':'):
            return False
        
        # Remove colon for analysis
        text = text[:-1]
        
        # Remove parenthetical content
        text = re.sub(r'\([^\)]*?\)', '', text)
        
        # Check if all non-stopword tokens are capitalized
        words = text.split()
        for word in words:
            if word.lower() in self.TITLE_STOPWORDS:
                continue
            if not word or not word[0].isupper():
                return False
        
        # Check for common non-title patterns
        if text in self.NON_TITLE_PATTERNS:
            return False
        
        return True
    
    def is_inline_title(self, text: str) -> bool:
        """
        Check if line starts with inline title (e.g., "Vitals: BP 120/80").
        
        Args:
            text: Text to check
            
        Returns:
            True if text starts with inline title
        """
        m = re.search(r'^([a-zA-Z ]+:) ', text)
        if not m:
            return False
        return self.is_title(m.groups()[0])
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize clinical text into sentence segments.
        
        Args:
            text: Raw clinical discharge summary text
            
        Returns:
            List of sentence/segment strings
        """
        # Step 1: Normalize long separator lines
        text = re.sub(r'---+', '\n\n-----\n\n', text)
        text = re.sub(r'___+', '\n\n_____\n\n', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Split on double newlines
        segments = text.split('\n\n')
        
        # Step 2: Separate section headers
        new_segments = []
        
        # Handle common header patterns at document start
        if len(segments) > 0:
            # "Admission Date: ... Discharge Date: ..."
            m1 = re.match(r'(Admission Date:) (.*) (Discharge Date:) (.*)', segments[0])
            if m1:
                new_segments += list(map(self.strip, m1.groups()))
                segments = segments[1:]
            
            # "Date of Birth: ... Sex: ..."
            if len(segments) > 0:
                m2 = re.match(r'(Date of Birth:) (.*) (Sex:) (.*)', segments[0])
                if m2:
                    new_segments += list(map(self.strip, m2.groups()))
                    segments = segments[1:]
        
        # Process each segment to extract headers
        for segment in segments:
            # Find all potential section headers
            possible_headers = re.findall(r'\n([A-Z][^\n:]+:)', '\n' + segment)
            headers = []
            for h in possible_headers:
                if self.is_title(h.strip()):
                    headers.append(h.strip())
            
            # Split segment by headers
            for h in headers:
                h = h.strip()
                
                # Find header in segment
                if h not in segment:
                    continue
                
                ind = segment.index(h)
                prefix = segment[:ind].strip()
                rest = segment[ind + len(h):].strip()
                
                # Add prefix if not empty
                if len(prefix) > 0:
                    new_segments.append(prefix)
                
                # Add header
                new_segments.append(h)
                
                # Continue with rest
                segment = rest
            
            # Add remaining segment
            if len(segment) > 0:
                new_segments.append(segment.strip())
        
        segments = list(new_segments)
        new_segments = []
        
        # Step 3: Split on underscore separators
        for segment in segments:
            subsections = segment.split('\n_____\n')
            new_segments.append(subsections[0])
            for ss in subsections[1:]:
                new_segments.append('_____')
                if ss:
                    new_segments.append(ss)
        
        segments = list(new_segments)
        new_segments = []
        
        # Step 4: Split on dash separators
        for segment in segments:
            subsections = segment.split('\n-----\n')
            new_segments.append(subsections[0])
            for ss in subsections[1:]:
                new_segments.append('-----')
                if ss:
                    new_segments.append(ss)
        
        segments = list(new_segments)
        new_segments = []
        
        # Step 5: Separate enumerated lists
        for segment in segments:
            # Check if segment contains numbered list
            if not re.search(r'\n\s*\d+\.', '\n' + segment):
                new_segments.append(segment)
                continue
            
            # Generalize for lists starting mid-segment
            segment = '\n' + segment
            
            # Find list start
            match = re.search(r'\n\s*(\d+)\.', '\n' + segment)
            if not match:
                new_segments.append(segment.strip())
                continue
            
            start = int(match.groups()[0])
            n = start
            while re.search(r'\n\s*%d\.' % n, segment):
                n += 1
            n -= 1
            
            # No valid list
            if n < 1:
                new_segments.append(segment.strip())
                continue
            
            # Split each list item
            for i in range(start, n + 1):
                match = re.search(r'(\n\s*\d+\.)', segment)
                if match:
                    matching_text = match.groups()[0]
                    prefix = segment[:segment.index(matching_text)].strip()
                    segment = segment[segment.index(matching_text):].strip()
                    if len(prefix) > 0:
                        new_segments.append(prefix)
            
            if len(segment) > 0:
                new_segments.append(segment.strip())
        
        segments = list(new_segments)
        new_segments = []
        
        # Step 6: Remove inline titles from larger segments
        for segment in segments:
            lines = segment.split('\n')
            buf = []
            
            for line in lines:
                if self.is_inline_title(line):
                    if len(buf) > 0:
                        new_segments.append('\n'.join(buf))
                    buf = []
                buf.append(line)
            
            if len(buf) > 0:
                new_segments.append('\n'.join(buf))
        
        segments = list(new_segments)
        new_segments = []
        
        # Step 7: Merge one-liner answers with their header
        N = len(segments)
        for i in range(N):
            if i == 0:
                new_segments.append(segments[i])
                continue
            
            # If current segment is one line, previous is a title, and current is not a title
            if (segments[i].count('\n') == 0 and
                self.is_title(segments[i - 1]) and
                not self.is_title(segments[i])):
                
                # If next segment is a title or this is the last segment
                if (i == N - 1) or (i < N - 1 and self.is_title(segments[i + 1])):
                    # Merge with previous
                    new_segments = new_segments[:-1]
                    new_segments.append(segments[i - 1] + ' ' + segments[i])
                else:
                    new_segments.append(segments[i])
            else:
                new_segments.append(segments[i])
        
        # Filter out empty segments
        final_segments = [s for s in new_segments if s.strip()]
        
        return final_segments
    
    def tokenize_sentences(self, text: str, use_nltk_fallback: bool = True) -> List[str]:
        """
        Tokenize into actual sentences (not just segments).
        
        Args:
            text: Input text
            use_nltk_fallback: Use NLTK sentence tokenizer on prose segments
            
        Returns:
            List of sentences
        """
        # First get clinical segments
        segments = self.tokenize(text)
        
        sentences = []
        
        if use_nltk_fallback:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        
        for segment in segments:
            # If segment is a title, keep as-is
            if self.is_title(segment):
                sentences.append(segment)
            # If segment is a separator, skip
            elif segment in ['-----', '_____']:
                continue
            # If segment contains multiple lines but no clear structure, use NLTK
            elif use_nltk_fallback and len(segment) > 100:
                try:
                    sub_sentences = nltk.sent_tokenize(segment)
                    sentences.extend(sub_sentences)
                except:
                    sentences.append(segment)
            else:
                sentences.append(segment)
        
        return sentences


def test_clinical_tokenizer():
    """Test the clinical tokenizer on sample text."""
    sample_text = """Admission Date: 2023-01-15 Discharge Date: 2023-01-20
Date of Birth: 1960-03-12 Sex: M

CHIEF COMPLAINT: Chest pain

HISTORY OF PRESENT ILLNESS:
The patient is a 63-year-old male who presented to the emergency department
with acute onset chest pain radiating to the left arm.

PAST MEDICAL HISTORY:
1. Hypertension
2. Type 2 Diabetes
3. Hyperlipidemia

PHYSICAL EXAM:
Vitals: BP 145/90 HR 88 RR 16 T 98.6F
General: Alert and oriented, in mild distress
CV: Regular rate and rhythm

DISCHARGE DIAGNOSIS: Acute myocardial infarction

DISCHARGE MEDICATIONS:
1. Aspirin 81mg daily
2. Metoprolol 25mg twice daily
"""
    
    tokenizer = ClinicalSentenceTokenizer()
    segments = tokenizer.tokenize(sample_text)
    
    print("Clinical Sentence Tokenizer Test")
    print("=" * 50)
    print(f"Found {len(segments)} segments:\n")
    
    for i, seg in enumerate(segments, 1):
        print(f"{i}. [{len(seg)} chars] {seg[:60]}{'...' if len(seg) > 60 else ''}")
    
    print("\n" + "=" * 50)
    
    # Test sentence tokenization
    sentences = tokenizer.tokenize_sentences(sample_text)
    print(f"\nFull sentence tokenization: {len(sentences)} sentences")
    

if __name__ == "__main__":
    test_clinical_tokenizer()

