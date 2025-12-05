#!/usr/bin/env python
"""
Test script for new preprocessing improvements.

Tests:
1. Clinical sentence tokenizer
2. Hierarchical ICD encoder
3. Enhanced preprocessor
4. Integration test

Run with: python scripts/test_improvements.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Sample clinical text
SAMPLE_TEXT = """Admission Date: [**2023-1-15**] Discharge Date: [**2023-1-20**]
Date of Birth: [**1960-3-12**] Sex: M

CHIEF COMPLAINT:
Chest pain

HISTORY OF PRESENT ILLNESS:
This is a 63-year-old male with history of hypertension, type 2 diabetes 
mellitus, and hyperlipidemia who presented to the emergency department with 
acute onset substernal chest pain radiating to the left arm.

PAST MEDICAL HISTORY:
1. Hypertension - well controlled on medications
2. Type 2 Diabetes Mellitus - HgbA1c 7.2%
3. Hyperlipidemia - on statin therapy

PHYSICAL EXAM:
Vitals: BP 145/90 HR 88 RR 16 T 98.6F
General: Alert and oriented x3, in mild distress
HEENT: Normocephalic, atraumatic
Neck: No JVD
CV: Regular rate and rhythm, no murmurs
Lungs: Clear to auscultation bilaterally
Abd: Soft, non-tender, non-distended

PERTINENT RESULTS:
Troponin: 2.5 (elevated)
CK-MB: 45 (elevated)
EKG: ST elevations in leads II, III, aVF

DISCHARGE DIAGNOSIS:
1. Acute inferior myocardial infarction
2. Hypertension
3. Type 2 Diabetes Mellitus

DISCHARGE MEDICATIONS:
1. Aspirin 81mg daily
2. Clopidogrel 75mg daily
3. Metoprolol 25mg twice daily
4. Lisinopril 10mg daily
5. Atorvastatin 80mg nightly
"""

# Sample ICD codes
SAMPLE_ICD_CODES = [
    ["410.71", "401.9", "250.00", "272.4"],  # AMI, HTN, DM, HLD
    ["480.9", "496", "491.21"],              # Pneumonia, COPD
    ["585.9", "403.90", "584.9"],            # CKD, renal HTN, AKI
    ["V45.81", "V58.61"],                    # Post-surgical status
]


def test_clinical_tokenizer():
    """Test 1: Clinical Sentence Tokenizer"""
    print("\n" + "=" * 70)
    print("TEST 1: Clinical Sentence Tokenizer")
    print("=" * 70)
    
    try:
        from data.clinical_tokenizer import ClinicalSentenceTokenizer
        
        tokenizer = ClinicalSentenceTokenizer()
        
        # Get segments
        segments = tokenizer.tokenize(SAMPLE_TEXT)
        print(f"✓ Found {len(segments)} segments")
        
        # Show first few
        print("\nFirst 5 segments:")
        for i, seg in enumerate(segments[:5], 1):
            preview = seg[:60].replace('\n', ' ')
            print(f"  {i}. {preview}{'...' if len(seg) > 60 else ''}")
        
        # Get sentences
        sentences = tokenizer.tokenize_sentences(SAMPLE_TEXT)
        print(f"\n✓ Found {len(sentences)} sentences")
        
        print("✅ Clinical tokenizer test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Clinical tokenizer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hierarchical_encoder():
    """Test 2: Hierarchical ICD Encoder"""
    print("\n" + "=" * 70)
    print("TEST 2: Hierarchical ICD Encoder")
    print("=" * 70)
    
    try:
        from data.hierarchical_encoder import HierarchicalICDEncoder
        
        # Create encoder
        encoder = HierarchicalICDEncoder(top_k_fine=None, min_frequency=1)
        encoder.fit(SAMPLE_ICD_CODES)
        
        print(f"✓ Fitted encoder on {len(SAMPLE_ICD_CODES)} samples")
        print(f"  Fine-grained codes: {encoder.num_fine_labels}")
        print(f"  Coarse-grained categories: {encoder.num_coarse_labels}")
        
        # Test parent mapping
        print("\nParent category mapping:")
        test_codes = ["410.71", "250.00", "480.9", "V45.81"]
        for code in test_codes:
            cat = encoder.get_parent_category(code)
            if cat:
                desc = encoder.get_category_description(cat)
                print(f"  {code:10s} → {cat:20s} ({desc[:40]}...)")
        
        # Transform
        fine_labels, coarse_labels = encoder.transform_hierarchical(SAMPLE_ICD_CODES)
        print(f"\n✓ Transformed to labels")
        print(f"  Fine shape: {fine_labels.shape}")
        print(f"  Coarse shape: {coarse_labels.shape}")
        
        # Show first sample
        print(f"\nSample 1: {SAMPLE_ICD_CODES[0]}")
        print(f"  Fine labels: {fine_labels[0].sum():.0f} active")
        print(f"  Coarse labels: {coarse_labels[0].sum():.0f} active")
        categories = encoder.inverse_transform_coarse(coarse_labels[:1])[0]
        print(f"  Categories: {categories}")
        
        print("✅ Hierarchical encoder test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Hierarchical encoder test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_preprocessor():
    """Test 3: Enhanced Preprocessor"""
    print("\n" + "=" * 70)
    print("TEST 3: Enhanced Preprocessor")
    print("=" * 70)
    
    try:
        from data.enhanced_preprocessing import EnhancedClinicalPreprocessor
        
        # Test without scispacy first (faster, more reliable)
        print("Testing without scispacy...")
        preprocessor = EnhancedClinicalPreprocessor(
            tokenizer_name="allenai/longformer-base-4096",
            max_length=512,
            use_scispacy=False,
            use_clinical_tokenizer=True,
            extract_headers=True,
        )
        
        doc = preprocessor.process_document(SAMPLE_TEXT, hadm_id=12345, tokenize=True)
        
        print(f"✓ Processed document {doc.hadm_id}")
        print(f"  Sections: {len(doc.sections)}")
        print(f"  Headers: {len(doc.headers)}")
        print(f"  Clinical sentences: {len(doc.clinical_sentences)}")
        print(f"  Tokens: {len(doc.token_ids)}")
        
        print("\nExtracted headers:")
        for h in doc.headers[:5]:
            print(f"  - {h}")
        
        print(f"\nProcessed text (first 150 chars):")
        print(f"  {doc.processed_text[:150]}...")
        
        # Try with scispacy if available
        print("\nTrying with scispacy...")
        try:
            preprocessor_sci = EnhancedClinicalPreprocessor(
                tokenizer_name="allenai/longformer-base-4096",
                max_length=512,
                use_scispacy=True,
                use_clinical_tokenizer=True,
                extract_headers=True,
            )
            
            doc_sci = preprocessor_sci.process_document(SAMPLE_TEXT, hadm_id=12345, tokenize=True)
            
            print(f"✓ Processed with scispacy")
            if doc_sci.vocabulary_tokens:
                print(f"  scispacy tokens: {len(doc_sci.vocabulary_tokens)}")
                print(f"  First 10: {doc_sci.vocabulary_tokens[:10]}")
        except Exception as e:
            print(f"⚠ scispacy test skipped: {e}")
            print("  (This is OK - scispacy is optional)")
        
        print("✅ Enhanced preprocessor test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced preprocessor test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test 4: Full Integration"""
    print("\n" + "=" * 70)
    print("TEST 4: Full Integration Test")
    print("=" * 70)
    
    try:
        from data.enhanced_preprocessing import EnhancedClinicalPreprocessor
        from data.hierarchical_encoder import HierarchicalICDEncoder
        
        # Initialize
        preprocessor = EnhancedClinicalPreprocessor(
            use_scispacy=False,
            use_clinical_tokenizer=True,
            extract_headers=True,
        )
        
        encoder = HierarchicalICDEncoder(top_k_fine=None)
        encoder.fit(SAMPLE_ICD_CODES)
        
        # Process multiple documents
        texts = [SAMPLE_TEXT] * 4
        hadm_ids = [12345, 12346, 12347, 12348]
        
        documents = []
        for text, hadm_id in zip(texts, hadm_ids):
            doc = preprocessor.process_document(text, hadm_id, tokenize=True)
            documents.append(doc)
        
        # Get labels
        fine_labels, coarse_labels = encoder.transform_hierarchical(SAMPLE_ICD_CODES)
        
        print(f"✓ Processed {len(documents)} documents")
        print(f"✓ Generated {fine_labels.shape[0]} label sets")
        print(f"  Fine-grained: {fine_labels.shape}")
        print(f"  Coarse-grained: {coarse_labels.shape}")
        
        # Verify shapes match
        assert len(documents) == fine_labels.shape[0]
        assert fine_labels.shape[0] == coarse_labels.shape[0]
        
        print("\n✓ All shapes match")
        print("✓ Ready for model training")
        
        print("✅ Integration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" TESTING NEW PREPROCESSING IMPROVEMENTS")
    print("=" * 70)
    
    results = {
        "Clinical Tokenizer": test_clinical_tokenizer(),
        "Hierarchical Encoder": test_hierarchical_encoder(),
        "Enhanced Preprocessor": test_enhanced_preprocessor(),
        "Integration": test_integration(),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:30s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to use improvements.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit(main())

