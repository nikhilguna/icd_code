#!/bin/bash
# Run all tests in sequence
# This script tests the entire ICD coding pipeline using mock data

set -e  # Exit on first error

echo "======================================================================"
echo "ICD CODING PROJECT - COMPLETE TEST SUITE"
echo "======================================================================"
echo ""
echo "This will test the entire pipeline using mock data."
echo "Total estimated time: 10-15 minutes"
echo ""
echo "Tests:"
echo "  1. Generate mock data"
echo "  2. Test preprocessing"
echo "  3. Test data pipeline"
echo "  4. Test CAML model"
echo "  5. Test LED model"
echo ""
read -p "Press Enter to continue..."

# Change to project root
cd "$(dirname "$0")/.."

echo ""
echo "======================================================================"
echo "PHASE 1: GENERATE MOCK DATA"
echo "======================================================================"
python scripts/generate_mock_data.py --num-samples 100 --output mock_data/raw/

echo ""
echo "======================================================================"
echo "PHASE 2: TEST PREPROCESSING"
echo "======================================================================"
python scripts/test_preprocessing.py --data-dir mock_data/raw/ --num-samples 10

echo ""
echo "======================================================================"
echo "PHASE 3: TEST DATA PIPELINE"
echo "======================================================================"
python scripts/test_data_pipeline.py --data-dir mock_data/raw/ --top-k 50

echo ""
echo "======================================================================"
echo "PHASE 4: TEST CAML MODEL"
echo "======================================================================"
python scripts/test_caml.py --num-labels 50

echo ""
echo "======================================================================"
echo "PHASE 5: TEST LED MODEL"
echo "======================================================================"
python scripts/test_led.py --num-labels 50 --max-length 512

echo ""
echo "======================================================================"
echo "ALL TESTS COMPLETE! ✅"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  ✅ Mock data generated"
echo "  ✅ Preprocessing validated"
echo "  ✅ Data pipeline working"
echo "  ✅ CAML model functional"
echo "  ✅ LED model functional"
echo ""
echo "Your ICD coding pipeline is ready!"
echo ""
echo "Next steps:"
echo "  1. Apply for MIMIC access (see MIMIC_ACCESS_GUIDE.md)"
echo "  2. Extract real MIMIC data"
echo "  3. Train models on real data"
echo ""
echo "For more details, see:"
echo "  - TESTING_PLAN.md (detailed testing strategy)"
echo "  - TESTING_QUICK_START.md (quick reference)"
echo ""

