#!/bin/bash
# Quick comparison of different DeCoMa configurations

echo "========================================================================"
echo "DeCoMa Configuration Comparison"
echo "========================================================================"
echo ""

echo ">>> Configuration 1: Current Optimal (z=4.0, type=col, min=3.0)"
echo "------------------------------------------------------------------------"
python3 run_decoma_smol.py \
    --z_score_threshold 4.0 \
    --detect_type col \
    --minimum_scale 3.0 \
    2>&1 | grep -A 15 "DETECTION RESULTS"

echo ""
echo ""
echo ">>> Configuration 2: Previous Default (z=4.0, type=col, min=2.0)"
echo "------------------------------------------------------------------------"
python3 run_decoma_smol.py \
    --z_score_threshold 4.0 \
    --detect_type col \
    --minimum_scale 2.0 \
    2>&1 | grep -A 15 "DETECTION RESULTS"

echo ""
echo ""
echo ">>> Configuration 3: Original Paper (z=3.0, type=both, min=0.0004)"
echo "------------------------------------------------------------------------"
python3 run_decoma_smol.py \
    --z_score_threshold 3.0 \
    --detect_type both \
    --minimum_scale 0.0004 \
    2>&1 | grep -A 15 "DETECTION RESULTS"
