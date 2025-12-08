#!/bin/bash

# Efficient transfer script - only sends necessary files to Great Lakes
# Excludes: venv, checkpoints, results, wandb, cache, etc.

set -e

echo "=========================================="
echo "Transferring ICD Project to Great Lakes"
echo "=========================================="
echo ""

# Configuration
REMOTE_USER="nikhilvg"
REMOTE_HOST="greatlakes-xfer.arc-ts.umich.edu"
REMOTE_PATH="icd_project"  # Don't use ~ in rsync, it's expanded automatically
LOCAL_PATH="."

echo "From: $(pwd)"
echo "To:   $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found"
    echo "Please run this script from the icd project directory"
    exit 1
fi

# Check if processed data exists
if [ ! -f "data/processed/mimic3_full.parquet" ]; then
    echo "⚠️  Warning: data/processed/mimic3_full.parquet not found"
    echo "You may need to process MIMIC data first"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Files to transfer:"
echo "  ✓ Source code (scripts/, models/, data/, training/)"
echo "  ✓ Processed MIMIC data (data/processed/)"
echo "  ✓ Configuration (requirements.txt)"
echo "  ✓ SLURM scripts (slurm_*.sh)"
echo ""
echo "⏭️  Excluding:"
echo "  ✗ Virtual environments (venv/)"
echo "  ✗ Checkpoints (checkpoints/)"
echo "  ✗ Results (results/)"
echo "  ✗ Wandb logs (wandb/)"
echo "  ✗ Cache files (__pycache__/)"
echo ""

# Start transfer
echo "🚀 Starting transfer..."
echo ""

rsync -avz --progress \
    --exclude-from='.transfer_exclude' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    -e ssh \
    "$LOCAL_PATH/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Transfer completed successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. SSH to Great Lakes:"
    echo "   ssh $REMOTE_USER@greatlakes.arc-ts.umich.edu"
    echo ""
    echo "2. Navigate and submit job:"
    echo "   cd ~/icd_project"
    echo "   mkdir -p logs"
    echo "   chmod +x slurm_train_led_quick.sh"
    echo "   sbatch slurm_train_led_quick.sh"
    echo ""
else
    echo ""
    echo "❌ Transfer failed"
    exit 1
fi

