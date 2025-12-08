#!/bin/bash
#SBATCH --job-name=led_mimic3_quick
#SBATCH --account=stats_dept1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/led_mimic3_quick_%j.log
#SBATCH --error=logs/led_mimic3_quick_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikhilvg@umich.edu

# Quick training script for faster turnaround (3 epochs, smaller batch)
# Good for testing before running the full 5-epoch training

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Load modules
module load python/3.11
module load cuda/12.1

# Navigate to project directory
cd $HOME/icd_project

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

# Install requirements
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Create directories
mkdir -p logs checkpoints/led_mimic3_quick results/led_mimic3_quick

# Set environment variables
export TOKENIZERS_PARALLELISM=false

# Print GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "Starting QUICK LED training (3 epochs)..."
echo ""

# Run quick training
python scripts/train_led.py \
    --data data/processed/mimic3_full.parquet \
    --output-dir checkpoints/led_mimic3_quick \
    --epochs 3 \
    --batch-size 8 \
    --gradient-accumulation 2 \
    --top-k-codes 50 \
    --max-length 4096 \
    --freeze-layers 9 \
    --device cuda

if [ $? -eq 0 ]; then
    echo "Training successful! Running evaluation..."
    
    python scripts/evaluate.py \
        --model led \
        --checkpoint checkpoints/led_mimic3_quick/best_model.pt \
        --data data/processed/mimic3_full.parquet \
        --output-dir results/led_mimic3_quick \
        --device cuda \
        --num-workers 4 \
        --batch-size 16
    
    if [ -f "results/led_mimic3_quick/results.json" ]; then
        echo ""
        echo "Results:"
        cat results/led_mimic3_quick/results.json
    fi
fi

echo ""
echo "Job End Time: $(date)"

deactivate

