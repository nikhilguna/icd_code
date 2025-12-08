#!/bin/bash
#SBATCH --job-name=led_mimic3_full
#SBATCH --account=stats_dept1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/led_mimic3_full_%j.log
#SBATCH --error=logs/led_mimic3_full_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikhilvg@umich.edu

# Print job info
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

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade requirements
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints/led_mimic3_full
mkdir -p results/led_mimic3_full

# Set environment variables for optimal performance
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Print GPU info
echo ""
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="
echo ""

# Run training
echo "Starting LED training on full MIMIC-III dataset..."
echo "Configuration:"
echo "  - Dataset: mimic3_full.parquet"
echo "  - Epochs: 5"
echo "  - Batch size: 8"
echo "  - Gradient accumulation: 2"
echo "  - Top-K codes: 50"
echo "  - Max length: 4096"
echo "  - Freeze layers: 9"
echo ""

python scripts/train_led.py \
    --data data/processed/mimic3_full.parquet \
    --output-dir checkpoints/led_mimic3_full \
    --epochs 5 \
    --batch-size 8 \
    --gradient-accumulation 2 \
    --top-k-codes 50 \
    --max-length 4096 \
    --freeze-layers 9 \
    --device cuda

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo ""
    
    # Run evaluation on test set
    echo "Running evaluation on test set..."
    python scripts/evaluate.py \
        --model led \
        --checkpoint checkpoints/led_mimic3_full/best_model.pt \
        --data data/processed/mimic3_full.parquet \
        --output-dir results/led_mimic3_full \
        --device cuda \
        --num-workers 4 \
        --batch-size 16
    
    EVAL_EXIT_CODE=$?
    
    if [ $EVAL_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Evaluation completed successfully!"
        echo "Results saved to: results/led_mimic3_full/"
        echo "=========================================="
        echo ""
        
        # Print results summary
        if [ -f "results/led_mimic3_full/results.json" ]; then
            echo "Results Summary:"
            cat results/led_mimic3_full/results.json
        fi
    else
        echo "Evaluation failed with exit code: $EVAL_EXIT_CODE"
    fi
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
fi

echo ""
echo "=========================================="
echo "Job End Time: $(date)"
echo "=========================================="

# Deactivate virtual environment
deactivate

