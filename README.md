# ICD Code Prediction from Clinical Documents

Multi-label ICD code prediction from ICU discharge summaries using CAML and Longformer architectures.

## Project Overview

This project implements automatic ICD code prediction from MIMIC-III clinical discharge summaries, framing it as a multi-label text classification task. We compare:

- **CAML**: Convolutional Attention for Multi-Label classification (CNN + per-label attention)
- **Longformer**: Long-document transformer for processing long clinical documents (up to 4k tokens)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Getting Started

### Data Preparation

This project uses local MIMIC-III data files. You need PhysioNet credentialed access to MIMIC:

1. **Apply for MIMIC Access** (see `MIMIC_ACCESS_GUIDE.md`)
2. **Download MIMIC-III Data** to `MIMIC_DATA/MIMIC-III/` directory
3. **Process Data** into training format

```bash
# Process MIMIC-III data (full dataset)
python scripts/process_mimic3_local.py --output data/processed/mimic3_full.parquet

# Or process a smaller sample for testing
python scripts/process_mimic3_local.py --output data/processed/mimic3_test.parquet --limit 5000
```

## Project Structure

```
icd/
├── data/                    # Data extraction and preprocessing
│   ├── preprocessing.py     # Text cleaning, section parsing
│   ├── enhanced_preprocessing.py  # Advanced clinical NLP
│   ├── clinical_tokenizer.py      # Clinical sentence tokenization
│   ├── hierarchical_encoder.py    # Hierarchical ICD encoding
│   ├── dataset.py           # PyTorch Dataset classes
│   ├── label_encoder.py     # Multi-label binarization
│   └── athena_extraction.py # Data loading utilities (local & AWS)
├── models/                  # Model architectures
│   ├── caml.py              # CAML implementation
│   └── longformer_classifier.py    # Longformer classifier
├── training/                # Training infrastructure
│   ├── trainer.py           # Training loop
│   └── losses.py            # Loss functions
├── evaluation/              # Evaluation metrics
│   └── metrics.py           # F1, P@k, ROC-AUC
├── utils/                   # Utilities
│   └── config.py            # Configuration management
├── configs/                 # Experiment configs
│   └── default.yaml
├── scripts/                 # Executable scripts
│   ├── process_mimic3_local.py  # Process local MIMIC-III data
│   ├── train_caml.py
│   ├── train_longformer.py
│   └── evaluate.py
├── MIMIC_DATA/             # Local MIMIC data (git-ignored)
│   └── MIMIC-III/
│       ├── NOTEEVENTS.csv.gz
│       ├── DIAGNOSES_ICD.csv.gz
│       └── ...
└── data/processed/          # Processed parquet files
    ├── mimic3_full.parquet
    └── mimic3_test.parquet
```

## Usage

### 1. Process Local MIMIC Data

```bash
# Process full MIMIC-III dataset
python scripts/process_mimic3_local.py \
    --output data/processed/mimic3_full.parquet

# Process test subset (5000 samples)
python scripts/process_mimic3_local.py \
    --output data/processed/mimic3_test.parquet \
    --limit 5000
```

### 2. Train CAML Model

```bash
python scripts/train_caml.py \
    --data data/processed/mimic3_full.parquet \
    --output-dir checkpoints/caml_mimic3 \
    --epochs 50 \
    --batch-size 32 \
    --top-k-codes 50 \
    --device cuda
```

### 3. Train Longformer Model

```bash
python scripts/train_longformer.py \
    --data data/processed/mimic3_full.parquet \
    --output-dir checkpoints/longformer_mimic3 \
    --epochs 10 \
    --batch-size 4 \
    --gradient-accumulation 8 \
    --top-k-codes 50 \
    --device cuda
```

### 4. Evaluate Models

```bash
python scripts/evaluate.py \
    --model caml \
    --checkpoint checkpoints/caml_mimic3/best_model.pt \
    --data data/processed/mimic3_full.parquet \
    --output-dir results/caml_mimic3
```

## Evaluation Metrics

- **Micro-F1 / Macro-F1**: Overall and per-class F1 scores
- **Precision@k**: Precision at top-k predictions (k=5, 10)
- **ROC-AUC**: Per-label AUC for top-50 frequent codes
- **Stratified Analysis**: Performance breakdown by label frequency (head/medium/tail)

## References

- Mullenbach et al. "Explainable Prediction of Medical Codes from Clinical Text" (CAML)
- Beltagy et al. "Longformer: The Long-Document Transformer"
- Johnson et al. "MIMIC-III, a freely accessible critical care database"

## Authors

Nikhil Gunaratnam, Pranav Rakasi  
University of Michigan - CS595 NLP Fall 2025
