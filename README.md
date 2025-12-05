# ICD Code Prediction from Clinical Documents

Multi-label ICD code prediction from ICU discharge summaries using CAML and Longformer architectures.

## Project Overview

This project implements automatic ICD code prediction from MIMIC-III/IV clinical discharge summaries, framing it as a multi-label text classification task. We compare:

- **CAML**: Convolutional Attention for Multi-Label classification (CNN + per-label attention)
- **LED**: Longformer Encoder-Decoder for long document understanding (up to 16k tokens)

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


### AWS Configuration

This project uses Amazon Athena to query MIMIC data. Configure your AWS credentials:

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region
```

Ensure you have access to the PhysioNet MIMIC data on AWS. See `MIMIC_ACCESS_GUIDE.md` for details.

## Project Structure

```
icd/
├── data/                    # Data extraction and preprocessing
│   ├── athena_extraction.py # SQL queries for MIMIC via Athena
│   ├── preprocessing.py     # Text cleaning, section parsing
│   ├── dataset.py           # PyTorch Dataset classes
│   └── label_encoder.py     # Multi-label binarization
├── models/                  # Model architectures
│   ├── caml.py              # CAML implementation
│   └── led_classifier.py    # LED classifier
├── training/                # Training infrastructure
│   ├── trainer.py           # Training loop
│   └── losses.py            # Loss functions
├── evaluation/              # Evaluation metrics
│   ├── metrics.py           # F1, P@k, ROC-AUC
│   └── cross_dataset.py     # Cross-dataset evaluation
├── interpretability/        # Interpretability analysis
│   ├── attention_analysis.py
│   ├── integrated_gradients.py
│   └── heatmaps.py
├── utils/                   # Utilities
│   └── config.py            # Configuration management
├── configs/                 # Experiment configs
│   └── default.yaml
└── scripts/                 # Executable scripts
    ├── extract_data.py
    ├── train_caml.py
    ├── train_led.py
    └── evaluate.py
```

## Usage

### 1. Extract Data from MIMIC

```bash
python scripts/extract_data.py --dataset mimic3 --output data/raw/
```

### 2. Train CAML Model

```bash
python scripts/train_caml.py --config configs/default.yaml
```

### 3. Train LED Model

```bash
python scripts/train_led.py --config configs/default.yaml
```

### 4. Evaluate Models

```bash
python scripts/evaluate.py --model caml --checkpoint checkpoints/caml_best.pt
```

## Evaluation Metrics

- **Micro-F1 / Macro-F1**: Overall and per-class F1 scores
- **Precision@k**: Precision at top-k predictions (k=5, 10)
- **ROC-AUC**: Per-label AUC for top-50 frequent codes
- **Stratified Analysis**: Performance breakdown by label frequency (head/medium/tail)

## Interpretability

- Token-level attention heatmaps
- Integrated Gradients attribution
- Section-level correlation analysis
- Highlight Overlap@k metrics

## References

- Mullenbach et al. "Explainable Prediction of Medical Codes from Clinical Text" (CAML)
- Beltagy et al. "Longformer: The Long-Document Transformer"
- Johnson et al. "MIMIC-III, a freely accessible critical care database"

## Authors

Nikhil Gunaratnam, Pranav Rakasi  
University of Michigan - CS595 NLP Fall 2025
