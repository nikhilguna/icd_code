#!/usr/bin/env python3
"""
Evaluation script for ICD code prediction models.

Usage:
    # Evaluate CAML on test set
    python scripts/evaluate.py --model caml \
        --checkpoint checkpoints/caml/best_model.pt \
        --data data/processed/mimic3.parquet
    
    # Cross-dataset evaluation
    python scripts/evaluate.py --model caml \
        --checkpoint checkpoints/caml/best_model.pt \
        --source-data data/processed/mimic3.parquet \
        --target-data data/processed/mimic4.parquet \
        --cross-dataset
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import load_mimic_data, create_dataloaders
from data.label_encoder import ICDLabelEncoder
from models.caml import CAML
from models.led_classifier import LEDClassifier
from evaluation.metrics import ICDMetrics, per_label_metrics, find_optimal_threshold
from evaluation.cross_dataset import evaluate_model_cross_dataset, analyze_prediction_shift

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(model_type: str, checkpoint_path: str, num_labels: int, device: str):
    """Load model from checkpoint."""
    # Load checkpoint to CPU first to avoid MPS device mapping issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if model_type == "caml":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        
        model = CAML(
            vocab_size=tokenizer.vocab_size,
            num_labels=num_labels,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
    elif model_type == "led":
        model = LEDClassifier(
            num_labels=num_labels,
            model_name="allenai/longformer-base-4096",
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_single_dataset(
    model,
    dataloader,
    label_encoder,
    device: str,
    output_dir: Optional[str] = None,
):
    """Evaluate model on a single dataset."""
    metrics = ICDMetrics(
        label_encoder=label_encoder,
        compute_auc=True,
        compute_stratified=True,
    )
    
    all_logits = []
    all_labels = []
    
    logger.info(f"Evaluating on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            metrics.update(outputs["logits"], labels, is_logits=True)
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(labels)
    
    results = metrics.compute()
    logger.info(str(results))
    
    # Concatenate for additional analysis
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Find optimal threshold
    best_thresh, best_f1 = find_optimal_threshold(logits, labels)
    logger.info(f"Optimal threshold: {best_thresh} (F1: {best_f1:.4f})")
    
    # Per-label metrics
    label_metrics = per_label_metrics(logits, labels, label_encoder)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save overall results
        with open(output_dir / "results.json", "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        
        # Save per-label metrics
        df = pd.DataFrame(label_metrics)
        df.to_csv(output_dir / "per_label_metrics.csv", index=False)
        
        logger.info(f"Saved results to {output_dir}")
    
    return results, label_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICD prediction model")
    
    parser.add_argument("--model", choices=["caml", "led"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, help="Path to test data")
    parser.add_argument("--encoder", type=str, help="Path to label encoder")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0=single thread, recommended for MPS)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto' (default), 'mps', 'cuda', or 'cpu'")
    parser.add_argument("--top-k-codes", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=4096)
    
    # Cross-dataset evaluation
    parser.add_argument("--cross-dataset", action="store_true")
    parser.add_argument("--source-data", type=str, help="Source dataset (MIMIC-III)")
    parser.add_argument("--target-data", type=str, help="Target dataset (MIMIC-IV)")
    
    args = parser.parse_args()
    
    # Auto-detect best device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
            logger.info("Auto-detected device: MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            args.device = "cuda"
            logger.info("Auto-detected device: CUDA (NVIDIA GPU)")
        else:
            args.device = "cpu"
            logger.info("Auto-detected device: CPU")
    else:
        logger.info(f"Using specified device: {args.device}")
    
    # Load label encoder
    if args.encoder:
        label_encoder = ICDLabelEncoder.load(args.encoder)
    else:
        # Try to load from checkpoint directory
        encoder_path = Path(args.checkpoint).parent / "label_encoder.pkl"
        if encoder_path.exists():
            label_encoder = ICDLabelEncoder.load(str(encoder_path))
        else:
            logger.error("Label encoder not found. Specify with --encoder")
            sys.exit(1)
    
    # Load model
    model = load_model(
        args.model,
        args.checkpoint,
        label_encoder.num_labels,
        args.device,
    )
    
    tokenizer_name = "allenai/longformer-base-4096"
    
    if args.cross_dataset:
        if not args.source_data or not args.target_data:
            parser.error("--source-data and --target-data required for cross-dataset eval")
        
        # Load source dataset
        source_df, _ = load_mimic_data(args.source_data, label_encoder=label_encoder)
        _, _, source_test, _ = create_dataloaders(
            source_df, label_encoder, tokenizer_name,
            max_length=args.max_length, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        
        # Load target dataset
        target_df, _ = load_mimic_data(args.target_data, label_encoder=label_encoder)
        _, _, target_test, _ = create_dataloaders(
            target_df, label_encoder, tokenizer_name,
            max_length=args.max_length, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        
        # Cross-dataset evaluation
        results = evaluate_model_cross_dataset(
            model, source_test, target_test, label_encoder, args.device
        )
        
        # Analyze prediction shift
        shift_df = analyze_prediction_shift(
            model, source_test, target_test, label_encoder, args.device
        )
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "cross_dataset_results.json", "w") as f:
            json.dump(results["comparison"], f, indent=2, default=str)
        
        shift_df.to_csv(output_dir / "prediction_shift.csv", index=False)
        
    else:
        if not args.data:
            parser.error("--data required for single dataset evaluation")
        
        # Load dataset
        df, _ = load_mimic_data(args.data, label_encoder=label_encoder)
        _, _, test_loader, _ = create_dataloaders(
            df, label_encoder, tokenizer_name,
            max_length=args.max_length, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        
        # Evaluate
        evaluate_single_dataset(
            model, test_loader, label_encoder, args.device, args.output_dir
        )


if __name__ == "__main__":
    from typing import Optional
    main()

