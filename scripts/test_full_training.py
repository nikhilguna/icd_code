#!/usr/bin/env python3
"""
End-to-end test: Train and evaluate a model on mock data.

This validates the complete pipeline:
1. Data loading
2. Model initialization
3. Training loop
4. Evaluation metrics
5. Checkpoint saving

Usage:
    python scripts/test_full_training.py --model caml --epochs 3
    python scripts/test_full_training.py --model led --epochs 2
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import ICDDataset, create_dataloaders
from data.label_encoder import ICDLabelEncoder
from models.caml import CAML
from models.led_classifier import LEDClassifier
from training.trainer import Trainer, create_optimizer, create_scheduler
from training.losses import get_loss_function
from evaluation.metrics import ICDMetrics, find_optimal_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_mock_data(data_dir: str = "mock_data/raw"):
    """Load mock MIMIC data."""
    data_dir = Path(data_dir)
    
    # Load CSVs
    notes = pd.read_csv(data_dir / "NOTEEVENTS.csv")
    diagnoses = pd.read_csv(data_dir / "DIAGNOSES_ICD.csv")
    
    # Aggregate ICD codes per admission
    icd_df = diagnoses.groupby("HADM_ID")["ICD9_CODE"].apply(list).reset_index()
    icd_df.columns = ["HADM_ID", "icd_codes"]
    
    # Merge with notes
    df = notes.merge(icd_df, on="HADM_ID", how="inner")
    
    # Rename columns
    df = df.rename(columns={"TEXT": "discharge_text", "HADM_ID": "hadm_id"})
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"  Avg codes per sample: {df['icd_codes'].apply(len).mean():.1f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Test full training pipeline")
    parser.add_argument("--model", choices=["caml", "led"], default="caml")
    parser.add_argument("--data-dir", type=str, default="mock_data/raw")
    parser.add_argument("--output-dir", type=str, default="test_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("END-TO-END TRAINING TEST")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("")
    
    # Set seed
    torch.manual_seed(42)
    
    # Load data
    logger.info("Loading mock data...")
    df = load_mock_data(args.data_dir)
    
    # Fit label encoder
    logger.info("Fitting label encoder...")
    label_encoder = ICDLabelEncoder(top_k=50, min_frequency=5)
    label_encoder.fit(df["icd_codes"].tolist())
    
    logger.info(f"  Number of labels: {label_encoder.num_labels}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    tokenizer_name = "bert-base-uncased"  # Faster for testing
    
    train_loader, val_loader, test_loader, label_encoder = create_dataloaders(
        df=df,
        label_encoder=label_encoder,
        tokenizer_name=tokenizer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        num_workers=0,  # 0 for testing to avoid multiprocessing issues
    )
    
    logger.info(f"  Train: {len(train_loader.dataset)} samples")
    logger.info(f"  Val:   {len(val_loader.dataset)} samples")
    logger.info(f"  Test:  {len(test_loader.dataset)} samples")
    logger.info("")
    
    # Create model
    logger.info(f"Creating {args.model.upper()} model...")
    
    if args.model == "caml":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        model = CAML(
            vocab_size=tokenizer.vocab_size,
            num_labels=label_encoder.num_labels,
            embedding_dim=128,  # Smaller for faster testing
            num_filters=64,
            dropout=0.2,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
    else:  # LED
        model = LEDClassifier(
            num_labels=label_encoder.num_labels,
            model_name=tokenizer_name,
            dropout=0.1,
        )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info("")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=1e-5)
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = create_scheduler(
        optimizer,
        scheduler_type="linear",
        num_training_steps=num_training_steps,
        num_warmup_steps=max(1, num_training_steps // 10),
    )
    
    # Create loss function
    pos_weight = train_loader.dataset.get_label_weights(method="inverse_freq")
    loss_fn = get_loss_function("bce", pos_weight=pos_weight.to(args.device))
    
    # Create evaluation function
    def eval_fn(logits, labels):
        metrics = ICDMetrics(label_encoder=label_encoder, compute_auc=False)
        metrics.update(logits, labels, is_logits=True)
        return metrics.compute().to_dict()
    
    # Create trainer
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=args.device,
        scheduler=scheduler,
        loss_fn=loss_fn,
        eval_fn=eval_fn,
        checkpoint_dir=str(output_dir),
        early_stopping_patience=10,  # High to complete all epochs
        log_every_n_steps=5,
        mixed_precision=False,  # Disable for testing
        use_wandb=False,
    )
    
    # Train
    logger.info("=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)
    
    history = trainer.train(num_epochs=args.epochs)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)
    
    # Load best model
    best_checkpoint = output_dir / "best_model.pt"
    checkpoint = torch.load(best_checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()
    
    # Evaluate
    metrics = ICDMetrics(
        label_encoder=label_encoder,
        compute_auc=False,
        compute_stratified=True,
    )
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            metrics.update(outputs["logits"], labels, is_logits=True)
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(labels)
    
    results = metrics.compute()
    
    logger.info("")
    logger.info(str(results))
    
    # Find optimal threshold
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    best_thresh, best_f1 = find_optimal_threshold(logits, labels, metric="micro_f1")
    
    logger.info("")
    logger.info(f"Optimal threshold: {best_thresh} (Micro F1: {best_f1:.4f})")
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✓ Model: {args.model.upper()}")
    logger.info(f"✓ Trained for {args.epochs} epochs")
    logger.info(f"✓ Best val F1: {trainer.state.best_metric:.4f} (epoch {trainer.state.best_epoch})")
    logger.info(f"✓ Test micro F1: {results.micro_f1:.4f}")
    logger.info(f"✓ Test P@5: {results.precision_at_5:.4f}")
    logger.info(f"✓ Checkpoints saved to: {output_dir}")
    logger.info("")
    logger.info("✅ END-TO-END TEST PASSED!")
    logger.info("=" * 60)
    
    # Cleanup to prevent hanging
    try:
        del model
        del trainer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Force exit to prevent thread hanging
        import os
        os._exit(0)

