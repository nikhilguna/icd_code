#!/usr/bin/env python3
"""
Train LED (Longformer) model for ICD code prediction.

Usage:
    python scripts/train_led.py --config configs/default.yaml
    python scripts/train_led.py --data data/processed/mimic3.parquet --epochs 10
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import load_mimic_data, create_dataloaders
from data.label_encoder import ICDLabelEncoder
from models.led_classifier import LEDClassifier
from training.trainer import Trainer, create_optimizer, create_scheduler
from training.losses import get_loss_function
from utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train LED model")
    
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data", type=str, help="Path to processed data parquet")
    parser.add_argument("--output-dir", type=str, default="checkpoints/led")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--top-k-codes", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--freeze-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--model-name", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        if args.data:
            data_path = args.data
        else:
            data_path = Path(config.data.processed_data_dir) / "mimic3.parquet"
    else:
        config = None
        data_path = args.data
    
    if not data_path or not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run data extraction and preprocessing first:")
        logger.info("  python scripts/extract_data.py ...")
        sys.exit(1)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df, label_encoder = load_mimic_data(
        data_path,
        top_k_codes=args.top_k_codes,
        min_frequency=10,
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader, label_encoder = create_dataloaders(
        df=df,
        label_encoder=label_encoder,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    logger.info(f"Train: {len(train_loader.dataset)} samples")
    logger.info(f"Val: {len(val_loader.dataset)} samples")
    logger.info(f"Test: {len(test_loader.dataset)} samples")
    logger.info(f"Num labels: {label_encoder.num_labels}")
    
    # Create model
    model = LEDClassifier(
        num_labels=label_encoder.num_labels,
        model_name=args.model_name,
        dropout=args.dropout,
        freeze_layers=args.freeze_layers,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        learning_rate=args.lr,
        weight_decay=0.01,
    )
    
    # Create scheduler
    num_training_steps = (len(train_loader) // args.gradient_accumulation) * args.epochs
    scheduler = create_scheduler(
        optimizer,
        scheduler_type="cosine",
        num_training_steps=num_training_steps,
        num_warmup_steps=int(num_training_steps * 0.1),
    )
    
    # Create loss function
    pos_weight = train_loader.dataset.get_label_weights(method="inverse_freq")
    loss_fn = get_loss_function("bce", pos_weight=pos_weight.to(args.device))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=args.device,
        scheduler=scheduler,
        loss_fn=loss_fn,
        gradient_accumulation_steps=args.gradient_accumulation,
        checkpoint_dir=args.output_dir,
        early_stopping_patience=3,
        use_wandb=args.wandb,
        mixed_precision=True,
    )
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        resume_from=args.resume,
    )
    
    # Save label encoder
    encoder_path = Path(args.output_dir) / "label_encoder.pkl"
    label_encoder.save(str(encoder_path))
    logger.info(f"Saved label encoder to {encoder_path}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    model.load_state_dict(
        torch.load(Path(args.output_dir) / "best_model.pt", weights_only=False)["model_state_dict"]
    )
    
    test_results = evaluate_model(model, test_loader, args.device)
    
    logger.info("Test Results:")
    for metric, value in test_results.items():
        logger.info(f"  {metric}: {value:.4f}")


def evaluate_model(model, dataloader, device):
    """Quick evaluation on a dataloader."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    import numpy as np
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return {
        "micro_f1": f1_score(all_labels, all_preds, average="micro", zero_division=0),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "micro_precision": precision_score(all_labels, all_preds, average="micro", zero_division=0),
        "micro_recall": recall_score(all_labels, all_preds, average="micro", zero_division=0),
    }


if __name__ == "__main__":
    main()

