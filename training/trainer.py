"""
Training infrastructure for ICD code prediction models.

Provides:
- Mixed precision training with AMP
- Gradient accumulation for large effective batch sizes
- Early stopping based on validation metrics
- Checkpoint management
- Logging to console/wandb
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Tracks training state for checkpointing."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    early_stop_counter: int = 0


class Trainer:
    """
    Trainer for ICD code prediction models.
    
    Supports CAML and LED models with:
    - Mixed precision training
    - Gradient accumulation
    - Early stopping
    - Checkpoint saving/loading
    - Metric logging
    
    Usage:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device="cuda",
        )
        
        trainer.train(num_epochs=50)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = True,
        checkpoint_dir: str = "checkpoints",
        save_top_k: int = 3,
        early_stopping_patience: int = 5,
        early_stopping_metric: str = "micro_f1",
        log_every_n_steps: int = 100,
        eval_fn: Optional[Callable] = None,
        use_wandb: bool = False,
        project_name: str = "icd-prediction",
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            device: Device to train on
            scheduler: Optional learning rate scheduler
            loss_fn: Loss function (if None, uses model's built-in loss)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            mixed_precision: Whether to use mixed precision training
            checkpoint_dir: Directory for saving checkpoints
            save_top_k: Number of best checkpoints to keep
            early_stopping_patience: Epochs without improvement before stopping
            early_stopping_metric: Metric to monitor for early stopping
            log_every_n_steps: Steps between logging
            eval_fn: Optional custom evaluation function
            use_wandb: Whether to log to W&B
            project_name: W&B project name
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision and device != "cpu"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_top_k = save_top_k
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.log_every_n_steps = log_every_n_steps
        self.eval_fn = eval_fn
        self.use_wandb = use_wandb
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Training state
        self.state = TrainingState()
        
        # Checkpoint tracking
        self.best_checkpoints: List[Dict[str, Any]] = []
        
        # Initialize W&B if requested
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=project_name)
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")
    
    def train(
        self,
        num_epochs: int,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Optional checkpoint path to resume from
            
        Returns:
            Dictionary with training history
        """
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
        }
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches: {len(self.val_loader)}")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"  Mixed precision: {self.mixed_precision}")
        
        start_epoch = self.state.epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.state.epoch = epoch
            
            # Training
            train_loss = self._train_epoch()
            history["train_loss"].append(train_loss)
            
            # Validation
            val_results = self._validate()
            history["val_loss"].append(val_results["loss"])
            history["val_metrics"].append(val_results["metrics"])
            
            # Get monitored metric
            current_metric = val_results["metrics"].get(
                self.early_stopping_metric, 0.0
            )
            
            # Log epoch results
            self._log_epoch(epoch, train_loss, val_results, current_metric)
            
            # Check for improvement
            if current_metric > self.state.best_metric:
                self.state.best_metric = current_metric
                self.state.best_epoch = epoch
                self.state.early_stop_counter = 0
                
                # Save best checkpoint
                self._save_checkpoint(
                    epoch, 
                    current_metric, 
                    is_best=True
                )
            else:
                self.state.early_stop_counter += 1
            
            # Save regular checkpoint
            self._save_checkpoint(epoch, current_metric)
            
            # Early stopping check
            if self.state.early_stop_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement for {self.early_stopping_patience} epochs)"
                )
                break
        
        # Final logging
        logger.info(f"Training complete!")
        logger.info(f"  Best {self.early_stopping_metric}: {self.state.best_metric:.4f}")
        logger.info(f"  Best epoch: {self.state.best_epoch}")
        
        return history
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.state.epoch + 1}",
            leave=False,
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.mixed_precision):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                if self.loss_fn is not None:
                    loss = self.loss_fn(outputs["logits"], labels)
                else:
                    loss = outputs["loss"]
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.state.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item() * self.gradient_accumulation_steps:.4f}"
            )
            
            # Log periodically
            if self.state.global_step % self.log_every_n_steps == 0:
                self._log_step(loss.item() * self.gradient_accumulation_steps)
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, Any]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with autocast(enabled=self.mixed_precision):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                if self.loss_fn is not None:
                    loss = self.loss_fn(outputs["logits"], labels)
                else:
                    loss = outputs["loss"]
            
            total_loss += loss.item()
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(labels.cpu())
        
        # Concatenate all predictions
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        if self.eval_fn is not None:
            metrics = self.eval_fn(all_logits, all_labels)
        else:
            metrics = self._compute_basic_metrics(all_logits, all_labels)
        
        return {
            "loss": total_loss / len(self.val_loader),
            "metrics": metrics,
            "logits": all_logits,
            "labels": all_labels,
        }
    
    def _compute_basic_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute basic classification metrics."""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)
        targets = labels.numpy().astype(int)
        
        # Micro and macro F1
        micro_f1 = f1_score(targets, preds, average="micro", zero_division=0)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        
        # Precision and recall
        micro_precision = precision_score(targets, preds, average="micro", zero_division=0)
        micro_recall = recall_score(targets, preds, average="micro", zero_division=0)
        
        return {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
        }
    
    def _log_step(self, loss: float) -> None:
        """Log training step."""
        if self.wandb_run is not None:
            import wandb
            wandb.log({
                "train/step_loss": loss,
                "train/global_step": self.state.global_step,
                "train/lr": self.optimizer.param_groups[0]["lr"],
            })
    
    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_results: Dict,
        current_metric: float,
    ) -> None:
        """Log epoch results."""
        logger.info(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_results['loss']:.4f}, "
            f"{self.early_stopping_metric}={current_metric:.4f}"
        )
        
        if self.wandb_run is not None:
            import wandb
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_results["loss"],
            }
            for k, v in val_results["metrics"].items():
                log_dict[f"val/{k}"] = v
            wandb.log(log_dict)
    
    def _save_checkpoint(
        self,
        epoch: int,
        metric: float,
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "state": self.state,
            "metric": metric,
        }
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        # Save regular checkpoint with metric tracking
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, ckpt_path)
        
        # Track checkpoint for top-k management
        self.best_checkpoints.append({
            "path": ckpt_path,
            "metric": metric,
            "epoch": epoch,
        })
        
        # Sort by metric (descending) and keep top-k
        self.best_checkpoints.sort(key=lambda x: x["metric"], reverse=True)
        
        while len(self.best_checkpoints) > self.save_top_k:
            to_remove = self.best_checkpoints.pop()
            if to_remove["path"].exists():
                to_remove["path"].unlink()
    
    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if checkpoint.get("scaler_state_dict") and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.state = checkpoint["state"]
        
        logger.info(f"Resumed from epoch {self.state.epoch}")


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    optimizer_type: str = "adamw",
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer with optional layer-wise learning rate decay.
    
    Args:
        model: Model to optimize
        learning_rate: Base learning rate
        weight_decay: Weight decay factor
        optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
        
    Returns:
        Configured optimizer
    """
    # Separate parameters that should/shouldn't have weight decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    if optimizer_type == "adamw":
        return AdamW(param_groups, lr=learning_rate, **kwargs)
    elif optimizer_type == "adam":
        return torch.optim.Adam(param_groups, lr=learning_rate, **kwargs)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(param_groups, lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 100,
    **kwargs,
) -> Any:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("cosine", "linear", "onecycle")
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        # Linear warmup then cosine decay
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )
        cosine = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_training_steps - num_warmup_steps,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[num_warmup_steps],
        )
    
    elif scheduler_type == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps,
        )
    
    elif scheduler_type == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            total_steps=num_training_steps,
            pct_start=num_warmup_steps / num_training_steps,
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

