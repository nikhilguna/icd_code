"""
Cross-dataset evaluation for ICD code prediction.

Evaluates models trained on MIMIC-III on MIMIC-IV to measure
robustness to temporal and stylistic distribution shift.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import ICDMetrics, EvaluationResults, per_label_metrics

logger = logging.getLogger(__name__)


class CrossDatasetEvaluator:
    """
    Evaluator for cross-dataset generalization.
    
    Trains on MIMIC-III and evaluates on MIMIC-IV to measure
    how well models generalize to newer clinical documentation styles.
    
    Usage:
        evaluator = CrossDatasetEvaluator(
            model=model,
            source_encoder=mimic3_encoder,
            device="cuda",
        )
        
        # Evaluate on MIMIC-IV
        results = evaluator.evaluate(mimic4_loader)
        
        # Compare with source performance
        comparison = evaluator.compare(mimic3_results, mimic4_results)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        source_encoder,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            source_encoder: Label encoder from source dataset (MIMIC-III)
            device: Device for inference
        """
        self.model = model.to(device)
        self.source_encoder = source_encoder
        self.device = device
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        threshold: float = 0.5,
    ) -> EvaluationResults:
        """
        Evaluate model on target dataset.
        
        Args:
            dataloader: DataLoader for target dataset
            threshold: Prediction threshold
            
        Returns:
            EvaluationResults with all metrics
        """
        self.model.eval()
        
        metrics = ICDMetrics(
            label_encoder=self.source_encoder,
            compute_auc=True,
            compute_stratified=True,
        )
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            metrics.update(
                outputs["logits"],
                labels,
                threshold=threshold,
                is_logits=True,
            )
        
        return metrics.compute()
    
    def compare(
        self,
        source_results: EvaluationResults,
        target_results: EvaluationResults,
    ) -> Dict[str, Any]:
        """
        Compare source and target dataset performance.
        
        Args:
            source_results: Results on source dataset (MIMIC-III)
            target_results: Results on target dataset (MIMIC-IV)
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            "source": source_results.to_dict(),
            "target": target_results.to_dict(),
            "delta": {},
            "retention": {},
        }
        
        # Compute deltas and retention ratios
        for key in source_results.to_dict():
            source_val = source_results.to_dict()[key]
            target_val = target_results.to_dict()[key]
            
            if source_val is not None and target_val is not None:
                comparison["delta"][key] = target_val - source_val
                
                if source_val > 0:
                    comparison["retention"][key] = target_val / source_val
                else:
                    comparison["retention"][key] = 0.0
        
        return comparison
    
    def evaluate_code_overlap(
        self,
        source_codes: List[List[str]],
        target_codes: List[List[str]],
    ) -> Dict[str, Any]:
        """
        Analyze ICD code overlap between datasets.
        
        Args:
            source_codes: ICD codes from source dataset
            target_codes: ICD codes from target dataset
            
        Returns:
            Dictionary with overlap statistics
        """
        # Get unique codes
        source_unique = set(code for codes in source_codes for code in codes)
        target_unique = set(code for codes in target_codes for code in codes)
        
        # Compute overlap
        overlap = source_unique & target_unique
        source_only = source_unique - target_unique
        target_only = target_unique - source_unique
        
        # Compute coverage
        source_coverage = len(overlap) / len(source_unique) if source_unique else 0
        target_coverage = len(overlap) / len(target_unique) if target_unique else 0
        
        return {
            "source_unique_codes": len(source_unique),
            "target_unique_codes": len(target_unique),
            "overlap_codes": len(overlap),
            "source_only_codes": len(source_only),
            "target_only_codes": len(target_only),
            "source_coverage": source_coverage,
            "target_coverage": target_coverage,
            "jaccard_similarity": len(overlap) / len(source_unique | target_unique),
        }
    
    def stratified_comparison(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Compare performance stratified by label frequency.
        
        Returns DataFrame with per-stratum comparison.
        """
        # Evaluate both datasets
        source_results = self.evaluate(source_loader, threshold)
        target_results = self.evaluate(target_loader, threshold)
        
        records = []
        
        for stratum in ["head", "medium", "tail"]:
            source_f1 = getattr(source_results, f"{stratum}_f1", 0.0)
            target_f1 = getattr(target_results, f"{stratum}_f1", 0.0)
            
            if source_f1 is not None and target_f1 is not None:
                records.append({
                    "stratum": stratum,
                    "source_f1": source_f1,
                    "target_f1": target_f1,
                    "delta": target_f1 - source_f1,
                    "retention": target_f1 / source_f1 if source_f1 > 0 else 0,
                })
        
        return pd.DataFrame(records)


def evaluate_model_cross_dataset(
    model: torch.nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    label_encoder,
    device: str = "cuda",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Convenience function for cross-dataset evaluation.
    
    Args:
        model: Trained model
        source_loader: DataLoader for source dataset (MIMIC-III)
        target_loader: DataLoader for target dataset (MIMIC-IV)
        label_encoder: Fitted label encoder
        device: Device for inference
        threshold: Prediction threshold
        
    Returns:
        Dictionary with evaluation results and comparison
    """
    evaluator = CrossDatasetEvaluator(
        model=model,
        source_encoder=label_encoder,
        device=device,
    )
    
    logger.info("Evaluating on source dataset (MIMIC-III)...")
    source_results = evaluator.evaluate(source_loader, threshold)
    
    logger.info("Evaluating on target dataset (MIMIC-IV)...")
    target_results = evaluator.evaluate(target_loader, threshold)
    
    comparison = evaluator.compare(source_results, target_results)
    
    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Dataset Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Source (MIMIC-III) Micro-F1: {source_results.micro_f1:.4f}")
    logger.info(f"Target (MIMIC-IV) Micro-F1:  {target_results.micro_f1:.4f}")
    logger.info(f"Delta:                        {comparison['delta']['micro_f1']:.4f}")
    logger.info(f"Retention:                    {comparison['retention']['micro_f1']:.2%}")
    logger.info("=" * 60)
    
    return {
        "source_results": source_results,
        "target_results": target_results,
        "comparison": comparison,
    }


def analyze_prediction_shift(
    model: torch.nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    label_encoder,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Analyze how predictions shift between datasets.
    
    Useful for understanding what types of codes become
    harder or easier to predict on the target dataset.
    """
    model.eval()
    
    def get_per_label_stats(loader):
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs["logits"].cpu())
                all_labels.append(labels)
        
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return per_label_metrics(logits, labels, label_encoder)
    
    source_stats = get_per_label_stats(source_loader)
    target_stats = get_per_label_stats(target_loader)
    
    # Combine into DataFrame
    records = []
    for s, t in zip(source_stats, target_stats):
        record = {
            "label_idx": s["label_idx"],
            "code": s.get("code", f"label_{s['label_idx']}"),
            "frequency": s.get("frequency", 0),
            "stratum": s.get("stratum", "unknown"),
            "source_f1": s.get("f1", 0),
            "target_f1": t.get("f1", 0),
            "source_support": s.get("support", 0),
            "target_support": t.get("support", 0),
        }
        record["f1_delta"] = record["target_f1"] - record["source_f1"]
        records.append(record)
    
    df = pd.DataFrame(records)
    df = df.sort_values("f1_delta", ascending=True)
    
    return df

