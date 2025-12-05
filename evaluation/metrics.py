"""
Evaluation metrics for multi-label ICD code prediction.

Includes:
- Micro/Macro F1, Precision, Recall
- Precision@k (P@5, P@10)
- ROC-AUC per label
- Frequency-stratified metrics (head/medium/tail)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    # Overall metrics
    micro_f1: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    macro_precision: float
    macro_recall: float
    
    # Ranking metrics
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    
    # AUC metrics
    micro_auc: Optional[float] = None
    macro_auc: Optional[float] = None
    per_label_auc: Optional[Dict[int, float]] = None
    
    # Stratified metrics
    head_f1: Optional[float] = None
    medium_f1: Optional[float] = None
    tail_f1: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        result = {
            "micro_f1": self.micro_f1,
            "macro_f1": self.macro_f1,
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "precision_at_5": self.precision_at_5,
            "precision_at_10": self.precision_at_10,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
        }
        
        if self.micro_auc is not None:
            result["micro_auc"] = self.micro_auc
        if self.macro_auc is not None:
            result["macro_auc"] = self.macro_auc
        if self.head_f1 is not None:
            result["head_f1"] = self.head_f1
            result["medium_f1"] = self.medium_f1
            result["tail_f1"] = self.tail_f1
        
        return result
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "Evaluation Results",
            "=" * 50,
            f"Micro F1:     {self.micro_f1:.4f}",
            f"Macro F1:     {self.macro_f1:.4f}",
            f"Micro Prec:   {self.micro_precision:.4f}",
            f"Micro Recall: {self.micro_recall:.4f}",
            f"P@5:          {self.precision_at_5:.4f}",
            f"P@10:         {self.precision_at_10:.4f}",
        ]
        
        if self.micro_auc is not None:
            lines.append(f"Micro AUC:    {self.micro_auc:.4f}")
            lines.append(f"Macro AUC:    {self.macro_auc:.4f}")
        
        if self.head_f1 is not None:
            lines.extend([
                "-" * 50,
                "Stratified F1:",
                f"  Head:   {self.head_f1:.4f}",
                f"  Medium: {self.medium_f1:.4f}",
                f"  Tail:   {self.tail_f1:.4f}",
            ])
        
        lines.append("=" * 50)
        return "\n".join(lines)


class ICDMetrics:
    """
    Comprehensive metrics calculator for ICD code prediction.
    
    Usage:
        metrics = ICDMetrics(label_encoder)
        
        # Add predictions
        metrics.update(predictions, labels)
        
        # Compute all metrics
        results = metrics.compute()
    """
    
    def __init__(
        self,
        label_encoder=None,
        compute_auc: bool = True,
        compute_stratified: bool = True,
    ):
        """
        Initialize metrics calculator.
        
        Args:
            label_encoder: Optional ICDLabelEncoder for stratified metrics
            compute_auc: Whether to compute AUC metrics
            compute_stratified: Whether to compute stratified metrics
        """
        self.label_encoder = label_encoder
        self.compute_auc = compute_auc
        self.compute_stratified = compute_stratified and label_encoder is not None
        
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated predictions."""
        self.all_probs: List[np.ndarray] = []
        self.all_preds: List[np.ndarray] = []
        self.all_labels: List[np.ndarray] = []
    
    def update(
        self,
        logits_or_probs: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        is_logits: bool = True,
    ) -> None:
        """
        Add batch of predictions.
        
        Args:
            logits_or_probs: Model outputs (logits or probabilities)
            labels: Ground truth labels
            threshold: Threshold for binary predictions
            is_logits: Whether input is logits (will apply sigmoid)
        """
        # Convert to numpy
        if isinstance(logits_or_probs, torch.Tensor):
            logits_or_probs = logits_or_probs.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Convert logits to probabilities
        if is_logits:
            probs = 1 / (1 + np.exp(-logits_or_probs))  # sigmoid
        else:
            probs = logits_or_probs
        
        # Threshold to get predictions
        preds = (probs >= threshold).astype(int)
        
        self.all_probs.append(probs)
        self.all_preds.append(preds)
        self.all_labels.append(labels.astype(int))
    
    def compute(self) -> EvaluationResults:
        """
        Compute all metrics.
        
        Returns:
            EvaluationResults with all computed metrics
        """
        # Concatenate all batches
        probs = np.concatenate(self.all_probs, axis=0)
        preds = np.concatenate(self.all_preds, axis=0)
        labels = np.concatenate(self.all_labels, axis=0)
        
        # Basic metrics
        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        micro_precision = precision_score(labels, preds, average="micro", zero_division=0)
        micro_recall = recall_score(labels, preds, average="micro", zero_division=0)
        macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
        macro_recall = recall_score(labels, preds, average="macro", zero_division=0)
        
        # Precision@k and Recall@k
        p_at_5, r_at_5 = self._precision_recall_at_k(probs, labels, k=5)
        p_at_10, r_at_10 = self._precision_recall_at_k(probs, labels, k=10)
        
        results = EvaluationResults(
            micro_f1=micro_f1,
            macro_f1=macro_f1,
            micro_precision=micro_precision,
            micro_recall=micro_recall,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            precision_at_5=p_at_5,
            precision_at_10=p_at_10,
            recall_at_5=r_at_5,
            recall_at_10=r_at_10,
        )
        
        # AUC metrics
        if self.compute_auc:
            try:
                micro_auc, macro_auc, per_label_auc = self._compute_auc(probs, labels)
                results.micro_auc = micro_auc
                results.macro_auc = macro_auc
                results.per_label_auc = per_label_auc
            except ValueError as e:
                logger.warning(f"Could not compute AUC: {e}")
        
        # Stratified metrics
        if self.compute_stratified:
            head_f1, medium_f1, tail_f1 = self._compute_stratified_f1(preds, labels)
            results.head_f1 = head_f1
            results.medium_f1 = medium_f1
            results.tail_f1 = tail_f1
        
        return results
    
    def _precision_recall_at_k(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        k: int,
    ) -> Tuple[float, float]:
        """
        Compute Precision@k and Recall@k.
        
        For each sample, take top-k predicted labels and compare to true labels.
        """
        n_samples = probs.shape[0]
        precisions = []
        recalls = []
        
        for i in range(n_samples):
            # Get top-k predictions
            top_k_idx = np.argsort(probs[i])[-k:]
            top_k_preds = np.zeros_like(probs[i])
            top_k_preds[top_k_idx] = 1
            
            # Count matches
            true_positives = (top_k_preds * labels[i]).sum()
            
            # Precision@k = TP / k
            precisions.append(true_positives / k)
            
            # Recall@k = TP / num_true_labels
            num_true = labels[i].sum()
            if num_true > 0:
                recalls.append(true_positives / num_true)
            else:
                recalls.append(0.0)
        
        return np.mean(precisions), np.mean(recalls)
    
    def _compute_auc(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[float, float, Dict[int, float]]:
        """
        Compute ROC-AUC metrics.
        
        Returns:
            micro_auc: Micro-averaged AUC
            macro_auc: Macro-averaged AUC
            per_label_auc: AUC for each label
        """
        n_labels = labels.shape[1]
        per_label_auc = {}
        valid_aucs = []
        
        for i in range(n_labels):
            # Skip labels with only one class
            if len(np.unique(labels[:, i])) < 2:
                continue
            
            auc = roc_auc_score(labels[:, i], probs[:, i])
            per_label_auc[i] = auc
            valid_aucs.append(auc)
        
        macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        
        # Micro AUC
        try:
            micro_auc = roc_auc_score(labels.ravel(), probs.ravel())
        except ValueError:
            micro_auc = 0.0
        
        return micro_auc, macro_auc, per_label_auc
    
    def _compute_stratified_f1(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute F1 stratified by label frequency.
        
        Returns:
            head_f1: F1 for frequent labels
            medium_f1: F1 for medium-frequency labels
            tail_f1: F1 for rare labels
        """
        if self.label_encoder is None:
            return 0.0, 0.0, 0.0
        
        stratum_f1s = {}
        
        for stratum, indices in self.label_encoder.stratum_indices.items():
            if not indices:
                stratum_f1s[stratum] = 0.0
                continue
            
            # Get subset of predictions and labels for this stratum
            stratum_preds = preds[:, indices]
            stratum_labels = labels[:, indices]
            
            # Compute F1
            f1 = f1_score(
                stratum_labels.ravel(),
                stratum_preds.ravel(),
                zero_division=0
            )
            stratum_f1s[stratum] = f1
        
        return (
            stratum_f1s.get("head", 0.0),
            stratum_f1s.get("medium", 0.0),
            stratum_f1s.get("tail", 0.0),
        )


def compute_metrics(
    logits: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    label_encoder=None,
) -> Dict[str, float]:
    """
    Convenience function to compute all metrics at once.
    
    Args:
        logits: Model output logits
        labels: Ground truth labels
        threshold: Threshold for binary predictions
        label_encoder: Optional encoder for stratified metrics
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = ICDMetrics(label_encoder=label_encoder)
    metrics.update(logits, labels, threshold=threshold, is_logits=True)
    results = metrics.compute()
    return results.to_dict()


def find_optimal_threshold(
    logits: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    metric: str = "micro_f1",
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Find optimal prediction threshold using grid search.
    
    Args:
        logits: Model output logits
        labels: Ground truth labels
        metric: Metric to optimize ("micro_f1", "macro_f1", etc.)
        thresholds: List of thresholds to try
        
    Returns:
        best_threshold: Optimal threshold
        best_score: Best metric score
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Convert to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    probs = 1 / (1 + np.exp(-logits))
    labels = labels.astype(int)
    
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        
        if metric == "micro_f1":
            score = f1_score(labels, preds, average="micro", zero_division=0)
        elif metric == "macro_f1":
            score = f1_score(labels, preds, average="macro", zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def per_label_metrics(
    logits: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    label_encoder=None,
    threshold: float = 0.5,
) -> List[Dict]:
    """
    Compute metrics for each individual label.
    
    Returns list of dicts with per-label statistics.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)
    
    n_labels = labels.shape[1]
    results = []
    
    for i in range(n_labels):
        label_info = {
            "label_idx": i,
            "support": int(labels[:, i].sum()),
            "predictions": int(preds[:, i].sum()),
        }
        
        if label_encoder is not None:
            code = label_encoder.idx_to_code.get(i, f"label_{i}")
            label_info["code"] = code
            label_info["frequency"] = label_encoder.code_frequencies.get(code, 0)
            label_info["stratum"] = label_encoder.code_strata.get(code, "unknown")
        
        # Compute metrics if there are positive samples
        if labels[:, i].sum() > 0:
            tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum()
            fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum()
            fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            label_info["precision"] = precision
            label_info["recall"] = recall
            label_info["f1"] = f1
            
            # AUC if both classes present
            if len(np.unique(labels[:, i])) == 2:
                label_info["auc"] = roc_auc_score(labels[:, i], probs[:, i])
        else:
            label_info["precision"] = 0.0
            label_info["recall"] = 0.0
            label_info["f1"] = 0.0
        
        results.append(label_info)
    
    return results

