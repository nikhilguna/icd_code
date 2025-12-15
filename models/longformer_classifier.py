"""
Longformer classifier for ICD code prediction.

Uses the Longformer architecture for handling long clinical documents
(up to 4,096 tokens) with sliding window + global attention.

Based on: "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    LongformerModel,
    LongformerConfig,
    AutoTokenizer,
    LongformerTokenizer,
)

logger = logging.getLogger(__name__)


class LongformerClassifier(nn.Module):
    """
    Longformer-based classifier for multi-label ICD code prediction.
    
    Uses the encoder portion of Longformer with:
    - Sliding window attention for local context
    - Global attention on [CLS] token for document-level representation
    - Classification head for multi-label prediction
    
    Usage:
        model = LongformerClassifier(
            num_labels=50,
            model_name="allenai/longformer-base-4096",
            freeze_layers=6,
        )
        
        outputs = model(input_ids, attention_mask)
        # outputs["logits"]: (batch, num_labels)
        # outputs["attentions"]: optional attention weights
    """
    
    def __init__(
        self,
        num_labels: int,
        model_name: str = "allenai/longformer-base-4096",
        hidden_dim: int = 768,
        dropout: float = 0.1,
        freeze_layers: int = 0,
        use_global_attention: bool = True,
        classifier_hidden_dim: Optional[int] = None,
        gradient_checkpointing: bool = False,
    ):
        """
        Initialize Longformer classifier.
        
        Args:
            num_labels: Number of ICD codes to predict
            model_name: HuggingFace model name for Longformer
            hidden_dim: Hidden dimension of Longformer (usually 768)
            dropout: Dropout probability for classifier
            freeze_layers: Number of bottom layers to freeze
            use_global_attention: Whether to use global attention on CLS
            classifier_hidden_dim: Hidden dim for classifier (None = no hidden layer)
            gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.use_global_attention = use_global_attention
        
        # Load pretrained Longformer
        logger.info(f"Loading Longformer: {model_name}")
        self.longformer = LongformerModel.from_pretrained(model_name)
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self.longformer.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Freeze bottom layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        
        if classifier_hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, num_labels),
            )
        else:
            self.classifier = nn.Linear(hidden_dim, num_labels)
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze the bottom N transformer layers."""
        # Freeze embeddings
        for param in self.longformer.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified number of encoder layers
        for i in range(min(num_layers, len(self.longformer.encoder.layer))):
            for param in self.longformer.encoder.layer[i].parameters():
                param.requires_grad = False
        
        logger.info(f"Frozen {num_layers} bottom layers + embeddings")
    
    def _init_classifier(self) -> None:
        """Initialize classifier weights."""
        if isinstance(self.classifier, nn.Sequential):
            for module in self.classifier:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(self.classifier.weight)
            if self.classifier.bias is not None:
                nn.init.zeros_(self.classifier.bias)
    
    def _create_global_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create global attention mask with attention on CLS token.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Regular attention mask (batch, seq_len)
            
        Returns:
            Global attention mask (batch, seq_len)
        """
        global_attention_mask = torch.zeros_like(attention_mask)
        
        if self.use_global_attention:
            # Set global attention on first token (CLS)
            global_attention_mask[:, 0] = 1
        
        return global_attention_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            global_attention_mask: Global attention mask (if None, created automatically)
            labels: Optional labels of shape (batch, num_labels) for loss computation
            output_attentions: Whether to output attention weights
            
        Returns:
            Dictionary with:
                - logits: Prediction logits (batch, num_labels)
                - loss: Optional BCE loss if labels provided
                - attentions: Optional attention weights
                - hidden_states: CLS hidden state
        """
        batch_size = input_ids.size(0)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Create global attention mask if not provided
        if global_attention_mask is None:
            global_attention_mask = self._create_global_attention_mask(
                input_ids, attention_mask
            )
        
        # Forward through Longformer
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )
        
        # Get CLS token representation
        # last_hidden_state: (batch, seq_len, hidden_dim)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)
        
        # Apply dropout and classify
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)  # (batch, num_labels)
        
        result = {
            "logits": logits,
            "hidden_states": cls_hidden,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            result["loss"] = loss
        
        if output_attentions:
            result["attentions"] = outputs.attentions
        
        return result
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            threshold: Threshold for binary predictions
            
        Returns:
            probabilities: Sigmoid probabilities (batch, num_labels)
            predictions: Binary predictions (batch, num_labels)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs["logits"])
            predictions = (probabilities >= threshold).float()
        
        return probabilities, predictions
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            layer: Which layer to extract attention from (-1 = last)
            
        Returns:
            Attention weights of shape (batch, num_heads, seq_len, seq_len)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids, 
                attention_mask, 
                output_attentions=True
            )
            
            if outputs.get("attentions") is not None:
                return outputs["attentions"][layer]
        
        return None
    
    @classmethod
    def from_config(
        cls,
        num_labels: int,
        config: Any,
    ) -> "LongformerClassifier":
        """
        Create LongformerClassifier from a config object.
        
        Args:
            num_labels: Number of labels
            config: Config object with Longformer settings
            
        Returns:
            Initialized LongformerClassifier
        """
        return cls(
            num_labels=num_labels,
            model_name=config.model_name,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            freeze_layers=config.freeze_layers,
            use_global_attention=config.use_global_attention,
        )


class LongformerWithLabelAttention(nn.Module):
    """
    Longformer variant with per-label attention (similar to CAML).
    
    Instead of using just the CLS token, this model computes
    label-specific attention over the Longformer outputs.
    """
    
    def __init__(
        self,
        num_labels: int,
        model_name: str = "allenai/longformer-base-4096",
        hidden_dim: int = 768,
        dropout: float = 0.1,
        freeze_layers: int = 0,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        
        # Load pretrained Longformer
        logger.info(f"Loading Longformer: {model_name}")
        self.longformer = LongformerModel.from_pretrained(model_name)
        
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # Per-label attention (similar to CAML)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U = nn.Linear(hidden_dim, num_labels, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)
    
    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze bottom layers."""
        for param in self.longformer.embeddings.parameters():
            param.requires_grad = False
        
        for i in range(min(num_layers, len(self.longformer.encoder.layer))):
            for param in self.longformer.encoder.layer[i].parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with label-wise attention."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Create global attention on CLS
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1
        
        # Forward through Longformer
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        
        # Label-wise attention
        projected = torch.tanh(self.W(hidden_states))  # (batch, seq_len, hidden_dim)
        scores = self.U(projected)  # (batch, seq_len, num_labels)
        scores = scores.transpose(1, 2)  # (batch, num_labels, seq_len)
        
        # Apply mask
        mask = attention_mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 0.0
        )
        
        # Compute context
        context = torch.bmm(attention_weights, hidden_states)  # (batch, num_labels, hidden_dim)
        context = self.dropout(context)
        
        # Classify
        logits = self.classifier(context).squeeze(-1)  # (batch, num_labels)
        
        result = {"logits": logits}
        
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            result["loss"] = loss
        
        if return_attention:
            result["attention"] = attention_weights
        
        return result


class LongformerForSequenceClassification(nn.Module):
    """
    Simpler Longformer classifier using mean pooling over all tokens.
    
    Alternative to CLS-based classification.
    """
    
    def __init__(
        self,
        num_labels: int,
        model_name: str = "allenai/longformer-base-4096",
        hidden_dim: int = 768,
        dropout: float = 0.1,
        pooling: str = "mean",  # "mean", "max", "cls"
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.pooling = pooling
        
        self.longformer = LongformerModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Global attention on CLS
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1
        
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Pooling
        if self.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.pooling == "mean":
            # Mean pooling with attention mask
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "max":
            # Max pooling with attention mask
            mask = attention_mask.unsqueeze(-1)
            hidden_states = hidden_states.masked_fill(mask == 0, float("-inf"))
            pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        result = {"logits": logits}
        
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            result["loss"] = loss
        
        return result

