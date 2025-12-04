"""
CAML (Convolutional Attention for Multi-Label Classification) implementation.

Based on: "Explainable Prediction of Medical Codes from Clinical Text" (Mullenbach et al., 2018)

Architecture:
1. Word embeddings (300d, optionally pretrained)
2. 1D CNN encoder with multiple filter sizes
3. Per-label attention mechanism
4. Sigmoid output for multi-label classification
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class LabelWiseAttention(nn.Module):
    """
    Label-wise attention mechanism.
    
    For each label, computes attention weights over the encoded sequence
    to produce a label-specific document representation.
    
    Attention: alpha_l = softmax(U_l @ tanh(W @ H))
    Context: c_l = alpha_l @ H
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_labels: int,
    ):
        """
        Args:
            hidden_dim: Dimension of encoder hidden states
            num_labels: Number of labels (ICD codes)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        # Shared projection: W @ H
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Per-label attention vectors: U_l
        self.U = nn.Linear(hidden_dim, num_labels, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute label-wise attention.
        
        Args:
            hidden_states: Encoder output of shape (batch, seq_len, hidden_dim)
            attention_mask: Mask of shape (batch, seq_len), 1 for valid positions
            
        Returns:
            context: Label-specific representations of shape (batch, num_labels, hidden_dim)
            attention_weights: Attention weights of shape (batch, num_labels, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project: (batch, seq_len, hidden_dim)
        projected = torch.tanh(self.W(hidden_states))
        
        # Compute attention scores: (batch, seq_len, num_labels)
        scores = self.U(projected)
        
        # Transpose to (batch, num_labels, seq_len)
        scores = scores.transpose(1, 2)
        
        # Apply mask if provided
        if attention_mask is not None:
            # Expand mask to (batch, 1, seq_len) for broadcasting
            mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Softmax over sequence dimension
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN from all-masked positions
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 0.0
        )
        
        # Compute context vectors: (batch, num_labels, hidden_dim)
        # attention_weights: (batch, num_labels, seq_len)
        # hidden_states: (batch, seq_len, hidden_dim)
        context = torch.bmm(attention_weights, hidden_states)
        
        return context, attention_weights


class CNNEncoder(nn.Module):
    """
    1D CNN encoder with multiple filter sizes.
    
    Applies parallel convolutions with different kernel sizes,
    then concatenates the outputs.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: List[int],
        dropout: float = 0.2,
    ):
        """
        Args:
            embedding_dim: Input embedding dimension
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes (e.g., [3, 5, 7])
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        # Create convolution layers for each filter size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs,
                padding=fs // 2,  # Same padding
            )
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension is num_filters * num_filter_sizes
        self.output_dim = num_filters * len(filter_sizes)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embeddings using CNN.
        
        Args:
            embeddings: Input of shape (batch, seq_len, embedding_dim)
            
        Returns:
            Encoded output of shape (batch, seq_len, output_dim)
        """
        # Transpose for Conv1d: (batch, embedding_dim, seq_len)
        x = embeddings.transpose(1, 2)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            h = F.relu(conv(x))  # (batch, num_filters, seq_len)
            conv_outputs.append(h)
        
        # Concatenate: (batch, output_dim, seq_len)
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Transpose back: (batch, seq_len, output_dim)
        output = concatenated.transpose(1, 2)
        
        return self.dropout(output)


class CAML(nn.Module):
    """
    CAML: Convolutional Attention for Multi-Label Classification.
    
    Full architecture for ICD code prediction from clinical text.
    
    Usage:
        model = CAML(
            vocab_size=50000,
            num_labels=50,
            embedding_dim=300,
            num_filters=256,
            filter_sizes=[3, 5, 7],
        )
        
        outputs = model(input_ids, attention_mask)
        # outputs["logits"]: (batch, num_labels)
        # outputs["attention"]: (batch, num_labels, seq_len)
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 300,
        num_filters: int = 256,
        filter_sizes: List[int] = None,
        dropout: float = 0.2,
        pad_token_id: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ):
        """
        Initialize CAML model.
        
        Args:
            vocab_size: Vocabulary size
            num_labels: Number of ICD codes to predict
            embedding_dim: Word embedding dimension
            num_filters: Number of CNN filters per filter size
            filter_sizes: List of CNN filter sizes
            dropout: Dropout probability
            pad_token_id: Padding token ID
            pretrained_embeddings: Optional pretrained embedding matrix
            freeze_embeddings: Whether to freeze embeddings
        """
        super().__init__()
        
        if filter_sizes is None:
            filter_sizes = [3, 5, 7]
        
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        
        # Word embeddings
        self.embeddings = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_token_id
        )
        
        if pretrained_embeddings is not None:
            self.embeddings.weight.data.copy_(pretrained_embeddings)
            logger.info("Loaded pretrained embeddings")
        
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False
            logger.info("Frozen embedding weights")
        
        self.embedding_dropout = nn.Dropout(dropout)
        
        # CNN encoder
        self.encoder = CNNEncoder(
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            dropout=dropout,
        )
        
        hidden_dim = self.encoder.output_dim
        
        # Label-wise attention
        self.attention = LabelWiseAttention(
            hidden_dim=hidden_dim,
            num_labels=num_labels,
        )
        
        # Output classifier
        self.classifier = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Mask of shape (batch, seq_len)
            labels: Optional labels of shape (batch, num_labels) for loss computation
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with:
                - logits: Prediction logits (batch, num_labels)
                - loss: Optional BCE loss if labels provided
                - attention: Optional attention weights (batch, num_labels, seq_len)
        """
        # Create attention mask from padding if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        
        # Embed tokens: (batch, seq_len, embedding_dim)
        embeddings = self.embeddings(input_ids)
        embeddings = self.embedding_dropout(embeddings)
        
        # Encode with CNN: (batch, seq_len, hidden_dim)
        hidden_states = self.encoder(embeddings)
        
        # Apply label-wise attention
        # context: (batch, num_labels, hidden_dim)
        # attention_weights: (batch, num_labels, seq_len)
        context, attention_weights = self.attention(hidden_states, attention_mask)
        
        # Classify each label: (batch, num_labels, 1) -> (batch, num_labels)
        logits = self.classifier(context).squeeze(-1)
        
        outputs = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            outputs["loss"] = loss
        
        if return_attention:
            outputs["attention"] = attention_weights
        
        return outputs
    
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
            outputs = self.forward(input_ids, attention_mask, return_attention=False)
            probabilities = torch.sigmoid(outputs["logits"])
            predictions = (probabilities >= threshold).float()
        
        return probabilities, predictions
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Attention weights of shape (batch, num_labels, seq_len)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, return_attention=True)
        
        return outputs["attention"]
    
    @classmethod
    def from_tokenizer(
        cls,
        tokenizer_name: str,
        num_labels: int,
        **kwargs,
    ) -> "CAML":
        """
        Create CAML model from a HuggingFace tokenizer.
        
        Args:
            tokenizer_name: Name of HuggingFace tokenizer
            num_labels: Number of labels
            **kwargs: Additional arguments for CAML
            
        Returns:
            Initialized CAML model
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        return cls(
            vocab_size=tokenizer.vocab_size,
            num_labels=num_labels,
            pad_token_id=tokenizer.pad_token_id or 0,
            **kwargs,
        )


class CAMLWithDescription(CAML):
    """
    CAML variant that incorporates ICD code descriptions.
    
    Uses label embeddings derived from ICD descriptions for improved
    attention computation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        label_embedding_dim: int = 128,
        **kwargs,
    ):
        super().__init__(vocab_size, num_labels, **kwargs)
        
        hidden_dim = self.encoder.output_dim
        
        # Replace attention with description-aware version
        self.label_embeddings = nn.Parameter(
            torch.randn(num_labels, label_embedding_dim)
        )
        
        self.label_projection = nn.Linear(label_embedding_dim, hidden_dim)
        
        # Modified attention using label embeddings
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with description-aware attention."""
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        
        # Embed and encode
        embeddings = self.embedding_dropout(self.embeddings(input_ids))
        hidden_states = self.encoder(embeddings)  # (batch, seq_len, hidden_dim)
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Project hidden states: (batch, seq_len, hidden_dim)
        projected = torch.tanh(self.W(hidden_states))
        
        # Project label embeddings: (num_labels, hidden_dim)
        label_keys = self.label_projection(self.label_embeddings)
        
        # Compute attention scores
        # projected: (batch, seq_len, hidden_dim)
        # label_keys: (num_labels, hidden_dim)
        # scores: (batch, num_labels, seq_len)
        scores = torch.einsum("bsh,lh->bls", projected, label_keys)
        
        # Apply mask
        mask = attention_mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 0.0
        )
        
        # Compute context: (batch, num_labels, hidden_dim)
        context = torch.bmm(attention_weights, hidden_states)
        
        # Classify
        logits = self.classifier(context).squeeze(-1)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            outputs["loss"] = loss
        
        if return_attention:
            outputs["attention"] = attention_weights
        
        return outputs

