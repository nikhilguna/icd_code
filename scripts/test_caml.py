#!/usr/bin/env python3
"""
Test CAML model architecture.

Tests:
- Model initialization
- Forward pass
- Backward pass
- Attention mechanism
- Prediction functionality
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from models.caml import CAML


def test_initialization(num_labels, vocab_size):
    """Test model initialization."""
    print("\n" + "="*60)
    print("TEST: Model Initialization")
    print("="*60)
    
    model = CAML(
        vocab_size=vocab_size,
        num_labels=num_labels,
        embedding_dim=128,  # Smaller for testing
        num_filters=64,
        filter_sizes=[3, 5, 7],
        dropout=0.2,
    )
    
    print(f"✓ Model initialized successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    # Check components
    assert hasattr(model, "embeddings")
    assert hasattr(model, "encoder")
    assert hasattr(model, "attention")
    assert hasattr(model, "classifier")
    
    print(f"\n✓ All components present")
    
    return model


def test_forward_pass(model, num_labels, vocab_size):
    """Test forward pass."""
    print("\n" + "="*60)
    print("TEST: Forward Pass")
    print("="*60)
    
    batch_size = 4
    seq_len = 512
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.zeros(batch_size, num_labels)
    labels[:, :5] = 1  # Set first 5 labels positive
    
    print(f"  Input shapes:")
    print(f"    input_ids: {input_ids.shape}")
    print(f"    attention_mask: {attention_mask.shape}")
    print(f"    labels: {labels.shape}")
    
    # Forward pass
    outputs = model(input_ids, attention_mask, labels, return_attention=True)
    
    print(f"\n✓ Forward pass successful")
    print(f"  Output keys: {list(outputs.keys())}")
    print(f"  Output shapes:")
    for key, value in outputs.items():
        if torch.is_tensor(value):
            print(f"    {key:15s}: {value.shape}")
        else:
            print(f"    {key:15s}: {type(value).__name__}")
    
    # Validate outputs
    assert "logits" in outputs
    assert "loss" in outputs
    assert "attention" in outputs
    
    assert outputs["logits"].shape == (batch_size, num_labels)
    assert outputs["attention"].shape == (batch_size, num_labels, seq_len)
    assert outputs["loss"].dim() == 0  # Scalar
    
    # Check value ranges
    print(f"\n  Logits range: [{outputs['logits'].min():.3f}, {outputs['logits'].max():.3f}]")
    print(f"  Loss value: {outputs['loss'].item():.4f}")
    
    assert torch.isfinite(outputs["logits"]).all(), "Non-finite values in logits"
    assert torch.isfinite(outputs["loss"]), "Non-finite loss"
    
    # Check attention sums to 1
    attention_sums = outputs["attention"].sum(dim=2)
    print(f"  Attention sums (should be ~1.0): [{attention_sums.min():.3f}, {attention_sums.max():.3f}]")
    assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-5)
    
    print(f"\n✓ All forward pass validations passed")
    
    return outputs


def test_backward_pass(model, num_labels, vocab_size):
    """Test backward pass."""
    print("\n" + "="*60)
    print("TEST: Backward Pass")
    print("="*60)
    
    batch_size = 4
    seq_len = 512
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.zeros(batch_size, num_labels)
    labels[:, :5] = 1
    
    # Forward pass
    outputs = model(input_ids, attention_mask, labels)
    loss = outputs["loss"]
    
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    print(f"\n✓ Backward pass successful")
    
    # Check gradients
    grad_norms = []
    params_with_grad = 0
    params_without_grad = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                params_with_grad += 1
            else:
                params_without_grad += 1
    
    print(f"  Parameters with gradients: {params_with_grad}")
    print(f"  Parameters without gradients: {params_without_grad}")
    
    if grad_norms:
        print(f"  Gradient norms:")
        print(f"    Mean: {sum(grad_norms) / len(grad_norms):.6f}")
        print(f"    Max: {max(grad_norms):.6f}")
        print(f"    Min: {min(grad_norms):.6f}")
    
    assert params_with_grad > 0, "No gradients computed!"
    assert all(torch.isfinite(param.grad).all() for param in model.parameters() if param.grad is not None), \
        "Non-finite gradients!"
    
    print(f"\n✓ All backward pass validations passed")
    
    # Zero gradients for next test
    model.zero_grad()
    
    return True


def test_prediction(model, num_labels, vocab_size):
    """Test prediction functionality."""
    print("\n" + "="*60)
    print("TEST: Prediction")
    print("="*60)
    
    batch_size = 4
    seq_len = 512
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Make predictions
    probabilities, predictions = model.predict(input_ids, attention_mask, threshold=0.5)
    
    print(f"✓ Predictions successful")
    print(f"  Probabilities shape: {probabilities.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    
    # Validate shapes
    assert probabilities.shape == (batch_size, num_labels)
    assert predictions.shape == (batch_size, num_labels)
    
    # Check value ranges
    assert probabilities.min() >= 0 and probabilities.max() <= 1, "Probabilities not in [0, 1]"
    assert predictions.min() >= 0 and predictions.max() <= 1, "Predictions not binary"
    assert set(predictions.unique().tolist()).issubset({0.0, 1.0}), "Predictions not binary"
    
    print(f"  Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    print(f"  Predictions range: [{predictions.min():.0f}, {predictions.max():.0f}]")
    print(f"  Predictions per sample: {predictions.sum(dim=1).tolist()}")
    
    print(f"\n✓ All prediction validations passed")
    
    return True


def test_training_steps(model, num_labels, vocab_size, num_steps=10):
    """Test multiple training steps."""
    print("\n" + "="*60)
    print("TEST: Training Steps")
    print("="*60)
    
    batch_size = 4
    seq_len = 512
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    
    print(f"  Running {num_steps} training steps...")
    
    for step in range(num_steps):
        # Create dummy batch
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.zeros(batch_size, num_labels)
        labels[:, :5] = 1
        
        # Forward
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 5 == 0:
            print(f"    Step {step:2d}: Loss = {loss.item():.4f}")
    
    print(f"\n✓ Training steps successful")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss change: {losses[-1] - losses[0]:.4f}")
    
    if losses[-1] < losses[0]:
        print(f"  Loss decreased: ✓")
    else:
        print(f"  Loss did not decrease (may be normal for random data)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test CAML model")
    parser.add_argument(
        "--num-labels",
        type=int,
        default=50,
        help="Number of labels",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30522,  # BERT vocab size
        help="Vocabulary size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()
    
    print("="*60)
    print("CAML MODEL TEST")
    print("="*60)
    print(f"Number of labels: {args.num_labels}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Device: {args.device}")
    
    try:
        # Test 1: Initialization
        model = test_initialization(args.num_labels, args.vocab_size)
        model = model.to(args.device)
        
        # Test 2: Forward pass
        test_forward_pass(model, args.num_labels, args.vocab_size)
        
        # Test 3: Backward pass
        test_backward_pass(model, args.num_labels, args.vocab_size)
        
        # Test 4: Prediction
        test_prediction(model, args.num_labels, args.vocab_size)
        
        # Test 5: Training steps
        test_training_steps(model, args.num_labels, args.vocab_size)
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✅ ALL TESTS PASSED")
    print("\nCAML model is working correctly!")
    print("\nNext steps:")
    print("  python scripts/test_led.py --num-labels", args.num_labels)
    
    # Cleanup to prevent hanging
    try:
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    import os
    os._exit(exit_code)

