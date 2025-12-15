#!/usr/bin/env python3
"""
Test Longformer classifier.

Tests:
- Model initialization with pretrained Longformer
- Forward pass with different sequence lengths
- Global attention mechanism
- Backward pass
- Memory usage with long sequences
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from models.longformer_classifier import LongformerClassifier


def test_initialization(num_labels, model_name="allenai/longformer-base-4096"):
    """Test model initialization."""
    print("\n" + "="*60)
    print("TEST: Model Initialization")
    print("="*60)
    
    print(f"  Loading pretrained Longformer: {model_name}")
    print(f"  This may take a minute on first run...")
    
    model = LongformerClassifier(
        num_labels=num_labels,
        model_name=model_name,
        freeze_layers=0,  # Don't freeze for testing
        dropout=0.1,
    )
    
    print(f"\n✓ Model initialized successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    # Check components
    assert hasattr(model, "longformer")
    assert hasattr(model, "classifier")
    
    print(f"\n✓ All components present")
    
    return model


def test_forward_pass_short(model, num_labels):
    """Test forward pass with short sequences."""
    print("\n" + "="*60)
    print("TEST: Forward Pass (Short Sequences)")
    print("="*60)
    
    batch_size = 2
    seq_len = 512
    
    # Create dummy input
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))  # Longformer vocab size
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.zeros(batch_size, num_labels)
    labels[:, :5] = 1
    
    print(f"  Input shapes:")
    print(f"    input_ids: {input_ids.shape}")
    print(f"    attention_mask: {attention_mask.shape}")
    print(f"    labels: {labels.shape}")
    
    # Forward pass
    outputs = model(input_ids, attention_mask, labels=labels)
    
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
    assert "hidden_states" in outputs
    
    assert outputs["logits"].shape == (batch_size, num_labels)
    assert outputs["hidden_states"].shape == (batch_size, 768)  # Longformer hidden dim
    assert outputs["loss"].dim() == 0  # Scalar
    
    # Check value ranges
    print(f"\n  Logits range: [{outputs['logits'].min():.3f}, {outputs['logits'].max():.3f}]")
    print(f"  Loss value: {outputs['loss'].item():.4f}")
    
    assert torch.isfinite(outputs["logits"]).all(), "Non-finite values in logits"
    assert torch.isfinite(outputs["loss"]), "Non-finite loss"
    
    print(f"\n✓ All forward pass validations passed")
    
    return outputs


def test_forward_pass_long(model, num_labels, seq_len=4096):
    """Test forward pass with long sequences (Longformer's strength)."""
    print("\n" + "="*60)
    print(f"TEST: Forward Pass (Long Sequences, length={seq_len})")
    print("="*60)
    
    batch_size = 1  # Smaller batch for long sequences
    
    # Create dummy input
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Testing Longformer's ability to handle long documents...")
    
    import time
    start = time.time()
    
    # Forward pass
    outputs = model(input_ids, attention_mask)
    
    elapsed = time.time() - start
    
    print(f"\n✓ Long sequence forward pass successful")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    # Check memory usage
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  Peak GPU memory: {memory_mb:.1f} MB")
        torch.cuda.reset_peak_memory_stats()
    
    assert outputs["logits"].shape == (batch_size, num_labels)
    
    print(f"\n✓ Long sequence handling validated")
    
    return True


def test_global_attention(model, num_labels):
    """Test global attention mechanism."""
    print("\n" + "="*60)
    print("TEST: Global Attention Mechanism")
    print("="*60)
    
    batch_size = 2
    seq_len = 512
    
    # Create dummy input
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Test global attention mask creation
    global_attention_mask = model._create_global_attention_mask(input_ids, attention_mask)
    
    print(f"  Global attention mask shape: {global_attention_mask.shape}")
    print(f"  Global attention mask:\n{global_attention_mask}")
    
    # Check that first token (CLS) has global attention
    assert global_attention_mask[0, 0] == 1, "CLS token should have global attention"
    assert global_attention_mask.sum() == batch_size, "Only CLS tokens should have global attention"
    
    print(f"\n✓ Global attention mask correctly created")
    print(f"  CLS tokens have global attention: ✓")
    
    # Test forward with explicit global attention
    outputs = model(input_ids, attention_mask, global_attention_mask=global_attention_mask)
    
    assert "logits" in outputs
    print(f"\n✓ Forward pass with global attention successful")
    
    return True


def test_backward_pass(model, num_labels):
    """Test backward pass."""
    print("\n" + "="*60)
    print("TEST: Backward Pass")
    print("="*60)
    
    batch_size = 2
    seq_len = 512
    
    # Create dummy input
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.zeros(batch_size, num_labels)
    labels[:, :5] = 1
    
    # Forward pass
    outputs = model(input_ids, attention_mask, labels=labels)
    loss = outputs["loss"]
    
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    print(f"\n✓ Backward pass successful")
    
    # Check gradients
    grad_norms = []
    params_with_grad = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            params_with_grad += 1
    
    print(f"  Parameters with gradients: {params_with_grad}")
    
    if grad_norms:
        print(f"  Gradient norms:")
        print(f"    Mean: {sum(grad_norms) / len(grad_norms):.6f}")
        print(f"    Max: {max(grad_norms):.6f}")
    
    assert params_with_grad > 0, "No gradients computed!"
    
    print(f"\n✓ All backward pass validations passed")
    
    # Zero gradients
    model.zero_grad()
    
    return True


def test_training_steps(model, num_labels, num_steps=5):
    """Test multiple training steps."""
    print("\n" + "="*60)
    print("TEST: Training Steps")
    print("="*60)
    
    batch_size = 2
    seq_len = 512
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    losses = []
    
    print(f"  Running {num_steps} training steps...")
    
    for step in range(num_steps):
        # Create dummy batch
        input_ids = torch.randint(0, 50265, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.zeros(batch_size, num_labels)
        labels[:, :5] = 1
        
        # Forward
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs["loss"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        print(f"    Step {step:2d}: Loss = {loss.item():.4f}")
    
    print(f"\n✓ Training steps successful")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss change: {losses[-1] - losses[0]:.4f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Longformer model")
    parser.add_argument(
        "--num-labels",
        type=int,
        default=50,
        help="Number of labels",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="allenai/longformer-base-4096",
        help="Longformer model name",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length to test (512 or 4096)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()
    
    print("="*60)
    print("LONGFORMER MODEL TEST")
    print("="*60)
    print(f"Number of labels: {args.num_labels}")
    print(f"Model: {args.model_name}")
    print(f"Max length: {args.max_length}")
    print(f"Device: {args.device}")
    
    if args.max_length > 512 and args.device == "cpu":
        print("\n⚠️  Warning: Testing long sequences on CPU will be slow!")
    
    try:
        # Test 1: Initialization
        model = test_initialization(args.num_labels, args.model_name)
        model = model.to(args.device)
        
        # Test 2: Forward pass (short)
        test_forward_pass_short(model, args.num_labels)
        
        # Test 3: Global attention
        test_global_attention(model, args.num_labels)
        
        # Test 4: Backward pass
        test_backward_pass(model, args.num_labels)
        
        # Test 5: Training steps
        test_training_steps(model, args.num_labels)
        
        # Test 6: Long sequences (if requested)
        if args.max_length > 512:
            test_forward_pass_long(model, args.num_labels, seq_len=args.max_length)
        
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
    print("\nLongformer model is working correctly!")
    print("\nNext steps:")
    print("  python scripts/test_training.py --model led --data-dir mock_data/raw/")
    
    # Cleanup to prevent hanging
    try:
        # Delete model to free memory and close threads
        del model
        # Force garbage collection
        import gc
        gc.collect()
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    # Force exit to prevent thread hanging
    import os
    os._exit(exit_code)

