"""
Test script to verify setup and basic functionality.
"""
import torch
import numpy as np
import os
import sys

print("=" * 60)
print("EchoFind Setup Test")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    import torch
    import torchaudio
    import librosa
    import numpy as np
    import sklearn
    import matplotlib
    print("   ✓ All required packages imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test config
print("\n2. Testing configuration...")
try:
    import config
    print(f"   ✓ Config loaded: SAMPLE_RATE={config.SAMPLE_RATE}, EMBEDDING_DIM={config.EMBEDDING_DIM}")
except Exception as e:
    print(f"   ✗ Config error: {e}")
    sys.exit(1)

# Test model creation
print("\n3. Testing model creation...")
try:
    from model import ResNetEncoder, SimCLRModel
    encoder = ResNetEncoder(embedding_dim=config.EMBEDDING_DIM)
    model = SimCLRModel()
    print(f"   ✓ Model created successfully")
    print(f"   ✓ Encoder output dim: {config.EMBEDDING_DIM}")
except Exception as e:
    print(f"   ✗ Model creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test forward pass
print("\n4. Testing forward pass...")
try:
    # Create dummy spectrogram (batch=2, channels=1, n_mels=128, time=215)
    dummy_input = torch.randn(2, 1, config.N_MELS, 215)
    with torch.no_grad():
        embedding = encoder(dummy_input)
        projection = model(dummy_input)
    
    assert embedding.shape == (2, config.EMBEDDING_DIM), f"Expected embedding shape (2, {config.EMBEDDING_DIM}), got {embedding.shape}"
    assert projection.shape == (2, config.PROJECTION_DIM), f"Expected projection shape (2, {config.PROJECTION_DIM}), got {projection.shape}"
    
    # Check normalization
    embedding_norm = torch.norm(embedding, dim=1)
    assert torch.allclose(embedding_norm, torch.ones_like(embedding_norm), atol=1e-5), "Embeddings not normalized!"
    
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Embedding shape: {embedding.shape}")
    print(f"   ✓ Embeddings are normalized: {embedding_norm.mean().item():.6f}")
except Exception as e:
    print(f"   ✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test loss
print("\n5. Testing NT-Xent loss...")
try:
    from loss import NTXentLoss
    loss_fn = NTXentLoss(temperature=config.TEMPERATURE)
    
    # Create dummy projections
    z1 = torch.randn(4, config.PROJECTION_DIM)
    z2 = torch.randn(4, config.PROJECTION_DIM)
    z1 = torch.nn.functional.normalize(z1, p=2, dim=1)
    z2 = torch.nn.functional.normalize(z2, p=2, dim=1)
    
    loss = loss_fn(z1, z2)
    assert loss.item() > 0, "Loss should be positive"
    print(f"   ✓ Loss computed successfully: {loss.item():.4f}")
except Exception as e:
    print(f"   ✗ Loss error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test augmentations
print("\n6. Testing augmentations...")
try:
    from augmentations import AudioAugmentationPipeline
    aug = AudioAugmentationPipeline()
    
    dummy_spec = torch.randn(1, config.N_MELS, 215)
    augmented = aug(dummy_spec)
    assert augmented.shape == dummy_spec.shape, "Augmentation changed shape!"
    print(f"   ✓ Augmentations work correctly")
except Exception as e:
    print(f"   ✗ Augmentation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test directory structure
print("\n7. Testing directory structure...")
try:
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(f"   ✓ Directories created: {config.WEIGHTS_DIR}, {config.RESULTS_DIR}")
except Exception as e:
    print(f"   ✗ Directory creation error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! Setup is correct.")
print("=" * 60)
print("\nNext steps:")
print("1. Place FMA-Small dataset in data/fma_small/")
print("2. Run: python train.py")
print("3. Run: python build_index.py")
print("4. Run: python evaluate.py")
print("5. Run: python visualize.py")
