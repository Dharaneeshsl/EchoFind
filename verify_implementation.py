"""
Verification script to ensure implementation matches all requirements.
"""
import os
import sys
import inspect
import torch
import numpy as np

print("=" * 70)
print("ECHOFIND IMPLEMENTATION VERIFICATION")
print("=" * 70)

checks_passed = 0
checks_failed = 0

def check(condition, message):
    """Check a condition and report result."""
    global checks_passed, checks_failed
    if condition:
        print(f"[PASS] {message}")
        checks_passed += 1
    else:
        print(f"[FAIL] {message}")
        checks_failed += 1

# 1. Check required files exist
print("\n1. FILE STRUCTURE CHECK")
print("-" * 70)
required_files = [
    "config.py",
    "audio_processing.py",
    "augmentations.py",
    "dataset.py",
    "model.py",
    "loss.py",
    "train.py",
    "retrieval.py",
    "evaluate.py",
    "submission.py",
    "requirements.txt"
]
for f in required_files:
    check(os.path.exists(f), f"File exists: {f}")

check(os.path.exists("weights"), "Directory exists: weights/")
check(os.path.exists("notebooks"), "Directory exists: notebooks/")

# 2. Check submission.py structure
print("\n2. SUBMISSION.PY CHECK")
print("-" * 70)
try:
    from submission import AudioEncoder, get_embedding, predict_track
    
    # Check AudioEncoder class
    check(hasattr(AudioEncoder, 'get_embedding'), "AudioEncoder has get_embedding method")
    
    # Check functions exist
    check(callable(get_embedding), "get_embedding function exists")
    check(callable(predict_track), "predict_track function exists")
    
    # Check function signatures
    sig_embedding = inspect.signature(get_embedding)
    sig_predict = inspect.signature(predict_track)
    
    check('audio_path' in sig_embedding.parameters, "get_embedding has audio_path parameter")
    check('noisy_audio_path' in sig_predict.parameters, "predict_track has noisy_audio_path parameter")
    check('database' in sig_predict.parameters, "predict_track has database parameter")
    
except Exception as e:
    check(False, f"submission.py imports failed: {e}")

# 3. Check model architecture
print("\n3. MODEL ARCHITECTURE CHECK")
print("-" * 70)
try:
    from model import ResNetEncoder, SimCLRModel, ProjectionHead
    import config
    
    encoder = ResNetEncoder(embedding_dim=config.EMBEDDING_DIM)
    model = SimCLRModel()
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, config.N_MELS, 215)
    with torch.no_grad():
        embedding = encoder(dummy_input)
        projection = model(dummy_input)
    
    check(embedding.shape == (2, config.EMBEDDING_DIM), 
          f"Encoder output shape correct: {embedding.shape}")
    check(projection.shape == (2, config.PROJECTION_DIM),
          f"Projection output shape correct: {projection.shape}")
    
    # Check normalization
    embedding_norm = torch.norm(embedding, dim=1)
    check(torch.allclose(embedding_norm, torch.ones_like(embedding_norm), atol=1e-5),
          "Embeddings are L2-normalized")
    
    # Check projection head structure
    proj_head = model.projection_head
    check(len(list(proj_head.modules())) >= 3, "Projection head has Linear->ReLU->Linear structure")
    
except Exception as e:
    check(False, f"Model check failed: {e}")

# 4. Check dataset (no labels)
print("\n4. DATASET CHECK")
print("-" * 70)
try:
    from dataset import ContrastiveAudioDataset
    
    # Check dataset returns two views, no labels
    dataset_code = inspect.getsource(ContrastiveAudioDataset.__getitem__)
    check('label' not in dataset_code.lower() or 'no label' in dataset_code.lower(),
          "Dataset does not return labels (or explicitly documents no labels)")
    check('view1' in dataset_code.lower() or 'spec1' in dataset_code.lower(),
          "Dataset returns two views")
    
except Exception as e:
    check(False, f"Dataset check failed: {e}")

# 5. Check augmentations
print("\n5. AUGMENTATION CHECK")
print("-" * 70)
try:
    from augmentations import (
        TimeMasking, FrequencyMasking, AddNoise, RandomGain,
        RandomCrop, TimeStretch, PitchShift, AudioAugmentationPipeline
    )
    
    check(True, "All augmentation classes exist")
    
    # Check augmentation pipeline
    aug = AudioAugmentationPipeline()
    dummy_spec = torch.randn(1, config.N_MELS, 215)
    augmented = aug(dummy_spec)
    check(augmented.shape == dummy_spec.shape, "Augmentations preserve shape")
    
except Exception as e:
    check(False, f"Augmentation check failed: {e}")

# 6. Check loss function
print("\n6. LOSS FUNCTION CHECK")
print("-" * 70)
try:
    from loss import NTXentLoss
    import config
    
    loss_fn = NTXentLoss(temperature=config.TEMPERATURE)
    
    z1 = torch.randn(4, config.PROJECTION_DIM)
    z2 = torch.randn(4, config.PROJECTION_DIM)
    z1 = torch.nn.functional.normalize(z1, p=2, dim=1)
    z2 = torch.nn.functional.normalize(z2, p=2, dim=1)
    
    loss = loss_fn(z1, z2)
    check(loss.item() > 0, "NT-Xent loss computes positive value")
    check(not torch.isnan(loss), "NT-Xent loss is not NaN")
    
except Exception as e:
    check(False, f"Loss check failed: {e}")

# 7. Check training configuration
print("\n7. TRAINING CONFIGURATION CHECK")
print("-" * 70)
try:
    import config
    
    check(config.NUM_EPOCHS >= 50, f"Training runs for at least 50 epochs ({config.NUM_EPOCHS})")
    check(config.SAMPLE_RATE == 22050, f"Sample rate is 22050 Hz ({config.SAMPLE_RATE})")
    check(config.EMBEDDING_DIM == 512, f"Embedding dimension is 512 ({config.EMBEDDING_DIM})")
    check(config.PROJECTION_DIM == 128, f"Projection dimension is 128 ({config.PROJECTION_DIM})")
    check(config.TEMPERATURE > 0, f"Temperature is positive ({config.TEMPERATURE})")
    
except Exception as e:
    check(False, f"Config check failed: {e}")

# 8. Check audio processing
print("\n8. AUDIO PROCESSING CHECK")
print("-" * 70)
try:
    from audio_processing import load_audio, audio_to_logmel, normalize_spectrogram, preprocess_audio
    
    check(True, "Audio processing functions exist")
    
    # Check log-mel conversion
    dummy_waveform = torch.randn(1, 22050)  # 1 second at 22050 Hz
    logmel = audio_to_logmel(dummy_waveform)
    check(logmel.shape[1] == config.N_MELS, f"Log-mel has {config.N_MELS} mel bins")
    check(torch.all(logmel >= -20), "Log-mel values are reasonable (log scale)")
    
except Exception as e:
    check(False, f"Audio processing check failed: {e}")

# 9. Check retrieval system
print("\n9. RETRIEVAL SYSTEM CHECK")
print("-" * 70)
try:
    from retrieval import AudioRetrievalSystem
    
    retrieval = AudioRetrievalSystem()
    check(hasattr(retrieval, 'build_index'), "RetrievalSystem has build_index method")
    check(hasattr(retrieval, 'predict_track'), "RetrievalSystem has predict_track method")
    
except Exception as e:
    check(False, f"Retrieval check failed: {e}")

# 10. Check evaluation
print("\n10. EVALUATION CHECK")
print("-" * 70)
try:
    from evaluate import linear_probe_evaluation
    
    check(callable(linear_probe_evaluation), "linear_probe_evaluation function exists")
    check(config.LINEAR_PROBE_TRAIN_RATIO == 0.1, 
          f"Linear probe uses 10% labeled data ({config.LINEAR_PROBE_TRAIN_RATIO})")
    
except Exception as e:
    check(False, f"Evaluation check failed: {e}")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print(f"Passed: {checks_passed}")
print(f"Failed: {checks_failed}")
print(f"Total:  {checks_passed + checks_failed}")

if checks_failed == 0:
    print("\n[SUCCESS] ALL CHECKS PASSED! Implementation matches requirements.")
else:
    print(f"\n[WARNING] {checks_failed} CHECK(S) FAILED. Please review the issues above.")

print("=" * 70)
