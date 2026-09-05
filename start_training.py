"""
Start training script with dynamic environment & dataset verification.
"""
import os
import sys
import glob
import torch
import subprocess
import config

print("=" * 70)
print("ECHOFIND - STARTING TRAINING")
print("=" * 70)
print("\nDataset & Environment Verification:")

# Dynamic audio file count
audio_files = glob.glob(os.path.join(config.DATA_DIR, "**", "*.mp3"), recursive=True)
num_files = len(audio_files)
if num_files > 0:
    print(f"[OK] Found {num_files} audio files in {config.DATA_DIR}")
else:
    print(f"[WARNING] No audio files found in {config.DATA_DIR}")

# Hardware acceleration check
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"[OK] CUDA GPU acceleration available: {device_name}")
else:
    print("[INFO] CUDA GPU unavailable. Training will run on CPU.")

print(f"[OK] Training hyperparameter NUM_EPOCHS = {config.NUM_EPOCHS}")
print(f"[OK] Training hyperparameter BATCH_SIZE = {config.BATCH_SIZE}")
print("\nStarting training process...\n")
print("=" * 70)

# Run training
sys.exit(subprocess.run([sys.executable, "train.py"]).returncode)
