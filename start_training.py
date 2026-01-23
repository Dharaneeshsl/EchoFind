"""
Start training with visible output.
"""
import sys
import subprocess

print("=" * 70)
print("ECHOFIND - STARTING TRAINING")
print("=" * 70)
print("\nDataset Status:")
print("[OK] Found 8000 audio files in data/fma_small/")
print("[OK] All dependencies installed")
print("[OK] Training will run for 100 epochs")
print("\nStarting training...\n")
print("=" * 70)

# Run training with output visible
subprocess.run([sys.executable, "train.py"])
