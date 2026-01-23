# Quick Start Guide

## 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py
```

## 2. Prepare Data

Place FMA-Small dataset in `data/fma_small/`:
```
data/
└── fma_small/
    ├── track1.mp3
    ├── track2.mp3
    └── ...
```

## 3. Train Encoder

```bash
python train.py
```

This will:
- Train self-supervised encoder (NO labels used)
- Save weights to `weights/encoder.pth`
- Take ~1-2 hours depending on GPU

## 4. Build Retrieval Index

```bash
python build_index.py
```

Creates index of all clean tracks for fast retrieval.

## 5. Evaluate

```bash
# Linear probe evaluation (10% labeled data)
python evaluate.py

# Generate visualizations
python visualize.py
```

## 6. Use Submission API

```python
from submission import AudioEncoder, get_embedding, predict_track

# Get embedding
embedding = get_embedding("audio.mp3")
print(f"Shape: {embedding.shape}")  # (512,)

# Predict track from noisy clip
database = {
    "track1.mp3": embedding1,
    "track2.mp3": embedding2
}
predicted = predict_track("noisy_audio.mp3", database)
print(f"Predicted: {predicted}")
```

## Troubleshooting

### No audio files found
- Check that `data/fma_small/` contains audio files
- Supported formats: .mp3, .wav, .flac, .ogg, .m4a

### CUDA out of memory
- Reduce `BATCH_SIZE` in `config.py`
- Use CPU: set `device = torch.device('cpu')` in training script

### FAISS not available
- System will fall back to brute-force search
- Install: `pip install faiss-cpu` (optional)

### Labels not found
- Linear probe evaluation requires genre labels
- Check FMA dataset structure or adapt `load_fma_labels()` in `evaluate.py`
