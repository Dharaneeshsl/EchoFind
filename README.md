# EchoFind - Self-Supervised Audio Representation Learning

Complete solution for the "Impulse 2026 – EchoFind" challenge: building a self-supervised audio encoder for Shazam-style retrieval and genre classification.

## Overview

This project implements a SimCLR-based contrastive learning system for audio representation learning:

- **SSL Method**: SimCLR (contrastive learning)
- **Encoder**: ResNet-18 adapted for spectrograms
- **Input**: Log-Mel Spectrograms
- **Embedding Dimension**: 512
- **Projection Dimension**: 128 (training only)

## Project Structure

```
EchoFind/
├── config.py              # Configuration parameters
├── audio_processing.py     # Audio preprocessing pipeline
├── augmentations.py        # Data augmentation functions
├── dataset.py             # PyTorch Dataset for contrastive learning
├── model.py               # ResNet-18 encoder and projection head
├── loss.py                # NT-Xent loss implementation
├── train.py               # Training script
├── retrieval.py           # Shazam-style retrieval system
├── evaluate.py            # Linear probe evaluation
├── visualize.py           # Visualization script
├── build_index.py         # Build retrieval index
├── submission.py          # Submission file with AudioEncoder class
├── requirements.txt       # Python dependencies
├── weights/               # Trained encoder weights (encoder.pth)
├── results/               # Evaluation results and visualizations
└── notebooks/            # Jupyter notebooks for visualization
    └── visualization.ipynb
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download FMA-Small dataset and place it in `data/fma_small/`

## Usage

### 1. Training

Train the self-supervised encoder:

```bash
python train.py
```

This will:
- Load FMA-Small dataset
- Apply augmentations (time/freq masking, noise, gain)
- Train encoder using SimCLR contrastive learning
- Save encoder weights to `weights/encoder.pth`

**Important**: Labels are NOT used during SSL training.

### 2. Build Retrieval Index

Build index of all clean audio tracks:

```bash
python build_index.py
```

This creates a FAISS index (or dictionary) for fast retrieval.

### 3. Evaluate Linear Probe

Evaluate encoder using linear probe on 10% labeled data:

```bash
python evaluate.py
```

This will:
- Extract embeddings from trained encoder
- Train linear classifier on 10% labeled data
- Report F1-score and classification metrics

### 4. Visualize Embeddings

Generate t-SNE and UMAP visualizations:

```bash
python visualize.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/visualization.ipynb
```

### 5. Retrieval

Use the retrieval system to identify tracks:

```python
from retrieval import AudioRetrievalSystem

# Initialize system
retrieval = AudioRetrievalSystem()
retrieval.build_index()

# Predict track from noisy clip
predictions = retrieval.predict_track("noisy_audio.mp3", top_k=5)
print(f"Predicted track: {predictions[0][0]}")
```

## Submission Format

The `submission.py` file contains:

- `AudioEncoder` class: Main encoder class
- `get_embedding(audio_path)`: Extract embedding from audio
- `predict_track(noisy_audio_path, database)`: Predict track ID

Example usage:

```python
from submission import AudioEncoder, get_embedding, predict_track

# Get embedding
embedding = get_embedding("audio.mp3")
print(f"Embedding shape: {embedding.shape}")

# Predict track
database = {
    "track1.mp3": embedding1,
    "track2.mp3": embedding2
}
predicted = predict_track("noisy_audio.mp3", database)
```

## Key Features

### Audio Processing
- Resample to 22050 Hz
- Convert to Log-Mel Spectrogram (128 mel bins)
- Normalize spectrograms

### Augmentations
- Random time masking
- Random frequency masking
- Additive Gaussian noise
- Random gain (volume scaling)
- Optional: pitch shift, time stretch

### Model Architecture
- ResNet-18 encoder (adapted for 1-channel input)
- Global average pooling
- L2-normalized embeddings (512-dim)
- Projection head: Linear → ReLU → Linear (128-dim, training only)

### Training
- NT-Xent loss with temperature=0.07
- Adam optimizer with cosine annealing
- Batch normalization
- Gradient stability

### Evaluation
- Linear probe on 10% labeled data
- F1-score metric
- t-SNE/UMAP visualization

## Configuration

Edit `config.py` to adjust:
- Audio processing parameters (sample rate, mel bins, etc.)
- Model architecture (embedding dim, projection dim)
- Training hyperparameters (batch size, learning rate, epochs)
- Augmentation parameters

## Notes

- **No label leakage**: Labels are only used for linear probe evaluation, never during SSL training
- **Deterministic embeddings**: Encoder outputs are L2-normalized and deterministic
- **Robust to noise**: Augmentations make the model robust to noisy queries
- **FAISS support**: Uses FAISS for fast similarity search (falls back to brute-force if unavailable)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchaudio
- librosa
- scikit-learn
- matplotlib, seaborn
- faiss-cpu (optional, for fast retrieval)
- umap-learn (optional, for visualization)

## License

This project is for the EchoFind challenge submission.
