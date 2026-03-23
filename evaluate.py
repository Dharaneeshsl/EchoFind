"""
Evaluation script for linear probe and semantic verification.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import os
import glob
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Dict, Optional
import config
from model import ResNetEncoder
from audio_processing import preprocess_audio


class LabeledAudioDataset(Dataset):
    """Dataset for evaluation with labels."""
    
    def __init__(self, audio_files, labels, track_ids):
        self.audio_files = audio_files
        self.labels = labels
        self.track_ids = track_ids
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        track_id = self.track_ids[idx]
        
        # Preprocess (same as training, but NO augmentation)
        spectrogram = preprocess_audio(audio_path, normalize=True)
        
        # Pad or crop to consistent length
        target_frames = int(5 * config.SAMPLE_RATE / config.HOP_LENGTH)
        if spectrogram.shape[2] < target_frames:
            pad_size = target_frames - spectrogram.shape[2]
            spectrogram = torch.nn.functional.pad(
                spectrogram, (0, pad_size), mode='constant', value=0
            )
        elif spectrogram.shape[2] > target_frames:
            start = (spectrogram.shape[2] - target_frames) // 2
            spectrogram = spectrogram[:, :, start:start + target_frames]
        
        return spectrogram, label, track_id


def extract_embeddings(
    encoder: ResNetEncoder,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract embeddings from dataset.
    
    Returns:
        embeddings, labels, track_ids
    """
    encoder.eval()
    embeddings = []
    labels = []
    track_ids = []
    
    with torch.no_grad():
        for spectrogram, label, track_id in tqdm(dataloader, desc="Extracting embeddings"):
            spectrogram = spectrogram.to(device)
            embedding = encoder(spectrogram)
            embeddings.append(embedding.cpu().numpy())
            labels.extend(label.numpy() if isinstance(label, torch.Tensor) else label)
            track_ids.extend(track_id)
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    track_ids = np.array(track_ids)
    
    return embeddings, labels, track_ids


def load_fma_labels(data_dir: str = config.DATA_DIR) -> Dict[str, int]:
    """
    Load genre labels from FMA dataset.
    This is a placeholder - you'll need to adapt based on actual FMA structure.
    """
    # FMA-Small structure: tracks.csv contains genre labels
    # This is a simplified version - adapt to your actual FMA structure
    labels = {}
    
    # Try to load from tracks.csv if available
    tracks_csv = os.path.join(data_dir, 'tracks.csv')
    if os.path.exists(tracks_csv):
        try:
            df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
            # FMA structure: ('track', 'genre_top') column contains genre
            if ('track', 'genre_top') in df.columns:
                for idx, row in df.iterrows():
                    genre = row[('track', 'genre_top')]
                    if pd.notna(genre):
                        # Convert genre to integer label
                        if genre not in labels:
                            labels[genre] = len(labels)
                        labels[idx] = labels[genre]
        except Exception as e:
            print(f"Error loading tracks.csv: {e}")
    
    # If no CSV, try to infer from directory structure
    if len(labels) == 0:
        print("Warning: Could not load labels from CSV. Using directory structure.")
        # FMA-Small might have genre folders
        genre_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        genre_to_label = {genre: i for i, genre in enumerate(sorted(genre_dirs))}
        
        audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
        for ext in audio_extensions:
            pattern = os.path.join(data_dir, '**', ext)
            for audio_file in glob.glob(pattern, recursive=True):
                # Try to extract genre from path
                rel_path = os.path.relpath(audio_file, data_dir)
                parts = rel_path.split(os.sep)
                if len(parts) > 1 and parts[0] in genre_to_label:
                    track_id = os.path.basename(audio_file)
                    labels[track_id] = genre_to_label[parts[0]]
    
    return labels


def linear_probe_evaluation(
    encoder_path: str = os.path.join(config.WEIGHTS_DIR, 'encoder.pth'),
    data_dir: str = config.DATA_DIR,
    train_ratio: float = config.LINEAR_PROBE_TRAIN_RATIO
):
    """
    Evaluate encoder using linear probe on genre classification.
    Uses only train_ratio (default 10%) of labeled data.
    """
    print("=" * 60)
    print("Linear Probe Evaluation")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load encoder
    encoder = ResNetEncoder(embedding_dim=config.EMBEDDING_DIM).to(device)
    if os.path.exists(encoder_path):
        checkpoint = torch.load(encoder_path, map_location=device)
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            encoder.load_state_dict(checkpoint)
        print(f"Loaded encoder from {encoder_path}")
    else:
        raise FileNotFoundError(f"Encoder not found at {encoder_path}")
    
    encoder.eval()
    
    # Load labels
    print("Loading labels...")
    label_dict = load_fma_labels(data_dir)
    
    if len(label_dict) == 0:
        print("Warning: No labels found. Skipping linear probe evaluation.")
        return None
    
    # Find audio files with labels
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
    audio_files = []
    labels = []
    track_ids = []
    
    for ext in audio_extensions:
        pattern = os.path.join(data_dir, '**', ext)
        for audio_file in glob.glob(pattern, recursive=True):
            track_id = os.path.basename(audio_file)
            if track_id in label_dict:
                audio_files.append(audio_file)
                labels.append(label_dict[track_id])
                track_ids.append(track_id)
    
    print(f"Found {len(audio_files)} labeled audio files")
    
    if len(audio_files) == 0:
        print("No labeled files found. Skipping evaluation.")
        return None
    
    # Create dataset
    dataset = LabeledAudioDataset(audio_files, labels, track_ids)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, labels_array, track_ids_array = extract_embeddings(encoder, dataloader, device)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels_array.shape}")
    print(f"Number of classes: {len(np.unique(labels_array))}")
    
    # Split data (10% for training, 90% for testing)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels_array,
        test_size=1.0 - train_ratio,
        random_state=config.RANDOM_SEED
    )
    
    print(f"Training set: {len(X_train)} samples ({train_ratio*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({(1-train_ratio)*100:.1f}%)")
    
    # Train linear probe
    print("Training linear probe...")
    linear_probe = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_SEED,
        multi_class='ovr'  # One-vs-rest for multi-class
    )
    linear_probe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = linear_probe.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Weighted F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'f1_score': f1,
        'embeddings': embeddings,
        'labels': labels_array,
        'track_ids': track_ids_array,
        'linear_probe': linear_probe
    }


if __name__ == "__main__":
    results = linear_probe_evaluation()
    if results:
        print(f"\nLinear probe F1-score: {results['f1_score']:.4f}")
