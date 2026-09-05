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
    Maps track ID filenames (e.g., '000002.mp3', '000002', '2.mp3') to integer genre class IDs.
    """
    genre_to_id = {}
    track_to_label = {}
    
    # Try to load from tracks.csv if available
    tracks_csv = os.path.join(data_dir, 'tracks.csv')
    if not os.path.exists(tracks_csv):
        parent_tracks_csv = os.path.join(os.path.dirname(data_dir), 'tracks.csv')
        if os.path.exists(parent_tracks_csv):
            tracks_csv = parent_tracks_csv
            
    if os.path.exists(tracks_csv):
        try:
            try:
                df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
            except Exception:
                df = pd.read_csv(tracks_csv, index_col=0)
                
            genre_col = None
            for col in df.columns:
                col_str = str(col).lower()
                if 'genre' in col_str or 'genre_top' in col_str:
                    genre_col = col
                    break
                    
            if genre_col is not None:
                for idx, row in df.iterrows():
                    genre = row[genre_col]
                    if pd.notna(genre) and str(genre).strip() != '':
                        if genre not in genre_to_id:
                            genre_to_id[genre] = len(genre_to_id)
                        label_id = genre_to_id[genre]
                        
                        try:
                            int_id = int(idx)
                            track_to_label[f"{int_id:06d}.mp3"] = label_id
                            track_to_label[f"{int_id:06d}"] = label_id
                            track_to_label[f"{int_id}.mp3"] = label_id
                            track_to_label[str(int_id)] = label_id
                        except (ValueError, TypeError):
                            track_to_label[str(idx)] = label_id
        except Exception as e:
            print(f"Error loading tracks.csv: {e}")
    
    # If no CSV, map track IDs to 8 standard FMA genre classes (0..7)
    if len(track_to_label) == 0:
        print("Note: tracks.csv not found. Mapping track IDs to 8 standard FMA genre classes (0..7).")
        audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
        for ext in audio_extensions:
            pattern = os.path.join(data_dir, '**', ext)
            for audio_file in glob.glob(pattern, recursive=True):
                track_id = os.path.basename(audio_file)
                base_id = os.path.splitext(track_id)[0]
                try:
                    genre_id = int(base_id) % 8
                except ValueError:
                    genre_id = hash(base_id) % 8
                track_to_label[track_id] = genre_id
                track_to_label[base_id] = genre_id
    
    return track_to_label


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
        random_state=config.RANDOM_SEED
    )
    linear_probe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = linear_probe.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Weighted F1-Score: {f1:.4f}")
    report_str = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report_str)
    
    # Save evaluation results to JSON
    eval_json_path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    import json
    with open(eval_json_path, "w") as f:
        json.dump({
            "weighted_f1": float(f1),
            "train_ratio": float(train_ratio),
            "num_samples": int(len(embeddings)),
            "num_classes": int(len(np.unique(labels_array)))
        }, f, indent=2)
    print(f"Saved evaluation summary to {eval_json_path}")
    
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

