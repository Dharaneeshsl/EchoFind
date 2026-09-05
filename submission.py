"""
Submission file for EchoFind challenge.
Contains AudioEncoder class and required functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import config
from model import ResNetEncoder
from audio_processing import preprocess_audio


class AudioEncoder:
    """
    Audio encoder for self-supervised representation learning.
    Produces deterministic, normalized embeddings.
    """
    
    def __init__(
        self,
        encoder_path: str = os.path.join("weights", "encoder.pth"),
        device: Optional[torch.device] = None
    ):
        """
        Initialize audio encoder.
        
        Args:
            encoder_path: Path to trained encoder weights
            device: PyTorch device (auto-detect if None)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Initialize encoder
        self.encoder = ResNetEncoder(embedding_dim=config.EMBEDDING_DIM).to(self.device)
        
        # Load weights if available
        if os.path.exists(encoder_path):
            checkpoint = torch.load(encoder_path, map_location=self.device)
            if 'encoder_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            else:
                self.encoder.load_state_dict(checkpoint)
        else:
            print(f"Warning: Encoder weights not found at {encoder_path}")
            print("Using randomly initialized encoder.")
        
        self.encoder.eval()
    
    def get_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract embedding from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Normalized embedding vector of shape (embedding_dim,)
        """
        # Preprocess audio
        spectrogram = preprocess_audio(audio_path, normalize=True)
        
        # Pad or crop to consistent length (5 seconds)
        if config.HOP_LENGTH == 0:
            raise ValueError("HOP_LENGTH cannot be zero in config")
        target_frames = int(5 * config.SAMPLE_RATE / config.HOP_LENGTH)
        
        if spectrogram.shape[2] < target_frames:
            pad_size = target_frames - spectrogram.shape[2]
            spectrogram = torch.nn.functional.pad(
                spectrogram, (0, pad_size), mode='constant', value=0
            )
        elif spectrogram.shape[2] > target_frames:
            start = (spectrogram.shape[2] - target_frames) // 2
            spectrogram = spectrogram[:, :, start:start + target_frames]
        
        # Extract embedding
        with torch.no_grad():
            spectrogram = spectrogram.unsqueeze(0).to(self.device)
            embedding = self.encoder(spectrogram)
            embedding_np = embedding.cpu().numpy().flatten()
        
        return embedding_np


_DEFAULT_ENCODER: Optional[AudioEncoder] = None
_DEFAULT_ENCODER_PATH: Optional[str] = None

def get_default_encoder(encoder_path: str = os.path.join("weights", "encoder.pth")) -> AudioEncoder:
    """Helper to reuse loaded encoder instance across query calls."""
    global _DEFAULT_ENCODER, _DEFAULT_ENCODER_PATH
    if _DEFAULT_ENCODER is None or _DEFAULT_ENCODER_PATH != encoder_path:
        _DEFAULT_ENCODER = AudioEncoder(encoder_path=encoder_path)
        _DEFAULT_ENCODER_PATH = encoder_path
    return _DEFAULT_ENCODER


def get_embedding(audio_path: str, encoder_path: str = os.path.join("weights", "encoder.pth")) -> np.ndarray:
    """
    Standalone function to get embedding from audio file.
    
    Args:
        audio_path: Path to audio file
        encoder_path: Path to encoder weights
    
    Returns:
        Normalized embedding vector
    """
    encoder = get_default_encoder(encoder_path=encoder_path)
    return encoder.get_embedding(audio_path)


def predict_track(
    noisy_audio_path: str,
    database: Dict[str, np.ndarray],
    encoder_path: str = os.path.join("weights", "encoder.pth")
) -> str:
    """
    Predict track ID from noisy audio clip using cosine similarity.
    
    Args:
        noisy_audio_path: Path to noisy audio clip
        database: Dictionary mapping track_id -> embedding
        encoder_path: Path to encoder weights
    
    Returns:
        Predicted track ID
    """
    encoder = get_default_encoder(encoder_path=encoder_path)
    query_embedding = encoder.get_embedding(noisy_audio_path)
    
    # Find best match using cosine similarity
    best_track_id = None
    best_similarity = -1.0
    
    for track_id, db_embedding in database.items():
        # Cosine similarity (embeddings are normalized)
        similarity = np.dot(query_embedding, db_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_track_id = track_id
    
    return best_track_id


# Example usage
if __name__ == "__main__":
    # Example: get embedding
    # embedding = get_embedding("path/to/audio.mp3")
    # print(f"Embedding shape: {embedding.shape}")
    
    # Example: predict track
    # database = {
    #     "track1.mp3": np.random.randn(512),
    #     "track2.mp3": np.random.randn(512)
    # }
    # predicted = predict_track("path/to/noisy_audio.mp3", database)
    # print(f"Predicted track: {predicted}")
    pass
