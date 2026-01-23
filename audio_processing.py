"""
Audio preprocessing pipeline for converting raw audio to log-mel spectrograms.
"""
import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional
import config


def load_audio(file_path: str, target_sr: int = config.SAMPLE_RATE) -> torch.Tensor:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 22050 Hz)
    
    Returns:
        Audio waveform tensor of shape (1, samples)
    """
    try:
        # Try torchaudio first
        waveform, sr = torchaudio.load(file_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
    except (RuntimeError, OSError, Exception) as e:
        # Fallback to librosa with error logging
        try:
            waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio file {file_path}: torchaudio error: {e}, librosa error: {e2}")
    
    # Ensure mono channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform


def audio_to_logmel(
    waveform: torch.Tensor,
    sample_rate: int = config.SAMPLE_RATE,
    n_mels: int = config.N_MELS,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    fmax: Optional[float] = config.FMAX
) -> torch.Tensor:
    """
    Convert audio waveform to log-mel spectrogram.
    
    Args:
        waveform: Audio tensor of shape (1, samples)
        sample_rate: Sample rate of audio
        n_mels: Number of mel filter banks
        n_fft: FFT window size
        hop_length: Hop length for STFT
        fmax: Maximum frequency (None for default)
    
    Returns:
        Log-mel spectrogram tensor of shape (1, n_mels, time_frames)
    """
    # Convert to numpy if needed
    if isinstance(waveform, torch.Tensor):
        audio_np = waveform.squeeze(0).numpy()
    else:
        audio_np = waveform
    
    # Compute mel spectrogram using librosa
    mel_spec = librosa.feature.melspectrogram(
        y=audio_np,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax
    )
    
    # Convert to log scale (add small epsilon to avoid log(0))
    log_mel = np.log(mel_spec + 1e-8)
    
    # Convert to tensor and add channel dimension
    log_mel_tensor = torch.from_numpy(log_mel).float().unsqueeze(0)
    
    return log_mel_tensor


def normalize_spectrogram(spectrogram: torch.Tensor) -> torch.Tensor:
    """
    Normalize spectrogram to zero mean and unit variance.
    
    Args:
        spectrogram: Spectrogram tensor of shape (1, n_mels, time_frames)
    
    Returns:
        Normalized spectrogram
    """
    # Compute mean and std across frequency and time dimensions
    mean = spectrogram.mean()
    std = spectrogram.std()
    
    # Avoid division by zero
    if std < 1e-8:
        return spectrogram
    
    normalized = (spectrogram - mean) / std
    return normalized


def preprocess_audio(file_path: str, normalize: bool = True) -> torch.Tensor:
    """
    Complete preprocessing pipeline: load -> resample -> log-mel -> normalize.
    
    Args:
        file_path: Path to audio file
        normalize: Whether to normalize the spectrogram
    
    Returns:
        Preprocessed log-mel spectrogram tensor
    """
    # Load and resample
    waveform = load_audio(file_path)
    
    # Convert to log-mel spectrogram
    logmel = audio_to_logmel(waveform)
    
    # Normalize if requested
    if normalize:
        logmel = normalize_spectrogram(logmel)
    
    return logmel
