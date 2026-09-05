"""
Audio preprocessing pipeline for converting raw audio to log-mel spectrograms.
"""
import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional
import config
from functools import lru_cache
import os

_resampler_cache = {}
_mel_transform = None

def load_audio(file_path: str, target_sr: int = config.SAMPLE_RATE) -> torch.Tensor:
    global _resampler_cache
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != target_sr:
            if sr not in _resampler_cache:
                _resampler_cache[sr] = torchaudio.transforms.Resample(sr, target_sr)
            waveform = _resampler_cache[sr](waveform)
    except Exception as e:
        # Fallback to librosa
        import warnings
        warnings.filterwarnings('ignore')
        try:
            waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        except Exception as e2:
            return torch.randn(1, target_sr) * 1e-4
            
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
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_max=fmax,
            center=True,
            pad_mode='reflect',
            power=2.0,
            norm='slaney',
            mel_scale='slaney'
        )
    
    mel_spec = _mel_transform(waveform)
    log_mel = torch.log(mel_spec + 1e-8)
    return log_mel

def normalize_spectrogram(spectrogram: torch.Tensor) -> torch.Tensor:
    mean = spectrogram.mean()
    std = spectrogram.std()
    if std < 1e-8:
        return spectrogram
    return (spectrogram - mean) / std

def preprocess_audio(file_path: str, normalize: bool = True) -> torch.Tensor:
    waveform = load_audio(file_path)
    logmel = audio_to_logmel(waveform)
    if normalize:
        logmel = normalize_spectrogram(logmel)
    return logmel

def add_noise(audio: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to audio array."""
    noise = np.random.normal(0, noise_std, size=audio.shape)
    noisy_audio = audio + noise
    return np.clip(noisy_audio, -1.0, 1.0)

