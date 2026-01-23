"""
Audio augmentation pipeline for contrastive learning.
Implements time masking, frequency masking, noise, gain, pitch shift, and time stretch.
"""
import torch
import torchaudio
import numpy as np
from typing import Tuple
import config


class TimeMasking:
    """Random time masking augmentation."""
    
    def __init__(self, time_mask_param: int = config.TIME_MASK_PARAM, num_masks: int = config.NUM_TIME_MASKS):
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply random time masking.
        
        Args:
            spectrogram: Tensor of shape (1, n_mels, time_frames)
        
        Returns:
            Masked spectrogram
        """
        spec = spectrogram.clone()
        time_frames = spec.shape[2]
        
        for _ in range(self.num_masks):
            t0 = np.random.randint(0, time_frames - self.time_mask_param)
            t1 = t0 + np.random.randint(0, self.time_mask_param)
            spec[:, :, t0:t1] = 0
        
        return spec


class FrequencyMasking:
    """Random frequency masking augmentation."""
    
    def __init__(self, freq_mask_param: int = config.FREQ_MASK_PARAM, num_masks: int = config.NUM_FREQ_MASKS):
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply random frequency masking.
        
        Args:
            spectrogram: Tensor of shape (1, n_mels, time_frames)
        
        Returns:
            Masked spectrogram
        """
        spec = spectrogram.clone()
        n_mels = spec.shape[1]
        
        for _ in range(self.num_masks):
            f0 = np.random.randint(0, n_mels - self.freq_mask_param)
            f1 = f0 + np.random.randint(0, self.freq_mask_param)
            spec[:, f0:f1, :] = 0
        
        return spec


class AddNoise:
    """Additive Gaussian noise augmentation."""
    
    def __init__(self, noise_std: float = config.NOISE_STD):
        self.noise_std = noise_std
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to spectrogram.
        
        Args:
            spectrogram: Tensor of shape (1, n_mels, time_frames)
        
        Returns:
            Noisy spectrogram
        """
        noise = torch.randn_like(spectrogram) * self.noise_std
        return spectrogram + noise


class RandomGain:
    """Random gain (volume scaling) augmentation."""
    
    def __init__(self, gain_min: float = config.GAIN_MIN, gain_max: float = config.GAIN_MAX):
        self.gain_min = gain_min
        self.gain_max = gain_max
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply random gain scaling.
        
        Args:
            spectrogram: Tensor of shape (1, n_mels, time_frames)
        
        Returns:
            Scaled spectrogram
        """
        gain = np.random.uniform(self.gain_min, self.gain_max)
        return spectrogram * gain


class PitchShift:
    """Pitch shift augmentation using librosa."""
    
    def __init__(self, n_steps: int = config.PITCH_SHIFT_RANGE):
        self.n_steps = n_steps
    
    def __call__(self, waveform: torch.Tensor, sample_rate: int = config.SAMPLE_RATE) -> torch.Tensor:
        """
        Apply pitch shift to waveform (must be applied before spectrogram conversion).
        
        Args:
            waveform: Audio tensor of shape (1, samples)
            sample_rate: Sample rate
        
        Returns:
            Pitch-shifted waveform
        """
        import librosa
        
        audio_np = waveform.squeeze(0).numpy()
        steps = np.random.randint(-self.n_steps, self.n_steps + 1)
        shifted = librosa.effects.pitch_shift(
            audio_np, sr=sample_rate, n_steps=steps
        )
        return torch.from_numpy(shifted).unsqueeze(0).float()


class RandomCrop:
    """Random time-domain cropping augmentation."""
    
    def __init__(self, crop_ratio: float = 0.8):
        """
        Initialize random crop augmentation.
        
        Args:
            crop_ratio: Ratio of original length to keep (default: 0.8, i.e., crop 20%)
        """
        self.crop_ratio = crop_ratio
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply random cropping to spectrogram (time-domain).
        
        Args:
            spectrogram: Tensor of shape (1, n_mels, time_frames)
        
        Returns:
            Cropped spectrogram
        """
        time_frames = spectrogram.shape[2]
        crop_length = int(time_frames * self.crop_ratio)
        
        if crop_length >= time_frames:
            return spectrogram
        
        # Random start position
        start = np.random.randint(0, time_frames - crop_length + 1)
        
        # Crop
        cropped = spectrogram[:, :, start:start + crop_length]
        
        # Pad back to original length (or return cropped if variable length is acceptable)
        # For consistency, we'll pad back
        if cropped.shape[2] < time_frames:
            pad_size = time_frames - cropped.shape[2]
            cropped = torch.nn.functional.pad(cropped, (0, pad_size), mode='constant', value=0)
        
        return cropped


class TimeStretch:
    """Time stretch augmentation using librosa."""
    
    def __init__(self, rate_range: Tuple[float, float] = config.TIME_STRETCH_RANGE):
        self.rate_min, self.rate_max = rate_range
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply time stretch to waveform (must be applied before spectrogram conversion).
        
        Args:
            waveform: Audio tensor of shape (1, samples)
        
        Returns:
            Time-stretched waveform
        """
        import librosa
        
        audio_np = waveform.squeeze(0).numpy()
        rate = np.random.uniform(self.rate_min, self.rate_max)
        stretched = librosa.effects.time_stretch(audio_np, rate=rate)
        return torch.from_numpy(stretched).unsqueeze(0).float()


class AudioAugmentationPipeline:
    """
    Complete augmentation pipeline for contrastive learning.
    Applies augmentations to spectrograms (time/freq masking, noise, gain, random crop).
    """
    
    def __init__(
        self,
        apply_time_mask: bool = True,
        apply_freq_mask: bool = True,
        apply_noise: bool = True,
        apply_gain: bool = True,
        apply_random_crop: bool = True,
        time_mask_param: int = config.TIME_MASK_PARAM,
        freq_mask_param: int = config.FREQ_MASK_PARAM,
        noise_std: float = config.NOISE_STD,
        gain_min: float = config.GAIN_MIN,
        gain_max: float = config.GAIN_MAX,
        num_time_masks: int = config.NUM_TIME_MASKS,
        num_freq_masks: int = config.NUM_FREQ_MASKS,
        crop_ratio: float = 0.8
    ):
        self.augmentations = []
        
        if apply_time_mask:
            self.augmentations.append(TimeMasking(time_mask_param, num_time_masks))
        if apply_freq_mask:
            self.augmentations.append(FrequencyMasking(freq_mask_param, num_freq_masks))
        if apply_noise:
            self.augmentations.append(AddNoise(noise_std))
        if apply_gain:
            self.augmentations.append(RandomGain(gain_min, gain_max))
        if apply_random_crop:
            self.augmentations.append(RandomCrop(crop_ratio))
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to spectrogram.
        
        Args:
            spectrogram: Tensor of shape (1, n_mels, time_frames)
        
        Returns:
            Augmented spectrogram
        """
        spec = spectrogram.clone()
        
        # Apply each augmentation with some probability
        for aug in self.augmentations:
            if np.random.random() > 0.5:  # 50% probability for each augmentation
                spec = aug(spec)
        
        return spec
