"""
Custom PyTorch Dataset for self-supervised contrastive learning.
Returns two different augmented views of the same audio sample (NO LABELS).
"""
import torch
from torch.utils.data import Dataset
import os
import glob
from typing import List, Optional
import config
from audio_processing import preprocess_audio
from augmentations import AudioAugmentationPipeline, PitchShift, TimeStretch
import numpy as np


class ContrastiveAudioDataset(Dataset):
    """
    Dataset for contrastive learning that returns two augmented views.
    NO LABELS are returned during SSL training.
    """
    
    def __init__(
        self,
        data_dir: str = config.DATA_DIR,
        augment: bool = True,
        use_pitch_shift: bool = False,
        use_time_stretch: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing FMA-Small audio files
            augment: Whether to apply augmentations
            use_pitch_shift: Whether to use pitch shift (applied to waveform)
            use_time_stretch: Whether to use time stretch (applied to waveform)
        """
        self.data_dir = data_dir
        self.augment = augment
        self.use_pitch_shift = use_pitch_shift
        self.use_time_stretch = use_time_stretch
        
        # Find all audio files
        self.audio_files = self._find_audio_files(data_dir)
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
        
        # Initialize augmentation pipeline
        if augment:
            self.spectrogram_aug = AudioAugmentationPipeline()
            if use_pitch_shift:
                self.pitch_shift = PitchShift()
            if use_time_stretch:
                self.time_stretch = TimeStretch()
        else:
            self.spectrogram_aug = None
    
    def _find_audio_files(self, data_dir: str) -> List[str]:
        """Find all audio files in directory tree."""
        audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            pattern = os.path.join(data_dir, '**', ext)
            audio_files.extend(glob.glob(pattern, recursive=True))
        
        return sorted(audio_files)
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get two augmented views of the same audio sample.
        
        Returns:
            Tuple of (view1, view2) - both are spectrograms of shape (1, n_mels, time_frames)
        """
        for _ in range(10):
            try:
                audio_path = self.audio_files[idx]
                
                # Load and preprocess audio
                if self.augment and (self.use_pitch_shift or self.use_time_stretch):
                    # For pitch shift and time stretch, we need to work with waveform first
                    import torchaudio
                    import librosa
                    
                    try:
                        waveform, sr = torchaudio.load(audio_path)
                        if sr != config.SAMPLE_RATE:
                            resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
                            waveform = resampler(waveform)
                    except:
                        waveform, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
                        waveform = torch.from_numpy(waveform).unsqueeze(0)
                    
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Apply waveform-level augmentations randomly
                    if self.use_pitch_shift and np.random.random() > 0.5:
                        waveform = self.pitch_shift(waveform)
                    if self.use_time_stretch and np.random.random() > 0.5:
                        waveform = self.time_stretch(waveform)
                    
                    # Convert to spectrogram
                    from audio_processing import audio_to_logmel, normalize_spectrogram
                    spec1 = audio_to_logmel(waveform)
                    spec1 = normalize_spectrogram(spec1)
                    
                    # Create second view (may apply different waveform augmentations)
                    waveform2 = waveform.clone()
                    if self.use_pitch_shift and np.random.random() > 0.5:
                        waveform2 = self.pitch_shift(waveform2)
                    if self.use_time_stretch and np.random.random() > 0.5:
                        waveform2 = self.time_stretch(waveform2)
                    
                    spec2 = audio_to_logmel(waveform2)
                    spec2 = normalize_spectrogram(spec2)
                else:
                    # Standard path: load directly to spectrogram
                    spec1 = preprocess_audio(audio_path, normalize=True).clone()
                    spec2 = preprocess_audio(audio_path, normalize=True).clone()
                
                # Apply spectrogram-level augmentations
                if self.augment and self.spectrogram_aug is not None:
                    spec1 = self.spectrogram_aug(spec1)
                    spec2 = self.spectrogram_aug(spec2)
                
                # Ensure consistent shape (pad or crop to fixed length if needed)
                # For now, we'll use variable length and handle in collate function
                return spec1, spec2
            except Exception as e:
                import numpy as np
                idx = np.random.randint(0, len(self.audio_files))
                
        dummy = torch.zeros(1, config.N_MELS, 215)
        return dummy, dummy


def collate_fn(batch):
    """
    Custom collate function to handle variable-length spectrograms.
    Pads to the maximum length in the batch.
    """
    view1_list, view2_list = zip(*batch)
    
    # Find maximum time dimension
    max_time = max(spec.shape[2] for spec in view1_list + view2_list)
    
    # Pad all spectrograms to max_time
    padded_view1 = []
    padded_view2 = []
    
    for spec1, spec2 in zip(view1_list, view2_list):
        # Pad view1
        if spec1.shape[2] < max_time:
            pad_size = max_time - spec1.shape[2]
            spec1 = torch.nn.functional.pad(spec1, (0, pad_size), mode='constant', value=0)
        elif spec1.shape[2] > max_time:
            spec1 = spec1[:, :, :max_time]
        
        # Pad view2
        if spec2.shape[2] < max_time:
            pad_size = max_time - spec2.shape[2]
            spec2 = torch.nn.functional.pad(spec2, (0, pad_size), mode='constant', value=0)
        elif spec2.shape[2] > max_time:
            spec2 = spec2[:, :, :max_time]
        
        padded_view1.append(spec1)
        padded_view2.append(spec2)
    
    # Stack into batches
    batch_view1 = torch.stack(padded_view1, dim=0)  # (batch, 1, n_mels, time)
    batch_view2 = torch.stack(padded_view2, dim=0)  # (batch, 1, n_mels, time)
    
    return batch_view1, batch_view2
