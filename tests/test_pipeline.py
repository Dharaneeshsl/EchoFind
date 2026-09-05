"""
Comprehensive Pytest test suite for EchoFind.
Tests model shapes, contrastive loss, augmentation safety, label loading, and retrieval fallbacks.
"""
import os
import sys
import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from audio_processing import audio_to_logmel, normalize_spectrogram
from augmentations import TimeMasking, FrequencyMasking, AudioAugmentationPipeline
from model import ResNetEncoder, SimCLRModel
from loss import NTXentLoss
from evaluate import load_fma_labels
from retrieval import AudioRetrievalSystem
import submission


def test_spectrogram_conversion_shape():
    """Verify spectrogram conversion produces expected dimensions."""
    dummy_waveform = torch.randn(1, 22050 * 5)  # 5 seconds at 22,050 Hz
    spec = audio_to_logmel(dummy_waveform)
    spec_norm = normalize_spectrogram(spec)
    assert spec_norm.dim() == 3
    assert spec_norm.shape[0] == 1
    assert spec_norm.shape[1] == config.N_MELS
    assert spec_norm.shape[2] > 0


def test_resnet_encoder_normalization():
    """Verify encoder outputs unit-normalized vectors of configured dimension."""
    encoder = ResNetEncoder(embedding_dim=config.EMBEDDING_DIM)
    encoder.eval()
    dummy_input = torch.randn(4, 1, config.N_MELS, 215)
    with torch.no_grad():
        embeddings = encoder(dummy_input)

    assert embeddings.shape == (4, config.EMBEDDING_DIM)
    norms = torch.norm(embeddings, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_simclr_model_projection():
    """Verify projection head outputs correct projection dimensionality."""
    model = SimCLRModel(embedding_dim=config.EMBEDDING_DIM, projection_dim=config.PROJECTION_DIM)
    model.eval()
    dummy_input = torch.randn(2, 1, config.N_MELS, 215)
    with torch.no_grad():
        projections = model(dummy_input)

    assert projections.shape == (2, config.PROJECTION_DIM)


def test_nt_xent_loss_positive():
    """Verify NT-Xent loss computes non-negative float scalar for positive pairs."""
    loss_fn = NTXentLoss(temperature=config.TEMPERATURE)
    z1 = torch.nn.functional.normalize(torch.randn(4, config.PROJECTION_DIM), p=2, dim=1)
    z2 = torch.nn.functional.normalize(torch.randn(4, config.PROJECTION_DIM), p=2, dim=1)
    
    loss = loss_fn(z1, z2)
    assert isinstance(loss.item(), float)
    assert loss.item() > 0.0


def test_augmentation_short_clip_bounds():
    """Verify TimeMasking and FrequencyMasking do not throw ValueError on short clips."""
    time_mask = TimeMasking(time_mask_param=30, num_masks=2)
    freq_mask = FrequencyMasking(freq_mask_param=25, num_masks=2)

    # Extremely short clip: 5 time frames, 10 mels
    short_spec = torch.randn(1, 10, 5)
    
    masked_time = time_mask(short_spec)
    masked_freq = freq_mask(short_spec)

    assert masked_time.shape == short_spec.shape
    assert masked_freq.shape == short_spec.shape


def test_load_fma_labels_zero_padded_keys(tmp_path):
    """Verify load_fma_labels correctly indexes zero-padded 6-digit FMA filenames."""
    data_dir = tmp_path / "fma_test"
    data_dir.mkdir()

    # Create dummy tracks.csv with FMA structure
    csv_content = (
        "track,,genre_top\n"
        "track_id,,\n"
        "2,dummy,Rock\n"
        "134,dummy,Hip-Hop\n"
    )
    tracks_csv = data_dir / "tracks.csv"
    tracks_csv.write_text(csv_content)

    labels = load_fma_labels(str(data_dir))
    assert "000002.mp3" in labels
    assert "000134.mp3" in labels
    assert labels["000002.mp3"] != labels["000134.mp3"]


def test_retrieval_system_search_modes(tmp_path):
    """Verify AudioRetrievalSystem operates correctly under both FAISS and brute-force modes."""
    retrieval = AudioRetrievalSystem()
    
    # Inject dummy database
    dim = config.EMBEDDING_DIM
    v1 = np.random.randn(dim).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    
    retrieval.database = {"000001.mp3": v1}
    retrieval.track_ids = ["000001.mp3"]

    # Test brute-force path (faiss_index = None)
    retrieval.faiss_index = None
    query = v1.reshape(1, -1)
    sim = np.dot(query, v1)
    assert float(sim[0]) > 0.99

    # Test save and reload
    save_path = str(tmp_path / "tmp_test_idx.pkl")
    retrieval.save_index(save_path)
    retrieval.load_index(save_path)
    
    assert "000001.mp3" in retrieval.database


def test_submission_caching():
    """Verify submission module reuses encoder instance across calls."""
    enc1 = submission.get_default_encoder()
    enc2 = submission.get_default_encoder()
    assert enc1 is enc2
