"""
Configuration file for EchoFind SSL training and evaluation.
"""
import os

# Audio processing
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMAX = 11025  # Nyquist frequency for 22050 Hz

# Model architecture
EMBEDDING_DIM = 512
PROJECTION_DIM = 128

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.07  # NT-Xent temperature
NUM_WORKERS = 0  # Set to 0 on Windows to avoid multiprocessing issues

# Augmentation parameters
TIME_MASK_PARAM = 20  # Maximum time masking length
FREQ_MASK_PARAM = 20  # Maximum frequency masking length
NUM_TIME_MASKS = 2
NUM_FREQ_MASKS = 2
NOISE_STD = 0.01
GAIN_MIN = 0.5
GAIN_MAX = 1.5
PITCH_SHIFT_RANGE = 2  # semitones
TIME_STRETCH_RANGE = (0.8, 1.2)

# Paths
DATA_DIR = "data/fma_small"
WEIGHTS_DIR = "weights"
RESULTS_DIR = "results"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Evaluation
LINEAR_PROBE_TRAIN_RATIO = 0.1  # 10% labeled data for linear probe
RANDOM_SEED = 42
