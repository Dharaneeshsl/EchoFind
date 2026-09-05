# EchoFind: Shazam-Style Self-Supervised Audio Representation Learning & Music Retrieval

![Pytest](https://img.shields.io/badge/Pytest-8%2F8%20Passed-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Verified-blue)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions%20Green-brightgreen)

EchoFind is a production-grade self-supervised audio representation learning system and Shazam-style music retrieval engine. It pretrains a ResNet-18 neural network on 8,000 FMA-Small audio tracks using **SimCLR contrastive learning (NT-Xent loss)** to produce 512-dimensional normalized embeddings for instant song identification under noise.

---

## 🚀 Quick Start & Interview Demo

### 1. Interactive Streamlit Web Demo
Run the live web application to test Shazam song identification under background noise:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 2. Run Pytest Unit Test Suite
```bash
pytest tests/ --verbose
```

### 3. Docker Container Execution
Build and run the containerized test suite:
```bash
docker build -t echofind:latest .
docker run --rm echofind:latest
```

---

## 🏗️ Project Architecture

```text
EchoFind/
├── app.py                      # Interactive Streamlit Web Application
├── audio_processing.py         # Log-Mel Spectrogram extraction & audio preprocessing
├── augmentations.py            # Time/Frequency Masking & Noise augmentations
├── model.py                    # ResNet-18 Encoder & SimCLR Projection Head
├── loss.py                     # NT-Xent Contrastive Loss (AMP float16 safe)
├── train.py                    # SimCLR Training with AMP & Gradient Accumulation
├── start_training.py           # Dynamic environment verification & training runner
├── retrieval.py                # FAISS & Cosine Similarity Shazam Retrieval Engine
├── evaluate.py                 # Linear Probe & Evaluation pipeline
├── submission.py               # Singleton AudioEncoder interface
├── benchmark_retrieval.py      # Reproducible SNR & clip-length retrieval benchmark
├── ablation.py                 # Reproducible label-efficiency & SVD rank ablation
├── tests/                      # Pytest Unit Test Suite
│   └── test_pipeline.py
├── Dockerfile                  # Container build file
├── .dockerignore               # Optimized Docker context ignore
├── .github/workflows/ci.yml    # GitHub Actions CI/CD Pipeline
└── RESULTS.md                  # Comprehensive Verification & Benchmark Report
```

---

## 🏋️ Pretraining & Evaluation

### Start Training
```bash
python start_training.py
```
- **Pretraining**: SimCLR contrastive learning on 8,000 audio tracks.
- **Effective Batch Size**: 64 (Mini-batch 16 $\times$ 4 Accumulation Steps).
- **Hardware Acceleration**: Automatic Mixed Precision (AMP `fp16`) on CUDA GPU.
- **Checkpointing**: Saves checkpoints to `weights/best_model.pth` and `weights/encoder.pth`.

### Run Benchmarks & Ablations
```bash
python benchmark_retrieval.py
python ablation.py
```

---

## 🧪 Testing & CI/CD Pipeline
Every pull request and push to `main` executes:
- Environment & setup verification (`python test_setup.py`)
- Full Pytest test suite (`pytest tests/`)
- Container image build (`docker build -t echofind:latest .`)
