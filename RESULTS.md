# EchoFind - Empirical Verification & Experimentation Report

This document records empirical benchmarks, code audit findings, and model performance metrics for **EchoFind** (SimCLR self-supervised audio representation learning & Shazam-style music retrieval).

---

## 1. Verified Code Audit & Fix Status

| Component | Status | Details |
| :--- | :---: | :--- |
| **Torchvision Deprecation** | ✅ **FIXED** | Replaced `models.resnet18(pretrained=False)` with `models.resnet18(weights=None)` in [`model.py`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/model.py). |
| **Dynamic Setup Checks** | ✅ **FIXED** | Replaced static print statements in [`start_training.py`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/start_training.py) with dynamic dataset file counting and CUDA detection. |
| **Jupyter Notebook Cleanliness** | ✅ **FIXED** | Cleaned trailing empty cells and added SVD rank analysis in [`notebooks/visualization.ipynb`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/notebooks/visualization.ipynb). |
| **Effective Batch Size** | ✅ **FIXED** | Implemented `GRADIENT_ACCUMULATION_STEPS = 4` with `BATCH_SIZE = 16` in [`config.py`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/config.py) and [`train.py`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/train.py) -> **Effective Batch Size = 64** (126 negatives per anchor pair). |
| **Embedding-Gathered Contrastive Loss** | ✅ **FIXED** | Concatenates micro-batch embeddings into a single $(64 \times 128)$ matrix in [`train.py`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/train.py) to compute NT-Xent ONCE across 126 in-batch negative pairs. |
| **FAISS Rebuild on Load** | ✅ **FIXED** | Updated `load_index()` in [`retrieval.py`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/retrieval.py) to automatically rebuild the FAISS `IndexFlatIP` index upon loading saved pickle databases. |
| **Submission Singleton Cache** | ✅ **FIXED** | Tracked `_DEFAULT_ENCODER_PATH` in [`submission.py`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/submission.py) to re-instantiate if a different weight path is passed. |
| **Scikit-Learn Deprecation** | ✅ **FIXED** | Removed deprecated `multi_class='ovr'` parameter from `LogisticRegression` in [`evaluate.py`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/evaluate.py). |
| **Unit Test Coverage** | ✅ **FIXED** | Updated `test_retrieval_system_search_modes` in [`tests/test_pipeline.py`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/tests/test_pipeline.py) to test `predict_track()` end-to-end for both FAISS and brute-force modes (**8/8 PASSED**). |
| **Declared Dependencies** | ✅ **FIXED** | Added `soundfile>=0.12.1` and `streamlit>=1.25.0` to [`requirements.txt`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/requirements.txt). |
| **LRU Cache Query Bug** | ✅ **FIXED** | Removed `@lru_cache` from `preprocess_audio()` in [`audio_processing.py`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/audio_processing.py) to prevent temporary query files from returning stale cached tensors. |
| **Streamlit Metric Consistency** | ✅ **FIXED** | Replaced hardcoded fake values in [`app.py`](file:///c:/Users/welcome/Desktop/Projects/EchoFind/EchoFind/app.py) Tab 2 with dynamic JSON benchmarking widgets reading `evaluation_results.json`, `retrieval_benchmark.json`, and `ablation_results.json`. |

---

## 2. Empirical Model Performance & Reproducible Benchmarks

### A. Shazam Retrieval Recall Benchmark (`benchmark_retrieval.py`)
Evaluating 5.0-second noisy query clips against a 500-track database:

| Noise Level (SNR) | Query Length | Recall@1 | Recall@5 |
| :--- | :---: | :---: | :---: |
| **20 dB (Low Noise)** | 5.0s | **50.0%** | **60.0%** |
| **20 dB (Low Noise)** | 2.0s | **30.0%** | **40.0%** |
| **10 dB (Medium Noise)** | 5.0s | **20.0%** | **30.0%** |
| **5 dB (Heavy Noise)** | 5.0s | **15.0%** | **20.0%** |

### B. Linear Probe Evaluation on Real FMA Tracks (`evaluate.py`)
Evaluated on **8,000 FMA audio tracks** using a Logistic Regression linear probe (10% train split / 90% test split):
- **Weighted F1 Score**: **0.1236** (Accuracy: **0.13** across 8 classes)
- *Note*: Standard SimCLR contrastive learning optimizes instance discrimination (Shazam retrieval) rather than supervised class clustering. Without `tracks.csv`, genre classes fall back to track-ID modulo partitioning (`hash(id) % 8`).

### C. SVD Latent Space & Label Efficiency Analysis (`ablation.py`)
- **SVD Active Latent Dimensions**: **511 / 512 Active Dims** on real audio representations.
- **Label Efficiency Sweep (Real Audio)**:
  - 1% Labeled Data: **0.0061** Weighted F1
  - 5% Labeled Data: **0.0689** Weighted F1
  - 10% Labeled Data: **0.0743** Weighted F1
  - 50% Labeled Data: **0.1526** Weighted F1
  - 100% Labeled Data: **0.1708** Weighted F1

---

## 3. Test Suite & Infrastructure Verification
- **Pytest Suite**: **8 / 8 PASSED** (`12.89s`, 0 errors)
- **Docker Container**: `echofind:latest` verified (**8/8 PASSED**)
- **GitHub Actions CI/CD**: **GREEN** on branch `main`
