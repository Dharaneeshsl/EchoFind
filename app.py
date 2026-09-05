"""
EchoFind Interactive Streamlit Application.
Self-Supervised Audio Representation & Shazam-Style Retrieval Demo.
"""
import os
import glob
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display

import config
from retrieval import AudioRetrievalSystem
from audio_processing import preprocess_audio, add_noise

st.set_page_config(
    page_title="EchoFind - Self-Supervised Music Retrieval",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 EchoFind: Shazam-Style Audio Representation & Retrieval")
st.markdown("""
**Powered by SimCLR Self-Supervised Contrastive Learning (ResNet-18)**  
*Trained on 8,000 FMA-Small Audio Tracks with PyTorch AMP Mixed Precision.*
""")

@st.cache_resource
def load_retrieval_system():
    weights_path = os.path.join(config.WEIGHTS_DIR, "encoder.pth")
    system = AudioRetrievalSystem(encoder_path=weights_path)
    audio_files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*", "*.mp3")))
    if len(audio_files) > 0:
        # Index 200 demo tracks for instant 1-second startup
        demo_files = audio_files[:200]
        system.track_ids = []
        embeddings = []
        with torch.no_grad():
            for f in demo_files:
                try:
                    spec = preprocess_audio(f, normalize=True)
                    target_frames = int(5 * config.SAMPLE_RATE / config.HOP_LENGTH)
                    if spec.shape[2] < target_frames:
                        spec = torch.nn.functional.pad(spec, (0, target_frames - spec.shape[2]))
                    elif spec.shape[2] > target_frames:
                        spec = spec[:, :, :target_frames]
                    spec = spec.unsqueeze(0).to(system.device)
                    emb = system.encoder(spec).cpu().numpy().flatten()
                    track_id = os.path.basename(f)
                    system.database[track_id] = emb
                    system.track_ids.append(track_id)
                except Exception:
                    continue
    return system, audio_files

with st.spinner("Initializing Model & Embedding Index (Fast Load)..."):
    system, audio_files = load_retrieval_system()

st.sidebar.header("🕹️ Demo Controls")
st.sidebar.write(f"**Database Size**: {len(system.track_ids)} tracks")

tab1, tab2, tab3 = st.tabs(["🔍 Audio Identification Demo", "📊 Model Metrics & Benchmark", "📄 Architecture Details"])

with tab1:
    st.header("Test Song Identification under Noise")
    
    if len(audio_files) == 0:
        st.warning("No audio files found in `data/fma_small/`. Please ensure audio files exist for live retrieval demo.")
    else:
        selected_file = st.selectbox("Select a track from dataset:", audio_files[:50])
        track_name = os.path.basename(selected_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Track")
            st.audio(selected_file)
            
            # Spectrogram plot
            spec = preprocess_audio(selected_file, normalize=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            librosa.display.specshow(spec.squeeze(0).numpy(), sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH, ax=ax)
            ax.set_title("Clean Spectrogram")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Noisy Query Clip")
            noise_std = st.slider("Background Noise Intensity (Std)", 0.0, 0.1, 0.02, 0.005)
            
            # Add noise & plot
            y, sr = librosa.load(selected_file, sr=config.SAMPLE_RATE, duration=5.0)
            y_noisy = add_noise(y, noise_std=noise_std)
            
            # Save temporary query
            temp_query = os.path.join(config.RESULTS_DIR, "temp_query.wav")
            import soundfile as sf
            sf.write(temp_query, y_noisy, sr)
            st.audio(temp_query)
            
            spec_noisy = preprocess_audio(temp_query, normalize=True)
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            librosa.display.specshow(spec_noisy.squeeze(0).numpy(), sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH, ax=ax2)
            ax2.set_title(f"Noisy Spectrogram (Noise Std = {noise_std})")
            st.pyplot(fig2)

        if st.button("🚀 Identify Song (Query Retrieval)"):
            with st.spinner("Extracting SimCLR 512-D Embedding & Searching Database..."):
                results = system.predict_track(temp_query, top_k=5)
                
                st.subheader("Top 5 Identified Matches:")
                matched = False
                clean_target = os.path.splitext(os.path.basename(track_name))[0]
                for rank, (pred_id, sim) in enumerate(results, 1):
                    clean_pred = os.path.splitext(os.path.basename(pred_id))[0]
                    is_match = (clean_pred == clean_target)
                    if is_match:
                        matched = True
                        st.success(f"**Rank {rank}**: `{pred_id}` — Similarity Score: **{sim:.4f}** (EXACT MATCH ✓)")
                    else:
                        st.write(f"**Rank {rank}**: `{pred_id}` — Similarity Score: {sim:.4f}")
                
                if matched:
                    st.balloons()

with tab2:
    st.header("Self-Supervised Learning Empirical Benchmarks")
    eval_file = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    bench_file = os.path.join(config.RESULTS_DIR, "retrieval_benchmark.json")
    ablation_file = os.path.join(config.RESULTS_DIR, "ablation_results.json")
    
    import json
    eval_data = json.load(open(eval_file)) if os.path.exists(eval_file) else None
    bench_data = json.load(open(bench_file)) if os.path.exists(bench_file) else None
    ablation_data = json.load(open(ablation_file)) if os.path.exists(ablation_file) else None

    # Summary Metrics Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        if eval_data and "weighted_f1" in eval_data:
            st.metric("Linear Probe Weighted F1", f"{eval_data['weighted_f1']:.4f}", help="Linear probe evaluated on 10% labeled data")
        else:
            st.metric("Linear Probe Weighted F1", "Running...", help="Evaluation script running")
    with m2:
        if bench_data and "20dB_noise" in bench_data and "recall@1" in bench_data["20dB_noise"]:
            st.metric("Shazam Recall@1 (20dB Noise)", f"{bench_data['20dB_noise']['recall@1']*100:.1f}%")
        else:
            st.metric("Shazam Recall@1 (20dB Noise)", "N/A")
    with m3:
        if ablation_data and "svd_active_dims" in ablation_data:
            st.metric("SVD Active Latent Dims", f"{ablation_data['svd_active_dims']} / {config.EMBEDDING_DIM}")
        else:
            st.metric("SVD Active Latent Dims", "N/A")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📊 Linear Probe & Classification (`evaluation_results.json`)")
        if eval_data:
            st.json(eval_data)
        else:
            st.info("Run `python evaluate.py` to generate linear probe evaluation metrics.")
            
        st.subheader("🎯 Shazam Retrieval Recall Benchmark (`retrieval_benchmark.json`)")
        if bench_data:
            st.json(bench_data)
        else:
            st.info("Run `python benchmark_retrieval.py` to generate Shazam retrieval metrics.")

    with col_b:
        st.subheader("🧪 SVD Latent Rank & Label Efficiency (`ablation_results.json`)")
        if ablation_data:
            st.json(ablation_data)
        else:
            st.info("Run `python ablation.py` to generate ablation metrics.")


with tab3:
    st.header("System Architecture")
    st.markdown("""
    ```
    +------------------+     +------------------------+     +-----------------------+
    | Clean Audio Track| --> | Log-Mel Spectrogram    | --> | ResNet-18 Encoder     |
    +------------------+     | (128 Mels, 22050 Hz)   |     | (512-D Embedding)     |
                             +------------------------+     +-----------------------+
                                                                        |
                                                                        v
                                                            +-----------------------+
                                                            | FAISS / Cosine Index  |
                                                            +-----------------------+
    ```
    """)
