"""
Reproducible Benchmark Script for EchoFind Retrieval System.
Evaluates Shazam-style audio retrieval recall across SNR noise levels and clip lengths.
"""
import os
import glob
import json
import torch
import numpy as np
import soundfile as sf
import config
from retrieval import AudioRetrievalSystem
from audio_processing import add_noise

def run_retrieval_benchmark(num_samples: int = 50):
    """
    Run empirical retrieval recall evaluation.
    
    Args:
        num_samples: Number of query tracks to sample for benchmark.
    """
    print("=" * 60)
    print("ECHOFIND - RETRIEVAL RECALL BENCHMARK")
    print("=" * 60)
    
    weights_path = os.path.join(config.WEIGHTS_DIR, "encoder.pth")
    system = AudioRetrievalSystem(encoder_path=weights_path)
    
    audio_files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*", "*.mp3")))
    if len(audio_files) == 0:
        print(f"No audio files found in {config.DATA_DIR}")
        return
        
    print(f"Indexing database of {min(500, len(audio_files))} tracks...")
    index_files = audio_files[:min(500, len(audio_files))]
    
    # Build database
    system.database = {}
    system.track_ids = []
    with torch.no_grad():
        for f in index_files:
            try:
                from audio_processing import preprocess_audio
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
                
    if system.faiss_index is None and len(system.database) > 0:
        try:
            import faiss
            emb_arr = np.array(list(system.database.values())).astype('float32')
            system.faiss_index = faiss.IndexFlatIP(emb_arr.shape[1])
            system.faiss_index.add(emb_arr)
        except ImportError:
            pass

    snr_levels = [20, 10, 5, 0]  # dB
    clip_lengths = [2.0, 5.0, 10.0]  # seconds
    
    results = {}
    eval_tracks = index_files[:min(num_samples, len(index_files))]
    
    print("\nRunning Evaluation Across Noise Levels (SNR) & Clip Lengths...\n")
    for snr in snr_levels:
        results[f"snr_{snr}dB"] = {}
        # Convert SNR to noise std estimate
        noise_std = 10.0 ** (-snr / 20.0) * 0.1
        
        for length in clip_lengths:
            correct_top1 = 0
            correct_top5 = 0
            total = 0
            
            for track_file in eval_tracks:
                target_id = os.path.basename(track_file)
                try:
                    import librosa
                    y, sr = librosa.load(track_file, sr=config.SAMPLE_RATE, duration=length)
                    y_noisy = add_noise(y, noise_std=noise_std)
                    
                    tmp_query = os.path.join(config.RESULTS_DIR, "bench_query.wav")
                    sf.write(tmp_query, y_noisy, sr)
                    
                    preds = system.predict_track(tmp_query, top_k=5)
                    top1_match = (preds[0][0] == target_id)
                    top5_match = any(p[0] == target_id for p in preds)
                    
                    if top1_match:
                        correct_top1 += 1
                    if top5_match:
                        correct_top5 += 1
                    total += 1
                except Exception:
                    continue
                    
            r1 = (correct_top1 / total) * 100 if total > 0 else 0.0
            r5 = (correct_top5 / total) * 100 if total > 0 else 0.0
            results[f"snr_{snr}dB"][f"len_{length}s"] = {"recall_at_1": r1, "recall_at_5": r5}
            print(f"SNR {snr:2d}dB | Clip {length:4.1f}s | Recall@1: {r1:5.1f}% | Recall@5: {r5:5.1f}%")
            
    # Save benchmark json
    out_path = os.path.join(config.RESULTS_DIR, "retrieval_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark completed & saved to {out_path}")

if __name__ == "__main__":
    run_retrieval_benchmark(num_samples=20)
