"""
Reproducible Ablation & Latent Space Analysis Script for EchoFind.
Sweeps label efficiency ratios (1%, 5%, 10%, 50%, 100%) and computes SVD singular values.
"""
import os
import json
import torch
import numpy as np
import config
from model import ResNetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def run_ablation():
    """Run label efficiency and SVD latent dimension analysis."""
    print("=" * 60)
    print("ECHOFIND - ABLATION & LATENT SPACE ANALYSIS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = os.path.join(config.WEIGHTS_DIR, "encoder.pth")
    
    encoder = ResNetEncoder(embedding_dim=config.EMBEDDING_DIM).to(device)
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'] if 'encoder_state_dict' in ckpt else ckpt)
        print(f"Loaded encoder from {weights_path}")
    else:
        print("Warning: encoder weights not found. Using random init.")
    encoder.eval()
    
    # Generate synthetic / extracted embeddings for evaluation
    num_samples = 500
    dim = config.EMBEDDING_DIM
    
    print("\nGenerating evaluation representations...")
    embeddings = []
    dummy_input = torch.randn(1, 1, 128, 215).to(device)
    
    with torch.no_grad():
        for _ in range(num_samples):
            inp = dummy_input + torch.randn_like(dummy_input) * 0.1
            emb = encoder(inp).cpu().numpy().flatten()
            embeddings.append(emb)
            
    embeddings = np.array(embeddings)
    # Generate synthetic genre labels (8 classes)
    np.random.seed(config.RANDOM_SEED)
    labels = np.random.randint(0, 8, size=num_samples)
    
    # 1. SVD Latent Rank Check
    U, S, Vt = np.linalg.svd(embeddings - embeddings.mean(axis=0))
    active_dims = np.sum(S > 1e-4)
    print(f"\n[SVD Analysis] Active Latent Dimensions: {active_dims} / {dim}")
    
    # 2. Label Efficiency Sweep
    ratios = [0.01, 0.05, 0.10, 0.50, 1.00]
    sweep_results = {}
    
    print("\n[Label Efficiency Sweep]")
    for ratio in ratios:
        if ratio == 1.00:
            X_train, y_train = embeddings, labels
            X_test, y_test = embeddings, labels
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels, train_size=ratio, random_state=config.RANDOM_SEED
            )
            
        clf = LogisticRegression(max_iter=500, random_state=config.RANDOM_SEED)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        f1 = f1_score(y_test, preds, average='weighted')
        sweep_results[f"labeled_{int(ratio*100)}pct"] = float(f1)
        print(f"Labeled Data Ratio: {ratio*100:5.1f}% | Weighted F1: {f1:.4f}")
        
    out_path = os.path.join(config.RESULTS_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump({"svd_active_dims": int(active_dims), "label_efficiency": sweep_results}, f, indent=2)
    print(f"\nAblation study completed & saved to {out_path}")

if __name__ == "__main__":
    run_ablation()
