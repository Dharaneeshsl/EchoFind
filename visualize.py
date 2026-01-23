"""
Visualization script for t-SNE and UMAP embeddings.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from typing import Optional
import config
from evaluate import linear_probe_evaluation

# Check UMAP availability
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. UMAP visualization will be skipped.")


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    save_path: Optional[str] = None
):
    """
    Visualize embeddings using t-SNE or UMAP.
    
    Args:
        embeddings: Embedding matrix of shape (n_samples, embedding_dim)
        labels: Label array of shape (n_samples,)
        method: 'tsne' or 'umap'
        save_path: Path to save plot (optional)
    """
    print(f"Visualizing embeddings using {method.upper()}...")
    
    # Reduce dimensionality
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=config.RANDOM_SEED, perplexity=30)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method.lower() == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is not installed. Install it with: pip install umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=config.RANDOM_SEED)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and colors
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.6,
            s=20
        )
    
    plt.title(f'Embedding Visualization ({method.upper()})', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.savefig(
            os.path.join(config.RESULTS_DIR, f'embeddings_{method}.png'),
            dpi=300,
            bbox_inches='tight'
        )
        print(f"Plot saved to {os.path.join(config.RESULTS_DIR, f'embeddings_{method}.png')}")
    
    plt.close()


def main():
    """Main visualization function."""
    # Run linear probe evaluation to get embeddings
    print("Running linear probe evaluation to extract embeddings...")
    results = linear_probe_evaluation()
    
    if results is None:
        print("Could not extract embeddings. Skipping visualization.")
        return
    
    embeddings = results['embeddings']
    labels = results['labels']
    
    # Subsample if too many points (for faster visualization)
    if len(embeddings) > 5000:
        print(f"Subsampling from {len(embeddings)} to 5000 points for visualization...")
        indices = np.random.choice(len(embeddings), 5000, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Visualize with t-SNE
    visualize_embeddings(
        embeddings,
        labels,
        method='tsne',
        save_path=os.path.join(config.RESULTS_DIR, 'embeddings_tsne.png')
    )
    
    # Visualize with UMAP
    try:
        visualize_embeddings(
            embeddings,
            labels,
            method='umap',
            save_path=os.path.join(config.RESULTS_DIR, 'embeddings_umap.png')
        )
    except Exception as e:
        print(f"UMAP visualization failed: {e}")
    
    print("Visualization completed!")


if __name__ == "__main__":
    main()
