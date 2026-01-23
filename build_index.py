"""
Script to build retrieval index from FMA-Small dataset.
"""
import os
import config
from retrieval import AudioRetrievalSystem


def main():
    """Build retrieval index."""
    print("=" * 60)
    print("Building Retrieval Index")
    print("=" * 60)
    
    # Initialize retrieval system
    retrieval_system = AudioRetrievalSystem()
    
    # Build index
    retrieval_system.build_index(
        data_dir=config.DATA_DIR,
        use_faiss=True
    )
    
    # Validate index was built successfully
    if len(retrieval_system.database) == 0:
        print("ERROR: No tracks were indexed!")
        print(f"Please check that audio files exist in {config.DATA_DIR}")
        import sys
        sys.exit(1)
    
    # Save index
    index_path = os.path.join(config.RESULTS_DIR, 'retrieval_index.pkl')
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    retrieval_system.save_index(index_path)
    
    print(f"\nIndex built and saved to {index_path}")
    print(f"Total tracks indexed: {len(retrieval_system.database)}")


if __name__ == "__main__":
    main()
