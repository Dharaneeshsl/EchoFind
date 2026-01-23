"""
Shazam-style retrieval system for identifying songs from noisy clips.
"""
import torch
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional
import config
from model import ResNetEncoder
from audio_processing import preprocess_audio
import faiss


class AudioRetrievalSystem:
    """
    Retrieval system for identifying tracks from noisy audio clips.
    """
    
    def __init__(
        self,
        encoder_path: str = os.path.join(config.WEIGHTS_DIR, 'encoder.pth'),
        device: Optional[torch.device] = None
    ):
        """
        Initialize retrieval system.
        
        Args:
            encoder_path: Path to trained encoder weights
            device: PyTorch device (auto-detect if None)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load encoder
        self.encoder = ResNetEncoder(embedding_dim=config.EMBEDDING_DIM).to(self.device)
        
        if os.path.exists(encoder_path):
            checkpoint = torch.load(encoder_path, map_location=self.device)
            if 'encoder_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            else:
                self.encoder.load_state_dict(checkpoint)
            print(f"Loaded encoder from {encoder_path}")
        else:
            print(f"Warning: Encoder weights not found at {encoder_path}")
            print("Using randomly initialized encoder.")
        
        self.encoder.eval()
        
        # Database: track_id -> embedding
        self.database: Dict[str, np.ndarray] = {}
        
        # FAISS index (optional, for faster search)
        self.faiss_index = None
        self.track_ids = []
    
    def build_index(
        self,
        data_dir: str = config.DATA_DIR,
        use_faiss: bool = True
    ):
        """
        Build index of all clean audio tracks.
        
        Args:
            data_dir: Directory containing FMA-Small audio files
            use_faiss: Whether to use FAISS for fast search
        """
        print("Building retrieval index...")
        
        # Find all audio files
        audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            pattern = os.path.join(data_dir, '**', ext)
            audio_files.extend(glob.glob(pattern, recursive=True))
        
        audio_files = sorted(audio_files)
        print(f"Found {len(audio_files)} audio files")
        
        # Extract embeddings
        embeddings = []
        self.track_ids = []
        
        with torch.no_grad():
            for audio_file in audio_files:
                try:
                    # Preprocess audio
                    spectrogram = preprocess_audio(audio_file, normalize=True)
                    
                    # Pad or crop to consistent length (e.g., 5 seconds)
                    # Assuming ~43 frames per second at hop_length=512, 22050 Hz
                    target_frames = int(5 * config.SAMPLE_RATE / config.HOP_LENGTH)  # ~215 frames
                    
                    if spectrogram.shape[2] < target_frames:
                        # Pad
                        pad_size = target_frames - spectrogram.shape[2]
                        spectrogram = torch.nn.functional.pad(
                            spectrogram, (0, pad_size), mode='constant', value=0
                        )
                    elif spectrogram.shape[2] > target_frames:
                        # Crop (take middle portion)
                        start = (spectrogram.shape[2] - target_frames) // 2
                        spectrogram = spectrogram[:, :, start:start + target_frames]
                    
                    # Add batch dimension
                    spectrogram = spectrogram.unsqueeze(0).to(self.device)
                    
                    # Extract embedding
                    embedding = self.encoder(spectrogram)
                    embedding_np = embedding.cpu().numpy().flatten()
                    
                    # Store
                    track_id = os.path.basename(audio_file)
                    self.database[track_id] = embedding_np
                    embeddings.append(embedding_np)
                    self.track_ids.append(track_id)
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        print(f"Indexed {len(self.database)} tracks")
        
        # Build FAISS index if requested
        if use_faiss and len(embeddings) > 0:
            try:
                embeddings_array = np.array(embeddings).astype('float32')
                dimension = embeddings_array.shape[1]
                
                # Use Inner Product index (for normalized vectors, IP = cosine similarity)
                # This is more efficient and accurate than L2 for normalized embeddings
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.faiss_index.add(embeddings_array)
                
                print(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
            except ImportError:
                print("FAISS not available, using brute-force search")
                self.faiss_index = None
    
    def predict_track(
        self,
        noisy_audio_path: str,
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Predict track ID from noisy audio clip.
        
        Args:
            noisy_audio_path: Path to noisy audio clip
            top_k: Number of top predictions to return
        
        Returns:
            List of (track_id, similarity_score) tuples, sorted by similarity
        """
        if len(self.database) == 0:
            raise ValueError("Database not built. Call build_index() first.")
        
        # Preprocess noisy audio with error handling
        try:
            spectrogram = preprocess_audio(noisy_audio_path, normalize=True)
        except Exception as e:
            raise ValueError(f"Failed to process audio file {noisy_audio_path}: {e}")
        
        # Pad or crop to consistent length
        target_frames = int(5 * config.SAMPLE_RATE / config.HOP_LENGTH)
        
        if spectrogram.shape[2] < target_frames:
            pad_size = target_frames - spectrogram.shape[2]
            spectrogram = torch.nn.functional.pad(
                spectrogram, (0, pad_size), mode='constant', value=0
            )
        elif spectrogram.shape[2] > target_frames:
            start = (spectrogram.shape[2] - target_frames) // 2
            spectrogram = spectrogram[:, :, start:start + target_frames]
        
        # Extract embedding
        with torch.no_grad():
            spectrogram = spectrogram.unsqueeze(0).to(self.device)
            query_embedding = self.encoder(spectrogram)
            query_embedding_np = query_embedding.cpu().numpy().flatten()
        
        # Search
        if self.faiss_index is not None:
            # Use FAISS with Inner Product (cosine similarity for normalized vectors)
            query_array = query_embedding_np.reshape(1, -1).astype('float32')
            similarities, indices = self.faiss_index.search(query_array, top_k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.track_ids):
                    track_id = self.track_ids[idx]
                    # For normalized vectors, inner product = cosine similarity
                    # FAISS IndexFlatIP returns similarities directly
                    results.append((track_id, float(similarity)))
        else:
            # Brute-force search
            similarities = []
            for track_id, db_embedding in self.database.items():
                # Cosine similarity (embeddings are normalized)
                similarity = np.dot(query_embedding_np, db_embedding)
                similarities.append((track_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            results = similarities[:top_k]
        
        return results
    
    def save_index(self, index_path: str):
        """Save index to disk."""
        import pickle
        
        index_data = {
            'database': self.database,
            'track_ids': self.track_ids
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str):
        """Load index from disk."""
        import pickle
        
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.database = index_data['database']
        self.track_ids = index_data['track_ids']
        
        print(f"Index loaded from {index_path}")
        print(f"Loaded {len(self.database)} tracks")
