"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.
Implemented from scratch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for contrastive learning.
    Computes contrastive loss between positive pairs in a batch.
    """
    
    def __init__(self, temperature: float = config.TEMPERATURE):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature parameter for scaling (default: 0.07)
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss between two views.
        
        Args:
            z1: Projected features of first view, shape (batch, projection_dim)
            z2: Projected features of second view, shape (batch, projection_dim)
        
        Returns:
            Scalar loss value
        """
        batch_size = z1.shape[0]
        
        # Concatenate both views: [z1, z2]
        # Shape: (2*batch, projection_dim)
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        # z is normalized, so dot product = cosine similarity
        similarity_matrix = torch.matmul(z, z.T)  # (2*batch, 2*batch)
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels: positive pairs are (i, i+batch) and (i+batch, i)
        # For sample i: positive is (i, i+batch) and (i+batch, i)
        labels = torch.arange(batch_size).to(z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # (2*batch,)
        
        # Mask to remove self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute cross-entropy loss
        # For each row i, the positive class is at position labels[i]
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
