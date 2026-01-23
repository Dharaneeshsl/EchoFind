"""
ResNet-18 encoder adapted for audio spectrograms and projection head for contrastive learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional
import config


class ResNetEncoder(nn.Module):
    """
    ResNet-18 encoder adapted for single-channel spectrogram input.
    Outputs normalized embeddings of dimension h=512.
    """
    
    def __init__(self, embedding_dim: int = config.EMBEDDING_DIM):
        """
        Initialize ResNet-18 encoder.
        
        Args:
            embedding_dim: Dimension of output embedding (default: 512)
        """
        super(ResNetEncoder, self).__init__()
        
        # Load pretrained ResNet-18 (we'll adapt it)
        resnet = models.resnet18(pretrained=False)
        
        # Replace first conv layer to accept single channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Use ResNet layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection to embedding dimension
        self.fc = nn.Linear(512, embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize conv1 weights (since input channels changed)."""
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input spectrogram tensor of shape (batch, 1, n_mels, time_frames)
        
        Returns:
            Normalized embeddings of shape (batch, embedding_dim)
        """
        # ResNet forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Project to embedding dimension
        x = self.fc(x)
        
        # L2 normalization (critical for contrastive learning)
        x = F.normalize(x, p=2, dim=1)
        
        return x


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps embeddings to projection space (used ONLY during training).
    Architecture: Linear -> ReLU -> Linear
    """
    
    def __init__(
        self,
        input_dim: int = config.EMBEDDING_DIM,
        projection_dim: int = config.PROJECTION_DIM,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize projection head.
        
        Args:
            input_dim: Input embedding dimension (default: 512)
            projection_dim: Output projection dimension (default: 128)
            hidden_dim: Hidden layer dimension (default: input_dim)
        """
        super(ProjectionHead, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head.
        
        Args:
            x: Input embeddings of shape (batch, input_dim)
        
        Returns:
            Projected features of shape (batch, projection_dim)
        """
        return self.projection(x)


class SimCLRModel(nn.Module):
    """
    Complete SimCLR model: Encoder + Projection Head.
    """
    
    def __init__(
        self,
        embedding_dim: int = config.EMBEDDING_DIM,
        projection_dim: int = config.PROJECTION_DIM
    ):
        """
        Initialize SimCLR model.
        
        Args:
            embedding_dim: Embedding dimension (default: 512)
            projection_dim: Projection dimension (default: 128)
        """
        super(SimCLRModel, self).__init__()
        
        self.encoder = ResNetEncoder(embedding_dim)
        self.projection_head = ProjectionHead(embedding_dim, projection_dim)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input spectrogram tensor
            return_embedding: If True, return embedding instead of projection
        
        Returns:
            Projected features (or embeddings if return_embedding=True)
        """
        # Get embedding from encoder
        embedding = self.encoder(x)
        
        if return_embedding:
            return embedding
        
        # Project to projection space
        projection = self.projection_head(embedding)
        
        # Normalize projection
        projection = F.normalize(projection, p=2, dim=1)
        
        return projection
