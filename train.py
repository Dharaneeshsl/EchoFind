"""
Training script for self-supervised contrastive learning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import config
from dataset import ContrastiveAudioDataset, collate_fn
from model import SimCLRModel
from loss import NTXentLoss


def set_seed(seed: int = config.RANDOM_SEED):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for view1, view2 in pbar:
        # Move to device
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Forward pass
        z1 = model(view1)
        z2 = model(view2)
        
        # Compute loss
        loss = criterion(z1, z2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for view1, view2 in tqdm(dataloader, desc="Validation"):
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            z1 = model(view1)
            z2 = model(view2)
            
            loss = criterion(z1, z2)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train():
    """Main training function."""
    # Set seed
    set_seed()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    train_dataset = ContrastiveAudioDataset(
        data_dir=config.DATA_DIR,
        augment=True,
        use_pitch_shift=False,  # Can enable if needed
        use_time_stretch=False  # Can enable if needed
    )
    print(f"Dataset size: {len(train_dataset)}")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("Initializing model...")
    model = SimCLRModel(
        embedding_dim=config.EMBEDDING_DIM,
        projection_dim=config.PROJECTION_DIM
    ).to(device)
    
    # Loss and optimizer
    criterion = NTXentLoss(temperature=config.TEMPERATURE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=1e-6
    )
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            print(f"New best loss: {best_loss:.4f}, saving checkpoint...")
            
            # Save encoder only (without projection head)
            encoder_state = {
                'epoch': epoch + 1,
                'encoder_state_dict': model.encoder.state_dict(),
                'loss': train_loss,
            }
            torch.save(encoder_state, os.path.join(config.WEIGHTS_DIR, 'encoder.pth'))
    
    print("\nTraining completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Encoder saved to: {os.path.join(config.WEIGHTS_DIR, 'encoder.pth')}")


if __name__ == "__main__":
    train()
