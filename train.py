"""
Training script for self-supervised contrastive learning with validation & AMP.
"""
import os
import sys
import json
import types
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch with AMP."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for view1, view2 in pbar:
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=config.USE_AMP):
            z1 = model(view1)
            z2 = model(view2)
            loss = criterion(z1, z2)
        
        if config.USE_AMP and device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate model with AMP."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for view1, view2 in tqdm(dataloader, desc="Validation"):
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                z1 = model(view1)
                z2 = model(view2)
                loss = criterion(z1, z2)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


class EarlyStopping:
    """Early stopping based on validation loss to prevent overfitting."""
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train():
    """Main training function."""
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading full dataset...")
    full_dataset = ContrastiveAudioDataset(
        data_dir=config.DATA_DIR,
        augment=True,
        use_pitch_shift=False,
        use_time_stretch=False
    )
    
    # Split into train and validation (90% train, 10% val)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.VAL_SPLIT * dataset_size))
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0)
    )
    
    # Create model
    print("Initializing SimCLR model...")
    model = SimCLRModel(
        embedding_dim=config.EMBEDDING_DIM,
        projection_dim=config.PROJECTION_DIM
    ).to(device)
    
    # Loss, optimizer, scaler
    criterion = NTXentLoss(temperature=config.TEMPERATURE)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler('cuda', enabled=config.USE_AMP)
    
    # Scheduler with 5 epoch warmup + Cosine Annealing
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS - warmup_epochs, eta_min=1e-6
    )
    
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return config.LEARNING_RATE * (epoch + 1) / warmup_epochs
        return None

    early_stopping = EarlyStopping(patience=15)
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(config.RESULTS_DIR, 'runs'))
    except ImportError:
        writer = None

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    start_epoch = 0

    checkpoint_path = os.path.join(config.WEIGHTS_DIR, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                best_val_loss = checkpoint.get('val_loss', float('inf'))
                print(f"[RESUMING] Successfully loaded checkpoint from Epoch {start_epoch} with Val Loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"[WARNING] Could not load existing checkpoint: {e}. Starting training from scratch.")
    
    print(f"Starting/Resuming full training from Epoch {start_epoch + 1} up to {config.NUM_EPOCHS} epochs...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        lr = get_lr(epoch)
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} | LR: {current_lr:.6f}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('LearningRate', current_lr, epoch)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if epoch >= warmup_epochs:
            scheduler.step()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}, saving checkpoint...")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(config.WEIGHTS_DIR, 'best_model.pth'))
            torch.save(model.encoder.state_dict(), os.path.join(config.WEIGHTS_DIR, 'encoder.pth'))
            
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered - model fully converged!")
            break
            
    if writer is not None:
        writer.close()

    with open(os.path.join(config.RESULTS_DIR, 'history.json'), 'w') as f:
        json.dump(history, f)
        
    print("\nFull training completed!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
