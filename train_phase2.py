#!/usr/bin/env python
"""Phase 2: Teacher retrain with quick-win config."""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sleep_staging.config import TrainConfig, EEGConfig
from sleep_staging.models import TeacherCRNN
from sleep_staging.data import build_dataloaders
from sleep_staging.preprocess import process_manifest, SleepEEGPreprocessor
from sleep_staging.losses import FocalLoss

print('=' * 70)
print('PHASE 2: TEACHER RETRAIN WITH QUICK-WIN CONFIG (Cache Rebuild)')
print('=' * 70)
print()

# Configuration
cfg = TrainConfig()
cfg.seq_len = 60
cfg.batch_size = 16
cfg.epochs = 50  # Reduced from 80 for faster testing
cfg.use_focal = True
cfg.focal_gamma = 2.0
cfg.label_smoothing = 0.1
cfg.mixup_alpha = 0.4
cfg.channel_dropout = 0.1
cfg.gaussian_noise_std = 0.01
cfg.time_shift_ms = 50.0

eeg_cfg = EEGConfig()

print(f'seq_len: {cfg.seq_len} (30-min context)')
print(f'batch_size: {cfg.batch_size}')
print(f'epochs: {cfg.epochs}')
print(f'focal_gamma: {cfg.focal_gamma}')
print()

# Preprocess manifest and build cache
print('Loading and preprocessing data...')
preprocessor = SleepEEGPreprocessor(eeg_cfg)
manifest_path = Path('data/manifests/sleep_edf_full.csv')
cache_dir = Path('data/cache')

# Load manifest
manifest_df = pd.read_csv(manifest_path)
print(f'Manifest: {len(manifest_df)} recordings')

try:
    specs, labels, subjects, _feats = process_manifest(
        manifest_df,
        preprocessor=preprocessor,
        cache_dir=str(cache_dir),
        augment=False
    )
    
    print(f'Loaded {len(specs)} samples')
    print(f'  specs shape: {specs.shape}')
    print(f'  labels shape: {labels.shape}')
    print(f'  subjects: {len(np.unique(subjects))} unique')
    print()
    
    # Split data (70% train, 15% val, 15% test)
    n = len(specs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    indices = np.random.RandomState(42).permutation(n)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    
    train_specs, val_specs = specs[train_idx], specs[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]
    train_subj, val_subj = subjects[train_idx], subjects[val_idx]
    
    print(f'Train: {len(train_specs)} samples')
    print(f'Val:   {len(val_specs)} samples')
    print()
    
    # Build dataloaders
    loaders = build_dataloaders(
        train_specs, train_labels, train_subj,
        val_specs, val_labels, val_subj,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    print()
    
except Exception as e:
    print(f'ERROR during preprocessing: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
model = TeacherCRNN(
    d_model=cfg.d_model,
    nhead=4,
    num_layers=2,
    dropout=0.1,
    in_channels=4,
    use_freq_branch=True
)
model = model.to(device)

# Optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = FocalLoss(gamma=cfg.focal_gamma, alpha=None)

print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
print()
print('=' * 70)
print('STARTING TRAINING')
print('=' * 70)
print()

# Training loop
best_val_loss = float('inf')
for epoch in range(cfg.epochs):
    model.train()
    train_loss = 0
    for batch_idx, (specs, labels) in enumerate(train_loader):
        specs = specs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for specs, labels in val_loader:
            specs = specs.to(device)
            labels = labels.to(device)
            logits = model(specs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch+1:3d}/{cfg.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'artifacts/teacher_improved_v2.pt')
        if (epoch + 1) % 10 == 0:
            print(f'  → Checkpoint saved (val_loss={val_loss:.4f})')

print()
print('=' * 70)
print('TRAINING COMPLETE')
print('=' * 70)
print(f'Best validation loss: {best_val_loss:.4f}')
print(f'Checkpoint: artifacts/teacher_improved_v2.pt')
