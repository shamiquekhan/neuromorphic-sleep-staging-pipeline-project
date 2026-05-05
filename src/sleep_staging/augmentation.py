"""Advanced augmentation strategies for sleep staging EEG data.

Implements:
  - CutMix: swap random segments between epochs (improves robustness)
  - Frequency masking: zero out random FFT bins (regularization)
  - Channel dropout: randomly drop channels (robustness)
  - Gaussian noise: add small noise to raw signal (robustness)
  - Time shift: small temporal jitter (±50ms, avoids label leakage)

Based on: mixup/cutmix theory, EEG augmentation practices, TinyML guidelines.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class CutMix:
    """CutMix augmentation for EEG sequences.
    
    Randomly swaps segments (20% of epoch length) between two epochs,
    updating labels as weighted mixture.
    
    From: Yun et al. 2019 (computer vision), adapted for time-series EEG.
    Expected gain: +1-2% accuracy.
    """
    
    def __init__(self, alpha: float = 0.5, segment_ratio: float = 0.2):
        """
        Args:
            alpha: mixing ratio for label blending (0.5 = equal mix)
            segment_ratio: fraction of epoch to swap (e.g. 0.2 = 20% of 3000 samples)
        """
        self.alpha = alpha
        self.segment_ratio = segment_ratio
    
    def __call__(self, spec: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix to batch.
        
        Args:
            spec: (B, T, C, L) or (B, T, L) raw signal
            labels: (B, T) or (B*T,) class labels
            
        Returns:
            augmented_spec, blended_labels
        """
        batch_size = spec.shape[0]
        
        # Randomly select pairs of epochs to mix
        idx = torch.randperm(batch_size)
        spec_mix = spec[idx]
        
        # Segment length to swap
        if spec.dim() == 4:
            seq_len, n_channels, signal_len = spec.shape[1:]
            seg_len = max(1, int(signal_len * self.segment_ratio))
            start = np.random.randint(0, signal_len - seg_len)
            
            # Swap segments
            spec = spec.clone()
            spec[:, :, :, start:start+seg_len] = spec_mix[:, :, :, start:start+seg_len]
        else:  # (B, T, L)
            seq_len, signal_len = spec.shape[1:]
            seg_len = max(1, int(signal_len * self.segment_ratio))
            start = np.random.randint(0, signal_len - seg_len)
            
            spec = spec.clone()
            spec[:, :, start:start+seg_len] = spec_mix[:, :, start:start+seg_len]
        
        # Blend labels
        lam = self.alpha  # how much of first sample to keep
        labels_mix = labels[idx]
        if labels.dim() == 1:  # (B*T,)
            # Don't blend here; let training handle sequence-level consistency
            pass
        else:  # (B, T)
            # Soft blending
            blended = lam * labels.float() + (1 - lam) * labels_mix.float()
            labels = blended
        
        return spec, labels


class FrequencyMask:
    """Frequency masking for FFT-based augmentation.
    
    Zero out random contiguous bins in FFT representation.
    Simulates frequency-domain robustness.
    
    From: SpecAugment (Park et al. 2019), adapted for EEG.
    Expected gain: +0.5-1.5% accuracy.
    """
    
    def __init__(self, mask_ratio: float = 0.1, n_freq_bins: int = 128):
        """
        Args:
            mask_ratio: fraction of bins to zero out
            n_freq_bins: total FFT bins (e.g., 128 from STFT)
        """
        self.mask_ratio = mask_ratio
        self.n_freq_bins = n_freq_bins
    
    def __call__(self, fft_mag: torch.Tensor) -> torch.Tensor:
        """Mask random frequency bins.
        
        Args:
            fft_mag: (..., n_freq_bins, time_steps) magnitude spectrogram
            
        Returns:
            masked spectrogram
        """
        n_mask = max(1, int(self.n_freq_bins * self.mask_ratio))
        start = np.random.randint(0, self.n_freq_bins - n_mask)
        
        fft_mag = fft_mag.clone()
        fft_mag[..., start:start+n_mask, :] = 0.0
        
        return fft_mag


class ChannelDropout:
    """Randomly drop EEG channels during training.
    
    Forces network to learn robust features without relying on one channel.
    Expected gain: +0.5-1% accuracy.
    
    Args:
        dropout_prob: probability to drop each channel (e.g., 0.1 = drop 10%)
    """
    
    def __init__(self, dropout_prob: float = 0.1):
        self.dropout_prob = dropout_prob
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Drop random channels.
        
        Args:
            spec: (..., n_channels, signal_len) or (..., n_channels, freq, time)
            
        Returns:
            augmented with dropped channels
        """
        if np.random.rand() < self.dropout_prob:
            spec = spec.clone()
            # spec shape: (B, T, C, L) — drop one of 4 channels
            n_channels = spec.shape[2] if spec.dim() == 4 else spec.shape[-2]
            drop_ch = np.random.randint(0, n_channels)
            spec[..., drop_ch, :] = 0.0
        
        return spec


class GaussianNoise:
    """Add small Gaussian noise to raw signal.
    
    Improves robustness to measurement noise and ADC quantization.
    Expected gain: +0.5-1% accuracy.
    
    Args:
        std: standard deviation of noise (e.g., 0.01 for ±1% of signal RMS)
    """
    
    def __init__(self, std: float = 0.01):
        self.std = std
    
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise during training only.
        
        Args:
            signal: raw EEG signal (..., L) or (..., C, L)
            
        Returns:
            noisy signal
        """
        noise = torch.randn_like(signal) * self.std
        return signal + noise


class TimeShift:
    """Apply random small time shifts (±50ms) to simulate jitter.
    
    Adds temporal robustness without violating causality.
    Expected gain: +0.3-0.8% accuracy.
    
    Args:
        shift_ms: maximum shift in milliseconds (±50ms typical)
        fs: sampling frequency (100 Hz typical)
    """
    
    def __init__(self, shift_ms: float = 50.0, fs: int = 100):
        self.shift_samples = max(1, int(shift_ms / 1000.0 * fs))  # convert ms to samples
    
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply random time shift.
        
        Args:
            signal: (..., signal_len)
            
        Returns:
            time-shifted signal
        """
        if self.shift_samples == 0:
            return signal
        
        shift = np.random.randint(-self.shift_samples, self.shift_samples + 1)
        if shift == 0:
            return signal
        
        signal = signal.clone()
        if shift > 0:
            signal[..., shift:] = signal[..., :-shift]
            signal[..., :shift] = 0.0
        else:
            signal[..., :shift] = signal[..., -shift:]
            signal[..., shift:] = 0.0
        
        return signal


class AugmentationPipeline:
    """Combine multiple augmentations with configurable probabilities.
    
    Typical pipeline: Mixup (p=0.5) → CutMix (p=0.3) → ChannelDropout (p=0.2) 
                    → GaussianNoise (p=0.3) → TimeShift (p=0.2)
    
    Expected cumulative gain: +2-4% accuracy.
    """
    
    def __init__(self, cfg):
        """Initialize pipeline from config.
        
        Args:
            cfg: TrainConfig with augmentation settings
        """
        self.cutmix = CutMix(alpha=cfg.cutmix_alpha, segment_ratio=cfg.cutmix_segment_ratio) \
            if cfg.cutmix_enabled else None
        self.freq_mask = FrequencyMask(mask_ratio=cfg.freq_mask_ratio, n_freq_bins=cfg.freq_bins)
        self.channel_dropout = ChannelDropout(dropout_prob=cfg.channel_dropout)
        self.gaussian_noise = GaussianNoise(std=cfg.gaussian_noise_std)
        self.time_shift = TimeShift(shift_ms=cfg.time_shift_ms, fs=cfg.fs)
        self.config = cfg
    
    def __call__(self, spec: torch.Tensor, labels: torch.Tensor = None, 
                 epoch: int = 0, training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply pipeline.
        
        Args:
            spec: raw signal (B, T, C, L) or spectrogram (B, T, F, W)
            labels: (B, T) or (B*T,)
            epoch: current epoch (to enable/disable based on schedule)
            training: only apply if training=True
            
        Returns:
            augmented_spec, (possibly blended) labels
        """
        if not training:
            return spec, labels
        
        # Skip augmentation during warmup (first 5 epochs)
        if epoch < self.config.mixup_start_epoch:
            return spec, labels
        
        # CutMix (sequential, updates labels)
        if self.cutmix is not None and np.random.rand() < 0.3:
            spec, labels = self.cutmix(spec, labels)
        
        # Channel dropout (on raw signal)
        if np.random.rand() < 0.2:
            spec = self.channel_dropout(spec)
        
        # Gaussian noise (on raw signal)
        if np.random.rand() < 0.3:
            spec = self.gaussian_noise(spec)
        
        # Time shift (on raw signal)
        if np.random.rand() < 0.2:
            spec = self.time_shift(spec)
        
        return spec, labels
