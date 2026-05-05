from dataclasses import dataclass


LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Movement time": 0,
}

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


@dataclass
class EEGConfig:
    fs: int = 100
    epoch_sec: int = 30
    f_low: float = 0.5
    f_high: float = 35.0
    notch_hz: float = 50.0

    @property
    def epoch_samples(self) -> int:
        return self.fs * self.epoch_sec


@dataclass
class TrainConfig:
    """
    Training configuration — optimised for raw signal pipeline.

    Key changes from literature:
      - seq_len=10: 10 × 30s = 5 min context (Li & Gao use 15, we use 10 for speed)
      - batch_size=16: raw signals (3000 samples) are larger than spectrograms
      - lr=1e-4: conservative for transformer stability
      - n_channels=4: EEG×2 + EOG + EMG (SSNet: +3-5% kappa)
      - signal_len=3000: 30s × 100Hz raw samples per epoch
    """
    seq_len: int = 60          # 60 × 30s = 30 min context — increased for longer temporal context
    batch_size: int = 16       # raw signals are larger — keep batch moderate
    epochs: int = 80           # more epochs for better convergence with focal loss
    lr: float = 1e-4           # learning rate
    weight_decay: float = 5e-5
    num_workers: int = 0       # Windows: always 0
    amp: bool = True
    profile_steps: int = 0
    patience: int = 15
    # Signal dimensions
    n_channels: int = 4        # EEG×2 + EOG + EMG
    signal_len: int = 3000     # 30s × 100Hz
    fs: int = 100
    epoch_sec: int = 30
    f_low: float = 0.5
    f_high: float = 35.0
    # Model dims
    d_model: int = 128
    hidden: int = 128
    # Kept for backward compat (not used in raw signal pipeline)
    freq_bins: int = 128
    time_bins: int = 29
    # Loss options
    use_focal: bool = True
    focal_gamma: float = 2.0              # increased from 1.5 for harder focus on N1/N2 boundary
    # Label smoothing & regularization
    label_smoothing: float = 0.1          # regularises hard labels, improves soft-target KD
    # Augmentation
    mixup_alpha: float = 0.4              # increased from 0.3 — more mixing improves robustness
    mixup_start_epoch: int = 5            # start mixup after epoch 5 (warmup phase)
    channel_dropout: float = 0.1          # randomly drop 10% of channels during training
    gaussian_noise_std: float = 0.01      # add small Gaussian noise to raw signals (σ=0.01)
    time_shift_ms: float = 50.0           # small random time shifts (±50ms) for augmentation
    freq_mask_ratio: float = 0.1          # mask 10% of FFT bins during training
    cutmix_enabled: bool = True           # enable CutMix (swap segments of 20% length)
    cutmix_alpha: float = 0.5             # CutMix mix ratio
    cutmix_segment_ratio: float = 0.2     # length of segments to swap (20% of epoch)
    # Training loop tweaks
    use_multiscale_loss: bool = False     # for future: multi-scale temporal modeling
    gradient_accumulation_steps: int = 2  # accumulate gradients for larger effective batch
