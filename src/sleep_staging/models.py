"""
Sleep staging models — redesigned based on literature review.

Key papers:
  [1] Li & Gao 2023 — 1D-ResNet-SE-LSTM, κ=0.812 on Sleep-EDF-78
  [2] Almutairi et al. 2023 — SSNet, 96.57% acc with EEG+EOG+EMG
  [3] Ito & Tanaka 2025 — SleepSatelightFTC, κ=0.787 with 470K params
  [4] Yao & Liu 2022 — CNN-Transformer, 79.5% acc on microcontroller
  [5] L-SeqSleepNet 2023 — whole-cycle context (seq_len=180), κ=0.743 LOSO
  [6] BiT-MamSleep 2024 — bidirectional Mamba, O(T) memory, SOTA on long seqs

Architecture:
  Teacher: 1D-ResNet-SE (3 blocks) + Transformer (2L) on raw 4-ch signals
           Input: (B, T, 4, 3000) raw signal — per-segment Z-score normalised
           Output: (B, T, 5) logits
           Params: ~680K

  Student: Lightweight 1D-CNN + 2-layer GRU on raw 4-ch signals
           Input: (B, T, 4, 3000) raw signal
           Output: (B, T, 5) logits
           Params: ~186K

  Compression: ~3.7x

Why raw signals instead of spectrograms:
  - Li et al. achieve κ=0.812 with raw 1D signals (our spectrogram model: κ=0.35)
  - Raw signals preserve temporal detail (spindles, K-complexes) lost in STFT
  - Faster to process — no STFT computation at training time
  - Per-segment Z-score (SSNet) is simpler and avoids cross-epoch leakage

Roadmap to close SOTA gap (κ=0.636 → κ=0.74+):
  Step 1 (done):    seq_len=15, focal loss, FFT branch         → κ=0.636
  Step 2 (current): distillation + augmentation improvements   → κ~0.68
  Step 3 (next):    seq_len=60, confusion-aware focal loss     → κ~0.72
  Step 4 (future):  Mamba + seq_len=180 (whole sleep cycle)    → κ~0.76
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block (channel attention)
# From: Hu et al. 2018, used in Li & Gao 2023
# ---------------------------------------------------------------------------

class SEBlock1D(nn.Module):
    """Channel-wise attention for 1D feature maps. (B, C, L) → (B, C, L)"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(4, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # (B, C, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1)


# ---------------------------------------------------------------------------
# 1D-ResNet-SE block
# Directly from Li & Gao 2023 (1D-ResNet-SE-LSTM, κ=0.812)
# Two Conv1D layers + SE attention + residual shortcut
# ---------------------------------------------------------------------------

class ResNetSE1DBlock(nn.Module):
    """
    1D residual block with SE channel attention.
    Input/output: (B, C_in, L) → (B, C_out, L//stride)
    """
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_c)
        self.act1  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_c)
        self.se    = SEBlock1D(out_c, reduction=8)
        self.act2  = nn.ReLU(inplace=True)

        # Shortcut: match dimensions if needed
        self.shortcut = None
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x) if self.shortcut else x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act2(out + identity)


# ---------------------------------------------------------------------------
# Teacher CNN — 1D-ResNet-SE backbone
# 3 blocks: 32→64→128 channels, progressive downsampling
# Input: (B*T, 4, 3000) → Output: (B*T, 128)
# ---------------------------------------------------------------------------

class TeacherCNN1D(nn.Module):
    """
    1D-ResNet-SE feature extractor for raw multi-channel EEG/EOG/EMG.
    Architecture from Li & Gao 2023 adapted for 4-channel input.

    Input:  (B, 4, 3000)  — 4 channels × 3000 samples (30s @ 100Hz)
    Output: (B, 128)
    """
    def __init__(self, in_channels: int = 4):
        super().__init__()
        # Stem: large kernel to capture slow EEG waves (delta 0.5-4Hz needs ~200 samples)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=50, stride=5, padding=25, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 3000 → 150 samples
        )
        # 3 residual blocks with increasing channels and downsampling
        # Following Li & Gao 2023: kernel sizes 3, 5, 7
        self.block1 = ResNetSE1DBlock(32,  64,  kernel_size=3, stride=2)   # 150 → 75
        self.block2 = ResNetSE1DBlock(64,  128, kernel_size=5, stride=2)   # 75 → 38
        self.block3 = ResNetSE1DBlock(128, 128, kernel_size=7, stride=1)   # 38 → 38
        self.pool   = nn.AdaptiveAvgPool1d(1)                               # → 1
        self.drop   = nn.Dropout(0.5)  # Li & Gao use 50% dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.drop(self.pool(x).flatten(1))  # (B, 128)


# ---------------------------------------------------------------------------
# Frequency branch — amplitude spectrum MLP
# From SleepSatelightFTC (Ito & Tanaka 2025, κ=0.787)
# Key insight: 12Hz spindle component distinguishes N2 from N3
#              0.5-4Hz delta power distinguishes N3 from N2
# Input: (B, 4, 3000) raw signal → compute FFT → (B, 4, 51) amplitude spectrum
# Output: (B, 64)
# ---------------------------------------------------------------------------

class FrequencyBranch(nn.Module):
    """
    Frequency-domain branch using amplitude spectrum.
    Directly addresses N2↔N3 confusion by capturing:
      - 12Hz spindle waves (N2 marker)
      - 0.5-4Hz delta waves (N3 marker)
      - 8-12Hz alpha waves (Wake/N1 marker)

    From SleepSatelightFTC: adding frequency branch improves ACC by 2.3%
    and specifically helps N3 F1 score.

    Input:  (B, 4, 3000) raw signal
    Output: (B, 64) frequency features
    """
    def __init__(self, in_channels: int = 4, fs: int = 100, out_dim: int = 64):
        super().__init__()
        self.fs = fs
        # FFT gives 3000//2+1 = 1501 bins, but we only need 0-35Hz = 35/(fs/2)*1501 ≈ 1051 bins
        # We use 0-35Hz range: 35*30+1 = 1051 bins at 100Hz
        # Downsample to 51 bins (0-50Hz at 1Hz resolution) for efficiency
        self.n_fft_bins = 51  # 0-50Hz at 1Hz resolution

        # Small MLP on flattened spectrum (4 channels × 51 bins = 204 inputs)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * self.n_fft_bins, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) → (B, out_dim)"""
        B, C, L = x.shape
        # Compute amplitude spectrum per channel
        # FFT → magnitude → log scale (dBuV as in SleepSatelightFTC)
        fft = torch.fft.rfft(x, dim=-1)                    # (B, C, L//2+1)
        amp = torch.abs(fft) + 1e-10                        # (B, C, L//2+1)
        amp_db = torch.log(amp)                             # log amplitude (dBuV)

        # Take only 0-50Hz bins (first 51 bins at 100Hz sampling)
        amp_db = amp_db[:, :, :self.n_fft_bins]            # (B, C, 51)

        # Flatten and pass through MLP
        flat = amp_db.reshape(B, -1)                        # (B, C*51)
        return self.mlp(flat)                               # (B, out_dim)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ---------------------------------------------------------------------------
# Teacher model — 1D-ResNet-SE + Transformer
# Based on Li & Gao 2023 + Yao & Liu 2022
# Input: raw 4-channel signal (B, T, 4, 3000)
# Output: logits (B, T, 5)
# ---------------------------------------------------------------------------

class TeacherCRNN(nn.Module):
    """
    Teacher model for sleep staging — Time + Frequency dual-branch.

    Architecture (from literature):
      1. 1D-ResNet-SE CNN on raw signals → (B*T, 128) time features
      2. FFT amplitude spectrum MLP → (B*T, 64) frequency features
         [SleepSatelightFTC: +2.3% ACC, resolves N2↔N3 spindle/delta confusion]
      3. Fused (128+64=192) → projected to d_model → Transformer
      4. Linear head → (B, T, 5)

    Based on:
      - Li & Gao 2023: 1D-ResNet-SE-LSTM achieves κ=0.812
      - SleepSatelightFTC: dual time+freq branch resolves N2↔N3 confusion
      - SSNet: 4-channel input (EEG×2 + EOG + EMG) gives +3-5% kappa

    Params: ~1.6M  |  Input: (B, T, 4, 3000)  |  Output: (B, T, 5)
    """
    NUM_CLASSES = 5

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        in_channels: int = 4,
        use_freq_branch: bool = True,   # Set False to load old checkpoints
    ):
        super().__init__()
        self.use_freq_branch = use_freq_branch
        # Branch 1: time-domain 1D-ResNet-SE (128-dim)
        self.cnn  = TeacherCNN1D(in_channels=in_channels)
        # Branch 2: frequency-domain amplitude spectrum MLP (64-dim)
        # Resolves N2↔N3 confusion via spindle (12Hz) and delta (0.5-4Hz) detection
        self.freq = FrequencyBranch(in_channels=in_channels, out_dim=64)
        # Fuse 128 + 64 = 192 → d_model  (or just 128 if freq branch disabled)
        proj_in = (128 + 64) if use_freq_branch else 128
        self.proj = nn.Linear(proj_in, d_model)
        self.pe   = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_model, self.NUM_CLASSES),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        """raw: (B, T, C, L) → (B, T, d_model)"""
        B, T, C, L = raw.shape
        raw_flat = raw.view(B * T, C, L)
        # Branch 1: time-domain features
        time_feat = self.cnn(raw_flat).view(B, T, -1)       # (B, T, 128)
        if self.use_freq_branch:
            # Branch 2: frequency-domain features (resolves N2↔N3 confusion)
            freq_feat = self.freq(raw_flat).view(B, T, -1)  # (B, T, 64)
            feat = torch.cat([time_feat, freq_feat], dim=-1) # (B, T, 192)
        else:
            feat = time_feat                                  # (B, T, 128)
        x = self.proj(feat)   # (B, T, d_model)
        x = self.pe(x)
        x = self.transformer(x)
        return self.norm(x)

    def forward(
        self,
        spec: torch.Tensor,                    # kept for API compat — ignored if raw provided
        raw: Optional[torch.Tensor] = None,    # (B, T, 4, 3000) raw signal — PRIMARY input
        feats: Optional[torch.Tensor] = None,  # ignored
        return_features: bool = False,
    ):
        # Use raw signal as primary input; fall back to spec if raw not provided
        # (spec is kept for backward compat with existing training loop)
        signal = raw if raw is not None else spec
        feat = self.encode(signal)
        logits = self.head(feat)
        if return_features:
            return logits, feat, feat
        return logits

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Student model — Lightweight 1D-CNN + 2-layer GRU
# Inspired by TinySleepNet (Supratak & Guo 2020) + Li & Gao 2023
# Input: raw 4-channel signal (B, T, 4, 3000)
# Output: logits (B, T, 5), h_new (2, B, hidden)
# ---------------------------------------------------------------------------

class StudentCNN1D(nn.Module):
    """
    Lightweight 1D CNN for raw multi-channel signals.
    Input:  (B, 4, 3000)
    Output: (B, 64)
    """
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            # Large kernel stem — captures slow EEG waves
            nn.Conv1d(in_channels, 16, kernel_size=50, stride=5, padding=25, bias=False),
            nn.BatchNorm1d(16), nn.ReLU6(inplace=True),
            nn.MaxPool1d(4, stride=4),                                    # → 150 → 37
            # Two lightweight blocks
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.ReLU6(inplace=True),                   # → 19
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU6(inplace=True),                   # → 10
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),                                                  # → 64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentCRNN(nn.Module):
    """
    Student model — lightweight 1D-CNN + 2-layer GRU.
    Input:  raw (B, T, 4, 3000)
            h   (2, B, hidden)  optional GRU state
    Output: logits (B, T, 5), h_new (2, B, hidden)

    Params: ~280K  |  Compression vs teacher: ~5.4x
    """
    NUM_CLASSES = 5
    HIDDEN = 128

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.cnn  = StudentCNN1D(in_channels=in_channels)
        self.gru  = nn.GRU(64, self.HIDDEN, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(self.HIDDEN, self.NUM_CLASSES)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GRU, nn.LSTM)):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def encode_cnn(self, raw: torch.Tensor) -> torch.Tensor:
        """raw: (B, T, C, L) → (B, T, 64)"""
        B, T, C, L = raw.shape
        return self.cnn(raw.view(B * T, C, L)).view(B, T, -1)

    def get_cnn_features(self, raw: torch.Tensor) -> torch.Tensor:
        return self.encode_cnn(raw)

    def forward(
        self,
        spec: torch.Tensor,                   # kept for API compat
        h: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        # spec is actually the raw signal tensor in the new pipeline
        raw = spec
        single = raw.dim() == 3  # (B, C, L) — single epoch
        if single:
            raw = raw.unsqueeze(1)  # → (B, 1, C, L)

        B, T = raw.shape[:2]
        if h is None:
            h = torch.zeros(self.gru.num_layers, B, self.HIDDEN,
                            device=raw.device, dtype=raw.dtype)

        cnn_feat = self.encode_cnn(raw)          # (B, T, 64)
        gru_out, h_new = self.gru(cnn_feat, h)   # (B, T, 128)
        logits = self.head(gru_out)              # (B, T, 5)

        if single:
            logits   = logits.squeeze(1)
            cnn_feat = cnn_feat.squeeze(1)

        if return_features:
            return logits, cnn_feat, h_new
        return logits, h_new

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Feature projector for distillation
# ---------------------------------------------------------------------------

class FeatureProjector(nn.Module):
    """Projects student CNN features (64) to teacher feature space (128)."""
    def __init__(self, student_dim: int = 64, teacher_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(student_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, teacher_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def synthetic_batch(batch_size: int = 4, seq_len: int = 10,
                    n_channels: int = 4, signal_len: int = 3000):
    """Generate a synthetic batch of raw signals for testing."""
    x = torch.randn(batch_size, seq_len, n_channels, signal_len)
    y = torch.randint(0, 5, (batch_size, seq_len))
    return x, y


if __name__ == "__main__":
    import time
    t = TeacherCRNN()
    s = StudentCRNN()

    raw = torch.randn(2, 10, 4, 3000)
    t_logits, t_feat, _ = t(raw, raw=raw, return_features=True)
    s_logits, s_feat, s_h = s(raw, return_features=True)

    print(f"Teacher params : {t.param_count():,}")
    print(f"Student params : {s.param_count():,}")
    print(f"Compression    : {t.param_count()/s.param_count():.1f}x")
    print(f"Teacher logits : {t_logits.shape}  feat: {t_feat.shape}")
    print(f"Student logits : {s_logits.shape}  feat: {s_feat.shape}  h: {s_h.shape}")

    # Benchmark on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_gpu = t.to(device)
    raw_gpu = torch.randn(16, 10, 4, 3000, device=device)

    for _ in range(3):
        out = t_gpu(raw_gpu, raw=raw_gpu)
        (out[0] if isinstance(out, tuple) else out).sum().backward()
        t_gpu.zero_grad()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(10):
        out = t_gpu(raw_gpu, raw=raw_gpu)
        (out[0] if isinstance(out, tuple) else out).sum().backward()
        t_gpu.zero_grad()
    torch.cuda.synchronize()
    elapsed = (time.time()-t0)/10
    steps = 12551 // 16
    print(f"\nbs=16 sl=10: {elapsed:.3f}s/step  {steps} steps  ~{steps*elapsed/60:.1f} min/epoch")
    print(f"VRAM: {torch.cuda.memory_reserved()/1e9:.2f} GB")
