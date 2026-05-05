"""
Preprocessing pipeline for sleep staging.

Key design decisions from literature:
  [SSNet] Almutairi et al. 2023:
    - Use all 4 channels: EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental
    - Per-segment Z-score normalization (NOT per-recording)
    - No bandpass filtering (SSNet uses raw signals directly)
    - Accuracy 96.57% with 4-channel input

  [1D-ResNet-SE-LSTM] Li & Gao 2023:
    - Raw single-channel EEG, no filtering
    - Per-segment normalization
    - κ=0.812 on Sleep-EDF-78

  [SleepSatelightFTC] Ito & Tanaka 2025:
    - Downsample to 50Hz (gamma waves < 1% of power, mostly noise)
    - Per-segment normalization
    - κ=0.787 with 470K params

Our approach:
  - 4 channels (EEG×2 + EOG + EMG) — SSNet proven +3-5% kappa
  - Bandpass 0.5-35Hz + notch 50Hz (removes powerline noise)
  - Per-segment Z-score normalization (avoids cross-epoch leakage)
  - Store raw signals (3000 samples per epoch) in cache
  - No STFT — raw signals fed directly to 1D-CNN
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel configuration — 4 channels: EEG×2, EOG, EMG
# SSNet paper shows +3-5% kappa from adding EOG+EMG vs EEG-only
# ---------------------------------------------------------------------------

CHANNEL_CONFIG: List[Tuple[str, str, str]] = [
    ("eeg1", "EEG Fpz-Cz",    "eeg"),
    ("eeg2", "EEG Pz-Oz",     "eeg"),
    ("eog",  "EOG horizontal", "eog"),
    ("emg",  "EMG submental",  "emg"),
]
N_CHANNELS = 4

STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Movement time": 0,
    "Sleep stage ?": -1,
}


# ---------------------------------------------------------------------------
# Multi-channel EDF loading
# ---------------------------------------------------------------------------

def load_multichannel_edf(
    psg_path: str,
    target_fs: int = 100,
    required_channels: Tuple[str, ...] = ("eeg1", "eeg2"),
) -> Tuple[np.ndarray, int, List[str]]:
    """
    Load up to 4 channels (EEG×2, EOG, EMG) from a PSG EDF file.

    Required channels (EEG×2) raise ValueError if missing.
    Optional channels (EOG, EMG) use a proxy signal if missing.

    Returns:
        signals  (4, n_samples) float32
        fs       int
        loaded   list of channel names actually loaded
    """
    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError as exc:
        raise ImportError("mne is required: pip install mne") from exc

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    available = {ch.upper(): ch for ch in raw.ch_names}
    n_samples = int(raw.times[-1] * target_fs) + 1

    signals: List[np.ndarray] = []
    loaded: List[str] = []
    loaded_sigs: dict = {}

    for name, ch_name, ch_type in CHANNEL_CONFIG:
        ch_upper = ch_name.upper()
        if ch_upper in available:
            actual_ch = available[ch_upper]
            raw_ch = raw.copy().pick_channels([actual_ch])
            if int(raw_ch.info["sfreq"]) != target_fs:
                raw_ch.resample(target_fs, npad="auto")
            sig = raw_ch.get_data()[0].astype(np.float32)
            signals.append(sig)
            loaded.append(name)
            loaded_sigs[name] = sig
        else:
            if name in required_channels:
                raise ValueError(
                    f"Required channel '{ch_name}' missing in {psg_path}. "
                    f"Available: {list(raw.ch_names)}"
                )
            # Proxy for optional channels
            if name == "eog" and "eeg2" in loaded_sigs:
                proxy = loaded_sigs["eeg2"] + np.random.randn(n_samples).astype(np.float32) * 0.01
                log.warning("EOG missing in %s — using EEG Pz-Oz proxy", psg_path)
            elif name == "emg":
                proxy = np.random.randn(n_samples).astype(np.float32) * 0.1
                log.warning("EMG missing in %s — using noise proxy", psg_path)
            else:
                proxy = np.zeros(n_samples, dtype=np.float32)
                log.warning("Channel '%s' missing in %s — zero-filled", ch_name, psg_path)
            signals.append(proxy)
            loaded.append(f"{name}_PROXY")

    return np.stack(signals, axis=0), target_fs, loaded  # (4, n_samples)


def load_hypnogram(hyp_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse hypnogram EDF annotations into onsets(sec) and integer stages."""
    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError as exc:
        raise ImportError("mne is required: pip install mne") from exc

    ann = mne.read_annotations(hyp_path)
    onsets, labels = [], []
    for onset, desc in zip(ann.onset, ann.description):
        stage = STAGE_MAP.get(str(desc).strip(), -1)
        onsets.append(onset)
        labels.append(stage)
    return np.array(onsets, dtype=float), np.array(labels, dtype=np.int8)


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def bandpass(signal: np.ndarray, fs: int, lo: float = 0.5, hi: float = 35.0) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def notch(signal: np.ndarray, fs: int, freq: float = 50.0) -> np.ndarray:
    from scipy.signal import filtfilt, iirnotch
    b, a = iirnotch(freq / (fs / 2), Q=30.0)
    return filtfilt(b, a, signal).astype(np.float32)


def zscore_epoch(epoch: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-epoch Z-score normalization.
    From SSNet (Almutairi et al. 2023): z = (S - mean(S)) / std(S)
    Applied per-epoch to avoid cross-epoch statistics leakage.
    """
    mu  = float(np.mean(epoch))
    std = float(np.std(epoch)) + eps
    return ((epoch - mu) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Main recording processor — returns RAW signals (not spectrograms)
# ---------------------------------------------------------------------------

def process_recording(
    psg_path: str,
    hyp_path: str,
    cfg,
    notch_freq: float = 50.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Process one PSG+hypnogram pair.

    Returns:
        raw_epochs  (N, 4, epoch_samples) float32  — per-epoch Z-score normalised
        labels      (N,)                  int8

    Design decisions:
      - Raw signals (not spectrograms) — Li & Gao 2023 achieves κ=0.812 this way
      - Bandpass 0.5-35Hz + notch 50Hz — removes powerline noise
      - Per-epoch Z-score — SSNet approach, avoids cross-epoch leakage
      - 4 channels — SSNet shows +3-5% kappa vs single EEG
    """
    epoch_samples = int(cfg.fs * cfg.epoch_sec)

    # Load 4 channels
    try:
        signals, fs, loaded = load_multichannel_edf(psg_path, target_fs=cfg.fs)
        log.debug("Loaded channels: %s", loaded)
    except Exception as exc:
        log.error("Failed to load PSG %s: %s", psg_path, exc)
        return None

    # Filter each channel — bandpass + notch
    # Note: NO recording-level normalization here (per-epoch Z-score applied below)
    filtered = []
    for sig in signals:
        try:
            sig = bandpass(sig, fs, lo=getattr(cfg, "f_low", 0.5), hi=getattr(cfg, "f_high", 35.0))
            sig = notch(sig, fs, freq=notch_freq)
        except Exception:
            pass  # use raw if filtering fails
        filtered.append(sig.astype(np.float32))
    signals = np.stack(filtered, axis=0)  # (4, n_samples)

    # Load hypnogram
    try:
        onsets, stages = load_hypnogram(hyp_path)
    except Exception as exc:
        log.error("Failed to load hypnogram %s: %s", hyp_path, exc)
        return None

    # Label alignment validation
    signal_duration_s = signals.shape[1] / fs
    expected_epochs   = signal_duration_s / cfg.epoch_sec
    if abs(len(stages) - expected_epochs) > 5:
        log.warning(
            "Label count mismatch in %s: signal=%.0f epochs, labels=%d",
            psg_path, expected_epochs, len(stages),
        )

    raw_epochs: List[np.ndarray] = []
    labels: List[int] = []

    for onset, stage in zip(onsets, stages):
        if stage < 0:
            continue
        start = int(onset * fs)
        end   = start + epoch_samples

        if start < 0:
            log.warning("Skipping epoch at onset=%.1fs — negative start", onset)
            continue
        if end > signals.shape[1]:
            break

        # Extract epoch for each channel and apply per-epoch Z-score
        ch_epochs = []
        for ch_sig in signals:
            epoch = ch_sig[start:end]
            epoch = zscore_epoch(epoch)  # per-epoch Z-score (SSNet)
            ch_epochs.append(epoch)

        epoch_raw = np.stack(ch_epochs, axis=0)  # (4, epoch_samples)
        raw_epochs.append(epoch_raw)
        labels.append(int(stage))

    if not raw_epochs:
        log.warning("No valid epochs extracted from %s", psg_path)
        return None

    return (
        np.stack(raw_epochs, axis=0).astype(np.float32),  # (N, 4, epoch_samples)
        np.array(labels, dtype=np.int8),                   # (N,)
    )


# ---------------------------------------------------------------------------
# Backward-compatible wrappers
# ---------------------------------------------------------------------------

@dataclass
class ProcessedBatch:
    specs: np.ndarray    # actually raw signals (N, 4, epoch_samples)
    labels: np.ndarray
    features: Optional[np.ndarray] = None


class SleepEEGPreprocessor:
    """Backward-compatible wrapper."""

    def __init__(self, cfg):
        self.cfg = cfg

    def process_recording(
        self,
        psg_path: str,
        hypnogram_path: str,
        channel: str = "EEG Fpz-Cz",
        augment: bool = False,
    ) -> ProcessedBatch:
        _ = channel
        out = process_recording(psg_path, hypnogram_path, self.cfg)
        if out is None:
            raise RuntimeError(f"Failed to process recording: {psg_path}")
        raw, labels = out
        if augment:
            raw = raw + np.random.randn(*raw.shape).astype(np.float32) * 0.03
        return ProcessedBatch(specs=raw, labels=labels.astype(np.int64))


def process_manifest(
    manifest_df: pd.DataFrame,
    preprocessor: SleepEEGPreprocessor,
    cache_dir: Optional[str] = "data/cache",
    augment: bool = False,
):
    """Process all recordings in a manifest DataFrame."""
    all_specs:    List[np.ndarray] = []
    all_labels:   List[np.ndarray] = []
    all_subjects: List[np.ndarray] = []

    cache_base = Path(cache_dir) if cache_dir else None
    if cache_base:
        cache_base.mkdir(parents=True, exist_ok=True)

    n_channels = int(getattr(preprocessor.cfg, "n_channels", 4))
    fs         = int(getattr(preprocessor.cfg, "fs", 100))
    epoch_sec  = int(getattr(preprocessor.cfg, "epoch_sec", 30))

    for _, row in manifest_df.iterrows():
        subject  = str(row["subject_id"])
        psg_path = str(row.get("psg_path", row.get("psg", "")))
        hyp_path = str(row.get("hypnogram_path", row.get("hypnogram", "")))
        if not psg_path or not hyp_path:
            continue

        cache_file = None
        if cache_base:
            stem = f"{subject}_{Path(psg_path).stem}_raw_c{n_channels}_fs{fs}"
            cache_file = cache_base / f"{stem}.npz"
            if cache_file.exists():
                npz = np.load(cache_file)
                all_specs.append(npz["specs"])
                all_labels.append(npz["labels"])
                all_subjects.append(np.full((len(npz["labels"]),), subject))
                continue

        batch = preprocessor.process_recording(psg_path, hyp_path, augment=augment)
        all_specs.append(batch.specs)
        all_labels.append(batch.labels)
        all_subjects.append(np.full((len(batch.labels),), subject))

        if cache_file is not None:
            np.savez_compressed(cache_file, specs=batch.specs, labels=batch.labels)

    if not all_specs:
        raise RuntimeError("No recordings were processed from manifest")

    return (
        np.concatenate(all_specs,    axis=0),
        np.concatenate(all_labels,   axis=0),
        np.concatenate(all_subjects, axis=0),
        np.zeros((sum(len(s) for s in all_specs), 52), dtype=np.float32),  # compat
    )
