from __future__ import annotations

import csv
import hashlib
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

log = logging.getLogger(__name__)

STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Movement time": 0,
}
NUM_CLASSES = 5
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def download_sleep_edf_cassette(raw_dir: str, subjects: Optional[List[int]] = None) -> List[str]:
    """Download Sleep-EDF Cassette subset via MNE PhysioNet helpers."""
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required for Sleep-EDF download: pip install mne") from exc

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        subjects = list(range(20))

    fetched = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=subjects,
        recording=[1, 2],
        path=str(raw_path),
        on_missing="warn",
    )
    flat = [str(p) for pair in fetched for p in pair]
    log.info("Downloaded/found %d Sleep-EDF files", len(flat))
    return flat


def build_manifest(raw_dir: str, out_csv: str) -> List[Dict]:
    """Scan raw_dir for PSG/Hypnogram EDF pairs and write a manifest CSV."""
    raw_dir_path = Path(raw_dir)
    records: List[Dict] = []
    seen_pairs = set()

    psgs = sorted(raw_dir_path.rglob("*PSG*.edf"))
    for psg in psgs:
        stem = psg.stem[:7]
        hyp_candidates = list(psg.parent.glob(f"{stem}*Hypnogram*.edf"))
        if not hyp_candidates:
            log.warning("No hypnogram found for %s, skipping.", psg.name)
            continue

        hyp = hyp_candidates[0]
        subject_id = stem[:6]
        night = stem[6:]
        pair_key = (subject_id, night)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        records.append(
            {
                "subject_id": subject_id,
                "night": night,
                "psg": str(psg),
                "hypnogram": str(hyp),
            }
        )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "night", "psg", "hypnogram"])
        writer.writeheader()
        writer.writerows(records)

    log.info("Manifest: %d recordings -> %s", len(records), out_csv)
    return records


def load_manifest(csv_path: str) -> List[Dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_label_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=NUM_CLASSES)
    return {STAGE_NAMES[i]: int(counts[i]) for i in range(NUM_CLASSES)}


def subject_split(
    records: List[Dict],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Subject-level split to prevent data leakage."""
    subjects = sorted({r["subject_id"] for r in records})
    if not subjects:
        return [], [], []
    if len(subjects) == 1:
        return records, records, records
    if len(subjects) == 2:
        train_s = {subjects[0]}
        val_s = {subjects[1]}
        test_s = {subjects[1]}

        def pick(sset):
            return [r for r in records if r["subject_id"] in sset]

        return pick(train_s), pick(val_s), pick(test_s)

    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_test = min(len(subjects) - 2, math.ceil(len(subjects) * test_frac))
    n_test = max(1, n_test)
    n_val = min(len(subjects) - n_test - 1, math.ceil(len(subjects) * val_frac))
    n_val = max(1, n_val)

    # Ensure at least two subjects remain for training; shrink val/test if needed
    while (len(subjects) - n_test - n_val) < 2 and (n_val + n_test) > 2:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    test_s = set(subjects[:n_test])
    val_s = set(subjects[n_test : n_test + n_val])
    train_s = set(subjects[n_test + n_val :])

    def pick(sset):
        return [r for r in records if r["subject_id"] in sset]

    return pick(train_s), pick(val_s), pick(test_s)


def _cache_key(psg_path: str, freq_bins: int = 128, n_channels: int = 4,
               fs: int = 100, epoch_sec: int = 30) -> str:
    """
    Cache key — backward compatible with existing cache files.
    Format: md5(psg_path|f{freq_bins})[:12]  — matches files already on disk.
    Non-default params (n_channels, fs, epoch_sec) are appended as suffix
    so changing them busts the cache without breaking existing files.
    """
    base = f"{psg_path}|f{freq_bins}"
    # Only append extra params when they differ from defaults
    # This preserves compatibility with the 152 existing cache files
    extras = []
    if n_channels != 4:
        extras.append(f"c{n_channels}")
    if fs != 100:
        extras.append(f"fs{fs}")
    if epoch_sec != 30:
        extras.append(f"ep{epoch_sec}")
    if extras:
        base += "|" + "|".join(extras)
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:12]


def load_or_process(
    psg_path: str,
    hyp_path: str,
    cache_dir: Optional[str],
    cfg,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Return (raw_epochs, labels) from cache or fresh preprocessing.
    raw_epochs shape: (N, 4, epoch_samples)  — per-epoch Z-score normalised
    Cache uses .npy files keyed by psg_path + config params.
    """
    from .preprocess import process_recording

    spec_f = label_f = None
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        n_channels = int(getattr(cfg, "n_channels", 4))
        fs         = int(getattr(cfg, "fs", 100))
        epoch_sec  = int(getattr(cfg, "epoch_sec", 30))
        epoch_samples = fs * epoch_sec
        # New cache key for raw signals — distinct from old spectrogram keys
        key_str = f"{psg_path}|raw|c{n_channels}|fs{fs}|ep{epoch_sec}"
        key     = hashlib.md5(key_str.encode("utf-8")).hexdigest()[:12]
        spec_f  = cache_dir_path / f"{key}_raw.npy"
        label_f = cache_dir_path / f"{key}_labels.npy"

        if spec_f.exists() and label_f.exists():
            try:
                raw    = np.load(spec_f,  allow_pickle=False)
                labels = np.load(label_f, allow_pickle=False)
                # Integrity checks for raw signal cache
                assert raw.ndim == 3,                        "bad ndim (expected 3)"
                assert raw.shape[1] == n_channels,           "channel mismatch"
                assert raw.shape[2] == epoch_samples,        "epoch_samples mismatch"
                assert len(raw) == len(labels),              "raw/label count mismatch"
                assert not np.any(np.isnan(raw[:5])),        "NaN in raw"
                return raw, labels
            except Exception as exc:
                log.warning("Corrupt/stale cache for %s (%s) — reprocessing", psg_path, exc)
                spec_f.unlink(missing_ok=True)
                label_f.unlink(missing_ok=True)

    result = process_recording(psg_path, hyp_path, cfg)
    if result is None:
        return None
    raw, labels = result  # (N, 4, epoch_samples), (N,)

    if cache_dir and spec_f is not None and label_f is not None:
        np.save(spec_f,  raw)
        np.save(label_f, labels)
    return raw, labels


class SleepSequenceDataset(Dataset):
    """Windowed sequence dataset for CRNN temporal input."""

    def __init__(
        self,
        records_or_specs,
        cfg_or_labels,
        subject_ids: Optional[np.ndarray] = None,
        seq_len: int = 30,
        stride: int = 1,
        cache_dir: Optional[str] = None,
        augment: bool = False,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        self.windows: List[Tuple[np.ndarray, np.ndarray]] = []
        self.flat_labels: List[int] = []

        if isinstance(records_or_specs, np.ndarray):
            specs = records_or_specs
            labels = np.asarray(cfg_or_labels)
            if subject_ids is None:
                raise ValueError("subject_ids are required when building from arrays")
            self._build_from_arrays(specs, labels, np.asarray(subject_ids))
        else:
            records = list(records_or_specs)
            cfg = cfg_or_labels
            self._build_from_records(records, cfg, cache_dir=cache_dir)

        log.info("Dataset: %s windows", f"{len(self.windows):,}")

    def _build_from_records(self, records: List[Dict], cfg, cache_dir: Optional[str]) -> None:
        for rec in records:
            result = load_or_process(rec["psg"], rec["hypnogram"], cache_dir, cfg)
            if result is None:
                continue
            specs, labels = result
            self._append_windows(specs, labels)

    def _build_from_arrays(self, specs: np.ndarray, labels: np.ndarray, subject_ids: np.ndarray) -> None:
        unique_subjects = np.unique(subject_ids)
        for sid in unique_subjects:
            idx = np.where(subject_ids == sid)[0]
            if idx.size == 0:
                continue
            idx = np.sort(idx)
            self._append_windows(specs[idx], labels[idx])

    def _append_windows(self, specs: np.ndarray, labels: np.ndarray) -> None:
        n = len(labels)
        # stride=5 reduces overlap from 97% to 83%, cuts leakage significantly
        # while still giving enough windows per recording
        effective_stride = max(1, self.stride)
        for start in range(0, n - self.seq_len + 1, effective_stride):
            end = start + self.seq_len
            w_spec = specs[start:end]
            w_lbl = labels[start:end]
            if np.any(w_lbl < 0):
                continue
            self.windows.append((w_spec, w_lbl))
            self.flat_labels.append(int(w_lbl[self.seq_len // 2]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        spec_np, lbl_np = self.windows[idx]
        spec = torch.from_numpy(spec_np.astype(np.float32))
        lbl = torch.from_numpy(lbl_np.astype(np.int64))
        if self.augment:
            spec = self._augment(spec)
        return spec, lbl

    def _augment(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Augmentation for raw EEG/EOG/EMG signals.
        spec shape: (T, C, L) — T epochs, C channels, L samples

        From literature:
          - Gaussian noise: robustness to electrode noise (all papers)
          - Amplitude scaling: subject-to-subject variability (SSNet)
          - Time shift: temporal invariance
          - Channel dropout: simulates bad electrode contact
          - CutOut (time-domain): random contiguous block zeroed [23] — helps N1/N2 boundary
          - Frequency scaling: slight pitch shift via resampling proxy
        """
        # Gaussian noise — small relative to signal amplitude
        if random.random() < 0.6:
            spec = spec + torch.randn_like(spec) * random.uniform(0.02, 0.06)

        # Amplitude scaling — EEG amplitude varies 2-3x across subjects
        if random.random() < 0.5:
            scale = random.uniform(0.80, 1.20)
            spec = spec * scale

        # Time shift (circular) — temporal invariance within epoch
        if random.random() < 0.3 and spec.dim() == 3:
            shift = random.randint(1, 50)  # up to 0.5s shift at 100Hz
            spec = torch.roll(spec, shifts=shift, dims=-1)

        # Channel dropout — simulates bad electrode contact
        if random.random() < 0.2 and spec.dim() == 3:
            ch = random.randint(0, spec.shape[1] - 1)
            spec[:, ch, :] = 0.0

        # CutOut (time-domain) — zero out a random contiguous block
        # From [23]: cutmix used ~20% of epoch length. Helps N1/N2 boundary.
        if random.random() < 0.4 and spec.dim() == 3:
            L = spec.shape[-1]
            cut_len = random.randint(L // 10, L // 5)  # 10-20% of signal
            cut_start = random.randint(0, L - cut_len)
            spec[:, :, cut_start:cut_start + cut_len] = 0.0

        # Sequence-level CutMix — swap a random epoch within the sequence
        # Helps model learn stage transitions (N1↔N2, N2↔N3)
        if random.random() < 0.2 and spec.dim() == 3 and spec.shape[0] > 2:
            T = spec.shape[0]
            i, j = random.sample(range(T), 2)
            # Swap two epochs in the sequence
            spec = spec.clone()
            spec[[i, j]] = spec[[j, i]]

        # Frequency masking — zero out a random frequency band in FFT domain
        # Helps model not over-rely on specific frequency bands (e.g. spindles)
        # Applied at sequence level (one random band for the whole window) for speed
        if random.random() < 0.25 and spec.dim() == 3:
            L = spec.shape[-1]
            n_fft = L // 2 + 1
            # Mask a random band of ~2-5 Hz width
            # At 100Hz, 3000 samples: 1 Hz = 30 FFT bins
            band_w = random.randint(2, 5) * 30
            band_w = min(band_w, n_fft // 4)
            band_start = random.randint(0, n_fft - band_w)
            # Vectorised: apply to all T epochs and all C channels at once
            # spec: (T, C, L) → rfft → zero band → irfft
            fft = torch.fft.rfft(spec, dim=-1)          # (T, C, n_fft)
            fft[:, :, band_start:band_start + band_w] = 0.0
            spec = torch.fft.irfft(fft, n=L, dim=-1)    # (T, C, L)

        return spec


class SyntheticSleepDataset(Dataset):
    """
    Random raw signal sequences with approximate Sleep-EDF class proportions.
    Generates (T, 4, 3000) raw signal tensors — matches new raw signal pipeline.
    """
    WEIGHTS = [0.18, 0.08, 0.45, 0.12, 0.17]

    def __init__(self, n_windows: int = 500, seq_len: int = 10,
                 n_channels: int = 4, signal_len: int = 3000,
                 # kept for backward compat — ignored
                 freq_bins: int = 128, time_bins: int = 29):
        self.n = n_windows
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.signal_len = signal_len
        labels = np.random.choice(NUM_CLASSES, size=n_windows * seq_len, p=self.WEIGHTS)
        self.labels = labels.reshape(n_windows, seq_len)
        self.flat_labels = [int(self.labels[i, seq_len // 2]) for i in range(n_windows)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # Raw signal: (T, C, L) — per-epoch Z-score already applied in preprocessing
        raw = torch.randn(self.seq_len, self.n_channels, self.signal_len)
        lbl = torch.from_numpy(self.labels[idx].astype(np.int64))
        return raw, lbl


def make_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = 8,
    num_workers: int = 0,
    balanced: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    import platform
    # Windows multiprocessing with DataLoader workers causes OSError/pickle crashes
    if platform.system() == "Windows":
        num_workers = 0
    train_sampler = None
    flat_labels = getattr(train_ds, "flat_labels", None)
    if balanced and flat_labels is not None:
        # Ensure integer dtype for bincount, handle empty lists robustly
        flat = np.asarray(flat_labels, dtype=np.int64)
        if flat.size > 0:
            counts = np.bincount(flat, minlength=NUM_CLASSES).astype(float)
            counts = np.maximum(counts, 1.0)
            class_w = 1.0 / counts
            sample_w = [float(class_w[l]) for l in flat]
            train_sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def build_sleep_edf_manifest(raw_dir: str, out_csv: Optional[str] = None) -> pd.DataFrame:
    """Backward-compatible manifest builder expected by CLI."""
    out_csv_final = out_csv or "data/manifests/sleep_edf_manifest.csv"
    records = build_manifest(raw_dir, out_csv_final)
    rows = [
        {
            "subject_id": r["subject_id"],
            "night": r["night"],
            "psg_path": r["psg"],
            "hypnogram_path": r["hypnogram"],
        }
        for r in records
    ]
    return pd.DataFrame(rows)


def read_manifest(path: str) -> pd.DataFrame:
    rows = load_manifest(path)
    normalized = []
    for r in rows:
        normalized.append(
            {
                "subject_id": r["subject_id"],
                "night": r.get("night", ""),
                "psg_path": r.get("psg_path", r.get("psg", "")),
                "hypnogram_path": r.get("hypnogram_path", r.get("hypnogram", "")),
            }
        )
    return pd.DataFrame(normalized)


def subject_level_split(
    subject_ids: np.ndarray,
    test_frac: float = 0.15,
    val_frac: float = 0.1,
    seed: int = 42,
):
    unique = np.array(sorted(set(subject_ids.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    n_test = max(1, int(len(unique) * test_frac))
    n_val = max(1, int(len(unique) * val_frac))

    # Ensure at least two subjects remain for training; shrink val/test if needed
    while (len(unique) - n_test - n_val) < 2 and (n_val + n_test) > 2:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    test_subjects = set(unique[:n_test].tolist())
    val_subjects = set(unique[n_test : n_test + n_val].tolist())

    idx = np.arange(len(subject_ids))
    test_idx = idx[np.isin(subject_ids, list(test_subjects))]
    val_idx = idx[np.isin(subject_ids, list(val_subjects))]
    train_idx = idx[~np.isin(idx, np.concatenate([val_idx, test_idx]))]
    return train_idx, val_idx, test_idx


def build_dataloaders(
    train_specs: np.ndarray,
    train_labels: np.ndarray,
    train_subjects: np.ndarray,
    val_specs: np.ndarray,
    val_labels: np.ndarray,
    val_subjects: np.ndarray,
    seq_len: int = 20,
    batch_size: int = 16,
) -> dict[str, DataLoader]:
    # stride=5 on train: reduces overlap from 97% → 75%, cuts data leakage
    # stride=1 on val: full coverage for accurate evaluation
    train_ds = SleepSequenceDataset(train_specs, train_labels, train_subjects,
                                    seq_len=seq_len, stride=5, augment=True)
    val_ds   = SleepSequenceDataset(val_specs,   val_labels,   val_subjects,
                                    seq_len=seq_len, stride=1, augment=False)

    train_dl, val_dl, _ = make_loaders(train_ds, val_ds, val_ds,
                                       batch_size=batch_size, balanced=True)
    return {"train": train_dl, "val": val_dl}
