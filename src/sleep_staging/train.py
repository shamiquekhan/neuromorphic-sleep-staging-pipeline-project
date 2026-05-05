from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from .config import TrainConfig
from .data import SyntheticSleepDataset, build_dataloaders, make_loaders, read_manifest, subject_level_split
from .models import TeacherCRNN
from .preprocess import SleepEEGPreprocessor, process_manifest
from .losses import FocalLoss

log = logging.getLogger(__name__)

NUM_CLASSES = 5

# ---------------------------------------------------------------------------
# Confusion-matrix-informed class weights
# Based on eval_results.json analysis (teacher model):
#   Wake:  recall=78%, FP=1700  → weight 1.0 (well-classified)
#   N1:    FP rate=13.6%, hardest class → weight 1.5 (Li & Gao 2023)
#   N2:    recall=58%, 2296 misclassified as N3 → weight 1.5 (boost recall)
#   N3:    FP=2740 (over-predicted), recall=88% → weight 0.5 (reduce over-prediction)
#   REM:   recall=62%, 522 misclassified as N1 → weight 1.5 (boost recall)
# ---------------------------------------------------------------------------

CONFUSION_INFORMED_WEIGHTS = np.array([
    1.0,   # Wake  — recall 78%, well-classified
    1.5,   # N1    — hardest class, transitional, Li & Gao use 1.5x
    1.5,   # N2    — recall only 58%, 2296 N2→N3 errors, needs strong boost
    0.5,   # N3    — over-predicted (FP=2740), reduce weight to discourage
    1.5,   # REM   — recall 62%, 522 REM→N1 errors, needs boost
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Class weight computation — confusion-informed + sqrt-dampened
# ---------------------------------------------------------------------------

def infer_class_weights_from_loader(loader, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Confusion-matrix-informed class weights.

    Combines:
    1. Sqrt-dampened inverse-frequency (handles class imbalance)
    2. Confusion-informed multipliers (addresses specific FP/FN patterns)

    From eval_results.json analysis:
      - N2 recall=58% → weight boosted to 1.4x
      - N3 FP=2740 (over-predicted) → weight reduced to 0.7x
      - N1 FP rate=13.6% → weight 1.5x (Li & Gao 2023 recommendation)
    """
    dataset = getattr(loader, "dataset", None)
    flat_labels = getattr(dataset, "flat_labels", None)
    if flat_labels is None:
        labels = []
        for _, batch_labels in loader:
            labels.extend(int(v) for v in batch_labels.view(-1).tolist())
        flat = np.asarray(labels, dtype=np.int64)
    else:
        flat = np.asarray(flat_labels, dtype=np.int64)

    counts = np.bincount(flat, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    total = counts.sum()

    # Sqrt-dampened inverse frequency (base)
    raw_weights = total / (num_classes * counts)
    freq_weights = np.sqrt(raw_weights)
    freq_weights = freq_weights / (freq_weights.mean() + 1e-8)

    # Apply confusion-informed multipliers
    combined = freq_weights * CONFUSION_INFORMED_WEIGHTS
    # Renormalize so mean = 1.0
    combined = combined / (combined.mean() + 1e-8)

    return torch.tensor(combined, dtype=torch.float32)


def compute_class_weights(loader, device: torch.device, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Compatibility wrapper — moves result to device."""
    cw = infer_class_weights_from_loader(loader, num_classes=num_classes)
    return cw.to(device)


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------

def mixup_batch(
    spec: torch.Tensor, labels: torch.Tensor, alpha: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return mixed (spec, lbl_a, lbl_b, lam). alpha=0.3 is gentler than 0.4."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(spec.size(0), device=spec.device)
    return lam * spec + (1.0 - lam) * spec[idx], labels, labels[idx], lam


def mixup_criterion(
    criterion_fn, logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
):
    return lam * criterion_fn(logits, y_a) + (1.0 - lam) * criterion_fn(logits, y_b)


# ---------------------------------------------------------------------------
# LR scheduler — linear warmup + cosine decay (SequentialLR)
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """
    Linear warmup to base_lr, then cosine decay to 1e-6.
    No max_lr multiplication — peak LR = optimizer's configured lr.
    Uses SequentialLR (stable, no custom lambda needed).
    """
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=max(1, warmup_steps),
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=1e-6,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[max(1, warmup_steps)],
    )


# ---------------------------------------------------------------------------
# AMP context — float32 on GTX 1650 (no bfloat16 support on Turing)
# ---------------------------------------------------------------------------

def _autocast_context(device: torch.device, amp_dtype: torch.dtype):
    """
    Use bfloat16 if supported (RTX 30xx+), else float32.
    GTX 1650 (Turing) does NOT support bfloat16 — falls back to float32.
    float16 is avoided because it causes silent NaN in transformer attention on Turing.
    """
    if device.type == "cuda" and amp_dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _get_amp_dtype(device: torch.device, amp_enabled: bool) -> torch.dtype:
    if not amp_enabled or device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16   # RTX 30xx, A-series — fast and NaN-free
    return torch.float32        # GTX 1650 — safe fallback


# ---------------------------------------------------------------------------
# CUDA kernel warmup — eliminates 39-min first epoch
# ---------------------------------------------------------------------------

def _cuda_warmup(
    model: nn.Module,
    device: torch.device,
    freq_bins: int = 128,
    time_bins: int = 29,
    seq_len: int = 10,
    n_channels: int = 4,
    signal_len: int = 3000,
) -> None:
    """
    Run 3 dummy forward+backward passes to compile CUDA kernels before training.
    Uses raw signal shape (B, T, C, L) matching the 1D-ResNet-SE model.
    seq_len must match the actual DataLoader output.
    """
    if device.type != "cuda":
        return
    log.info("Warming up CUDA kernels (3 passes, seq_len=%d, signal_len=%d)...", seq_len, signal_len)
    # Raw signal: (B, T, C, L)
    dummy_raw = torch.randn(2, seq_len, n_channels, signal_len, device=device)
    dummy_lbl = torch.zeros(2, seq_len, dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(3):
        out = model(dummy_raw, raw=dummy_raw)
        logits = out[0] if isinstance(out, tuple) else out
        B, T, C = logits.shape
        loss = criterion(logits.view(B * T, C), dummy_lbl.view(B * T))
        loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    log.info("CUDA warmup complete — subsequent epochs will run at full speed.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seq_ce_loss(criterion: nn.Module, logits: torch.Tensor, labels: torch.Tensor):
    """Flatten sequence logits/labels and compute CE."""
    B, T, C = logits.shape
    return criterion(logits.view(B * T, C), labels.view(B * T))


def seq_focal_ce_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 1.5,
) -> torch.Tensor:
    """
    Focal CE loss — down-weights easy examples, focuses on hard ones.
    From MSDFN (Duan et al. 2021) and SSNet (Almutairi et al. 2023).
    gamma=1.5: moderate focusing (2.0 is standard, 1.5 is gentler for sleep staging).
    Particularly helps with N1/N2 boundary confusion.
    """
    B, T, C = logits.shape
    flat_logits = logits.view(B * T, C)
    flat_labels = labels.view(B * T)

    # Standard CE (with class weights from criterion)
    ce = criterion(flat_logits, flat_labels)

    # Focal weight: (1 - p_t)^gamma
    with torch.no_grad():
        probs = torch.softmax(flat_logits.detach(), dim=-1)
        p_t = probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t).pow(gamma)

    # Apply focal weight to per-sample CE
    per_sample_ce = torch.nn.functional.cross_entropy(
        flat_logits, flat_labels,
        weight=criterion.weight if hasattr(criterion, 'weight') else None,
        reduction='none',
    )
    focal_loss = (focal_weight * per_sample_ce).mean()
    return focal_loss


def _unpack_teacher_logits(output):
    return output[0] if isinstance(output, tuple) else output


def _save_teacher_checkpoint(model: nn.Module, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, out_path)


def _load_teacher_checkpoint(model: nn.Module, out_path: str, device: torch.device) -> None:
    ckpt = torch.load(out_path, map_location=device)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_teacher(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg,
    device: torch.device,
    out_path: str = "artifacts/teacher.pt",
    use_mixup: bool = True,    # Re-enabled — delayed until epoch 5 to allow convergence first
    grad_accum: int = 2,       # HARDCODED — never change to 4
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    # ── CUDA warmup — infer seq_len from actual loader, not config ──────────
    n_channels  = int(getattr(cfg, "n_channels", 4))
    signal_len  = int(getattr(cfg, "signal_len", 3000))
    try:
        _sample, _ = next(iter(train_loader))
        actual_seq_len  = _sample.shape[1]
        actual_sig_len  = _sample.shape[-1]
        del _sample
    except Exception:
        actual_seq_len = int(getattr(cfg, "seq_len", 10))
        actual_sig_len = signal_len
    _cuda_warmup(
        model, device,
        seq_len=actual_seq_len,
        n_channels=n_channels,
        signal_len=actual_sig_len,
    )

    # ── File logging ──────────────────────────────────────────────────────────
    try:
        log_path = Path(out_path).parent / "train_history.log"
        already = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(log_path.resolve())
            for h in log.handlers
        )
        if not already:
            fh = logging.FileHandler(log_path, mode="a")
            fh.setLevel(logging.INFO)
            root_handlers = logging.getLogger().handlers
            if root_handlers:
                fh.setFormatter(root_handlers[0].formatter)
            log.addHandler(fh)
    except Exception:
        pass

    # ── Class weights (sqrt-dampened) ─────────────────────────────────────────
    class_weights = compute_class_weights(train_loader, device)
    log.info("Class weights (sqrt-norm): %s", str(class_weights.cpu().numpy().round(3).tolist()))
    # Use FocalLoss if configured, else standard CrossEntropy
    if getattr(cfg, "use_focal", False):
        log.info("Using FocalLoss (gamma=%.2f)", getattr(cfg, "focal_gamma", 1.5))
        criterion = FocalLoss(gamma=getattr(cfg, "focal_gamma", 1.5), alpha=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ── AMP — float32 on GTX 1650, bfloat16 on newer GPUs ────────────────────
    amp_enabled = bool(getattr(cfg, "amp", True))
    amp_dtype = _get_amp_dtype(device, amp_enabled)
    log.info("AMP dtype: %s", str(amp_dtype))
    # GradScaler only needed for float16; float32/bfloat16 don't need it
    use_scaler = amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ── Scheduler — linear warmup + cosine decay ──────────────────────────────
    steps_per_epoch = math.ceil(max(1, len(train_loader)) / max(1, grad_accum))
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = max(1, steps_per_epoch * max(1, cfg.epochs // 10))  # 10% warmup
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    best_kappa = -1.0
    patience_ctr = 0
    early_stop_patience = int(getattr(cfg, "patience", 15))

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        data_wait_total = 0.0
        step_time_total = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        prev_end = time.time()

        # Delayed mixup: only after epoch 5 to allow initial convergence
        use_mixup_this_epoch = use_mixup and (epoch > 5)

        for step, (spec, labels) in enumerate(train_loader, 1):
            data_wait_total += time.time() - prev_end
            spec = spec.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            step_start = time.time()
            with _autocast_context(device, amp_dtype):
                if use_mixup_this_epoch:
                    spec, lbl_a, lbl_b, lam = mixup_batch(spec, labels, alpha=0.3)
                else:
                    lbl_a, lbl_b, lam = labels, labels, 1.0

                logits = _unpack_teacher_logits(model(spec, raw=spec))

                if use_mixup_this_epoch:
                    loss = mixup_criterion(
                        lambda l, y: seq_ce_loss(criterion, l, y),
                        logits, lbl_a, lbl_b, lam,
                    )
                else:
                    # Use sequence CE with selected criterion (CrossEntropy or FocalLoss)
                    loss = seq_ce_loss(criterion, logits, labels)

            loss = loss / max(1, grad_accum)
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % max(1, grad_accum) == 0 or step == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item()) * max(1, grad_accum)
            n_batches += 1
            step_time_total += time.time() - step_start
            prev_end = time.time()

        train_loss = epoch_loss / max(1, n_batches)
        val_acc, val_kappa = evaluate_quick(model, val_loader, device)
        elapsed = time.time() - t0

        # Get current LR safely from both SequentialLR and LambdaLR
        try:
            current_lr = scheduler.get_last_lr()[0]
        except Exception:
            current_lr = optimizer.param_groups[0]["lr"]

        log.info(
            "Epoch %03d/%d  loss=%.4f  val_acc=%.4f  val_kappa=%.4f  lr=%.2e  "
            "t=%.1fs  data=%.3fs  step=%.3fs  mixup=%s",
            epoch, cfg.epochs, train_loss, val_acc, val_kappa, current_lr,
            elapsed,
            data_wait_total / max(1, n_batches),
            step_time_total / max(1, n_batches),
            "ON" if use_mixup_this_epoch else "off",
        )

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            patience_ctr = 0
            _save_teacher_checkpoint(model, out_path)
            log.info("  >> Saved best teacher (kappa=%.4f)", best_kappa)
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                log.info("Early stopping at epoch %d (patience=%d)", epoch, early_stop_patience)
                break

    _load_teacher_checkpoint(model, out_path, device)
    log.info("Teacher training done. Best val kappa=%.4f", best_kappa)
    return model


def _cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    """Pure numpy Cohen's kappa — no sklearn dependency."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    n = cm.sum()
    if n == 0:
        return 0.0
    po = cm.diagonal().sum() / n
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    pe = (row_sums * col_sums).sum() / (n * n)
    return float((po - pe) / (1.0 - pe + 1e-10))


def evaluate_quick(model: nn.Module, loader, device: torch.device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for spec, labels in loader:
            spec = spec.to(device, non_blocking=True)
            logits = _unpack_teacher_logits(model(spec, raw=spec))
            all_pred.extend(logits.argmax(-1).cpu().numpy().ravel())
            all_true.extend(labels.numpy().ravel())

    pred = np.array(all_pred)
    true = np.array(all_true)
    acc  = float((pred == true).mean())
    kappa = _cohen_kappa(true, pred)
    return acc, kappa


def train_teacher_synthetic(cfg: TrainConfig, device: str = "cpu", save_path: str = "artifacts/teacher.pt") -> None:
    model = TeacherCRNN()
    train_ds = SyntheticSleepDataset(n_windows=max(200, cfg.batch_size * 40), seq_len=cfg.seq_len)
    val_ds = SyntheticSleepDataset(n_windows=max(80, cfg.batch_size * 10), seq_len=cfg.seq_len)
    train_loader, val_loader, _ = make_loaders(train_ds, val_ds, val_ds, batch_size=cfg.batch_size, num_workers=0, balanced=True)
    train_teacher(model=model, train_loader=train_loader, val_loader=val_loader, cfg=cfg,
                  device=torch.device(device), out_path=save_path, use_mixup=True, grad_accum=2)


def train_teacher_real(manifest_path: str, cfg: TrainConfig, eeg_cfg, device: str = "cpu",
                       save_path: str = "artifacts/teacher.pt") -> None:
    manifest = read_manifest(manifest_path)
    preprocessor = SleepEEGPreprocessor(eeg_cfg)
    specs, labels, subjects, _feats = process_manifest(manifest, preprocessor=preprocessor, augment=False)
    tr_idx, val_idx, _ = subject_level_split(subjects)
    dls = build_dataloaders(specs[tr_idx], labels[tr_idx], subjects[tr_idx],
                            specs[val_idx], labels[val_idx], subjects[val_idx],
                            seq_len=cfg.seq_len, batch_size=cfg.batch_size)
    model = TeacherCRNN()
    train_teacher(model=model, train_loader=dls["train"], val_loader=dls["val"], cfg=cfg,
                  device=torch.device(device), out_path=save_path, use_mixup=True, grad_accum=2)
