from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import SyntheticSleepDataset, build_dataloaders, make_loaders, read_manifest, subject_level_split
from .models import FeatureProjector, StudentCRNN, TeacherCRNN
from .preprocess import SleepEEGPreprocessor, process_manifest
from .train import evaluate_quick, infer_class_weights_from_loader, seq_ce_loss

log = logging.getLogger(__name__)

NUM_CLASSES = 5


def _autocast_context(device: torch.device, enabled: bool):
    """
    Use float32 on GTX 1650 (Turing — no bfloat16, float16 causes NaN in attention).
    Use bfloat16 on RTX 30xx+ for speed without NaN risk.
    """
    if not enabled or device.type != "cuda":
        return nullcontext()
    if torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    # GTX 1650: fall back to float32 — safe, no NaN
    return nullcontext()


def _load_ckpt_flexible(model: nn.Module, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"warning: partial checkpoint load for {ckpt_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )


def _teacher_logits_feat(teacher: nn.Module, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # spec is actually raw signal (B, T, C, L) in the new pipeline
    out = teacher(spec, raw=spec, return_features=True)
    if isinstance(out, tuple):
        return out[0], out[1]
    raise RuntimeError("Teacher forward did not return expected tuple")


def rkd_distance_loss(s_feat: torch.Tensor, t_feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Relational KD distance loss using normalized pairwise distances."""

    def pdist(feat: torch.Tensor) -> torch.Tensor:
        sq = feat.pow(2).sum(1, keepdim=True)
        dist = sq + sq.T - 2 * feat @ feat.T
        dist = dist.clamp(min=0).sqrt()
        dist = dist.clamp(max=1e8)  # Clip to prevent inf
        mu = dist[dist > 0].mean() if (dist > 0).any() else 1.0
        mu = mu + eps
        return dist / mu

    try:
        s_dist = pdist(s_feat)
        t_dist = pdist(t_feat)
        loss = F.smooth_l1_loss(s_dist, t_dist)
        # Fallback to zero if NaN
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=s_feat.device)
        return loss
    except Exception:
        return torch.tensor(0.0, device=s_feat.device)


class DistillationLoss(nn.Module):
    def __init__(
        self,
        student_feat_dim: int = 96,
        teacher_feat_dim: int = 256,
        temperature: float = 6.0,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.1,
        delta: float = 0.1,
        class_weights: torch.Tensor | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        if class_weights is None:
            cw = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
        else:
            cw = class_weights.to(device)
        # label_smoothing=0.05: regularises hard-label CE, improves soft-target distillation
        self.ce = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()
        self.projector = FeatureProjector(student_feat_dim, teacher_feat_dim).to(device)

    def forward(
        self,
        s_logits: torch.Tensor,
        t_logits: torch.Tensor,
        s_feat: torch.Tensor,
        t_feat: torch.Tensor,
        labels: torch.Tensor,
        focal_gamma: float = 1.5,
    ) -> dict:
        batch, seq_len, n_classes = s_logits.shape

        # Focal CE: down-weights easy examples, focuses on hard N1/N2 boundary
        # gamma=1.5: gentler than standard 2.0, better for sleep staging
        B, T, C = s_logits.shape
        flat_logits = s_logits.view(B * T, C)
        flat_labels = labels.view(B * T)
        base_ce = self.ce(flat_logits, flat_labels)
        with torch.no_grad():
            probs = torch.softmax(flat_logits.detach(), dim=-1)
            p_t = probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
            focal_w = (1.0 - p_t).pow(focal_gamma)
        per_sample_ce = F.cross_entropy(
            flat_logits, flat_labels,
            weight=self.ce.weight if hasattr(self.ce, 'weight') else None,
            reduction='none',
        )
        ce_loss = (focal_w * per_sample_ce).mean()
        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            ce_loss = base_ce  # fallback to plain CE
        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            ce_loss = s_logits[0, 0, 0] * 0.0 + 0.5  # Safe default with grad
        
        # KL: soft-target matching
        with torch.no_grad():
            t_logits_clip = t_logits.clamp(min=-10, max=10)
        s_logits_clip = s_logits.clamp(min=-10, max=10)
        
        s_soft = F.log_softmax(s_logits_clip.view(batch * seq_len, n_classes) / self.T, dim=-1)
        t_soft = F.softmax(t_logits_clip.view(batch * seq_len, n_classes) / self.T, dim=-1)
        
        if torch.isnan(s_soft).any() or torch.isnan(t_soft).any():
            kl_loss = s_logits[0, 0, 0] * 0.0
        else:
            kl_loss = self.kl(s_soft, t_soft) * (self.T**2)
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = s_logits[0, 0, 0] * 0.0

        # Feature MSE — L2-normalised (bounded [0,2], never NaN)
        # cosine-space MSE = 1 - cosine_similarity, safe and stable
        if self.gamma > 0:
            try:
                s_proj = self.projector(s_feat.reshape(batch * seq_len, -1))
                t_flat = t_feat.reshape(batch * seq_len, -1).detach()
                s_norm = F.normalize(s_proj, p=2, dim=-1)
                t_norm = F.normalize(t_flat, p=2, dim=-1)
                feat_loss = F.mse_loss(s_norm, t_norm)
                if torch.isnan(feat_loss) or torch.isinf(feat_loss):
                    feat_loss = s_logits[0, 0, 0] * 0.0
            except Exception:
                feat_loss = s_logits[0, 0, 0] * 0.0
        else:
            feat_loss = s_logits[0, 0, 0] * 0.0

        rkd_loss = s_logits[0, 0, 0] * 0.0

        # Compute total with proper scalar handling
        ce_scalar = ce_loss if not torch.isnan(ce_loss) else torch.ones(1, device=s_logits.device)[0] * 0.5
        kl_scalar = kl_loss if not torch.isnan(kl_loss) else torch.zeros(1, device=s_logits.device)[0]
        fl_scalar = feat_loss if not torch.isnan(feat_loss) else torch.zeros(1, device=s_logits.device)[0]

        total = self.alpha * ce_scalar + self.beta * kl_scalar + self.gamma * fl_scalar
        
        return {
            "total": total,
            "ce": float(ce_loss.detach().item()) if not torch.isnan(ce_loss) else 0.5,
            "kl": float(kl_loss.detach().item()) if not torch.isnan(kl_loss) else 0.0,
            "feat": float(feat_loss.detach().item()) if not torch.isnan(feat_loss) else 0.0,
            "rkd": float(rkd_loss.detach().item()) if not torch.isnan(rkd_loss) else 0.0,
        }


def _eval_student(student: nn.Module, loader, device: torch.device):
    from .train import _cohen_kappa

    student.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for spec, labels in loader:
            spec = spec.to(device)
            logits, _, _ = student(spec, return_features=True)
            preds = logits.argmax(-1).cpu().numpy().ravel()
            truth = labels.numpy().ravel()
            all_pred.extend(preds)
            all_true.extend(truth)

    all_pred_arr = np.array(all_pred)
    all_true_arr = np.array(all_true)
    acc   = float((all_pred_arr == all_true_arr).mean())
    kappa = _cohen_kappa(all_true_arr, all_pred_arr)
    return acc, kappa


def distill_student(
    teacher: nn.Module,
    student: nn.Module,
    train_loader,
    val_loader,
    cfg,
    device: torch.device,
    out_path: str = "artifacts/student.pt",
    teacher_ckpt_path: str = "",   # Always reload from disk to ensure consistency
    grad_accum: int = 2,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Initialize augmentation pipeline ──────────────────────────────────────
    from .augmentation import AugmentationPipeline
    aug_pipeline = AugmentationPipeline(cfg)
    log.info("Augmentation pipeline initialized (CutMix, ChannelDropout, GaussianNoise, TimeShift)")

    # ── Always reload teacher from checkpoint ─────────────────────────────────
    # Root cause of "student beats teacher": distillation was running against
    # the in-memory teacher (epoch 22, κ=0.437) while the saved checkpoint was
    # from epoch 3 (κ=0.397). Reloading ensures the student learns from the
    # exact same model that was evaluated.
    if teacher_ckpt_path and Path(teacher_ckpt_path).exists():
        ckpt = torch.load(teacher_ckpt_path, map_location=str(device))
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        teacher.load_state_dict(state, strict=False)
        log.info("Reloaded teacher from checkpoint: %s", teacher_ckpt_path)
    else:
        log.warning("No teacher checkpoint path provided — using in-memory weights")

    teacher = teacher.to(device).eval()
    student = student.to(device)
    class_weights = infer_class_weights_from_loader(train_loader).to(device)
    amp_enabled = bool(getattr(cfg, "amp", True) and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Temperature schedule: T_start=8 → T_end=4 over training
    T_start = 8.0
    T_end   = 4.0

    criterion = DistillationLoss(
        student_feat_dim=64,
        teacher_feat_dim=getattr(cfg, "d_model", 128),
        temperature=T_start,  # will be updated per epoch
        alpha=getattr(cfg, "kd_alpha", 0.3),
        beta=getattr(cfg, "kd_beta", 0.4),
        gamma=getattr(cfg, "kd_gamma", 0.2),   # re-enabled with L2 normalisation
        delta=getattr(cfg, "kd_delta", 0.1),
        class_weights=class_weights,
        device=device,
    )

    all_params = list(student.parameters()) + list(criterion.projector.parameters())
    optimizer = AdamW(all_params, lr=getattr(cfg, "distill_lr", 5e-4), weight_decay=1e-4)

    total_steps = max(1, math.ceil(max(1, len(train_loader)) / max(1, grad_accum)) * cfg.epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    best_kappa = -1.0
    patience_ctr = 0
    early_stop_patience = int(getattr(cfg, "patience", 15))

    for epoch in range(1, cfg.epochs + 1):
        # Temperature annealing: T_start=8 → T_end=4 over training
        # High T early = soft labels (exploration), low T late = harder labels (discrimination)
        T_current = T_start - (T_start - T_end) * (epoch / max(1, cfg.epochs))
        criterion.T = T_current

        # α/β curriculum: start with more soft KD (β=0.5), shift to hard CE (α=0.5) late
        # Early: β=0.5, α=0.3 (soft labels dominate) → Late: β=0.3, α=0.5 (hard labels dominate)
        progress = (epoch - 1) / max(1, cfg.epochs)
        alpha_current = 0.3 + 0.2 * progress  # 0.3 → 0.5
        beta_current  = 0.5 - 0.2 * progress  # 0.5 → 0.3
        criterion.alpha = alpha_current
        criterion.beta  = beta_current

        student.train()
        criterion.projector.train()
        running = {k: 0.0 for k in ("total", "ce", "kl", "feat", "rkd")}
        n_batches = 0
        data_wait_total = 0.0
        step_time_total = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        prev_end = time.time()

        for step, (spec, labels) in enumerate(train_loader, 1):
            data_wait_total += time.time() - prev_end
            spec = spec.to(device, non_blocking=amp_enabled)
            labels = labels.to(device, non_blocking=amp_enabled)
            hard_labels = labels.long()

            # Apply augmentation pipeline during training
            spec, labels = aug_pipeline(spec, labels, epoch=epoch, training=True)

            step_start = time.time()
            with _autocast_context(device, amp_enabled):
                with torch.no_grad():
                    t_logits, t_feat = _teacher_logits_feat(teacher, spec)

                s_logits, s_feat, _ = student(spec, return_features=True)

                # Base distillation losses (Focal CE + KL)
                losses = criterion(s_logits, t_logits, s_feat, t_feat, hard_labels, focal_gamma=1.5)

                # RKD: only enable after warmup epochs to avoid explosion
                use_rkd = (epoch > 2) and (getattr(criterion, "delta", 0.0) > 0.0)
                rkd_val = torch.tensor(0.0, device=spec.device)
                try:
                    if use_rkd:
                        b, seq_len, _ = s_feat.shape
                        s_flat = s_feat.view(b * seq_len, -1)
                        t_flat = t_feat.view(b * seq_len, -1)
                        s_proj = criterion.projector(s_flat)
                        rkd_val = rkd_distance_loss(s_proj, t_flat)
                    else:
                        rkd_val = torch.tensor(0.0, device=spec.device)
                except Exception:
                    rkd_val = torch.tensor(0.0, device=spec.device)

                # Combine losses (keep main path safe from NaN)
                total = losses["total"]
                if use_rkd:
                    total = total + getattr(criterion, "delta", 0.0) * rkd_val

            # Guard against NaN/Inf before backward
            if not torch.isfinite(total):
                optimizer.zero_grad()
                log.warning("Non-finite loss encountered at epoch %d step %d — skipping step", epoch, step)
                continue

            loss_val = total / max(1, grad_accum)
            scaler.scale(loss_val).backward()

            if step % max(1, grad_accum) == 0 or step == len(train_loader):
                # Clip student and projector separately to stabilise training
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                nn.utils.clip_grad_norm_(criterion.projector.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Accumulate losses - convert all to float scalars immediately
            running["total"] += float(total.item()) if torch.isfinite(total) else 0.0
            running["ce"] += losses["ce"]
            running["kl"] += losses["kl"]
            running["feat"] += losses["feat"]
            running["rkd"] += float(rkd_val.item()) if torch.isfinite(rkd_val) else 0.0
            n_batches += 1
            step_time_total += time.time() - step_start
            prev_end = time.time()

        avg = {k: v / max(1, n_batches) for k, v in running.items()}
        val_acc, val_kappa = _eval_student(student, val_loader, device)
        data_wait_avg = data_wait_total / max(1, n_batches)
        step_time_avg = step_time_total / max(1, n_batches)

        log.info(
            "Distill %03d/%d total=%.4f ce=%.3f kl=%.3f feat=%.3f rkd=%.3f val_acc=%.4f val_kappa=%.4f t=%.1fs data_wait=%.3fs step=%.3fs",
            epoch,
            cfg.epochs,
            avg["total"],
            avg["ce"],
            avg["kl"],
            avg["feat"],
            avg["rkd"],
            val_acc,
            val_kappa,
            time.time() - t0,
            data_wait_avg,
            step_time_avg,
        )

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            patience_ctr = 0
            torch.save({"model_state": student.state_dict()}, out_path)
            log.info("Saved best student (kappa=%.4f)", best_kappa)
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    _load_ckpt_flexible(student, out_path, str(device))
    log.info("Distillation done. Best val kappa=%.4f", best_kappa)
    return student


def distill_synthetic(
    epochs: int = 3,
    device: str = "cpu",
    teacher_ckpt: str = "artifacts/teacher.pt",
    save_path: str = "artifacts/student.pt",
) -> None:
    class Cfg:
        pass

    cfg = Cfg()
    cfg.epochs = epochs
    cfg.patience = min(5, epochs)
    cfg.d_model = 256

    train_ds = SyntheticSleepDataset(n_windows=320, seq_len=30)
    val_ds = SyntheticSleepDataset(n_windows=100, seq_len=30)
    train_loader, val_loader, _ = make_loaders(train_ds, val_ds, val_ds, batch_size=8, balanced=True)

    teacher = TeacherCRNN().to(device)
    if Path(teacher_ckpt).exists():
        _load_ckpt_flexible(teacher, teacher_ckpt, device)

    student = StudentCRNN().to(device)
    distill_student(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=torch.device(device),
        out_path=save_path,
        grad_accum=2,
    )


def distill_real(
    manifest_path: str,
    eeg_cfg,
    epochs: int = 10,
    device: str = "cpu",
    teacher_ckpt: str = "artifacts/teacher.pt",
    save_path: str = "artifacts/student.pt",
    batch_size: int = 8,
    seq_len: int = 30,
) -> None:
    if not Path(teacher_ckpt).exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")

    manifest = read_manifest(manifest_path)
    pre = SleepEEGPreprocessor(eeg_cfg)
    specs, labels, subjects, _feats = process_manifest(manifest, preprocessor=pre, augment=True)
    tr_idx, val_idx, _ = subject_level_split(subjects)

    dls = build_dataloaders(
        specs[tr_idx],
        labels[tr_idx],
        subjects[tr_idx],
        specs[val_idx],
        labels[val_idx],
        subjects[val_idx],
        seq_len=seq_len,
        batch_size=batch_size,
    )

    class Cfg:
        pass

    cfg = Cfg()
    cfg.epochs = epochs
    cfg.patience = min(10, max(3, epochs // 2))
    cfg.d_model = 256

    teacher = TeacherCRNN().to(device)
    _load_ckpt_flexible(teacher, teacher_ckpt, device)
    student = StudentCRNN().to(device)

    distill_student(
        teacher=teacher,
        student=student,
        train_loader=dls["train"],
        val_loader=dls["val"],
        cfg=cfg,
        device=torch.device(device),
        out_path=save_path,
        grad_accum=2,
    )
