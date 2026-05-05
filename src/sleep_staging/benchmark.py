"""
LOSO benchmark for Sleep-EDF.

Implements true leave-one-subject-out, per-fold JSON, aggregated summary,
per-class F1 averaging, GPU-aware device selection and progress logging.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import types

import numpy as np
import torch

log = logging.getLogger(__name__)

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def _split_loso(records: List[Dict], test_subject: str):
    train_pool = [r for r in records if r["subject_id"] != test_subject]
    if not train_pool:
        return [], [], []

    train_subjects = sorted({r["subject_id"] for r in train_pool})
    # pick a small val subject from the train pool
    val_subject = train_subjects[-1] if len(train_subjects) > 1 else train_subjects[0]

    train_records = [r for r in train_pool if r["subject_id"] != val_subject]
    val_records = [r for r in train_pool if r["subject_id"] == val_subject]
    test_records = [r for r in records if r["subject_id"] == test_subject]

    if not train_records:
        train_records = val_records
    if not val_records:
        val_records = test_records

    return train_records, val_records, test_records


def run_loso_benchmark(
    manifest: str,
    seq_len: int = 30,
    teacher_epochs: int = 30,
    distill_epochs: int = 40,
    batch_size: int = 16,
    teacher_lr: float = 3e-4,
    distill_lr: float = 5e-4,
    max_folds: Optional[int] = None,
    artifacts_dir: str = "artifacts/loso",
):
    """Run LOSO across subjects listed in `manifest`.

    Saves per-fold `fold_results.json` in `artifacts_dir` and returns summary dict.
    """
    from .data import load_manifest, SleepSequenceDataset, make_loaders
    from .models import TeacherCRNN, StudentCRNN
    from .train import train_teacher
    from .distill import distill_student
    from .evaluate import collect_preds_teacher, collect_preds_student, compute_metrics

    records = load_manifest(manifest)
    subjects = sorted({r["subject_id"] for r in records})
    if max_folds:
        subjects = subjects[: max(1, max_folds)]

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_fold_results: List[Dict] = []
    existing_results_path = out_dir / "loso_results.json"
    if existing_results_path.exists():
        try:
            with open(existing_results_path, "r", encoding="utf-8") as f:
                existing_payload = json.load(f)
            existing_fold_results = list(existing_payload.get("fold_results", []))
        except Exception:
            existing_fold_results = []

    completed_subjects = {f.get("test_subject") for f in existing_fold_results if f.get("test_subject")}
    fold_results = [f for f in existing_fold_results if f.get("test_subject")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    t0 = time.time()

    for fold_idx, test_subject in enumerate(subjects, start=1):
        if test_subject in completed_subjects:
            log.info("Skipping completed LOSO fold test_subject=%s", test_subject)
            continue

        train_records, val_records, test_records = _split_loso(records, test_subject)
        if not train_records or not val_records or not test_records:
            log.warning("Skipping fold subject=%s due to insufficient records", test_subject)
            continue

        log.info("LOSO fold %d/%d test_subject=%s", fold_idx, len(subjects), test_subject)

        cfg = types.SimpleNamespace(
            seq_len=seq_len,
            batch_size=batch_size,
            epochs=teacher_epochs,
            distill_epochs=distill_epochs,
            lr=teacher_lr,
            distill_lr=distill_lr,
            cache_dir="data/cache",
            num_workers=0,
            weight_decay=1e-3,
            use_focal=True,
            focal_gamma=2.0,
            amp=True,
            patience=15,
        )

        train_ds = SleepSequenceDataset(train_records, cfg, cfg.seq_len, cache_dir=cfg.cache_dir, augment=True)
        val_ds = SleepSequenceDataset(val_records, cfg, cfg.seq_len, cache_dir=cfg.cache_dir, augment=False)
        test_ds = SleepSequenceDataset(test_records, cfg, cfg.seq_len, cache_dir=cfg.cache_dir, augment=False)

        if len(train_ds) == 0 or len(test_ds) == 0:
            log.warning("Fold %s: empty datasets, skipping", test_subject)
            continue

        train_loader, val_loader, test_loader = make_loaders(
            train_ds,
            val_ds,
            test_ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

        # Train teacher
        teacher = TeacherCRNN().to(device)
        teacher_ckpt = out_dir / f"teacher_{test_subject}.pt"
        train_teacher(teacher, train_loader, val_loader, cfg, device, out_path=str(teacher_ckpt))

        # Distill
        student = StudentCRNN().to(device)
        student_ckpt = out_dir / f"student_{test_subject}.pt"
        distill_student(teacher, student, train_loader, val_loader, cfg, device, out_path=str(student_ckpt))

        # Evaluate
        t_pred, t_true = collect_preds_teacher(teacher, test_loader, device)
        s_pred, s_true = collect_preds_student(student, test_loader, device)

        teacher_metrics = compute_metrics(t_true, t_pred)
        student_metrics = compute_metrics(s_true, s_pred)

        # Save fold result
        fold_result = {
            "fold": fold_idx,
            "test_subject": test_subject,
            "n_train_windows": len(train_ds),
            "n_test_windows": len(test_ds),
            "teacher": teacher_metrics,
            "student": student_metrics,
        }
        (out_dir / f"fold_{fold_idx:02d}_{test_subject}").mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"fold_{fold_idx:02d}_{test_subject}" / "fold_results.json", "w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=2)

        fold_results.append(fold_result)

    if not fold_results:
        raise RuntimeError("LOSO produced no valid folds")

    # Aggregate metrics
    def _mean_std(vals):
        return float(np.mean(vals)), float(np.std(vals))

    summary = {"folds": len(fold_results)}
    for model in ("teacher", "student"):
        accs = [f[model]["accuracy"] for f in fold_results]
        kps = [f[model]["kappa"] for f in fold_results]
        mfs = [f[model]["macro_f1"] for f in fold_results]
        wf = [f[model].get("weighted_f1", 0.0) for f in fold_results]

        acc_m, acc_s = _mean_std(accs)
        kp_m, kp_s = _mean_std(kps)
        mf_m, mf_s = _mean_std(mfs)
        wf_m, wf_s = _mean_std(wf)

        # per-class f1
        per_class = {}
        for s in STAGE_NAMES:
            vals = []
            for f in fold_results:
                pc = f[model].get("per_class", {})
                if s in pc and "f1" in pc[s]:
                    vals.append(pc[s]["f1"])
            per_class[s] = {"f1_mean": float(np.mean(vals)) if vals else 0.0, "f1_std": float(np.std(vals)) if vals else 0.0}

        summary[model] = {
            "accuracy_mean": acc_m,
            "accuracy_std": acc_s,
            "kappa_mean": kp_m,
            "kappa_std": kp_s,
            "macro_f1_mean": mf_m,
            "macro_f1_std": mf_s,
            "weighted_f1_mean": wf_m,
            "weighted_f1_std": wf_s,
            "per_class": per_class,
        }

    summary["n_subjects"] = len(subjects)
    summary["n_folds"] = len(fold_results)
    summary["total_time_min"] = round((time.time() - t0) / 60.0, 1)

    # Save summary
    with open(out_dir / "loso_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "fold_results": fold_results}, f, indent=2)

    log.info("Saved LOSO benchmark -> %s", out_dir / "loso_results.json")
    return {"summary": summary, "folds": fold_results}
