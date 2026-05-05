from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .data import SyntheticSleepDataset, build_dataloaders, make_loaders, read_manifest, subject_level_split
from .models import StudentCRNN, TeacherCRNN
from .preprocess import SleepEEGPreprocessor, process_manifest

log = logging.getLogger(__name__)

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
NUM_CLASSES = 5


def _load_ckpt_flexible(model: nn.Module, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"warning: partial checkpoint load for {ckpt_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Pure numpy metrics — no sklearn dependency."""
    from .train import _cohen_kappa

    n = len(y_true)
    acc   = float((y_true == y_pred).mean())
    kappa = _cohen_kappa(y_true, y_pred)

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
            cm[t, p] += 1

    # Per-class precision, recall, F1
    prec = np.zeros(NUM_CLASSES)
    rec  = np.zeros(NUM_CLASSES)
    f1   = np.zeros(NUM_CLASSES)
    sup  = cm.sum(axis=1).astype(float)

    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec[i] = tp / (tp + fp + 1e-10)
        rec[i]  = tp / (tp + fn + 1e-10)
        f1[i]   = 2 * prec[i] * rec[i] / (prec[i] + rec[i] + 1e-10)

    macro_f1    = float(f1.mean())
    total_sup   = sup.sum()
    weighted_f1 = float((f1 * sup / max(1.0, total_sup)).sum())

    per_class = {}
    for i, name in enumerate(STAGE_NAMES):
        per_class[name] = {
            "precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1":        float(f1[i]),
            "support":   int(sup[i]),
        }

    return {
        "accuracy":     acc,
        "kappa":        kappa,
        "macro_f1":     macro_f1,
        "weighted_f1":  weighted_f1,
        "per_class":    per_class,
        "confusion_matrix": cm,
    }


@torch.no_grad()
def collect_preds_teacher(model: nn.Module, loader, device: torch.device):
    model.eval()
    all_pred, all_true = [], []
    for spec, labels in loader:
        spec = spec.to(device)
        out = model(spec, raw=spec, return_features=True)
        logits = out[0] if isinstance(out, tuple) else out
        preds = logits.argmax(-1).cpu().numpy().ravel()
        all_pred.extend(preds)
        all_true.extend(labels.numpy().ravel())
    return np.array(all_pred), np.array(all_true)


@torch.no_grad()
def collect_preds_student(model: nn.Module, loader, device: torch.device):
    model.eval()
    all_pred, all_true = [], []
    for spec, labels in loader:
        spec = spec.to(device)
        logits, _, _ = model(spec, return_features=True)
        preds = logits.argmax(-1).cpu().numpy().ravel()
        all_pred.extend(preds)
        all_true.extend(labels.numpy().ravel())
    return np.array(all_pred), np.array(all_true)


def print_metrics(metrics: dict, model_name: str = "Model"):
    cm = metrics.get("confusion_matrix")
    print(f"\n{'=' * 60}")
    print(f"  {model_name} Evaluation")
    print(f"{'=' * 60}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Cohen's kappa : {metrics['kappa']:.4f}")
    print(f"  Macro F1      : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1   : {metrics['weighted_f1']:.4f}")
    print("\n  Per-class metrics:")
    print(f"  {'Stage':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-' * 50}")
    for name, vals in metrics["per_class"].items():
        print(f"  {name:<8} {vals['precision']:>10.4f} {vals['recall']:>10.4f} {vals['f1']:>10.4f} {vals['support']:>10}")

    if cm is not None:
        print("\n  Confusion matrix (rows=true, cols=pred):")
        header = "  " + "".join(f"{n:>6}" for n in STAGE_NAMES)
        print(header)
        for i, row in enumerate(cm):
            print(f"  {STAGE_NAMES[i]:<5}" + "".join(f"{v:>6}" for v in row))
    print(f"{'=' * 60}\n")


def evaluate(
    teacher: Optional[nn.Module],
    student: Optional[nn.Module],
    test_loader,
    device: torch.device,
    save_dir: Optional[str] = "artifacts",
):
    results: dict[str, dict] = {}

    if teacher is not None:
        t_pred, t_true = collect_preds_teacher(teacher, test_loader, device)
        t_metrics = compute_metrics(t_true, t_pred)
        print_metrics(t_metrics, model_name="Teacher")
        results["teacher"] = t_metrics

    if student is not None:
        s_pred, s_true = collect_preds_student(student, test_loader, device)
        s_metrics = compute_metrics(s_true, s_pred)
        print_metrics(s_metrics, model_name="Student")
        results["student"] = s_metrics

    if save_dir:
        out = Path(save_dir) / "eval_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        def serialise(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            raise TypeError(f"Not serialisable: {type(obj)}")

        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=serialise)
        log.info("Saved evaluation results -> %s", out)

    _print_benchmark_targets(results)
    return results


def _print_benchmark_targets(results: dict):
    print("\nExpected performance on real Sleep-EDF (LOSO benchmark):")
    print("  +-------------+----------+----------+----------+")
    print("  | Model       | Accuracy | Kappa    | Macro F1 |")
    print("  +-------------+----------+----------+----------+")
    print("  | Teacher*    | ~83-87%  | ~0.77    | ~0.77    |")
    print("  | Student*    | ~79-83%  | ~0.73    | ~0.73    |")
    print("  +-------------+----------+----------+----------+")

    for name, vals in results.items():
        acc = vals.get("accuracy", 0.0)
        kap = vals.get("kappa", 0.0)
        mf1 = vals.get("macro_f1", 0.0)
        mode = "(synthetic)" if acc < 0.50 else "(real data)"
        print(f"  | {name.capitalize():<11} | {acc:>7.2%} | {kap:>8.4f} | {mf1:>8.4f} | {mode}")

    print("  +-------------+----------+----------+----------+")
    print("  * Targets for 20-subject Sleep-EDF Cassette LOSO split.")
    if any(v.get("accuracy", 0.0) < 0.50 for v in results.values()):
        print("\n  Low accuracy indicates synthetic/random-label evaluation.")
        print("  Run in real mode with a valid manifest for meaningful metrics.\n")


def evaluate_synthetic(
    student_ckpt: str = "artifacts/student.pt",
    device: str = "cpu",
    batch_size: int = 8,
    seq_len: int = 30,
) -> dict:
    student = StudentCRNN().to(device)
    _load_ckpt_flexible(student, student_ckpt, device)

    ds = SyntheticSleepDataset(n_windows=120, seq_len=seq_len)
    _, val_loader, _ = make_loaders(ds, ds, ds, batch_size=batch_size, balanced=False)

    results = evaluate(
        teacher=None,
        student=student,
        test_loader=val_loader,
        device=torch.device(device),
        save_dir="artifacts",
    )
    return results.get("student", {})


def evaluate_real(
    manifest_path: str,
    eeg_cfg,
    student_ckpt: str = "artifacts/student.pt",
    device: str = "cpu",
    seq_len: int = 30,
    batch_size: int = 8,
) -> dict:
    if not Path(student_ckpt).exists():
        raise FileNotFoundError(f"Student checkpoint not found: {student_ckpt}")

    manifest = read_manifest(manifest_path)
    pre = SleepEEGPreprocessor(eeg_cfg)
    specs, labels, subjects, _feats = process_manifest(manifest, preprocessor=pre, augment=False)
    _, _, test_idx = subject_level_split(subjects)

    dls = build_dataloaders(
        specs[test_idx],
        labels[test_idx],
        subjects[test_idx],
        specs[test_idx],
        labels[test_idx],
        subjects[test_idx],
        seq_len=seq_len,
        batch_size=batch_size,
    )

    student = StudentCRNN().to(device)
    _load_ckpt_flexible(student, student_ckpt, device)

    results = evaluate(
        teacher=None,
        student=student,
        test_loader=dls["val"],
        device=torch.device(device),
        save_dir="artifacts",
    )
    return results.get("student", {})


def subject_finetune(
    model: nn.Module,
    test_loader,
    device: torch.device,
    finetune_frac: float = 0.10,
    epochs: int = 5,
    lr: float = 1e-5,
) -> nn.Module:
    """
    Fine-tune the last temporal head (GRU + head) on a small fraction of a test subject's data.

    Procedure:
      - Collect all (spec, label) from the loader and randomly sample `finetune_frac` portion
      - Freeze CNN parameters; train only GRU + head
      - Return the updated model (in-place)
    """
    model.train()
    device = torch.device(device)

    # Collect all samples from loader
    all_spec = []
    all_lbl = []
    for spec, lbl in test_loader:
        all_spec.append(spec)
        all_lbl.append(lbl)
    if not all_spec:
        raise RuntimeError("Test loader yielded no data for fine-tuning")

    spec_all = torch.cat(all_spec, dim=0)
    lbl_all = torch.cat(all_lbl, dim=0)

    n_total = spec_all.shape[0]
    n_ft = max(1, int(n_total * finetune_frac))
    idx = torch.randperm(n_total)[:n_ft]

    ft_spec = spec_all[idx]
    ft_lbl = lbl_all[idx]

    ft_ds = torch.utils.data.TensorDataset(ft_spec, ft_lbl)
    ft_loader = torch.utils.data.DataLoader(ft_ds, batch_size=8, shuffle=True, num_workers=0)

    # Freeze CNN; enable GRU + head
    for name, p in model.named_parameters():
        if name.startswith("cnn"):
            p.requires_grad = False
        else:
            p.requires_grad = True

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.to(device)
    for ep in range(epochs):
        for spec_b, lbl_b in ft_loader:
            spec_b = spec_b.to(device)
            lbl_b = lbl_b.to(device)
            out = model(spec_b)
            logits = out[0] if isinstance(out, tuple) else out
            # normalize shape (B, T, C) and reshape
            if logits.dim() == 2:
                # single-epoch model returned (B, C)
                loss = crit(logits, lbl_b)
            else:
                B, T, C = logits.shape
                loss = crit(logits.view(B * T, C), lbl_b.view(B * T))
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Unfreeze all parameters
    for p in model.parameters():
        p.requires_grad = True

    model.to(device)
    model.eval()
    return model
