from __future__ import annotations

import argparse
import logging
import os
import sys
import types
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def default_cfg(**overrides) -> types.SimpleNamespace:
    cfg = types.SimpleNamespace(
        fs=100,
        epoch_sec=30,
        freq_bins=128,
        time_bins=29,
        seq_len=15,
        d_model=128,
        hidden=128,
        batch_size=32,
        epochs=60,
        lr=1e-4,
        weight_decay=5e-5,
        patience=15,
        distill_epochs=40,
        distill_lr=5e-4,
        kd_temperature=6.0,
        kd_alpha=0.7,
        kd_beta=0.3,
        kd_gamma=0.0,
        kd_delta=0.0,
        val_frac=0.15,
        test_frac=0.15,
        cache_dir="data/cache",
        # Windows multiprocessing with DataLoader workers is unreliable;
        # num_workers=0 runs loading in the main process — safe on all platforms
        num_workers=0,
        amp=True,
        profile_steps=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _load_model_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except RuntimeError as exc:
        msg = str(exc)
        if "deserialize object on CUDA device" in msg or "device_count() is 0" in msg:
            log.warning("Falling back to CPU checkpoint load for %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        else:
            raise
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.warning(
            "Partial checkpoint load for %s (missing=%d unexpected=%d)",
            ckpt_path,
            len(missing),
            len(unexpected),
        )


def get_loaders(args, cfg):
    if args.mode == "synthetic":
        from sleep_staging.data import SyntheticSleepDataset, make_loaders

        n_windows = getattr(args, "synthetic_n", 1000)
        train_ds = SyntheticSleepDataset(
            n_windows=n_windows,
            seq_len=cfg.seq_len,
            freq_bins=cfg.freq_bins,
            time_bins=cfg.time_bins,
        )
        val_ds = SyntheticSleepDataset(
            n_windows=max(1, n_windows // 5),
            seq_len=cfg.seq_len,
            freq_bins=cfg.freq_bins,
            time_bins=cfg.time_bins,
        )
        test_ds = SyntheticSleepDataset(
            n_windows=max(1, n_windows // 5),
            seq_len=cfg.seq_len,
            freq_bins=cfg.freq_bins,
            time_bins=cfg.time_bins,
        )
        return make_loaders(
            train_ds,
            val_ds,
            test_ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    from sleep_staging.data import SleepSequenceDataset, load_manifest, make_loaders, subject_split

    records = load_manifest(args.manifest)
    # Allow overriding cache dir from CLI (args) so users can pass --cache-dir
    if hasattr(args, "cache_dir") and args.cache_dir:
        cfg.cache_dir = args.cache_dir

    train_r, val_r, test_r = subject_split(records, cfg.val_frac, cfg.test_frac)
    log.info(
        "Subjects: train=%d  val=%d  test=%d",
        len({r["subject_id"] for r in train_r}),
        len({r["subject_id"] for r in val_r}),
        len({r["subject_id"] for r in test_r}),
    )

    train_ds = SleepSequenceDataset(train_r, cfg, seq_len=cfg.seq_len, cache_dir=cfg.cache_dir, augment=True)
    val_ds = SleepSequenceDataset(val_r, cfg, seq_len=cfg.seq_len, cache_dir=cfg.cache_dir, augment=False)
    test_ds = SleepSequenceDataset(test_r, cfg, seq_len=cfg.seq_len, cache_dir=cfg.cache_dir, augment=False)
    return make_loaders(train_ds, val_ds, test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

def cmd_build_manifest(args):
    from sleep_staging.data import build_manifest

    records = build_manifest(args.raw_dir, args.manifest)
    log.info("Built manifest with %d recordings -> %s", len(records), args.manifest)


def cmd_audit_data(args):
    from sleep_staging.data import read_manifest, summarize_label_counts, subject_level_split
    from sleep_staging.preprocess import SleepEEGPreprocessor, process_manifest

    cfg = default_cfg(batch_size=args.batch_size)
    manifest = read_manifest(args.manifest)
    pre = SleepEEGPreprocessor(cfg)
    specs, labels, subjects, _feats = process_manifest(manifest, preprocessor=pre, augment=False)

    train_idx, val_idx, test_idx = subject_level_split(subjects)
    splits = {
        "train": labels[train_idx],
        "val": labels[val_idx],
        "test": labels[test_idx],
    }

    log.info("Dataset audit for %s", args.manifest)
    for split_name, split_labels in splits.items():
        counts = summarize_label_counts(split_labels)
        total = int(len(split_labels))
        log.info("%s: %d epochs", split_name.capitalize(), total)
        for stage, count in counts.items():
            pct = (count / total * 100.0) if total else 0.0
            log.info("  %-5s %8d (%5.1f%%)", stage, count, pct)


def _build_training_cfg(args, cfg_overrides=None):
    cfg_overrides = cfg_overrides or {}
    return default_cfg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=getattr(args, "lr", 3e-4),
        distill_lr=getattr(args, "lr", 5e-4),
        num_workers=args.num_workers,
        amp=args.amp,
        profile_steps=args.profile_steps,
        patience=args.patience,
        **cfg_overrides,
    )


def cmd_train_teacher(args):
    from sleep_staging.models import TeacherCRNN
    from sleep_staging.train import train_teacher

    cfg = _build_training_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    train_loader, val_loader, _ = get_loaders(args, cfg)
    model = TeacherCRNN()
    log.info("Teacher params: %s", f"{model.param_count():,}")
    train_teacher(model, train_loader, val_loader, cfg, device, out_path=args.teacher_ckpt)


def cmd_distill(args):
    from sleep_staging.distill import distill_student
    from sleep_staging.models import StudentCRNN, TeacherCRNN

    cfg = _build_training_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect teacher architecture from checkpoint
    try:
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        proj_in = state.get("proj.weight", torch.zeros(1, 128)).shape[1]
        use_freq = (proj_in == 192)
        log.info("Teacher checkpoint proj_in=%d → use_freq_branch=%s", proj_in, use_freq)
    except Exception:
        use_freq = True  # default to new architecture

    teacher = TeacherCRNN(use_freq_branch=use_freq).to(device)
    _load_model_checkpoint(teacher, args.teacher_ckpt, device)
    student = StudentCRNN().to(device)
    log.info("Student params: %s", f"{student.param_count():,}")

    train_loader, val_loader, _ = get_loaders(args, cfg)
    distill_student(
        teacher, student, train_loader, val_loader, cfg, device,
        out_path=args.student_ckpt,
        teacher_ckpt_path=args.teacher_ckpt
    )


def cmd_evaluate(args):
    from sleep_staging.evaluate import evaluate
    from sleep_staging.models import StudentCRNN, TeacherCRNN

    cfg = default_cfg(batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_loaders(args, cfg)

    teacher, student = None, None
    if Path(args.teacher_ckpt).exists():
        teacher = TeacherCRNN().to(device)
        _load_model_checkpoint(teacher, args.teacher_ckpt, device)

    if Path(args.student_ckpt).exists():
        student = StudentCRNN().to(device)
        _load_model_checkpoint(student, args.student_ckpt, device)

    evaluate(teacher, student, test_loader, device, save_dir=args.artifacts_dir)


def cmd_quantize(args):
    from sleep_staging.export import quantize_student
    from sleep_staging.models import StudentCRNN

    student = StudentCRNN()
    _load_model_checkpoint(student, args.student_ckpt, torch.device("cpu"))
    quantize_student(student, args.quant_out)


def cmd_export_onnx(args):
    from sleep_staging.export import export_onnx_static
    from sleep_staging.models import StudentCRNN

    student = StudentCRNN()
    _load_model_checkpoint(student, args.student_ckpt, torch.device("cpu"))
    student.eval()
    export_onnx_static(student, args.onnx_out)


def cmd_export_tflite(args):
    from sleep_staging.export import full_export_pipeline
    from sleep_staging.models import StudentCRNN

    student = StudentCRNN()
    _load_model_checkpoint(student, args.student_ckpt, torch.device("cpu"))
    student.eval()
    full_export_pipeline(student, artifacts_dir=args.artifacts_dir, firmware_dir=args.firmware_dir)


def cmd_export_firmware(args):
    from sleep_staging.export import tflite_to_c_array

    tflite_to_c_array(args.tflite, args.cc_out)


def cmd_build_all(args):
    log.info("=" * 60)
    log.info("FULL PIPELINE: train -> distill -> evaluate -> export")
    log.info("=" * 60)

    args_t = types.SimpleNamespace(**vars(args))
    args_t.epochs = getattr(args, "teacher_epochs", args.epochs)
    cmd_train_teacher(args_t)

    args_d = types.SimpleNamespace(**vars(args))
    args_d.epochs = getattr(args, "distill_epochs", int(args.epochs * 1.3))
    cmd_distill(args_d)

    cmd_evaluate(args)
    cmd_export_onnx(args)
    cmd_export_tflite(args)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)


def cmd_benchmark_loso(args):
    from sleep_staging.benchmark import run_loso_benchmark

    run_loso_benchmark(
        manifest=args.manifest,
        seq_len=args.seq_len,
        teacher_epochs=args.teacher_epochs,
        distill_epochs=args.distill_epochs,
        batch_size=args.batch_size,
        teacher_lr=args.teacher_lr,
        distill_lr=args.distill_lr,
        max_folds=args.max_folds,
        artifacts_dir=args.artifacts_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sleep_staging", description="Neuromorphic sleep staging pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
        sp.add_argument("--manifest", default="data/manifests/sleep_edf.csv")
        sp.add_argument("--batch-size", dest="batch_size", type=int, default=8)
        sp.add_argument("--num-workers", dest="num_workers", type=int, default=0)
        sp.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
        sp.add_argument("--profile-steps", dest="profile_steps", type=int, default=0)
        sp.add_argument("--patience", type=int, default=12)
        sp.add_argument("--teacher-ckpt", dest="teacher_ckpt", default="artifacts/teacher.pt")
        sp.add_argument("--student-ckpt", dest="student_ckpt", default="artifacts/student.pt")
        sp.add_argument("--artifacts-dir", dest="artifacts_dir", default="artifacts")
        sp.add_argument("--firmware-dir", dest="firmware_dir", default="firmware/src")
        sp.add_argument("--cache-dir", dest="cache_dir", default="data/cache")

    bm = sub.add_parser("build-manifest", help="Scan EDF dir and write manifest CSV")
    bm.add_argument("--raw-dir", dest="raw_dir", required=True)
    bm.add_argument("--manifest", required=True)

    ad = sub.add_parser("audit-data", help="Print label balance for the current manifest")
    add_common(ad)

    ba = sub.add_parser("build-all", help="Run full pipeline")
    add_common(ba)
    ba.add_argument("--epochs", type=int, default=30)
    ba.add_argument("--lr", type=float, default=3e-4)
    ba.add_argument("--onnx-out", dest="onnx_out", default="artifacts/student_static.onnx")

    tt = sub.add_parser("train-teacher")
    add_common(tt)
    tt.add_argument("--epochs", type=int, default=30)
    tt.add_argument("--lr", type=float, default=3e-4)

    dd = sub.add_parser("distill")
    add_common(dd)
    dd.add_argument("--epochs", type=int, default=40)
    dd.add_argument("--lr", type=float, default=5e-4)

    ev = sub.add_parser("evaluate")
    add_common(ev)

    qz = sub.add_parser("quantize")
    qz.add_argument("--student-ckpt", dest="student_ckpt", required=True)
    qz.add_argument("--quant-out", dest="quant_out", default="artifacts/student_int8.pt")

    eo = sub.add_parser("export-onnx")
    eo.add_argument("--student-ckpt", dest="student_ckpt", default="artifacts/student.pt")
    eo.add_argument("--onnx-out", dest="onnx_out", default="artifacts/student_static.onnx")

    et = sub.add_parser("export-tflite")
    add_common(et)

    ef = sub.add_parser("export-firmware")
    ef.add_argument("--tflite", required=True)
    ef.add_argument("--cc-out", dest="cc_out", default="firmware/src/student_model_data.cc")

    bl = sub.add_parser("benchmark-loso", help="Run leave-one-subject-out benchmark")
    bl.add_argument("--manifest", default="data/manifests/sleep_edf.csv")
    bl.add_argument("--seq-len", dest="seq_len", type=int, default=30)
    bl.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    bl.add_argument("--teacher-epochs", dest="teacher_epochs", type=int, default=8)
    bl.add_argument("--distill-epochs", dest="distill_epochs", type=int, default=10)
    bl.add_argument("--teacher-lr", dest="teacher_lr", type=float, default=3e-4)
    bl.add_argument("--distill-lr", dest="distill_lr", type=float, default=5e-4)
    bl.add_argument("--max-folds", dest="max_folds", type=int, default=None)
    bl.add_argument("--artifacts-dir", dest="artifacts_dir", default="artifacts/loso")

    return parser


COMMAND_MAP = {
    "build-manifest": cmd_build_manifest,
    "audit-data": cmd_audit_data,
    "build-all": cmd_build_all,
    "train-teacher": cmd_train_teacher,
    "distill": cmd_distill,
    "evaluate": cmd_evaluate,
    "quantize": cmd_quantize,
    "export-onnx": cmd_export_onnx,
    "export-tflite": cmd_export_tflite,
    "export-firmware": cmd_export_firmware,
    "benchmark-loso": cmd_benchmark_loso,
}


def main():
    parser = build_parser()
    args = parser.parse_args()
    fn = COMMAND_MAP.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)
    fn(args)


if __name__ == "__main__":
    main()
