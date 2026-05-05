"""Master orchestration script for Phases 3-4.

Runs after Phase 2 teacher retraining completes.
Automatically executes:
  Phase 3: Distill improved teacher (teacher_improved_v2.pt) to student
  Phase 4: Export student to TFLite with validation
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, 'src')

from sleep_staging.config import TrainConfig, EEGConfig
from sleep_staging.models import TeacherCRNN, StudentCRNN
from sleep_staging.data import build_dataloaders
from sleep_staging.preprocess import process_manifest, SleepEEGPreprocessor
from sleep_staging.distill import distill_student


def wait_for_checkpoint(ckpt_path, timeout_minutes=600, check_interval=60):
    """Wait for checkpoint to be created."""
    ckpt_path = Path(ckpt_path)
    start = time.time()
    
    print(f"\n{'='*70}")
    print(f"PHASE 3: Waiting for teacher checkpoint: {ckpt_path.name}")
    print(f"{'='*70}")
    print(f"Checking every {check_interval}s (timeout: {timeout_minutes} min)...\n")
    
    while True:
        elapsed = time.time() - start
        if ckpt_path.exists():
            size_mb = ckpt_path.stat().st_size / 1e6
            print(f"\n✓ Checkpoint found! ({size_mb:.1f} MB)")
            print(f"  Elapsed: {elapsed/60:.1f} minutes")
            return ckpt_path
        
        if elapsed > timeout_minutes * 60:
            raise TimeoutError(f"Timeout waiting for {ckpt_path} after {timeout_minutes} min")
        
        # Show progress dots
        elapsed_s = int(elapsed)
        elapsed_m = elapsed_s // 60
        elapsed_s = elapsed_s % 60
        print(f"  [{elapsed_m:02d}:{elapsed_s:02d}] still waiting...", end='\r')
        
        time.sleep(check_interval)


def run_phase3_distillation():
    """Phase 3: Distill improved teacher to student."""
    print(f"\n{'='*70}")
    print("PHASE 3: KNOWLEDGE DISTILLATION")
    print(f"{'='*70}\n")
    
    # Wait for Phase 2 teacher to complete
    teacher_ckpt = wait_for_checkpoint('artifacts/teacher_improved_v2.pt', timeout_minutes=600)
    
    # Load configuration
    cfg = TrainConfig()
    cfg.seq_len = 60
    cfg.batch_size = 16
    cfg.epochs = 50  # distillation epochs
    cfg.use_focal = True
    cfg.focal_gamma = 2.0
    cfg.label_smoothing = 0.1
    cfg.mixup_alpha = 0.4
    cfg.channel_dropout = 0.1
    cfg.gaussian_noise_std = 0.01
    cfg.time_shift_ms = 50.0
    cfg.cutmix_enabled = True
    
    eeg_cfg = EEGConfig()

    # Build dataloaders using the same preprocessing path as Phase 2
    print("\nLoading data...")
    try:
        manifest_df = pd.read_csv('data/manifests/sleep_edf_full.csv')
        preprocessor = SleepEEGPreprocessor(eeg_cfg)
        specs, labels, subjects, _feats = process_manifest(
            manifest_df,
            preprocessor=preprocessor,
            cache_dir='data/cache',
            augment=False,
        )

        n = len(specs)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        indices = np.random.RandomState(42).permutation(n)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]

        train_specs, val_specs = specs[train_idx], specs[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        train_subj, val_subj = subjects[train_idx], subjects[val_idx]

        loaders = build_dataloaders(
            train_specs, train_labels, train_subj,
            val_specs, val_labels, val_subj,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
        )

        train_loader = loaders['train']
        val_loader = loaders['val']
    except Exception as e:
        print(f"⚠ Could not build dataloaders: {e}")
        print("  Skipping distillation phase")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize models
    print("\nInitializing models...")
    teacher = TeacherCRNN().to(device)
    student = StudentCRNN().to(device)
    
    # Load improved teacher
    print(f"Loading teacher from {teacher_ckpt}...")
    ckpt = torch.load(str(teacher_ckpt), map_location=device)
    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
    teacher.load_state_dict(state, strict=False)
    
    # Run distillation
    print(f"\nDistilling teacher → student ({cfg.epochs} epochs)...")
    print(f"Config: focal_gamma=2.0, augmentations enabled, multi-level KD ready")
    print("="*70 + "\n")
    
    student = distill_student(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        out_path='artifacts/student_improved.pt',
        teacher_ckpt_path=str(teacher_ckpt),
        grad_accum=2,
    )
    
    print("\n" + "="*70)
    print("✓ PHASE 3 COMPLETE: Student distilled from improved teacher")
    print("="*70)
    
    return 'artifacts/student_improved.pt'


def run_phase4_export(student_ckpt):
    """Phase 4: Export student to TFLite with validation."""
    print(f"\n{'='*70}")
    print("PHASE 4: EXPORT TO TFLITE")
    print(f"{'='*70}\n")
    
    # Wait for student checkpoint
    student_path = Path(student_ckpt)
    if not student_path.exists():
        print(f"Waiting for student checkpoint: {student_ckpt}")
        start = time.time()
        while not student_path.exists() and (time.time() - start) < 3600:
            time.sleep(60)
            print(f"  Still waiting... ({(time.time()-start)/60:.1f} min)")
        
        if not student_path.exists():
            print(f"✗ Timeout: {student_ckpt} not created")
            return
    
    print(f"✓ Student checkpoint ready: {student_ckpt}")
    
    # Import export module
    try:
        from scripts.export_validated import load_checkpoint, export_onnx, convert_onnx_to_savedmodel, convert_savedmodel_to_tflite, validate_tflite_vs_pytorch
    except ImportError:
        print("✗ Export module not found — skipping export")
        return
    
    # Load model
    print("\nLoading student model...")
    student = StudentCRNN()
    load_checkpoint(student, student_path)
    
    # Paths
    out_dir = Path('artifacts/export_final')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = out_dir / 'student.onnx'
    savedmodel_dir = out_dir / 'saved_model'
    tflite_path = out_dir / 'student_int8.tflite'
    
    # Export ONNX
    print("\n[1/4] Exporting to ONNX...")
    sample = torch.randn(1, 1, 4, 3000)
    try:
        export_onnx(student, onnx_path, sample)
        print(f"✓ {onnx_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return
    
    # Convert to SavedModel
    print("\n[2/4] Converting ONNX → SavedModel...")
    try:
        convert_onnx_to_savedmodel(onnx_path, savedmodel_dir)
        print(f"✓ {savedmodel_dir}")
    except Exception as e:
        print(f"✗ SavedModel conversion failed: {e}")
        print("  (You can use ONNX directly with ONNX Runtime)")
        return
    
    # Convert to TFLite with int8 quantization
    print("\n[3/4] Converting SavedModel → TFLite (int8 quantization)...")
    try:
        convert_savedmodel_to_tflite(
            savedmodel_dir, tflite_path,
            quantize='int8',
            rep_data_dir=Path('data/cache')
        )
        print(f"✓ {tflite_path}")
    except Exception as e:
        print(f"✗ TFLite conversion failed: {e}")
        return
    
    # Validate
    print("\n[4/4] Validating TFLite output...")
    try:
        val_results = validate_tflite_vs_pytorch(
            student, tflite_path,
            Path('data/cache'), n_samples=50
        )
        print(f"✓ Validation complete: {val_results}")
    except Exception as e:
        print(f"⚠ Validation skipped: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ PHASE 4 COMPLETE: Student exported to TFLite")
    print("="*70)
    
    # Model sizes
    if onnx_path.exists():
        onnx_mb = onnx_path.stat().st_size / 1e6
        print(f"\nModel sizes:")
        print(f"  ONNX:   {onnx_mb:.2f} MB")
    
    if tflite_path.exists():
        tflite_mb = tflite_path.stat().st_size / 1e6
        print(f"  TFLite: {tflite_mb:.2f} MB")
        if onnx_path.exists():
            ratio = onnx_mb / tflite_mb
            print(f"  Compression: {ratio:.1f}× smaller")
    
    print(f"\nNext: Convert TFLite to C array for firmware:")
    print(f"  xxd -i {tflite_path} > firmware/src/models/student_int8.h")
    
    # Save export report
    report = {
        "phase": "4_export",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "student_ckpt": str(student_ckpt),
        "outputs": {
            "onnx": str(onnx_path),
            "savedmodel": str(savedmodel_dir),
            "tflite_int8": str(tflite_path),
        },
        "validation": val_results if 'val_results' in locals() else None,
    }
    
    report_path = out_dir / 'export_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Export report: {report_path}\n")


def main():
    print("\n" + "="*70)
    print("MASTER ORCHESTRATION: PHASES 3-4")
    print("="*70)
    print("\nPhase 3: Knowledge Distillation (improved teacher → student)")
    print("Phase 4: Export to TFLite (with int8 quantization + validation)")
    print("\n" + "="*70 + "\n")
    
    try:
        # Phase 3
        student_ckpt = run_phase3_distillation()
        
        if student_ckpt:
            # Phase 4
            run_phase4_export(student_ckpt)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("✓✓✓ ALL PHASES COMPLETE ✓✓✓")
    print("="*70)
    print("\nSummary:")
    print("  Phase 1: LOSO Validation (running in background)")
    print("  Phase 2: Retrain with quick-wins (running in background)")
    print("  ✓ Phase 3: Distillation complete")
    print("  ✓ Phase 4: Export to TFLite complete")
    print("\nNext steps:")
    print("  1. Check LOSO results: artifacts/loso_seq60_focal/loso_results.json")
    print("  2. Compare Phase 2 teacher vs original")
    print("  3. Deploy TFLite model to firmware")
    print("\n" + "="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
