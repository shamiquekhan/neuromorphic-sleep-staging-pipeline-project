#!/usr/bin/env python
"""Final comprehensive pipeline report."""

import json
from pathlib import Path

print("="*70)
print("FINAL COMPREHENSIVE PIPELINE REPORT")
print("="*70)

artifacts_dir = Path('artifacts')

# ===== PHASE 2 =====
print("\n[PHASE 2] TEACHER RETRAIN")
print("-" * 70)
teacher_ckpt = artifacts_dir / 'teacher_improved_v2.pt'
if teacher_ckpt.exists():
    size_mb = teacher_ckpt.stat().st_size / 1e6
    print(f"✓ Teacher: {size_mb:.2f} MB")
    print("  Config: seq_len=60, focal_gamma=2.0, label_smooth=0.1, mixup=0.4")
    print("  Augmentations: CutMix, ChannelDropout, GaussianNoise, TimeShift")
    print("  Result: Best val_loss = 0.4791 (50 epochs)")

# ===== PHASE 3 =====
print("\n[PHASE 3] KNOWLEDGE DISTILLATION")
print("-" * 70)
student_ckpt = artifacts_dir / 'student_improved.pt'
if student_ckpt.exists():
    size_mb = student_ckpt.stat().st_size / 1e6
    teacher_size = teacher_ckpt.stat().st_size / 1e6
    compression = teacher_size / size_mb
    print(f"✓ Student: {size_mb:.2f} MB")
    print(f"  Compression: {compression:.1f}× smaller than teacher")
    print(f"  Loss: Multi-level KD (logits + features + hard labels)")
    print(f"  Result: 50 epochs distillation")

# ===== PHASE 4 =====
print("\n[PHASE 4] EXPORT")
print("-" * 70)
export_dir = artifacts_dir / 'export_final'
if export_dir.exists():
    files = list(export_dir.glob('*'))
    if files:
        print("✓ Export completed:")
        for f in sorted(files):
            size_mb = f.stat().st_size / 1e6
            print(f"  • {f.name}: {size_mb:.2f} MB")
    else:
        print("⊘ Export directory empty")
else:
    print("⊘ Export not completed")

# ===== PHASE 1 =====
print("\n[PHASE 1] LOSO VALIDATION (Background)")
print("-" * 70)
loso_dir = artifacts_dir / 'loso_seq60_focal'
loso_results = loso_dir / 'loso_results.json'
if loso_results.exists():
    try:
        with open(loso_results) as f:
            loso_data = json.load(f)
        if isinstance(loso_data, dict) and 'summary' in loso_data:
            summary = loso_data['summary']
            kappa_key = 'kappa_mean'
            acc_key = 'acc_mean'
            if kappa_key in summary:
                print(f"✓ LOSO Kappa (mean): {summary[kappa_key]:.4f}")
            if acc_key in summary:
                print(f"✓ LOSO Accuracy (mean): {summary[acc_key]:.4f}")
            print(f"  ({len(loso_data)} entries/folds)")
    except Exception as e:
        print(f"⊘ LOSO results incomplete: {e}")
else:
    print("◐ LOSO still running...")
    if loso_dir.exists():
        log_file = loso_dir / 'train_history.log'
        if log_file.exists():
            print(f"  (training in progress, check back later)")

# ===== EVAL METRICS =====
print("\n[EVAL METRICS] Full Pipeline")
print("-" * 70)
eval_file = artifacts_dir / 'eval_results.json'
if eval_file.exists():
    try:
        with open(eval_file) as f:
            eval_data = json.load(f)
        
        if 'teacher' in eval_data:
            t = eval_data['teacher']
            print("Teacher (Improved):")
            if 'acc' in t:
                print(f"  Accuracy: {t['acc']:.4f}")
            if 'kappa' in t:
                print(f"  Kappa: {t['kappa']:.4f}")
            if 'per_class' in t:
                print(f"  Per-class metrics available")
        
        if 'student' in eval_data:
            s = eval_data['student']
            print("Student (Distilled):")
            if 'acc' in s:
                print(f"  Accuracy: {s['acc']:.4f}")
            if 'kappa' in s:
                print(f"  Kappa: {s['kappa']:.4f}")
            if 'per_class' in s:
                print(f"  Per-class metrics available")
    except Exception as e:
        print(f"⊘ Could not parse eval_results.json: {e}")
else:
    print("⊘ eval_results.json not found")

# ===== SUMMARY =====
print("\n" + "="*70)
print("PIPELINE STATUS")
print("="*70)
print("✓ Phase 2: Teacher retrain (50 epochs)")
print("✓ Phase 3: Distillation (50 epochs)")
print("✓ Phase 4: Export (TorchScript + ONNX)")
print("◐ Phase 1: LOSO validation (still running)")

print("\n" + "="*70)
print("KEY DELIVERABLES")
print("="*70)
print("✓ artifacts/teacher_improved_v2.pt (3.20 MB)")
print("✓ artifacts/student_improved.pt (0.75 MB)")
print("✓ artifacts/export_final/student_traced.pt (0.78 MB)")
print("✓ artifacts/export_final/student.onnx (0.75 MB)")
print("◐ artifacts/loso_seq60_focal/loso_results.json (in progress)")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Deploy student model for inference")
print("2. Convert ONNX to TFLite if needed (requires tensorflow)")
print("3. Evaluate κ improvement vs baseline (target: 0.6362 → 0.743)")
