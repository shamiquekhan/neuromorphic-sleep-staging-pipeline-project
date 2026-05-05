#!/usr/bin/env python
"""Phase 4: Simple export student to portable formats."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, 'src')
from sleep_staging.models import StudentCRNN

print("="*70)
print("PHASE 4: SIMPLE EXPORT")
print("="*70)

student_path = Path('artifacts/student_improved.pt')
if not student_path.exists():
    print(f"✗ Student checkpoint not found: {student_path}")
    sys.exit(1)

print(f"✓ Student checkpoint ready: {student_path}")

# Load student
print("\nLoading student model...")
student = StudentCRNN()
ckpt = torch.load(str(student_path), map_location='cpu')
state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
state = {k.replace('module.', ''): v for k, v in state.items()}
student.load_state_dict(state, strict=False)
student = student.eval()

# Paths
out_dir = Path('artifacts/export_final')
out_dir.mkdir(parents=True, exist_ok=True)

pt_traced = out_dir / 'student_traced.pt'
onnx_path = out_dir / 'student.onnx'

# Option 1: TorchScript trace
print("\n[1/2] Creating TorchScript trace...")
try:
    sample = torch.randn(1, 1, 4, 3000)
    traced = torch.jit.trace(student, sample)
    torch.jit.save(traced, str(pt_traced))
    print(f"✓ {pt_traced}")
except Exception as e:
    print(f"✗ Tracing failed: {e}")

# Option 2: Try ONNX export with fallback
print("\n[2/2] Attempting ONNX export (torch.onnx)...")
try:
    sample = torch.randn(1, 1, 4, 3000)
    torch.onnx.export(
        student,
        sample,
        str(onnx_path),
        opset_version=12,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}},
        do_constant_folding=True,
        verbose=False,
    )
    print(f"✓ {onnx_path}")
except Exception as e:
    print(f"⚠ ONNX export failed (skipping): {e}")
    print("   Note: ONNX requires onnxruntime or related packages")

# Summary
print("\n" + "="*70)
print("✓ PHASE 4 COMPLETE")
print("="*70)

if pt_traced.exists():
    trace_mb = pt_traced.stat().st_size / 1e6
    print(f"\nExported artifacts:")
    print(f"  TorchScript: {trace_mb:.2f} MB → {pt_traced}")

if onnx_path.exists():
    onnx_mb = onnx_path.stat().st_size / 1e6
    print(f"  ONNX:       {onnx_mb:.2f} MB → {onnx_path}")

if not pt_traced.exists() and not onnx_path.exists():
    print("⚠ Warning: No export artifacts created")
    sys.exit(1)

print(f"\n✓ Export complete!")
