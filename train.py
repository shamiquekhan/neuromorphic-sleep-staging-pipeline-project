#!/usr/bin/env python
"""
Full Sleep-EDF training pipeline — raw signal architecture (1D-ResNet-SE + Transformer).

Based on literature:
  - Li & Gao 2023 (1D-ResNet-SE-LSTM): κ=0.812 on Sleep-EDF-78
  - Almutairi et al. 2023 (SSNet): 96.57% acc with EEG+EOG+EMG raw signals
  - Ito & Tanaka 2025 (SleepSatelightFTC): κ=0.787 with 470K params

Key changes from spectrogram approach:
  - Input: raw signals (B, T, 4, 3000) instead of spectrograms (B, T, 4, 128, 29)
  - Per-epoch Z-score normalization (SSNet) instead of global recording norm
  - 1D-ResNet-SE CNN instead of 2D ResNet
  - Cache stores raw signals — first run rebuilds cache (~20 min)
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT     = Path(__file__).parent
PYTHON   = r"C:\Users\shami\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"
MANIFEST = "data/manifests/sleep_edf_full.csv"
CACHE    = "data/cache"
BATCH    = "16"   # raw signals (3000 samples) are larger than spectrograms

env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "src")

def run(label, cmd):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}\n", flush=True)
    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed (exit {result.returncode})")
        sys.exit(result.returncode)

# ── Step 1: Teacher ──────────────────────────────────────────────────────────
# NOTE: First run will take ~25 min to build raw signal cache (153 recordings)
# Subsequent runs load from cache in ~3 seconds
run("STEP 1/4 — Train Teacher (80 epochs, dual time+freq branch, confusion weights)", [
    PYTHON, "-u", "-m", "sleep_staging.cli", "train-teacher",
    "--mode",         "real",
    "--manifest",     MANIFEST,
    "--epochs",       "80",
    "--batch-size",   BATCH,
    "--patience",     "20",
    "--lr",           "1e-4",
    "--num-workers",  "0",
    "--teacher-ckpt", "artifacts/teacher.pt",
    "--cache-dir",    CACHE,
])

# ── Step 2: Distil student ───────────────────────────────────────────────────
run("STEP 2/4 — Distil Student (50 epochs)", [
    PYTHON, "-u", "-m", "sleep_staging.cli", "distill",
    "--mode",         "real",
    "--manifest",     MANIFEST,
    "--epochs",       "50",
    "--batch-size",   BATCH,
    "--patience",     "15",
    "--lr",           "3e-4",
    "--num-workers",  "0",
    "--teacher-ckpt", "artifacts/teacher.pt",
    "--student-ckpt", "artifacts/student.pt",
    "--cache-dir",    CACHE,
])

# ── Step 3: Evaluate ─────────────────────────────────────────────────────────
run("STEP 3/4 — Evaluate Teacher + Student", [
    PYTHON, "-u", "-m", "sleep_staging.cli", "evaluate",
    "--mode",         "real",
    "--manifest",     MANIFEST,
    "--batch-size",   "32",
    "--num-workers",  "0",
    "--teacher-ckpt", "artifacts/teacher.pt",
    "--student-ckpt", "artifacts/student.pt",
    "--cache-dir",    CACHE,
    "--artifacts-dir","artifacts",
])

# ── Step 4: Export ───────────────────────────────────────────────────────────
run("STEP 4/4 — Export TFLite + Firmware C array", [
    PYTHON, "-u", "-m", "sleep_staging.cli", "export-tflite",
    "--mode",         "real",
    "--manifest",     MANIFEST,
    "--student-ckpt", "artifacts/student.pt",
    "--artifacts-dir","artifacts",
    "--firmware-dir", "firmware/src",
    "--cache-dir",    CACHE,
])

print(f"\n{'='*72}")
print("  ALL STEPS COMPLETE")
print(f"  Teacher  : artifacts/teacher.pt")
print(f"  Student  : artifacts/student.pt")
print(f"  Eval     : artifacts/eval_results.json")
print(f"  Firmware : firmware/src/student_model_data.cc")
print(f"{'='*72}\n")
