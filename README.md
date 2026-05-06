# Neuromorphic Sleep Stage Scoring

End-to-end sleep stage classification pipeline for the Sleep-EDF dataset, with a teacher-student CRNN workflow, subject-level splitting, model export, and firmware handoff for embedded deployment.

## Overview

This project implements a practical pipeline for EEG sleep staging:

- Build a manifest from paired PSG and hypnogram EDF files.
- Preprocess raw Sleep-EDF recordings into cached epoch tensors.
- Train a teacher CRNN on real or synthetic data.
- Distill a smaller student model from the teacher.
- Evaluate with accuracy, Cohen's kappa, and class reports.
- Export to ONNX and quantized formats for downstream deployment.
- Prepare firmware assets under `firmware/` for MCU integration.

The code is organized as a Python package in `src/sleep_staging/`, with the main CLI exposed through `sleep_staging.cli` and the `sleep-staging` console script defined in `pyproject.toml`.

## Key Features

- Sleep-EDF manifest builder with `subject_id`, `night`, `psg`, and `hypnogram` columns.
- Raw-signal preprocessing with bandpass filtering, notch filtering, normalization, and epoch extraction.
- Subject-level train/validation/test splitting to reduce leakage.
- Teacher and student CRNN models built for raw EEG sequences.
- Synthetic mode for fast smoke testing without real EDF data.
- Knowledge distillation pipeline for compact student models.
- Evaluation and benchmarking, including leave-one-subject-out runs.
- Export paths for ONNX, post-training quantization, and firmware packaging.

## Requirements

- Python 3.10 or newer.
- PyTorch 2.3+.
- NumPy, SciPy, pandas, scikit-learn, MNE, ONNX, and related export dependencies.
- Optional CUDA-capable GPU for faster teacher training and distillation.

If you are using Windows, keep `num_workers=0` unless you have explicitly verified multi-process data loading on your machine.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

If you prefer the package entry point after installation, the CLI is also available as:

```powershell
sleep-staging --help
```

## Data Preparation

The real pipeline expects Sleep-EDF cassette recordings with paired PSG and hypnogram EDF files.

1. Place the raw EDF files under a directory such as `data/raw/sleep_edf`.
2. Build a manifest CSV from the raw directory.
3. Use the manifest for training, evaluation, distillation, and export.

Example:

```powershell
python -m sleep_staging.cli build-manifest --raw-dir data/raw/sleep_edf --manifest data/manifests/sleep_edf.csv
python -m sleep_staging.cli audit-data --mode real --manifest data/manifests/sleep_edf.csv
```

The manifest builder writes one row per recording pair and the split logic works at the subject level.

## Quick Start

Run a synthetic end-to-end smoke test first if you want to verify the environment without waiting for real data processing:

```powershell
python -m sleep_staging.cli build-all --mode synthetic --epochs 2
```

For a real-data run, use your manifest and keep the first pass conservative while you verify the data path and checkpoint locations:

```powershell
python -m sleep_staging.cli train-teacher --mode real --manifest data/manifests/sleep_edf.csv --epochs 30
python -m sleep_staging.cli distill --mode real --manifest data/manifests/sleep_edf.csv --epochs 40
python -m sleep_staging.cli evaluate --mode real --manifest data/manifests/sleep_edf.csv
```

## CLI Reference

### Manifest and data checks

```powershell
python -m sleep_staging.cli build-manifest --raw-dir data/raw/sleep_edf --manifest data/manifests/sleep_edf.csv
python -m sleep_staging.cli audit-data --mode real --manifest data/manifests/sleep_edf.csv
```

### Training

```powershell
python -m sleep_staging.cli train-teacher --mode real --manifest data/manifests/sleep_edf.csv --epochs 30
python -m sleep_staging.cli distill --mode real --manifest data/manifests/sleep_edf.csv --epochs 40
```

### Evaluation and compression

```powershell
python -m sleep_staging.cli evaluate --mode real --manifest data/manifests/sleep_edf.csv
python -m sleep_staging.cli quantize --student-ckpt artifacts/student.pt --quant-out artifacts/student_int8.pt
python -m sleep_staging.cli export-onnx --student-ckpt artifacts/student.pt --onnx-out artifacts/student_static.onnx
python -m sleep_staging.cli export-tflite --mode real --manifest data/manifests/sleep_edf.csv
python -m sleep_staging.cli export-firmware --tflite artifacts/export_final/student_int8.tflite --cc-out firmware/src/student_model_data.cc
```

### Benchmarking

```powershell
python -m sleep_staging.cli benchmark-loso --manifest data/manifests/sleep_edf.csv --max-folds 20
```

### Full pipeline

```powershell
python -m sleep_staging.cli build-all --mode real --manifest data/manifests/sleep_edf.csv --epochs 30
```

That command runs the teacher training, student distillation, evaluation, ONNX export, and the export step that prepares the deployment artifacts.

## Project Layout

```text
src/sleep_staging/      Core package code: data, preprocessing, models, training, export, CLI
scripts/                Convenience scripts and orchestration helpers
data/                   Raw files, manifests, and cached processed epochs
artifacts/              Checkpoints, reports, evaluation outputs, and exports
firmware/               MCU-facing scaffolding and generated model assets
```

Important modules:

- `src/sleep_staging/data.py`: manifest creation, Sleep-EDF loading, cache management, and dataset splits.
- `src/sleep_staging/preprocess.py`: recording-level preprocessing and epoch generation.
- `src/sleep_staging/models.py`: teacher and student CRNN definitions.
- `src/sleep_staging/train.py`: teacher training loop.
- `src/sleep_staging/distill.py`: student distillation pipeline.
- `src/sleep_staging/evaluate.py`: metrics and reports.
- `src/sleep_staging/export.py`: quantization and export helpers.
- `src/sleep_staging/benchmark.py`: leave-one-subject-out benchmarking.
- `src/sleep_staging/cli.py`: command-line entry point.

## Inputs and Outputs

### Expected inputs

- Raw Sleep-EDF PSG EDF files.
- Matching hypnogram EDF files.
- A manifest CSV generated by the project tooling.

### Common outputs

- `artifacts/teacher.pt`: teacher checkpoint.
- `artifacts/student.pt`: distilled student checkpoint.
- `artifacts/student_static.onnx`: exported student ONNX model.
- `artifacts/student_int8.pt`: quantized checkpoint path used by the project.
- `artifacts/export_final/`: export artifacts generated by the deployment pipeline.
- `artifacts/loso/` or `artifacts/loso_seq60_focal/`: benchmark results for LOSO runs.

## Notes on the Pipeline

- The pipeline operates on raw EEG windows shaped like `(B, T, 4, 3000)`.
- The default preprocessing cache is `data/cache/`.
- Windows multiprocessing is intentionally conservative; `num_workers=0` is the default in the CLI config.
- Subject-level splitting is used to avoid leakage across train, validation, and test sets.
- The synthetic mode is useful for validating the code path when the real EDF corpus is not available.

## Troubleshooting

- If a command cannot find EDF files, verify the manifest paths first.
- If preprocessing seems slow, confirm that the cache directory exists and that cached `.npy` files are being reused.
- If CUDA is unavailable, the code falls back to CPU execution.
- If ONNX export fails, make sure the student checkpoint exists and is loadable on CPU.

## Reproducibility Tips

- Keep raw data paths stable after building the manifest.
- Use the same manifest for training, distillation, and evaluation.
- Save the generated `artifacts/` directory if you want to compare models across runs.

