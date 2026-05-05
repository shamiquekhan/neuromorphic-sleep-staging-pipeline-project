# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-06

### Added

- **End-to-end Teacher-Student pipeline** for sleep stage classification
  - Teacher CRNN model (1.6M parameters) trained on Sleep-EDF
  - Student CRNN model (3.7× compression) via knowledge distillation
  - κ=0.636 (Teacher, fixed-split), κ=0.668 (Student, fixed-split)

- **Data pipeline**
  - Sleep-EDF manifest builder with subject-level splitting
  - Raw-signal preprocessing: bandpass filter, notch filter, robust normalization
  - Cached epoch generation for fast training
  - Synthetic mode for smoke testing without real EDF files

- **Training and evaluation**
  - Teacher training with focal loss and augmentations
  - Knowledge distillation with multi-level loss (logits + features + relations)
  - Evaluation with accuracy, Cohen's kappa, per-class F1, confusion matrix
  - Leave-one-subject-out (LOSO) benchmarking for honest generalization measurement

- **Model export**
  - ONNX export for student model
  - Post-training INT8 quantization
  - TFLite conversion pipeline for embedded deployment (4× compression)
  - Firmware scaffolding for MCU integration

- **CLI interface**
  - 11 subcommands for building manifests, training, distilling, evaluating, and exporting
  - Synthetic and real-data modes
  - Configurable hyperparameters (epochs, batch size, learning rates, etc.)

- **Documentation**
  - Comprehensive README with setup, data prep, CLI reference, and troubleshooting
  - CONTRIBUTING.md with PR process and code style guidelines
  - CONTRIBUTOR_ISSUES.md with 5 ready-to-post good-first-issues
  - Architecture documentation (ARCHITECTURE.md ready for community contribution)

- **Project structure**
  - Modular Python package in `src/sleep_staging/`
  - 15 core modules: data, preprocess, models, train, distill, evaluate, export, benchmark, CLI, etc.
  - Artifact tracking for reproducibility
  - Firmware directory for embedded integration

### Known Limitations

- Windows multiprocessing with DataLoader may be unreliable; `num_workers=0` is default
- TFLite export requires either `ai-edge-torch` or `onnx2tf` (automatic fallback)
- Firmware integration requires board-specific drivers (not included)
- Current SOTA gap: κ=0.636 vs. SOTA κ=0.787 (SleepSatelightFTC); roadmap includes Mamba architecture

### Tech Stack

- **Core**: Python 3.10+, PyTorch 2.3+, NumPy, SciPy, pandas, scikit-learn
- **Signal processing**: MNE, scipy.signal
- **Export**: ONNX, TFLite, ai-edge-torch / onnx2tf
- **Data**: Sleep-EDF Cassette (78 subjects), roadmap includes ISRUC-Sleep, MASS
- **Testing**: pytest (framework ready, tests forthcoming via good-first-issues)

### Future Roadmap

- [ ] Add ISRUC-Sleep dataset support
- [ ] Implement Mamba temporal encoder for improved context modeling
- [ ] Self-supervised pretraining on unlabeled PSG data
- [ ] Per-fold confusion matrix visualization
- [ ] Unit tests for all modules
- [ ] Windows UTF-8 logging fix
- [ ] Interactive web UI for inference
- [ ] Docker containerization for reproducibility

---

## [Unreleased]

### Planned

- Support for additional datasets (MASS, PhysioNet sleep cohorts)
- Improved documentation with tutorials
- Performance optimization for ARM MCU targets
