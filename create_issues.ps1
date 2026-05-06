#!/usr/bin/env pwsh

Set-Location 'c:\Project\CNN-ECG'

$issues = @(
    @{
        Title = "Add Unit Tests for preprocess.py"
        Labels = "good first issue,testing"
        Body = "The preprocessing module lacks unit tests. Write tests for bandpass_filter(), robust_normalize(), epoch_iterator(), and process_recording().

Steps:
1. Create tests/test_preprocess.py
2. Write synthetic 30-min EEG signal using NumPy
3. Test that bandpass_filter(fs=100, lo=0.5, hi=30) removes DC
4. Verify robust_normalize() has mean ≈ 0, std ≈ 1
5. Check epoch_iterator() yields exactly 60 epochs

Expected: pytest tests/test_preprocess.py -v shows 4 passed"
    },
    @{
        Title = "Add Support for ISRUC-Sleep Dataset"
        Labels = "good first issue,enhancement"
        Body = "Add ISRUC-Sleep dataset support. ISRUC-Sleep has 200 Hz sampling (vs 100 Hz) and different annotations.

Steps:
1. Download ISRUC-Sleep Session 1
2. Create src/sleep_staging/isruc_loader.py
3. Map stages: {1: Wake, 2: N1, 3: N2, 4: N3, 5: REM}
4. Resample 200 Hz to 100 Hz using scipy.signal.resample()
5. Write manifest builder

Expected: raw.shape = (120, 4, 3000), labels = array([0, 1, 2, ...])"
    },
    @{
        Title = "Fix Windows-Specific Logging Bug (UTF-8 Encoding)"
        Labels = "good first issue,bug"
        Body = "On Windows, logger writes UTF-8 but encodes as cp1252, causing UnicodeEncodeError with Greek letters like κ.

Reproduce:
1. On Windows, run: python -m sleep_staging.cli train-teacher --mode synthetic --epochs 1
2. Observe: UnicodeEncodeError: 'cp1252' codec can't encode character '\u03ba'

Fix:
In src/sleep_staging/train.py, add at top of train_teacher():
\`\`\`python
import sys, io
if sys.platform == 'win32' and not hasattr(sys.stdout, 'reconfigure'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
\`\`\`

After fix, logs should print cleanly without errors."
    },
    @{
        Title = "Add Confusion Matrix Visualization after Evaluation"
        Labels = "good first issue,visualization"
        Body = "After evaluation, create confusion matrix heatmap showing predicted vs true sleep stages.

Steps:
1. In src/sleep_staging/evaluate.py, after computing predictions:
2. Use sklearn.metrics.confusion_matrix() and seaborn.heatmap()
3. Save to artifacts/confusion_matrix.png

Expected: Clean heatmap with diagonal values high and off-diagonals showing confusions"
    },
    @{
        Title = "Document Pipeline Architecture with ASCII/Mermaid Diagram"
        Labels = "good first issue,documentation"
        Body = "Create ARCHITECTURE.md with visual guide showing data flow: EDF → Preprocess → Cache → DataLoader → Model → Export.

Include:
- Data Flow diagram
- Model Architecture (Teacher + Student)
- Training Pipeline (Train → Distill → Export)
- Module Dependency Graph

Example: Use Mermaid diagrams or ASCII art showing how 15 modules interact."
    },
    @{
        Title = "Implement Focal Loss and Compare with Cross-Entropy"
        Labels = "intermediate,research"
        Body = "Implement focal loss for handling class imbalance. Sleep stages are imbalanced (Wake/REM rarer than N2).

Steps:
1. Create src/sleep_staging/focal_loss.py with FocalLoss class
2. Update CLI to accept --loss-type focal or --loss-type ce
3. Train two models with CE and Focal (α=0.25, γ=2.0)
4. Compare: κ score, per-class F1, training curves

Expected output:
CE Loss: κ = 0.636, N1 F1 = 0.45
Focal Loss: κ = 0.655, N1 F1 = 0.52 (improvement on minority classes)"
    },
    @{
        Title = "Implement SWDF Loss (Soft Weighted Dice Focal)"
        Labels = "intermediate,research"
        Body = "SWDF combines Dice, Focal, and sample weighting for severe class imbalance.

Steps:
1. Create src/sleep_staging/swdf_loss.py
2. Compute class weights from training manifest
3. Train three models: CE, Focal, SWDF
4. Report per-class F1 and macro F1

Expected: SWDF should improve minority class performance (N1, N3, REM)"
    },
    @{
        Title = "Add Real-Time Per-Epoch Confusion Matrix During Training"
        Labels = "intermediate,visualization"
        Body = "Track confusion matrix evolution during training. Add visualization every 5 epochs to see how confusions change.

Steps:
1. In src/sleep_staging/train.py, compute validation confusion matrix every 5 epochs
2. Save as cm_epoch_000.png, cm_epoch_005.png, etc.
3. Create animated GIF: convert -delay 50 cm_epoch_*.png cm_evolution.gif

Expected: Animated visualization showing confusion patterns improving over time"
    },
    @{
        Title = "Add Attention Heatmap Visualization for Model Interpretability"
        Labels = "intermediate,visualization"
        Body = "Visualize which EEG signal parts the model focuses on when classifying stages.

Steps:
1. Modify TeacherCRNN to return attention weights
2. Create visualization showing temporal/frequency focus per stage
3. Test on 5-10 real samples; verify N2 attention focuses on sleep spindles

Expected: Heatmaps validating model attends to clinically relevant features"
    },
    @{
        Title = "Add Cross-Dataset Generalization Test (Synthetic → Real)"
        Labels = "intermediate,research"
        Body = "Train on synthetic data, evaluate on real Sleep-EDF. Measure domain gap.

Steps:
1. Train: python -m sleep_staging.cli train-teacher --mode synthetic --epochs 30
2. Evaluate: on real Sleep-EDF test set
3. Compare: κ (synthetic train) vs κ (real train)
4. Document findings in DOMAIN_GAP_ANALYSIS.md

Expected:
- Synthetic-trained, real-tested: κ = 0.42 (poor generalization)
- Real-trained, real-tested: κ = 0.636 (good fit)
- Domain gap: -0.216"
    },
    @{
        Title = "Implement Mamba Temporal Encoder for Better Long-Range Modeling"
        Labels = "advanced,research"
        Body = "Replace Transformer with Mamba (state-space model) for improved long-range temporal modeling. Mamba scales better than Transformer.

Steps:
1. Install: pip install mamba-ssm
2. Create src/sleep_staging/mamba_encoder.py
3. Update TeacherCRNN to use MambaTemporalEncoder
4. Train and compare: Teacher (Transformer) vs Mamba Teacher

Expected:
- Transformer: κ = 0.636, training time = 8h
- Mamba: κ = 0.672, training time = 6h (faster + better)"
    },
    @{
        Title = "Implement Self-Supervised Pretraining on Unlabeled PSG Data"
        Labels = "advanced,research"
        Body = "Use SimCLR contrastive learning to pretrain on unlabeled PSG before supervised finetuning.

Steps:
1. Create src/sleep_staging/ssl_pretraining.py with ContrastiveLoss
2. Implement EEG augmentations (temporal jitter, amplitude scaling, freq masking, noise)
3. Pretrain on unlabeled Sleep-EDF for 50 epochs
4. Finetune on labeled subset; vary labeled data fraction

Expected:
- 100% labeled: κ = 0.636 (baseline)
- 50% labeled + SSL: κ = 0.650 (good)
- 25% labeled + SSL: κ = 0.610 (vs 0.520 supervised)"
    },
    @{
        Title = "Implement TFLite INT8 Quantization with Validation"
        Labels = "advanced,optimization"
        Body = "Full TFLite INT8 quantization with representative dataset selection and validation. Target: <2% accuracy loss, 4× compression.

Steps:
1. Implement intelligent dataset selection (stratified by stage)
2. Quantize student model to INT8
3. Validate: PyTorch vs TFLite output comparison
4. Benchmark on ARM hardware (emulated if needed)

Expected:
- Size: 54 MB → 13 MB (4.2× compression)
- Latency: 120 ms → 40 ms (3.0× speedup, Cortex-M4)
- Accuracy drop: <2%"
    },
    @{
        Title = "Implement Multi-Dataset Training and Zero-Shot Transfer"
        Labels = "advanced,research"
        Body = "Train on Sleep-EDF + ISRUC-Sleep + MASS jointly; evaluate zero-shot transfer. Measure generalization across datasets.

Steps:
1. Implement loaders for ISRUC-Sleep and MASS
2. Create MultiDatasetLoader for balanced sampling
3. Train Teacher on all three datasets for 50 epochs
4. Evaluate zero-shot: test on each dataset

Expected:
- Multi-dataset: κ = 0.64 on Sleep-EDF, 0.58 on ISRUC, 0.56 on MASS
- Single-dataset (Sleep-EDF only): 0.636 on Sleep-EDF, 0.42 on ISRUC (domain gap)"
    },
    @{
        Title = "Implement Model Ensemble and Uncertainty Quantification"
        Labels = "advanced,research"
        Body = "Create ensemble of 5 Student models; implement uncertainty via Monte Carlo Dropout.

Steps:
1. Train ensemble on bootstrap resamples of training data
2. Implement MC Dropout for uncertainty estimates
3. Evaluate ensemble performance: κ should improve ~4%
4. Analyze calibration: high-confidence predictions should have high accuracy

Expected:
- Single Student: κ = 0.668
- Ensemble (5 models): κ = 0.695 (+4%)
- Uncertainty correlation: high σ → low accuracy (good calibration)"
    }
)

foreach ($issue in $issues) {
    Write-Host "Creating: $($issue.Title)" -ForegroundColor Green
    gh issue create --title $issue.Title --label $issue.Labels --body $issue.Body
    Start-Sleep -Milliseconds 500
}

Write-Host "All 15 issues created successfully!" -ForegroundColor Cyan
