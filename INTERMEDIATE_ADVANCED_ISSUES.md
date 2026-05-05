# Intermediate and Advanced Issues for Contributors

Copy each issue body into a GitHub issue. Adjust titles and labels as needed.

---

## Intermediate-Tier Issues (⭐⭐)

### Issue #6: Implement Focal Loss and Compare with Cross-Entropy

**Labels:** `intermediate`, `enhancement`, `loss-functions`, `research`

**Difficulty:** ⭐⭐ (Intermediate)

**Description:**

The current training uses standard cross-entropy loss, but focal loss is known to handle class imbalance better by focusing on hard examples. This issue covers implementing focal loss and comparing its performance against the baseline.

**Why this matters:**
Sleep stages are imbalanced (e.g., Wake and REM are rarer than N2). Focal loss can improve minority class performance without resampling.

**Steps to Reproduce:**

1. Create `src/sleep_staging/focal_loss.py` with a `FocalLoss` class:
   ```python
   class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0):
           """Focal Loss for addressing class imbalance.
           
           Args:
               alpha: Weight for rare classes (0.25 typical).
               gamma: Focus parameter (2.0 typical, higher = more focus on hard examples).
           """
       
       def forward(self, logits, labels):
           """Compute focal loss."""
           # pt = probability of true class
           # focal_loss = -alpha * (1 - pt)^gamma * log(pt)
   ```

2. Update `src/sleep_staging/cli.py` to accept `--loss-type focal` or `--loss-type ce`.

3. Train two models:
   - Teacher with CE loss for 30 epochs
   - Teacher with Focal loss (α=0.25, γ=2.0) for 30 epochs
   
4. Compare results:
   - κ score
   - Per-class F1 scores
   - Training curve

**Expected Output:**

```
Model Comparison:
CE Loss:        κ = 0.636, N1 F1 = 0.45, N2 F1 = 0.72, N3 F1 = 0.68
Focal Loss:     κ = 0.655, N1 F1 = 0.52, N2 F1 = 0.73, N3 F1 = 0.71
```

**Resources:**
- Focal Loss paper: https://arxiv.org/abs/1708.02002
- PyTorch implementation ref: https://github.com/clcarwin/focal_loss_pytorch

**How to Submit:**
- Fork, create `loss/add-focal-loss`, implement the loss and comparison, and open a PR.

---

### Issue #7: Implement Weighted Sampling and SWDF Loss

**Labels:** `intermediate`, `enhancement`, `loss-functions`, `research`

**Difficulty:** ⭐⭐ (Intermediate)

**Description:**

SWDF (Soft Weighted Dice Focal) loss combines Dice loss, focal loss, and sample weighting to handle severe class imbalance. Implement SWDF and compare against CE and Focal losses.

**Why this matters:**
In real sleep labs, Wake and Movement time are much more common than N3. SWDF is designed to handle this extreme imbalance better.

**Steps to Reproduce:**

1. Create `src/sleep_staging/swdf_loss.py`:
   ```python
   class SWDFLoss(nn.Module):
       """Soft Weighted Dice Focal Loss for severe class imbalance."""
       
       def __init__(self, alpha=0.5, beta=0.5, gamma=2.0):
           """
           Args:
               alpha: Weight for Dice loss component.
               beta: Weight for Focal loss component.
               gamma: Focal loss focus parameter.
           """
       
       def forward(self, logits, labels, class_weights=None):
           """Compute SWDF loss with optional class weighting."""
           # dice_loss = 1 - (2*TP) / (2*TP + FP + FN)
           # focal_loss = -alpha * (1-pt)^gamma * log(pt)
           # swdf = alpha * dice_loss + beta * focal_loss
   ```

2. Compute class weights from the training manifest:
   ```python
   class_counts = {label: count for label, count in summarize_label_counts(train_labels)}
   class_weights = max(class_counts.values()) / np.array(list(class_counts.values()))
   ```

3. Train three models and compare:
   - CE loss (baseline)
   - Focal loss
   - SWDF loss with class weighting

4. Report per-class F1 and macro F1.

**Expected Output:**

```
Per-Class F1 Comparison:
              Wake   N1    N2    N3    REM   Macro F1
CE Loss:     0.85  0.45  0.72  0.68  0.70  0.68
Focal Loss:  0.86  0.52  0.73  0.71  0.72  0.71
SWDF Loss:   0.87  0.58  0.74  0.73  0.74  0.73
```

**Resources:**
- SWDF Loss paper: https://arxiv.org/abs/2006.04009
- Dice Loss: https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient

**How to Submit:**
- Fork, create `loss/add-swdf-loss`, implement, compare, and open a PR.

---

### Issue #8: Add Real-Time Per-Epoch Confusion Matrix During Training

**Labels:** `intermediate`, `visualization`, `monitoring`

**Difficulty:** ⭐⭐ (Intermediate)

**Description:**

Currently, confusion matrices are computed only at the end of training. Add per-epoch validation confusion matrices to track how confusion patterns evolve during training.

**Why this matters:**
Seeing which stage confusions improve/worsen over time helps debug training dynamics. N2 vs. N3 confusion often persists—visualizing this helps understand when the model learns to distinguish them.

**Steps to Reproduce:**

1. In `src/sleep_staging/train.py`, after validation loop:
   ```python
   from sklearn.metrics import confusion_matrix
   import seaborn as sns
   import matplotlib.pyplot as plt
   
   def plot_confusion_matrix(labels, preds, epoch, save_dir):
       cm = confusion_matrix(labels, preds)
       plt.figure(figsize=(6, 5))
       sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES)
       plt.title(f'Validation Confusion Matrix - Epoch {epoch}')
       plt.tight_layout()
       plt.savefig(f'{save_dir}/cm_epoch_{epoch:03d}.png', dpi=100)
       plt.close()
   ```

2. Call this function every 5 epochs during validation.

3. Create an animated GIF showing CM evolution:
   ```bash
   convert -delay 50 cm_epoch_*.png cm_evolution.gif
   ```

**Expected Output:**

- `artifacts/cm_epoch_000.png`, `cm_epoch_005.png`, ..., `cm_epoch_030.png`
- Animated GIF showing confusion patterns improving over time

**Resources:**
- ImageMagick for GIF creation: https://imagemagick.org/
- Matplotlib animation: https://matplotlib.org/stable/gallery/animation/

**How to Submit:**
- Fork, edit `src/sleep_staging/train.py`, test with synthetic data, and open a PR.

---

### Issue #9: Add Attention Heatmap Visualization for Model Interpretability

**Labels:** `intermediate`, `visualization`, `interpretability`

**Difficulty:** ⭐⭐ (Intermediate)

**Description:**

Add attention visualizations to show which parts of the EEG signal (time windows or frequency bands) the model focuses on when classifying each sleep stage.

**Why this matters:**
Clinicians need to understand what the model "sees". Attention maps help validate that the model is learning clinically relevant features (e.g., sleep spindles for N2).

**Steps to Reproduce:**

1. Modify `TeacherCRNN` in `src/sleep_staging/models.py` to return attention weights:
   ```python
   class TeacherCRNN(nn.Module):
       def forward(self, x, return_attention=False):
           # ... existing code ...
           if return_attention:
               return logits, attention_weights
           return logits
   ```

2. Create a visualization function in `src/sleep_staging/export.py`:
   ```python
   def visualize_attention(model, sample_batch, output_dir):
       """Plot attention maps for a batch of EEG samples."""
       logits, attn = model(sample_batch, return_attention=True)
       
       for i in range(len(sample_batch)):
           fig, ax = plt.subplots(figsize=(12, 4))
           im = ax.imshow(attn[i].cpu().detach().numpy(), cmap='hot', aspect='auto')
           ax.set_xlabel('Time (epochs)')
           ax.set_ylabel('Frequency Band')
           ax.set_title(f'Attention Map - Stage {STAGE_NAMES[logits[i].argmax()]}')
           plt.colorbar(im)
           plt.savefig(f'{output_dir}/attn_{i}.png')
           plt.close()
   ```

3. Test on 5–10 real test samples and verify that N2 attention focuses on sleep spindles, etc.

**Expected Output:**

- Heatmaps showing temporal/frequency focus per stage
- Visual validation that model attends to clinically relevant features

**Resources:**
- Attention mechanisms: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- Interpretability tutorial: https://github.com/jacobgil/pytorch-grad-cam

**How to Submit:**
- Fork, update models.py and export.py, test, and open a PR.

---

### Issue #10: Add Cross-Dataset Generalization Test (Synthetic → Real)

**Labels:** `intermediate`, `research`, `evaluation`

**Difficulty:** ⭐⭐ (Intermediate)

**Description:**

Train on synthetic data and evaluate on real Sleep-EDF. This measures how well the synthetic pipeline prepares for real data and identifies domain gaps.

**Why this matters:**
Synthetic data is fast for development, but if the model trained on it doesn't generalize to real data, we have a domain gap problem. Quantifying this helps improve synthetic data generation.

**Steps to Reproduce:**

1. Train a model on synthetic data:
   ```bash
   python -m sleep_staging.cli train-teacher --mode synthetic --epochs 30 --teacher-ckpt artifacts/teacher_synthetic.pt
   ```

2. Evaluate on real Sleep-EDF test set:
   ```bash
   python -m sleep_staging.cli evaluate --mode real --manifest data/manifests/sleep_edf.csv --teacher-ckpt artifacts/teacher_synthetic.pt
   ```

3. Compare results:
   ```
   Trained on Synthetic, Evaluated on Real:
   κ = 0.42 (poor generalization)
   
   Trained on Real, Evaluated on Real:
   κ = 0.636 (good fit)
   
   Domain gap: -0.216
   ```

4. Document findings in `DOMAIN_GAP_ANALYSIS.md`:
   - Which stages transfer well?
   - Which stages suffer?
   - Recommendations for synthetic data improvement

**Expected Output:**

- Report showing domain gap quantification
- Per-class analysis of cross-dataset performance
- Recommendations for improving synthetic data realism

**Resources:**
- Domain adaptation: https://arxiv.org/abs/1502.03167
- Transfer learning: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

**How to Submit:**
- Fork, create evaluation script, analyze results, document findings, and open a PR.

---

## Advanced-Tier Issues (⭐⭐⭐)

### Issue #11: Implement Mamba Temporal Encoder

**Labels:** `advanced`, `research`, `architecture`, `sota`

**Difficulty:** ⭐⭐⭐ (Advanced)

**Description:**

Replace the Transformer layer in the Teacher model with a Mamba block for improved long-range temporal modeling. Mamba is a state-space model that scales better than Transformer for long sequences.

**Why this matters:**
Sleep staging benefits from long-range context (up to 30 minutes of history). Mamba handles this more efficiently than Transformer attention, potentially improving κ to 0.70+.

**Steps to Reproduce:**

1. Install Mamba:
   ```bash
   pip install mamba-ssm
   ```

2. Create `src/sleep_staging/mamba_encoder.py`:
   ```python
   from mamba_ssm import Mamba
   
   class MambaTemporalEncoder(nn.Module):
       def __init__(self, d_model=128, seq_len=60):
           super().__init__()
           self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
           self.seq_len = seq_len
       
       def forward(self, x):
           # x shape: (B, seq_len, d_model)
           return self.mamba(x)
   ```

3. Update `TeacherCRNN` to use `MambaTemporalEncoder` instead of `TransformerEncoder`.

4. Train and compare:
   - Original Teacher (Transformer): κ = 0.636
   - Mamba-based Teacher: κ = ? (expected 0.665–0.685)

5. Report:
   - Training time comparison
   - Memory usage comparison
   - κ improvement
   - Per-class F1

**Expected Output:**

```
Model Comparison:
Transformer Teacher:  κ = 0.636, training time = 8h
Mamba Teacher:        κ = 0.672, training time = 6h (faster!)
Memory: 12 GB → 10 GB (reduced)
```

**Resources:**
- Mamba paper: https://arxiv.org/abs/2312.00752
- Mamba implementation: https://github.com/state-spaces/mamba
- State-space models: https://en.wikipedia.org/wiki/State-space_representation

**How to Submit:**
- Fork, create mamba_encoder.py, update models.py, train and benchmark, and open a PR with results.

---

### Issue #12: Implement Self-Supervised Pretraining on Unlabeled PSG Data

**Labels:** `advanced`, `research`, `pretraining`, `self-supervised`

**Difficulty:** ⭐⭐⭐ (Advanced)

**Description:**

Use contrastive learning (SimCLR or MoCo) to pretrain the model on unlabeled polysomnography data before supervised finetuning. This can improve downstream performance and reduce label requirements.

**Why this matters:**
Unlabeled PSG data is abundant; leveraging it via self-supervised pretraining can reduce the need for labeled Sleep-EDF data and improve generalization.

**Steps to Reproduce:**

1. Create `src/sleep_staging/ssl_pretraining.py` with a contrastive loss (SimCLR):
   ```python
   class ContrastiveLoss(nn.Module):
       def __init__(self, temperature=0.07):
           """NT-Xent (normalized temperature-scaled cross-entropy) loss."""
       
       def forward(self, z_i, z_j):
           # z_i, z_j: projection outputs from two augmented views
           # Compute cosine similarity and NT-Xent loss
   ```

2. Implement data augmentations for EEG:
   ```python
   class EEGAugmentation:
       - Temporal jitter (±50 ms)
       - Amplitude scaling (0.9–1.1×)
       - Frequency masking (random band zeroing)
       - Gaussian noise (σ=0.01)
   ```

3. Pretrain on unlabeled Sleep-EDF data for 50 epochs, then finetune on labeled subset.

4. Compare:
   - Supervised baseline (κ = 0.636)
   - SSL pretrained + finetuned (κ = ? expected 0.665–0.685)

5. Vary labeled data fraction and measure label efficiency:
   - 100% labeled: κ = 0.636
   - 50% labeled: κ = ?
   - 25% labeled: κ = ?

**Expected Output:**

```
Label Efficiency Analysis:
                100% Labels  50% Labels  25% Labels
Supervised:     0.636        0.580       0.520
SSL Pretrained: 0.672        0.650       0.610
Improvement:    +3.6%        +12%        +17%
```

**Resources:**
- SimCLR paper: https://arxiv.org/abs/2002.05709
- MoCo paper: https://arxiv.org/abs/1911.05722
- PyTorch SSL: https://github.com/lightly-ai/lightly

**How to Submit:**
- Fork, implement SSL pretraining pipeline, run label efficiency experiments, document results, and open a PR.

---

### Issue #13: Implement TFLite INT8 Quantization with Validation

**Labels:** `advanced`, `optimization`, `deployment`, `quantization`

**Difficulty:** ⭐⭐⭐ (Advanced)

**Description:**

Implement full TFLite INT8 quantization pipeline with representative dataset selection and validation. Ensure <2% accuracy loss and measure latency improvement on ARM hardware.

**Why this matters:**
Embedded deployment requires 4–8× model compression. INT8 quantization achieves this, but representative dataset quality directly impacts quantized model accuracy.

**Steps to Reproduce:**

1. Update `src/sleep_staging/export.py` with intelligent dataset selection:
   ```python
   def select_representative_dataset(manifest, cache_dir, n_samples=500):
       """Select diverse representative samples for quantization calibration."""
       # Strategy: stratified sampling by sleep stage
       # - 100 Wake samples
       # - 100 N1 samples
       # - 100 N2 samples
       # - 100 N3 samples
       # - 100 REM samples
       # Returns: generator yielding (raw, label) tuples
   ```

2. Implement quantization with validation:
   ```python
   def quantize_with_validation(student_ckpt, representative_data, output_path):
       # Load model
       # Run through representative dataset for calibration
       # Quantize to INT8
       # Validate: PyTorch vs TFLite output comparison
       # Measure: accuracy drop, model size, latency
   ```

3. Benchmark on ARM hardware (if available, or use emulation):
   ```bash
   # On Cortex-M4 or similar
   latency_fp32 = 120ms  # original model
   latency_int8 = 40ms   # quantized model
   speedup = 3.0x
   ```

4. Generate validation report:
   - Per-class accuracy before/after quantization
   - Confusion matrix comparison
   - Latency improvement
   - Size reduction (e.g., 54 MB → 13 MB)

**Expected Output:**

```
TFLite INT8 Quantization Report
================================
Model Size:          54 MB → 13 MB (4.2× compression)
Latency (Cortex-M4): 120 ms → 40 ms (3.0× speedup)
Accuracy Drop:       κ = 0.668 → 0.656 (1.8% loss)

Per-Class Accuracy:
                 FP32    INT8   Drop
Wake:           0.85    0.84   -1.2%
N1:             0.52    0.50   -3.8%
N2:             0.73    0.72   -1.4%
N3:             0.71    0.70   -1.4%
REM:            0.74    0.73   -1.4%

Status: ✅ PASS (<2% target)
```

**Resources:**
- TFLite quantization: https://www.tensorflow.org/lite/performance/post_training_quantization
- Representative dataset: https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/quantization/keras/quantizers
- ARM CMSIS-NN: https://github.com/ARM-software/CMSIS_5

**How to Submit:**
- Fork, implement quantization pipeline, validate, measure, document results, and open a PR.

---

### Issue #14: Add Multi-Dataset Training and Zero-Shot Transfer

**Labels:** `advanced`, `research`, `generalization`, `multi-dataset`

**Difficulty:** ⭐⭐⭐ (Advanced)

**Description:**

Train on multiple sleep datasets (Sleep-EDF + ISRUC-Sleep + MASS) jointly and evaluate zero-shot transfer performance. This measures how well the model generalizes across datasets.

**Why this matters:**
Real-world deployment requires robustness across different sleep labs with different equipment. Multi-dataset training improves generalization and identifies dataset-specific biases.

**Steps to Reproduce:**

1. Implement dataset loaders for:
   - Sleep-EDF Cassette (existing)
   - ISRUC-Sleep (Issue #2)
   - MASS (to implement)

2. Create `src/sleep_staging/multidataset_loader.py`:
   ```python
   class MultiDatasetLoader:
       def __init__(self, manifests_dict):
           # manifests_dict = {
           #     'sleep_edf': manifest1,
           #     'isruc': manifest2,
           #     'mass': manifest3
           # }
       
       def __iter__(self):
           # Interleave batches from each dataset
           # Optionally: balanced sampling per dataset
   ```

3. Train Teacher on all three datasets:
   ```bash
   python -m sleep_staging.cli train-multi \
       --manifests sleep_edf:data/manifests/sleep_edf.csv \
                   isruc:data/manifests/isruc.csv \
                   mass:data/manifests/mass.csv \
       --epochs 50
   ```

4. Evaluate zero-shot transfer:
   ```
   Trained on: Sleep-EDF + ISRUC + MASS
   
   Test on:
   - Sleep-EDF:  κ = 0.64
   - ISRUC:      κ = 0.58 (lower, different equipment)
   - MASS:       κ = 0.56 (lower, different protocol)
   
   vs. single-dataset training:
   - Trained on Sleep-EDF only → Test on ISRUC: κ = 0.42 (domain gap)
   
   Multi-dataset advantage: +16% generalization
   ```

5. Analyze per-stage transfer success.

**Expected Output:**

- Multi-dataset training framework
- Zero-shot transfer benchmark report
- Recommendations for robust sleep staging models

**Resources:**
- Multi-task learning: https://arxiv.org/abs/1807.06358
- Domain generalization: https://arxiv.org/abs/2103.02579
- MASS dataset: https://github.com/akaraspt/sleep_stage_annotation_v2

**How to Submit:**
- Fork, implement multi-dataset framework, train, benchmark across datasets, and open a PR.

---

### Issue #15: Implement Model Ensemble and Uncertainty Quantification

**Labels:** `advanced`, `research`, `ensemble`, `uncertainty`

**Difficulty:** ⭐⭐⭐ (Advanced)

**Description:**

Create an ensemble of Student models trained on different data splits and implement uncertainty quantification (Monte Carlo Dropout or Bayesian estimates) to provide confidence scores for predictions.

**Why this matters:**
Clinical deployment requires confidence scores. Ensembles improve robustness; uncertainty estimates help clinicians know when the model is uncertain (e.g., N2 vs. N3 boundary epochs).

**Steps to Reproduce:**

1. Implement ensemble training:
   ```python
   def train_ensemble(manifest, n_models=5, epochs=40):
       """Train multiple students on bootstrap resamples."""
       for i in range(n_models):
           train_records = bootstrap_sample(manifest)
           student = StudentCRNN()
           train_student(student, train_records, epochs)
           save_model(student, f'artifacts/student_ensemble_{i}.pt')
   ```

2. Implement Monte Carlo Dropout for uncertainty:
   ```python
   class StudentCRNNWithDropout(StudentCRNN):
       def forward(self, x, n_mc_samples=10, training=False):
           # During inference, run n_mc_samples forward passes with dropout enabled
           # Return mean and std of outputs
           outputs = [self._forward_with_dropout(x) for _ in range(n_mc_samples)]
           mean_logits = np.mean(outputs, axis=0)
           std_logits = np.std(outputs, axis=0)
           return mean_logits, std_logits
   ```

3. Evaluate ensemble performance:
   ```
   Single Student:    κ = 0.668
   Ensemble (5 models): κ = 0.695 (+4%)
   
   Uncertainty Calibration:
   - Predictions with σ < 0.1: κ = 0.82 (high confidence, correct)
   - Predictions with σ > 0.2: κ = 0.45 (low confidence, often wrong)
   ```

4. Visualize uncertainty on confusion matrix:
   - Mark cells with high uncertainty
   - Help clinicians focus on borderline cases

**Expected Output:**

- Ensemble checkpoint: 5 student models
- Uncertainty quantification framework
- Calibration analysis
- Clinical use case: "flagged epochs" for manual review

**Resources:**
- Ensemble methods: https://en.wikipedia.org/wiki/Ensemble_learning
- MC Dropout: https://arxiv.org/abs/1506.02142
- Uncertainty quantification: https://arxiv.org/abs/2003.06957

**How to Submit:**
- Fork, implement ensemble and uncertainty pipeline, benchmark, analyze calibration, and open a PR.

---

## How to Post These Issues on GitHub

1. Go to your repo: https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline-project
2. Click **Issues** → **New Issue**
3. For each intermediate issue (#6–#10), copy the **Description** section and set labels:
   - `intermediate`, plus category labels (`loss-functions`, `visualization`, etc.)
4. For each advanced issue (#11–#15), set labels:
   - `advanced`, plus category labels (`research`, `optimization`, etc.)
5. Leave **Assignee** blank (let contributors self-assign)

---

## Contributor Roadmap

**Week 1–2 (Good First Issues):**
- #1: Unit tests for preprocess.py
- #2: ISRUC-Sleep loader
- #3: Windows UTF-8 fix
- #4: Confusion matrix viz
- #5: Architecture diagram

**Week 3–4 (Intermediate Issues):**
- #6: Focal Loss comparison
- #7: SWDF Loss implementation
- #8: Per-epoch confusion matrices
- #9: Attention heatmaps
- #10: Domain gap analysis

**Week 5–8 (Advanced Issues):**
- #11: Mamba encoder (3–4 weeks)
- #12: SSL pretraining (3–4 weeks)
- #13: TFLite quantization (2–3 weeks)
- #14: Multi-dataset training (3–4 weeks)
- #15: Ensemble + uncertainty (3–4 weeks)

---

**Total: 15 issues covering Beginner → Intermediate → Advanced.**

This gives a clear progression path for contributors and ensures your repo can scale from individual fixes to substantial architecture improvements. 🚀
