# Good First Issues for Contributors

Copy each issue body into a GitHub issue. Adjust titles and labels as needed.

---

## Issue #1: Add Unit Tests for `preprocess.py`

**Labels:** `good first issue`, `testing`, `help wanted`

**Difficulty:** ⭐ (Beginner)

**Description:**

The preprocessing module (`src/sleep_staging/preprocess.py`) lacks unit tests. This issue covers writing tests for the core preprocessing pipeline:

- `bandpass_filter()` — verify correct frequency range extraction
- `robust_normalize()` — verify Z-score normalization with outlier handling
- `epoch_iterator()` — verify correct 30-second epoch chunking
- `process_recording()` — end-to-end test with synthetic EDF inputs

**Why this matters:**
We need confidence that preprocessing doesn't silently corrupt signal integrity. Tests catch bugs that visual inspection misses.

**Steps to Reproduce:**

1. Create `tests/test_preprocess.py` in the repo.
2. Write a synthetic 30-min EEG signal using NumPy.
3. Test that `bandpass_filter(fs=100, lo=0.5, hi=30)` removes DC and high-frequency noise.
4. Verify `robust_normalize()` output has mean ≈ 0, std ≈ 1.
5. Check `epoch_iterator()` yields exactly 60 epochs from a 30-min recording.

**Expected Output:**

```bash
pytest tests/test_preprocess.py -v
# Should show: 4 passed in 0.5s
```

**Resources:**
- Preprocessing module: `src/sleep_staging/preprocess.py` (lines 1–200)
- Example test style: pytest docs (https://docs.pytest.org)

**How to Submit:**
- Fork the repo, create a branch `tests/add-preprocess-tests`, commit your tests, and open a PR.
- Ensure all tests pass locally before submitting.

---

## Issue #2: Add Support for ISRUC-Sleep Dataset

**Labels:** `good first issue`, `enhancement`, `data`

**Difficulty:** ⭐⭐ (Beginner–Intermediate)

**Description:**

The pipeline currently supports only Sleep-EDF. Adding [ISRUC-Sleep](https://www.isruc.inesctec.pt/resources/downloads/) support allows comparison with a second public dataset and increases contributor interest.

**Why this matters:**
ISRUC-Sleep has different EEG sampling rates (200 Hz vs Sleep-EDF's 100 Hz) and annotation schemes. Supporting it tests our pipeline's flexibility.

**Steps to Reproduce:**

1. Download ISRUC-Sleep Session 1 (≈10 recordings, ~1 GB).
2. Create `src/sleep_staging/isruc_loader.py` with a `load_isruc_recording(edf_path, hypnogram_path)` function.
3. Map ISRUC sleep stages to AASM: `{1: Wake, 2: N1, 3: N2, 4: N3, 5: REM}`.
4. Resample 200 Hz signals down to 100 Hz using `scipy.signal.resample()`.
5. Write a manifest builder similar to `data.py`'s `build_manifest()`.

**Expected Output:**

```python
from sleep_staging.isruc_loader import load_isruc_recording

raw, labels = load_isruc_recording("path/to/SC1_01_0800.edf", "path/to/SC1_01_0800_hypnogram.edf")
print(raw.shape)   # (120, 4, 3000) — 120 epochs
print(labels)      # array([0, 1, 2, 3, 4, 0, ...])
```

**Resources:**
- ISRUC-Sleep docs: https://www.isruc.inesctec.pt/
- Resampling example: https://scipy.io/docs/scipy.signal.resample/
- Existing Sleep-EDF loader: `src/sleep_staging/data.py` lines 50–150

**How to Submit:**
- Fork, create `isruc/add-isruc-support`, implement the loader, test on 2–3 recordings, and open a PR.

---

## Issue #3: Fix Windows-Specific Logging Bug in `train.py`

**Labels:** `good first issue`, `bug`, `windows`

**Difficulty:** ⭐ (Beginner)

**Description:**

On Windows, the logger writes UTF-8 characters (e.g., "κ=0.636") to the console but encodes as cp1252 by default, causing UnicodeEncodeError. The fix is a one-liner that forces UTF-8 output.

**Why this matters:**
Windows contributors encounter crashes during training with cryptic errors. This fix improves the developer experience.

**Steps to Reproduce:**

1. On Windows, run: `python -m sleep_staging.cli train-teacher --mode synthetic --epochs 1`
2. Wait for the first progress line.
3. Observe: `UnicodeEncodeError: 'cp1252' codec can't encode character '\u03ba'`

**Expected Fix:**

In `src/sleep_staging/train.py` at the top of the `train_teacher()` function, add:

```python
import sys
import io

if sys.platform == "win32" and not hasattr(sys.stdout, 'reconfigure'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

After the fix, logs should print cleanly.

**Resources:**
- Similar fix: https://github.com/pytorch/pytorch/issues/57776
- Python codec docs: https://docs.python.org/3/library/codecs.html

**How to Submit:**
- Fork, edit `src/sleep_staging/train.py`, test on Windows, commit, and open a PR.

---

## Issue #4: Add Confusion Matrix Visualization after Evaluation

**Labels:** `good first issue`, `visualization`, `enhancement`

**Difficulty:** ⭐⭐ (Beginner–Intermediate)

**Description:**

After evaluation, create a confusion matrix heatmap showing predicted vs. true sleep stages. This helps identify which stages the model confuses (e.g., N2 vs. N3).

**Why this matters:**
Confusion matrices are standard in ML publications. Our eval reports should include one to help debug model behavior.

**Steps to Reproduce:**

1. In `src/sleep_staging/evaluate.py`, after computing predictions, add:

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Student Model Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150)
```

2. Test by running: `python -m sleep_staging.cli evaluate --mode synthetic`
3. Verify `artifacts/confusion_matrix.png` appears.

**Expected Output:**

A clean heatmap saved to `artifacts/confusion_matrix.png` with diagonal values high (correct predictions) and off-diagonals showing common confusions.

**Resources:**
- Confusion matrix docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- Seaborn heatmap: https://seaborn.pydata.org/generated/seaborn.heatmap.html

**How to Submit:**
- Fork, edit `src/sleep_staging/evaluate.py`, run synthetic evaluation, commit the visualization code, and open a PR.

---

## Issue #5: Document the Full Pipeline Architecture with ASCII Diagram

**Labels:** `good first issue`, `documentation`

**Difficulty:** ⭐ (Beginner)

**Description:**

Create a visual guide showing how data flows through the pipeline: data loading → preprocessing → train/distill → export. Include ASCII art or Mermaid diagram in `ARCHITECTURE.md`.

**Why this matters:**
New contributors are confused about how the 15 modules interact. A clear diagram reduces onboarding time and encourages deeper understanding.

**Steps to Reproduce:**

1. Create `ARCHITECTURE.md` in the repo root.
2. Include sections:
   - **Data Flow**: EDF → Preprocess → Cache → DataLoader → Model
   - **Model Architecture**: Teacher (CNN + Transformer) → Student (1D-ResNet-SE + GRU)
   - **Training Pipeline**: Train Teacher → Evaluate Teacher → Distill Student → Export ONNX → TFLite
   - **Module Dependency Graph**: Which modules import which
3. Add ASCII art or a Mermaid diagram (see example below).

**Example Mermaid Diagram:**

```
graph TD
    A[Sleep-EDF Files] --> B[build-manifest]
    B --> C[Manifest CSV]
    C --> D[preprocess.py]
    D --> E[Cache .npy files]
    E --> F[DataLoader]
    F --> G[train.py - Teacher]
    G --> H[teacher.pt]
    H --> I[distill.py]
    F --> I
    I --> J[student.pt]
    J --> K[evaluate.py]
    K --> L[Metrics + Confusion Matrix]
    J --> M[export.py]
    M --> N[ONNX + TFLite]
```

**Expected Output:**

A well-formatted `ARCHITECTURE.md` file that contributors can reference when understanding how their issue fits into the larger system.

**Resources:**
- Mermaid diagram editor: https://mermaid.live
- Example architecture docs: https://github.com/pytorch/pytorch/blob/master/ARCHITECTURE.md

**How to Submit:**
- Fork, create `ARCHITECTURE.md`, include a clear diagram, and open a PR.

---

## How to Post These Issues on GitHub

1. Go to your GitHub repo: https://github.com/shamiquekhan/neuromorphic-sleep-staging-pipeline-project
2. Click **Issues** → **New Issue**
3. For each issue above, copy the **Description** section into the issue body
4. Set the **Labels** listed at the top
5. Leave **Assignee** blank (let contributors self-assign)
6. Click **Submit new issue**

---

## Next Steps

- Add these 5 issues to GitHub (should take ~10 min)
- Create a `CONTRIBUTING.md` file (optional but recommended) explaining the PR process
- Announce the program on social media / forums if seeking contributors

You're now **contributor-ready**! 🚀
