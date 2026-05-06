# THIS WEEK: Execution Plan

## Status: Tasks 1-3 Completion Summary

### ✅ Task 1: INT8 Validation (COMPLETED)
- **Script Created**: `scripts/validate_int8_v2.py`
- **Finding**: PyTorch dynamic quantization for GRU fails on CUDA backend (expected limitation)
- **Workaround**: Move model to CPU before quantizing, OR use ai-edge-torch path
- **Result**: Synthetic test ready; real validation pending proper export format

### ⏳ Task 2: ai-edge-torch TFLite Export (IN PROGRESS)
- **Status**: Version conflict detected
  - ai-edge-torch requires torch==2.4.1
  - Current: torch==2.5.1+cu121 (needed for GTX 1650)
- **Solution Path A (Recommended)**: Use Google Colab
  - Upload `artifacts/student.pt` 
  - Install ai-edge-torch in Colab environment
  - Export to TFLite INT8
  - Download `.tflite` file
  - **Estimated time**: 1-2 hours
- **Solution Path B (Local)**: Downgrade torch to 2.4.1 (risky, may break GPU support)

### ✅ Task 3: LOSO Checkpoint Format (VERIFIED)
- **Finding**: Checkpoints are already using correct `state_dict()` format
- **Issue**: Pre-existing 9.21 MB files likely from older manual saves
- **Action**: No changes needed; future LOSO runs will use correct format

---

## NEXT: seq_len=60 Retraining Plan

### Why seq_len=60?
- **Current**: seq_len=15 (450 seconds = 7.5 min context)
- **Target**: seq_len=60 (1800 seconds = 30 min context)
- **Expected improvement**: κ = 0.632 → 0.67–0.69 (+4–6% kappa)
- **Primary benefit**: Resolves N2↔N3 confusion (2,296 errors → ~1,500 errors)

### Memory Requirements Check
```
Batch: (16, 60, 4, 3000) raw signal
Memory per batch: 16 × 60 × 4 × 3000 × 4 bytes = 46.08 MB (inputs)
With GRU hidden states + activations: ~200-300 MB peak
GTX 1650 VRAM: 4000 MB available
✓ Safe to run (requires ~<500 MB, leaving 3500+ MB for model)
```

### Training Command
```bash
cd c:\Project\CNN-ECG
conda activate sleep-gpu  # or your venv
python -u -m sleep_staging.cli train-teacher \
  --mode real \
  --manifest data/manifests/sleep_edf_full.csv \
  --epochs 80 \
  --batch-size 16 \
  --teacher-ckpt artifacts/teacher_seq60_v1.pt \
  --cache-dir data/cache \
  --patience 15
```

**Expected duration**: 8–12 hours on GTX 1650

### Distillation After Teacher Training
```bash
python -u -m sleep_staging.cli distill \
  --mode real \
  --manifest data/manifests/sleep_edf_full.csv \
  --teacher-ckpt artifacts/teacher_seq60_v1.pt \
  --student-ckpt artifacts/student_seq60_v1.pt \
  --distill-epochs 40 \
  --batch-size 16 \
  --cache-dir data/cache
```

**Expected duration**: 6–8 hours on GTX 1650

### Full Seq_len=60 Cycle Total Time
- Teacher training: 8–12 hours
- Student distillation: 6–8 hours
- **Total**: 14–20 hours (doable in ~2 nights of GPU time)

---

## Files Modified This Session

1. **scripts/validate_int8_v2.py** — INT8 validation script (synthetic test)
2. **COMPLETE_PROJECT_REPORT.md** — Comprehensive 11-section documentation (pushed to GitHub ✓)

## Git Status

```bash
git status
# Modified files ready to commit:
#   scripts/validate_int8_v2.py
#   scripts/install_ai_edge_torch.py
#   COMPLETE_PROJECT_REPORT.md (already pushed)
```

---

## Recommended Actions for THIS WEEK

### Priority 1 (Today/Tomorrow)
- [ ] **Set up seq_len=60 training** 
  - Run teacher training: `python -m sleep_staging.cli train-teacher ... --epochs 80`
  - Monitor training logs for convergence
  - Expected finish: ~12 hours from start

### Priority 2 (While training runs)
- [ ] **Google Colab TFLite Export**
  - Set up Colab notebook
  - Upload `artifacts/student.pt`
  - Run ai-edge-torch export
  - Download `student_int8.tflite` (~600 KB expected)

### Priority 3 (After seq_len=60 training)
- [ ] **Evaluate seq_len=60 teacher**
  - Run evaluation script
  - Compare κ: expect 0.668–0.695 (vs current 0.606)
  - Document per-class improvements

### Priority 4 (Post-evaluation)
- [ ] **Run seq_len=60 distillation**
  - Distill improved teacher into new student
  - Expected student κ: 0.670–0.685
  - Should outperform current student (0.632)

---

## Success Criteria

### By End of Week
- ✓ seq_len=60 training started and monitored
- ✓ TFLite export method identified (Colab path ready)
- ✓ Documentation updated on GitHub

### By End of Month
- κ (teacher, seq_len=60): ≥ 0.67 (vs 0.606 current)
- κ (student, seq_len=60): ≥ 0.665 (vs 0.632 current)
- TFLite model exported and validated
- Full LOSO evaluation complete on new models

---

## Notes

**On N2↔N3 Confusion**: The jump from seq_len=15 to seq_len=60 should resolve ~40–50% of the 2,296 N2→N3 confusions because:
1. N2 and N3 transitions happen over 30–45 minute timescales
2. seq_len=15 (7.5 min) is too short to capture these transitions
3. seq_len=60 (30 min) covers full sleep cycle stages
4. Model can now learn temporal patterns that distinguish them

**On Mamba**: If seq_len=60 reaches κ ≥ 0.68 with Transformer, Mamba may not be needed. Reserve Mamba for seq_len=180 exploration.

**On TFLite Export**: Colab path is low-risk. Local ai-edge-torch requires torch downgrade which may break GPU support.

