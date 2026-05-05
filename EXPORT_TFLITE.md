# Exporting the student model to TFLite

This repository includes `scripts/export_to_tflite.py` which exports the student PyTorch model to ONNX 12 TensorFlow SavedModel 12 TFLite.

Quick steps (recommended):

1) Create a Python environment (conda recommended) and install core deps:

```powershell
python -m pip install -r requirements.txt
python -m pip install onnx-tf tensorflow
```

2) Run the exporter (float32 TFLite):

```powershell
python scripts/export_to_tflite.py --checkpoint artifacts/student.pt --output-dir artifacts/export
```

3) Run with float16 quantization:

```powershell
python scripts/export_to_tflite.py --checkpoint artifacts/student.pt --output-dir artifacts/export --quantize float16
```

4) Run with int8 quantization (requires representative `.npy` files):

```powershell
# prepare small representative set: save a few raw arrays shaped (4,3000) or (N,4,3000)
python scripts/export_to_tflite.py --checkpoint artifacts/student.pt --output-dir artifacts/export --quantize int8 --rep-data data/cache
```

Notes & troubleshooting:
- If ONNX->SavedModel conversion fails, you can use the ONNX model directly with ONNX Runtime on-device.
- INT8 quantization needs a small set (50-500) of representative examples; it does not require labels.
- For microcontroller deployment, consider also converting to C array (there's `scripts/bin_to_c_array.py` in the repo).
