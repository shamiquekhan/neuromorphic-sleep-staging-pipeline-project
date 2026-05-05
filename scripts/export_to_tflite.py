"""Export student model to ONNX -> TensorFlow SavedModel -> TFLite.

Usage:
  python scripts/export_to_tflite.py \
      --checkpoint artifacts/student.pt \
      --output-dir artifacts/export \
      --quantize int8 \
      --rep-data data/cache

This script attempts the following steps:
  1. Load `StudentCRNN` from `src.sleep_staging.models` and load checkpoint.
  2. Export to ONNX (dynamic batch and seq dims).
  3. Convert ONNX -> TensorFlow SavedModel (requires onnx and onnx-tf).
  4. Convert SavedModel -> TFLite (optional post-training quantization).

Notes:
  - Requires heavy optional dependencies: `onnx`, `onnx-tf`, `tensorflow`.
  - INT8 quantization requires a representative dataset directory with .npy raw arrays
    (shaped like (4, 3000) or (N,4,3000)). A small subset is sufficient.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleep_staging.models import StudentCRNN


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("model") or k.startswith("student") for k in ckpt.keys()):
        # try to find a state_dict-like entry
        for k in ("model_state_dict", "state_dict", "student_state_dict", "net"]):
            if k in ckpt:
                state = ckpt[k]
                break
        else:
            state = ckpt
    else:
        state = ckpt

    # if keys are prefixed like 'module.' strip them
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)
    return model


def export_onnx(model: torch.nn.Module, out_path: Path, sample_input: torch.Tensor):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    # model.forward expects spec/raw as first arg (raw shaped (B,T,C,L))
    input_names = ["raw"]
    output_names = ["logits","h"]
    dynamic_axes = {"raw": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}}
    torch.onnx.export(
        model,
        (sample_input,),
        str(out_path),
        opset_version=13,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"Exported ONNX -> {out_path}")


def convert_onnx_to_savedmodel(onnx_path: Path, savedmodel_dir: Path):
    try:
        import onnx
        from onnx_tf.backend import prepare
    except Exception as e:
        raise RuntimeError("onnx and onnx-tf are required for ONNX->SavedModel conversion. Install with `pip install onnx onnx-tf`") from e

    savedmodel_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(onnx_path))
    tf_rep = prepare(model)
    tf_rep.export_graph(str(savedmodel_dir))
    print(f"Converted ONNX -> SavedModel at {savedmodel_dir}")


def convert_savedmodel_to_tflite(savedmodel_dir: Path, tflite_path: Path, quantize: str = None, rep_data_dir: Path = None):
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("tensorflow is required for SavedModel->TFLite conversion. Install with `pip install tensorflow`") from e

    converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))
    if quantize is None:
        tflite_model = converter.convert()
    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
    elif quantize == "int8":
        if rep_data_dir is None:
            raise RuntimeError("INT8 quantization requires --rep-data directory with .npy samples")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_dataset_gen():
            # iterate over .npy files in rep_data_dir
            import numpy as np
            files = list(Path(rep_data_dir).glob("**/*.npy"))
            if not files:
                raise RuntimeError("No .npy files found in representative data directory")
            for f in files:
                arr = np.load(str(f))
                # arr can be (4,3000) or (N,4,3000)
                if arr.ndim == 2:
                    inp = arr[np.newaxis, np.newaxis, ...].astype("float32")
                elif arr.ndim == 3:
                    inp = arr[np.newaxis, ...].astype("float32")
                else:
                    continue
                # yield as list of tensors matching converter input
                yield [inp]

        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
    else:
        raise RuntimeError(f"Unknown quantize option: {quantize}")

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(tflite_path), "wb") as f:
        f.write(tflite_model)
    print(f"Wrote TFLite model to {tflite_path}")


def find_representative_files(rep_dir: Path):
    return list(rep_dir.glob("**/*.npy"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/student.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/export"))
    p.add_argument("--quantize", choices=(None, "int8", "float16"), default=None)
    p.add_argument("--rep-data", type=Path, default=None,
                   help="Directory with .npy representative raw samples for int8 quantization")
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    model = StudentCRNN()
    if args.checkpoint.exists():
        try:
            load_checkpoint(model, args.checkpoint)
            print(f"Loaded checkpoint {args.checkpoint}")
        except Exception as e:
            print("Warning: failed to load checkpoint exactly — proceeding with model defaults", e)
    else:
        print(f"Checkpoint {args.checkpoint} not found — exporting randomly initialized model")

    # Dummy input: (B, T, C, L) -> we'll export with batch=1, seq=1
    sample = torch.randn(1, 1, 4, 3000)

    onnx_path = out / "student.onnx"
    savedmodel_dir = out / "saved_model"
    tflite_path = out / "student.tflite"

    export_onnx(model, onnx_path, sample)

    try:
        convert_onnx_to_savedmodel(onnx_path, savedmodel_dir)
    except Exception as e:
        print("ONNX->SavedModel conversion failed:", e)
        print("You can stop here and use the ONNX model directly with ONNX Runtime on-device.")
        return

    try:
        convert_savedmodel_to_tflite(savedmodel_dir, tflite_path, quantize=args.quantize, rep_data_dir=args.rep_data)
    except Exception as e:
        print("SavedModel->TFLite conversion failed:", e)
        print("You can still use the SavedModel in TF or try converting manually.")
        return


if __name__ == "__main__":
    main()
