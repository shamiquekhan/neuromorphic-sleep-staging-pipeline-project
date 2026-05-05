"""Enhanced export with validation and benchmarking.

After distillation completes, run:
  python scripts/export_validated.py \
      --checkpoint artifacts/student_final.pt \
      --output-dir artifacts/export \
      --quantize int8 \
      --rep-data data/cache \
      --validate \
      --benchmark
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleep_staging.models import StudentCRNN


def load_checkpoint(model, ckpt_path):
    """Load model weights flexibly."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    
    # Strip 'module.' prefix if present
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    return model


def export_onnx(model, onnx_path, sample):
    """Export the student model to ONNX."""
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    torch.onnx.export(
        model,
        (sample,),
        str(onnx_path),
        opset_version=13,
        input_names=["raw"],
        output_names=["logits", "h"],
        dynamic_axes={"raw": {0: "batch", 1: "seq"}},
        do_constant_folding=True,
    )
    return onnx_path


def convert_onnx_to_savedmodel(onnx_path, savedmodel_dir):
    """Convert ONNX to TensorFlow SavedModel via onnx-tf."""
    import onnx
    from onnx_tf.backend import prepare

    onnx_path = Path(onnx_path)
    savedmodel_dir = Path(savedmodel_dir)
    savedmodel_dir.mkdir(parents=True, exist_ok=True)

    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(savedmodel_dir))
    return savedmodel_dir


def convert_savedmodel_to_tflite(savedmodel_dir, tflite_path, quantize="int8", rep_data_dir=Path("data/cache")):
    """Convert SavedModel to TFLite, optionally with int8 quantization."""
    import tensorflow as tf

    savedmodel_dir = Path(savedmodel_dir)
    tflite_path = Path(tflite_path)
    tflite_path.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))

    if quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep_data():
            files = sorted(Path(rep_data_dir).glob("*_raw.npy"))[:500]
            for f in files:
                arr = np.load(str(f)).astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, np.newaxis, ...]
                elif arr.ndim == 3:
                    arr = arr[np.newaxis, ...]
                else:
                    continue
                yield [arr]

        converter.representative_dataset = rep_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(str(tflite_path), "wb") as f:
        f.write(tflite_model)
    return tflite_path


def validate_tflite_vs_pytorch(pt_model, tflite_path, test_dir, n_samples=50):
    """Compare outputs: PyTorch vs TFLite quantized."""
    try:
        import tensorflow as tf
    except ImportError:
        print("⊘ TensorFlow not available — skipping validation")
        return {}
    
    test_dir = Path(test_dir)
    files = sorted(test_dir.glob("*_raw.npy"))[:n_samples]
    if not files:
        print(f"⊘ No test samples in {test_dir}")
        return {}
    
    print(f"\n  Validating on {len(files)} samples...")
    
    # TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']
    
    pt_model.eval()
    diffs = []
    
    with torch.no_grad():
        for f in files:
            arr = np.load(str(f)).astype(np.float32)
            if arr.ndim == 1:
                continue
            if arr.ndim == 2:
                arr = arr[np.newaxis, np.newaxis, ...]  # (C,L) -> (1,1,C,L)
            elif arr.ndim == 3:
                arr = arr[np.newaxis, ...]  # (N,C,L) -> (1,N,C,L)
            else:
                continue
            
            # PyTorch
            inp = torch.from_numpy(arr)
            out_pt = pt_model(inp)
            if isinstance(out_pt, tuple):
                out_pt = out_pt[0]
            out_pt = torch.softmax(out_pt, dim=-1).numpy()
            
            # TFLite
            interpreter.set_tensor(input_idx, arr)
            interpreter.invoke()
            out_tf = interpreter.get_tensor(output_idx)
            if out_tf.dtype == np.uint8:
                out_tf = out_tf.astype(np.float32) / 255.0
            
            diffs.append(np.abs(out_pt - out_tf).mean())
    
    if diffs:
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        print(f"  Output difference: mean={mean_diff:.6f}, max={max_diff:.6f}")
        if mean_diff > 0.05:
            print(f"  ⚠ Quantization may have degraded accuracy")
        else:
            print(f"  ✓ Quantization acceptable")
        return {"mean_diff": float(mean_diff), "max_diff": float(max_diff)}
    return {}


def benchmark_sizes(tflite_path, onnx_path=None):
    """Compare model sizes."""
    print(f"\n  Model sizes:")
    sizes = {}
    
    if Path(tflite_path).exists():
        mb = Path(tflite_path).stat().st_size / 1e6
        sizes["tflite"] = round(mb, 2)
        print(f"    TFLite:  {mb:.2f} MB")
    
    if onnx_path and Path(onnx_path).exists():
        mb = Path(onnx_path).stat().st_size / 1e6
        sizes["onnx"] = round(mb, 2)
        print(f"    ONNX:    {mb:.2f} MB")
    
    sizes["pytorch_est"] = 54.0  # Student ~54MB
    if sizes.get("tflite"):
        ratio = sizes["pytorch_est"] / sizes["tflite"]
        print(f"    Compression: {ratio:.1f}× vs PyTorch")
        sizes["compression_ratio"] = round(ratio, 1)
    
    return sizes


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/student_final.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/export_enhanced"))
    p.add_argument("--quantize", default="int8", choices=["none", "float16", "int8"])
    p.add_argument("--rep-data", type=Path, default=Path("data/cache"))
    p.add_argument("--validate", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()
    
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("ENHANCED EXPORT: PyTorch → ONNX → SavedModel → TFLite (with validation)")
    print("="*70)
    
    # Load model
    print("\n[1/4] Loading student model...")
    model = StudentCRNN()
    if args.checkpoint.exists():
        load_checkpoint(model, args.checkpoint)
        print(f"✓ Loaded {args.checkpoint}")
    else:
        print(f"⊘ {args.checkpoint} not found")
    
    sample = torch.randn(1, 1, 4, 3000)
    onnx_path = out / "student.onnx"
    savedmodel_dir = out / "saved_model"
    tflite_path = out / f"student_{args.quantize}.tflite"
    
    # Export ONNX
    print("\n[2/4] Exporting to ONNX...")
    try:
        model.eval()
        torch.onnx.export(
            model, (sample,), str(onnx_path),
            opset_version=13,
            input_names=["raw"],
            output_names=["logits", "h"],
            dynamic_axes={"raw": {0: "batch", 1: "seq"}},
            do_constant_folding=True,
        )
        print(f"✓ {onnx_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return
    
    # SavedModel
    print("\n[3/4] Converting → SavedModel...")
    try:
        import onnx
        from onnx_tf.backend import prepare
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(savedmodel_dir))
        print(f"✓ {savedmodel_dir}")
    except Exception as e:
        print(f"✗ SavedModel conversion failed: {e}")
        print("  (You can skip TFLite and use ONNX directly)")
        return
    
    # TFLite
    print("\n[4/4] Converting → TFLite...")
    try:
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))
        
        if args.quantize == "int8":
            print(f"  Quantizing to int8...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            def rep_data():
                files = list(args.rep_data.glob("*_raw.npy"))[:500]
                for f in files:
                    arr = np.load(str(f)).astype(np.float32)
                    if arr.ndim == 2:
                        arr = arr[np.newaxis, np.newaxis, ...]
                    elif arr.ndim == 3:
                        arr = arr[np.newaxis, ...]
                    else:
                        continue
                    yield [arr]
            
            converter.representative_dataset = rep_data
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        elif args.quantize == "float16":
            print(f"  Quantizing to float16...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(tflite_path), "wb") as f:
            f.write(tflite_model)
        print(f"✓ {tflite_path}")
    except Exception as e:
        print(f"✗ TFLite conversion failed: {e}")
        return
    
    # Validation & benchmarking
    results = {
        "checkpoint": str(args.checkpoint),
        "quantization": args.quantize,
        "outputs": {
            "onnx": str(onnx_path),
            "savedmodel": str(savedmodel_dir),
            "tflite": str(tflite_path),
        }
    }
    
    if args.validate:
        print("\n[VALIDATION] Comparing PyTorch vs TFLite...")
        results["validation"] = validate_tflite_vs_pytorch(
            model, tflite_path, args.rep_data
        )
    
    if args.benchmark:
        print("\n[BENCHMARKING] Model sizes...")
        results["sizes"] = benchmark_sizes(tflite_path, onnx_path)
    
    # Save report
    report = out / "export_report.json"
    with open(report, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Report: {report}")
    
    print("\n" + "="*70)
    print("✓ EXPORT COMPLETE")
    print("="*70)
    print(f"\nNext: Convert to C array for firmware:")
    print(f"  xxd -i {tflite_path} > firmware/src/models/student.h\n")


if __name__ == "__main__":
    main()
