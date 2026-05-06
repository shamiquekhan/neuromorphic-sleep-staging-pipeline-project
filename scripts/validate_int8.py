"""
Validate INT8 quantization accuracy against FP32 baseline.
Expected acceptable loss: < 0.02 (< 2% accuracy drop).
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sleep_staging.models import StudentCRNN
from sleep_staging.config import TrainConfig, EEGConfig
from sleep_staging.data import make_loaders, load_manifest
from sleep_staging.evaluate import evaluate_model

model_cfg = TrainConfig()


def validate_int8_quantization():
    """Compare INT8 vs FP32 predictions on test set."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load test set
    print("Loading test data...")
    manifest_path = "data/manifests/sleep_edf_full.csv"
    records = load_manifest(manifest_path)
    
    _, _, test_ds, _ = make_loaders(
        records, 
        split=(0.70, 0.15, 0.15), 
        mode="subject_level",
        batch_size=16,
        num_workers=0,
        seq_len=model_cfg.seq_len,
    )
    
    # Load FP32 baseline
    print(f"Loading FP32 student from artifacts/student.pt...")
    student_fp32 = StudentCRNN(model_cfg).to(device)
    ckpt = torch.load("artifacts/student.pt", map_location=device, weights_only=True)
    student_fp32.load_state_dict(ckpt)
    student_fp32.eval()
    
    # Create INT8 quantized version (dynamic quantization)
    print("Creating INT8 dynamic quantization...")
    student_int8 = torch.quantization.quantize_dynamic(
        student_fp32, 
        {torch.nn.Linear, torch.nn.GRU}, 
        dtype=torch.qint8
    )
    student_int8.eval()
    
    # Evaluate both
    print("\n" + "="*60)
    print("Evaluating FP32 baseline...")
    print("="*60)
    metrics_fp32, _ = evaluate_model(student_fp32, test_ds, device=device, verbose=True)
    
    print("\n" + "="*60)
    print("Evaluating INT8 quantized model...")
    print("="*60)
    metrics_int8, _ = evaluate_model(student_int8, test_ds, device=device, verbose=True)
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON: FP32 vs INT8")
    print("="*60)
    
    acc_drop = metrics_fp32["accuracy"] - metrics_int8["accuracy"]
    kappa_drop = metrics_fp32["kappa"] - metrics_int8["kappa"]
    
    print(f"Accuracy:       FP32={metrics_fp32['accuracy']:.4f}  INT8={metrics_int8['accuracy']:.4f}  Δ={acc_drop:+.4f}")
    print(f"Kappa:          FP32={metrics_fp32['kappa']:.4f}  INT8={metrics_int8['kappa']:.4f}  Δ={kappa_drop:+.4f}")
    print(f"Macro F1:       FP32={metrics_fp32['macro_f1']:.4f}  INT8={metrics_int8['macro_f1']:.4f}  Δ={metrics_fp32['macro_f1'] - metrics_int8['macro_f1']:+.4f}")
    
    # Decision
    acceptable_threshold = 0.02  # 2% accuracy drop
    if acc_drop < acceptable_threshold:
        print(f"\n✅ PASS: Accuracy drop {acc_drop:.4f} < {acceptable_threshold:.4f} threshold")
        print("INT8 quantization is acceptable for deployment.")
    else:
        print(f"\n❌ FAIL: Accuracy drop {acc_drop:.4f} >= {acceptable_threshold:.4f} threshold")
        print("INT8 quantization causes too much accuracy loss.")
        print("Consider Quantization-Aware Training (QAT) instead.")
    
    # Save results
    results = {
        "fp32_accuracy": float(metrics_fp32["accuracy"]),
        "int8_accuracy": float(metrics_int8["accuracy"]),
        "accuracy_drop": float(acc_drop),
        "fp32_kappa": float(metrics_fp32["kappa"]),
        "int8_kappa": float(metrics_int8["kappa"]),
        "kappa_drop": float(kappa_drop),
        "pass": bool(acc_drop < acceptable_threshold),
    }
    
    import json
    with open("artifacts/int8_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to artifacts/int8_validation_results.json")
    
    return results


if __name__ == "__main__":
    validate_int8_quantization()
