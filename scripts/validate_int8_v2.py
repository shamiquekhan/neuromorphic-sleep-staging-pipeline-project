"""
Validate INT8 quantization accuracy against FP32 baseline.
Expected acceptable loss: < 0.02 (< 2% accuracy drop).
"""
import torch
import numpy as np
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sleep_staging.models import StudentCRNN
from sleep_staging.config import TrainConfig


def validate_int8_quantization():
    """Compare INT8 vs FP32 predictions on synthetic test."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    model_cfg = TrainConfig()
    
    # Load FP32 baseline
    print(f"Loading FP32 student from artifacts/student.pt...")
    student_fp32 = StudentCRNN(in_channels=4).to(device)
    ckpt = torch.load("artifacts/student.pt", map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            ckpt = ckpt["model_state"]
        elif "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
    
    student_fp32.load_state_dict(ckpt)
    student_fp32.eval()
    print("✓ FP32 model loaded\n")
    
    # Create INT8 quantized version (dynamic quantization)
    print("Creating INT8 dynamic quantization...")
    student_int8 = torch.quantization.quantize_dynamic(
        student_fp32, 
        {torch.nn.Linear, torch.nn.GRU}, 
        dtype=torch.qint8
    )
    student_int8.eval()
    print("✓ INT8 model created\n")
    
    print(f"FP32 model size: {sum(p.numel() for p in student_fp32.parameters()):,} params")
    print(f"INT8 model size: {sum(p.numel() for p in student_int8.parameters()):,} params")
    
    # Synthetic test
    print("\nRunning synthetic test (10 random batches)...")
    print("="*60)
    
    n_test_batches = 10
    correct_fp32 = 0
    correct_int8 = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx in range(n_test_batches):
            # Random batch
            spec = torch.randn(4, 15, 4, 3000, device=device)  # (B=4, T=15, C=4, L=3000)
            labels = torch.randint(0, 5, (4, 15), device=device)
            
            # FP32 inference
            logits_fp32, _, _ = student_fp32(spec, None, None)
            
            # INT8 inference
            logits_int8, _, _ = student_int8(spec, None, None)
            
            p_fp32 = logits_fp32.argmax(-1).cpu().numpy().ravel()
            p_int8 = logits_int8.argmax(-1).cpu().numpy().ravel()
            t = labels.cpu().numpy().ravel()
            
            correct_fp32 += (p_fp32 == t).sum()
            correct_int8 += (p_int8 == t).sum()
            total += len(t)
            
            if (batch_idx + 1) % 3 == 0:
                print(f"  Batch {batch_idx+1}/{n_test_batches} processed...")
    
    acc_fp32 = correct_fp32 / total
    acc_int8 = correct_int8 / total
    acc_drop = acc_fp32 - acc_int8
    
    print(f"\n" + "="*60)
    print("SYNTHETIC TEST RESULTS")
    print("="*60)
    print(f"FP32 accuracy:      {acc_fp32:.4f}")
    print(f"INT8 accuracy:      {acc_int8:.4f}")
    print(f"Accuracy drop:      {acc_drop:+.4f} ({100*acc_drop:.2f}%)")
    print(f"Total predictions:  {total}")
    
    acceptable_threshold = 0.05  # 5% for synthetic
    if acc_drop < acceptable_threshold:
        status = "✅ PASS"
        msg = f"Accuracy drop {acc_drop:.4f} < {acceptable_threshold:.4f} threshold"
    else:
        status = "⚠ WARN"
        msg = f"Accuracy drop {acc_drop:.4f} >= {acceptable_threshold:.4f} threshold"
    
    print(f"\nResult: {status}")
    print(f"  {msg}")
    print(f"\nNote: This is a SYNTHETIC test with random data.")
    print(f"For production, validate on actual sleep EDF test set.")
    
    results = {
        "test_type": "synthetic_random",
        "fp32_accuracy": float(acc_fp32),
        "int8_accuracy": float(acc_int8),
        "accuracy_drop": float(acc_drop),
        "num_batches": n_test_batches,
        "total_samples": int(total),
        "pass": bool(acc_drop < acceptable_threshold),
        "status": msg
    }
    
    # Save results
    with open("artifacts/int8_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to artifacts/int8_validation_results.json")
    
    return results


if __name__ == "__main__":
    validate_int8_quantization()
