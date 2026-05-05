"""Real-time status dashboard for all 4 phases."""

import json
import time
from pathlib import Path
from datetime import datetime

def get_phase_status():
    """Check status of all phases."""
    status = {
        "timestamp": datetime.now().isoformat(),
        "phases": {}
    }
    
    # Phase 1: LOSO
    loso_results = Path('artifacts/loso_seq60_focal/loso_results.json')
    if loso_results.exists():
        with open(loso_results) as f:
            data = json.load(f)
        status["phases"]["1_loso"] = {
            "status": "✓ COMPLETE",
            "kappa_mean": data.get('summary', {}).get('mean_kappa'),
            "accuracy_mean": data.get('summary', {}).get('mean_accuracy'),
            "n_folds": data.get('summary', {}).get('n_folds'),
            "path": str(loso_results)
        }
    else:
        status["phases"]["1_loso"] = {
            "status": "🔄 RUNNING",
            "message": "Waiting for LOSO benchmark to complete (~3-4 hours)"
        }
    
    # Phase 2: Teacher retrain
    teacher_v2 = Path('artifacts/teacher_improved_v2.pt')
    if teacher_v2.exists():
        size_mb = teacher_v2.stat().st_size / 1e6
        status["phases"]["2_teacher_retrain"] = {
            "status": "✓ COMPLETE",
            "checkpoint": "artifacts/teacher_improved_v2.pt",
            "size_mb": round(size_mb, 1),
            "config": {
                "seq_len": 60,
                "focal_gamma": 2.0,
                "mixup_alpha": 0.4,
                "label_smoothing": 0.1,
                "augmentations": ["CutMix", "ChannelDropout", "GaussianNoise", "TimeShift"]
            }
        }
    else:
        status["phases"]["2_teacher_retrain"] = {
            "status": "🔄 RUNNING",
            "message": "Training with quick-win config (~8 hours)"
        }
    
    # Phase 3: Distillation
    student_improved = Path('artifacts/student_improved.pt')
    if student_improved.exists():
        size_mb = student_improved.stat().st_size / 1e6
        status["phases"]["3_distillation"] = {
            "status": "✓ COMPLETE",
            "checkpoint": "artifacts/student_improved.pt",
            "size_mb": round(size_mb, 1)
        }
    else:
        status["phases"]["3_distillation"] = {
            "status": "⏳ WAITING for Phase 2",
            "message": "Will auto-start once teacher_improved_v2.pt is ready"
        }
    
    # Phase 4: Export
    tflite_model = Path('artifacts/export_final/student_int8.tflite')
    if tflite_model.exists():
        size_mb = tflite_model.stat().st_size / 1e6
        status["phases"]["4_export"] = {
            "status": "✓ COMPLETE",
            "outputs": {
                "onnx": "artifacts/export_final/student.onnx",
                "savedmodel": "artifacts/export_final/saved_model",
                "tflite_int8": str(tflite_model)
            },
            "tflite_size_mb": round(size_mb, 1),
            "message": "Ready for firmware deployment"
        }
    else:
        status["phases"]["4_export"] = {
            "status": "⏳ WAITING for Phase 3",
            "message": "Will auto-start after distillation completes"
        }
    
    return status


def print_dashboard():
    """Print nicely formatted status dashboard."""
    status = get_phase_status()
    
    print("\n" + "="*80)
    print(f"PROGRESS DASHBOARD - {status['timestamp']}")
    print("="*80)
    
    for phase_name, phase_info in status["phases"].items():
        phase_num = phase_name.split("_")[0]
        phase_label = {
            "1": "LOSO Validation",
            "2": "Teacher Retrain",
            "3": "Distillation",
            "4": "Export to TFLite"
        }.get(phase_num, phase_name)
        
        phase_status = phase_info.get("status", "UNKNOWN")
        
        print(f"\n[Phase {phase_num}] {phase_label}")
        print(f"Status: {phase_status}")
        
        if "message" in phase_info:
            print(f"Info: {phase_info['message']}")
        
        if "kappa_mean" in phase_info:
            kappa = phase_info.get("kappa_mean")
            acc = phase_info.get("accuracy_mean")
            folds = phase_info.get("n_folds")
            print(f"Results: κ={kappa:.4f} ± σ, acc={acc:.4f}, folds={folds}")
        
        if "checkpoint" in phase_info:
            size = phase_info.get("size_mb", "?")
            print(f"Checkpoint: {phase_info['checkpoint']} ({size} MB)")
        
        if "config" in phase_info:
            cfg = phase_info["config"]
            print(f"Config: focal_γ={cfg['focal_gamma']}, mixup_α={cfg['mixup_alpha']}")
            aug = ", ".join(cfg.get("augmentations", []))
            print(f"Augmentations: {aug}")
        
        if "outputs" in phase_info:
            out = phase_info["outputs"]
            tflite_mb = phase_info.get("tflite_size_mb", "?")
            print(f"TFLite: {out['tflite_int8']} ({tflite_mb} MB)")
    
    print("\n" + "="*80)
    print("LEGEND: ✓ COMPLETE | 🔄 RUNNING | ⏳ WAITING")
    print("="*80 + "\n")
    
    return status


def save_progress_report(status):
    """Save detailed progress report to JSON."""
    report_path = Path('artifacts/PROGRESS_REPORT.json')
    with open(report_path, 'w') as f:
        json.dump(status, f, indent=2)
    print(f"✓ Progress report saved: {report_path}\n")


if __name__ == "__main__":
    status = print_dashboard()
    save_progress_report(status)
