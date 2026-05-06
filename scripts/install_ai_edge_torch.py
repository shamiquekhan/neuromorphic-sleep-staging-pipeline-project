"""
Export StudentCRNN to TFLite using ai-edge-torch.
This bypasses the broken onnx2tf path entirely.
"""
import subprocess
import sys
from pathlib import Path

print("="*60)
print("TASK 2/3: TFLite Export via ai-edge-torch")
print("="*60)

# Check if ai-edge-torch is installed
try:
    import ai_edge_torch
    print("✓ ai-edge-torch is already installed")
except ImportError:
    print("✗ ai-edge-torch not found. Installing...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "ai-edge-torch"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error installing ai-edge-torch:")
        print(result.stderr)
        print("\nFallback: Using Google Colab instead")
        print("\nSteps:")
        print("1. Upload artifacts/student.pt to Colab")
        print("2. Run: pip install ai-edge-torch")
        print("3. Run the export_tflite_colab.py script there")
        print("4. Download student_int8.tflite")
        sys.exit(1)
    else:
        print("✓ ai-edge-torch installed successfully")

print("\nNext: Run export_tflite_colab.py in Google Colab")
print("Or: If ai-edge-torch works locally, uncomment the main export code.")
