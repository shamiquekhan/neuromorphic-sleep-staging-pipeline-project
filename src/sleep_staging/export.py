from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class StudentStreamingWrapper(nn.Module):
    """Wrap StudentCRNN for static single-step streaming export."""

    def __init__(self, student: nn.Module):
        super().__init__()
        self.student = student

    def forward(self, spec_frame: torch.Tensor, h_in: torch.Tensor):
        logits, _, h_out = self.student(spec_frame, h_in, return_features=True)
        return logits, h_out


def export_tflite_ai_edge(
    student: nn.Module,
    out_path: str = "artifacts/student.tflite",
    freq_bins: int = 128,
    time_bins: int = 29,
    n_channels: int = 4,
    hidden: int = 128,
    gru_layers: int = 2,
):
    """Convert PyTorch model directly to TFLite via ai-edge-torch."""
    try:
        import ai_edge_torch
    except ImportError as exc:
        raise ImportError(
            "ai-edge-torch not found. Install with: pip install ai-edge-torch "
            "(Python 3.9-3.11, TensorFlow >=2.17)."
        ) from exc

    wrapper = StudentStreamingWrapper(student).eval()
    spec_sample = torch.randn(1, 1, n_channels, freq_bins, time_bins)
    h_sample = torch.zeros(gru_layers, 1, hidden)  # (num_layers, batch, hidden)

    log.info("Converting with ai-edge-torch (PyTorch -> TFLite directly)...")
    edge_model = ai_edge_torch.convert(wrapper, (spec_sample, h_sample))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(out))
    size_kb = out.stat().st_size / 1024.0
    log.info("TFLite model saved -> %s (%.1f KB)", out, size_kb)
    return str(out)


def export_onnx_static(
    student: nn.Module,
    out_path: str = "artifacts/student_static.onnx",
    freq_bins: int = 128,
    time_bins: int = 29,
    n_channels: int = 4,
    hidden: int = 128,
    gru_layers: int = 2,
    opset: int = 17,
):
    """Export static-shape ONNX in single-step streaming mode."""
    wrapper = StudentStreamingWrapper(student).eval()

    # 4-channel input, fully static shapes
    spec_sample = torch.randn(1, 1, n_channels, freq_bins, time_bins)
    h_sample = torch.zeros(gru_layers, 1, hidden)  # (num_layers, batch, hidden)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (spec_sample, h_sample),
        str(out),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["spectrogram", "h_in"],
        output_names=["logits", "h_out"],
        dynamic_axes=None,  # FULLY STATIC — critical for TFLite conversion
    )

    import onnx
    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    size_kb = out.stat().st_size / 1024.0
    log.info("Static ONNX saved -> %s (%.1f KB)", out, size_kb)
    return str(out)


def convert_onnx_to_tflite_onnx2tf(
    onnx_path: str,
    out_path: str = "artifacts/student.tflite",
    quantize: bool = True,
    freq_bins: int = 69,
    time_bins: int = 29,
):
    """Convert static ONNX to TFLite using onnx2tf."""
    import subprocess
    import sys

    out_file = Path(out_path)
    tf_dir = out_file.parent / "student_tf"

    cmd = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        onnx_path,
        "-o",
        str(tf_dir),
        "-osd",
        "-b",
        "1",
    ]
    if quantize:
        cmd += ["-cind", "spectrogram", f"[[[[1,1,1,{freq_bins},{time_bins}]]]]", "0", "1"]

    log.info("Running onnx2tf conversion...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"onnx2tf failed:\n{result.stderr}")

    tflites = list(tf_dir.glob("*.tflite"))
    if not tflites:
        raise FileNotFoundError(f"No .tflite found in {tf_dir}")

    import shutil

    out_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tflites[0], out_file)
    log.info("TFLite saved -> %s", out_file)
    return str(out_file)


def quantize_student(student: nn.Module, out_path: str = "artifacts/student_int8.pt"):
    """Apply dynamic quantization to GRU and Linear layers."""
    import torch.quantization as tq

    # Check for NaN weights that would break quantization
    has_nan = False
    for name, param in student.named_parameters():
        if param.isnan().any():
            log.warning("NaN detected in %s, skipping quantization", name)
            has_nan = True
            break

    if has_nan:
        log.warning("NaN weights found; skipping quantization, saving as-is")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": student.state_dict()}, out)
        size_kb = out.stat().st_size / 1024.0
        log.info("Unquantized student saved -> %s (%.1f KB)", out, size_kb)
        return student

    student_q = tq.quantize_dynamic(student.cpu(), {nn.GRU, nn.Linear}, dtype=torch.qint8)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student_q.state_dict(), out)
    size_kb = out.stat().st_size / 1024.0
    log.info("INT8 quantized student -> %s (%.1f KB)", out, size_kb)
    return student_q


def tflite_to_c_array(
    tflite_path: str,
    out_path: str = "firmware/src/student_model_data.cc",
    var_name: str = "student_model_data",
):
    """Convert TFLite FlatBuffer bytes into C array for TFLite Micro."""
    data = Path(tflite_path).read_bytes()
    lines = [
        f"// Auto-generated from {Path(tflite_path).name}",
        f"// Size: {len(data):,} bytes",
        '#include "sleep_inference.h"',
        "",
        f"const unsigned char {var_name}[] = {{",
    ]
    for i in range(0, len(data), 16):
        chunk = data[i : i + 16]
        hex_str = ", ".join(f"0x{byte:02x}" for byte in chunk)
        lines.append(f"  {hex_str},")
    lines += [
        "};",
        f"const unsigned int {var_name}_len = {len(data)};",
    ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("C array written -> %s (%.1f KB)", out, len(data) / 1024.0)


def full_export_pipeline(
    student: nn.Module,
    artifacts_dir: str = "artifacts",
    firmware_dir: str = "firmware/src",
    freq_bins: int = 128,
    time_bins: int = 29,
    n_channels: int = 4,
    hidden: int = 128,
    gru_layers: int = 2,
):
    """Quantize, export TFLite, and generate firmware C array."""
    art = Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)

    quantize_student(student, str(art / "student_int8.pt"))

    tflite_path = str(art / "student.tflite")
    try:
        export_tflite_ai_edge(student, tflite_path, freq_bins, time_bins, n_channels, hidden, gru_layers)
    except ImportError:
        log.warning("ai-edge-torch not available, falling back to static ONNX -> onnx2tf")
        onnx_path = export_onnx_static(student, str(art / "student_static.onnx"),
                                        freq_bins, time_bins, n_channels, hidden, gru_layers)
        try:
            convert_onnx_to_tflite_onnx2tf(onnx_path, tflite_path,
                                            freq_bins=freq_bins, time_bins=time_bins)
        except Exception as exc:
            log.error("onnx2tf conversion failed: %s", exc)
            log.error("Install ai-edge-torch for reliable conversion: pip install ai-edge-torch")
            return

    tflite_to_c_array(tflite_path, str(Path(firmware_dir) / "student_model_data.cc"))
    log.info("Full export pipeline complete")


# Backward-compatible helper expected by older flows.
def export_student_to_onnx(student_ckpt: str, output_path: str = "artifacts/student_static.onnx", opset_version: int = 17) -> str:
    from .models import StudentCRNN

    model = StudentCRNN()
    ckpt = torch.load(student_ckpt, map_location="cpu")
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return export_onnx_static(model, out_path=output_path, opset=opset_version)
