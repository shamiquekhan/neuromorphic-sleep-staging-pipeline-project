from __future__ import annotations

from pathlib import Path
import copy

import torch
import torch.nn as nn


def dynamic_quantize_gru_linear(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to recurrent and linear layers."""
    return torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.GRU, nn.Linear},
        dtype=torch.qint8,
    )


def prune_linear_weights(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Simple unstructured pruning on linear layers for size reduction experiments."""
    import torch.nn.utils.prune as prune

    out = copy.deepcopy(model)
    for mod in out.modules():
        if isinstance(mod, nn.Linear):
            prune.l1_unstructured(mod, name="weight", amount=amount)
            prune.remove(mod, "weight")
    return out


def save_model_state(model: nn.Module, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)


def quantize_checkpoint(student_ckpt: str, output_path: str = "artifacts/student_int8.pt") -> str:
    from .models import StudentCRNN

    model = StudentCRNN()
    ckpt = torch.load(student_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    q_model = dynamic_quantize_gru_linear(model)
    save_model_state(q_model, output_path)
    return output_path
