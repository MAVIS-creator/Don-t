"""CPU smoke test for a Sewer-ML v3 model image.

Exercises model construction, a real forward pass, ONNX export, validation, and
ONNX Runtime parity without touching campaign data or requesting a GPU.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from train_v3 import make_model


def run(architecture: str, image_size: int = 224) -> float:
    torch.manual_seed(42)
    model = make_model(architecture, pretrained=False).cpu().eval()
    example = torch.randn(1, 3, image_size, image_size)
    with torch.inference_mode():
        native = model(example).numpy()
    if native.shape != (1, 9) or not np.isfinite(native).all():
        raise RuntimeError(f"Invalid {architecture} output: shape={native.shape}")

    with tempfile.TemporaryDirectory(prefix="sewerml-smoke-") as directory:
        model_path = Path(directory) / "model.onnx"
        torch.onnx.export(
            model,
            example,
            str(model_path),
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )
        onnx.checker.check_model(str(model_path))
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        exported = session.run(None, {"image": example.numpy()})[0]

    parity = float(np.max(np.abs(native - exported)))
    if not np.isfinite(parity) or parity > 1e-4:
        raise RuntimeError(f"ONNX parity failed for {architecture}: {parity}")
    print(f"CPU smoke passed: architecture={architecture}, onnx_parity={parity:.8f}")
    return parity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("efficientnet_b0", "convnext_tiny"), required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    run(args.architecture, args.image_size)


if __name__ == "__main__":
    main()
