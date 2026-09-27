from __future__ import annotations

import shutil
import sys
from pathlib import Path

import coremltools as ct
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from speciesnet_adapter import IMAGE_SIZE, load_source


def main() -> None:
    model, _, _ = load_source()
    sample = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced = torch.jit.trace(model, sample, strict=False)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name="images", shape=sample.shape, scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="out0")],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )
    destination = ROOT / "models" / "coreml" / "speciesnet.mlpackage"
    if destination.exists():
        shutil.rmtree(destination)
    converted.save(destination)
    print(destination)


if __name__ == "__main__":
    main()

