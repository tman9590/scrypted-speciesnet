from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from speciesnet_adapter import IMAGE_SIZE, load_source


FILES = {
    "onnx": ["speciesnet.onnx"],
    "openvino": ["speciesnet.xml", "speciesnet.bin"],
    "ncnn": ["speciesnet.ncnn.param", "speciesnet.ncnn.bin"],
    "coreml": [
        "speciesnet.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        "speciesnet.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        "speciesnet.mlpackage/Manifest.json",
    ],
}


def main() -> None:
    _, labels, info = load_source()
    common = {
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "model": "resnet",
        "labels": {str(index): label for index, label in enumerate(labels)},
    }
    root_config = {
        **common,
        "source": {
            "name": "Google SpeciesNet",
            "version": info["version"],
            "model_id": "google/speciesnet/pyTorch/v4.0.3b/1",
        },
        "backends": {backend: {"files": files} for backend, files in FILES.items()},
    }
    (ROOT / "config.json").write_text(json.dumps(root_config, indent=2) + "\n", encoding="utf-8")
    for backend, files in FILES.items():
        directory = ROOT / "models" / backend
        directory.mkdir(parents=True, exist_ok=True)
        config = {**common, "files": files}
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Generated configs for {len(labels)} SpeciesNet labels")


if __name__ == "__main__":
    main()

