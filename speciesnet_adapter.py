from __future__ import annotations

import collections
import json
from pathlib import Path

import torch


MODEL_ID = "google/speciesnet/pyTorch/v4.0.3b/1"
IMAGE_SIZE = 480


class ScryptedSpeciesNet(torch.nn.Module):
    """Adapt Scrypted's NCHW float input to SpeciesNet's NHWC float graph."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images.permute(0, 2, 3, 1))


def download_model() -> Path:
    import kagglehub

    return Path(kagglehub.model_download(MODEL_ID))


def load_source(model_dir: Path | None = None) -> tuple[ScryptedSpeciesNet, list[str], dict]:
    model_dir = model_dir or download_model()
    info = json.loads((model_dir / "info.json").read_text(encoding="utf-8"))
    model = torch.load(model_dir / info["classifier"], map_location="cpu", weights_only=False)
    model.eval()
    raw_labels = (model_dir / info["classifier_labels"]).read_text(encoding="utf-8").splitlines()
    labels = display_labels(raw_labels)
    return ScryptedSpeciesNet(model).eval(), labels, info


def display_labels(raw_labels: list[str]) -> list[str]:
    parts = [label.split(";") for label in raw_labels]
    common = [row[-1] or next(value for value in reversed(row[1:-1]) if value) for row in parts]
    duplicates = {name for name, count in collections.Counter(common).items() if count > 1}
    result = []
    for name, row in zip(common, parts):
        if name in duplicates:
            scientific = " ".join(value for value in row[4:6] if value)
            name = f"{name} ({scientific or row[0][:8]})"
        result.append(name)
    duplicate_candidates = {
        name for name, count in collections.Counter(result).items() if count > 1
    }
    result = [
        f"{name} [{row[0][:8]}]" if name in duplicate_candidates else name
        for name, row in zip(result, parts)
    ]
    if len(result) != len(set(result)):
        raise ValueError("SpeciesNet labels cannot be made unique")
    return result
