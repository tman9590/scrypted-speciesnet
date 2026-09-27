# SpeciesNet for Scrypted

This repository packages Google's **SpeciesNet v4.0.3b whole-image classifier**
for Scrypted's ONNX, OpenVINO, CoreML, and NCNN object-detection backends. It
keeps the official 2,498-output order and exposes readable, unique labels.

## Why SpeciesNet

SpeciesNet is purpose-built for camera traps rather than generic web photos. It
uses EfficientNetV2-M and was trained by Google on more than 65 million images.
Google reports that its ensemble finds 99.4% of animal-containing images; 83%
of results reach species level, and 94.5% of those species-level predictions
are correct on held-out camera-trap projects.

BioCLIP 2.5 Huge is the strongest current zero-shot biological foundation model,
but its ViT-H/14 graph is impractical for continuous inference across all four
Scrypted backends. SpeciesNet is the higher-confidence deployment choice for
fixed-label camera-trap classification.

## Install in Scrypted

1. Install the appropriate Scrypted backend: ONNX (NVIDIA), OpenVINO
   (Intel/AMD), CoreML (Apple Silicon), or NCNN (Vulkan-capable systems).
2. In that backend plugin, create a model device.
3. Use the direct config URL for the installed backend:

   | Backend | Model URL |
   | --- | --- |
   | ONNX | `https://media.githubusercontent.com/media/tman9590/scrypted-speciesnet/v1.0.1/models/onnx/config.json` |
   | OpenVINO | `https://media.githubusercontent.com/media/tman9590/scrypted-speciesnet/v1.0.1/models/openvino/config.json` |
   | CoreML | `https://media.githubusercontent.com/media/tman9590/scrypted-speciesnet/v1.0.1/models/coreml/config.json` |
   | NCNN | `https://media.githubusercontent.com/media/tman9590/scrypted-speciesnet/v1.0.1/models/ncnn/config.json` |

4. Select the new classifier in Scrypted NVR.

The direct `media.githubusercontent.com` URLs are required because the model
artifacts use Git LFS; GitHub's ordinary raw URLs return LFS pointer text. The
root `config.json` is the canonical manifest, and backend configs are generated
compatibility projections.

## Rebuild

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements-build.txt
.venv\Scripts\python scripts\generate_configs.py
.venv\Scripts\python scripts\export_models.py --pnnx path\to\pnnx.exe
.venv\Scripts\python scripts\validate_exports.py
```

CoreML export runs on macOS through GitHub Actions.

## Important limitations

- This adapter deploys the official whole-image classifier. It does not include
  SpeciesNet's MegaDetector stage, location geofencing, or taxonomic roll-up.
- Scrypted resizes the frame to 480x480. That is close to, but not identical to,
  SpeciesNet's original aspect-aware vertical center crop.
- SpeciesNet's published accuracy describes the complete Google ensemble, not
  this backend adapter. Validate on footage from your own cameras.
- A classification result has no bounding box. Use a normal Scrypted detector
  alongside this classifier when spatial animal boxes are required.

## Licensing and attribution

Adapter code is Apache-2.0. SpeciesNet code and model are published by Google
under Apache-2.0. See the upstream repository and model card for citations and
limitations: https://github.com/google/cameratrapai
