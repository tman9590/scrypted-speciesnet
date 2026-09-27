# Model card

## Model

- Source: Google SpeciesNet v4.0.3b
- Architecture: EfficientNetV2-M
- Input: 480x480 RGB
- Outputs: 2,498 fixed logits
- Training domain: globally distributed camera-trap imagery

## Adapter scope

The graph is unchanged except for an NCHW-to-NHWC input permutation required by
Scrypted backends. Backend exports must preserve top-1 agreement with the
official PyTorch checkpoint on deterministic parity inputs.

This repository does not claim that converted runtime parity establishes field
accuracy. Deployment evaluation should include empty frames, people, vehicles,
daylight, infrared, rain, blur, occlusion, small animals, and local species.

