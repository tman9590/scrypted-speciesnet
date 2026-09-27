from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from speciesnet_adapter import IMAGE_SIZE, load_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", choices=["onnx", "openvino", "ncnn", "coreml"], default=["onnx", "openvino", "ncnn"])
    parser.add_argument("--report", type=Path, default=ROOT / "validation.json")
    args = parser.parse_args()
    model, labels, _ = load_source()
    rng = np.random.default_rng(403)
    samples = [rng.integers(0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8).astype(np.float32) / 255 for _ in range(3)]
    runners = {}
    if "onnx" in args.backends:
        import onnxruntime as ort

        session = ort.InferenceSession(str(ROOT / "models" / "onnx" / "speciesnet.onnx"), providers=["CPUExecutionProvider"])
        key = session.get_inputs()[0].name
        runners["onnx"] = lambda value: session.run(None, {key: value})[0]
    if "openvino" in args.backends:
        import openvino as ov

        compiled = ov.Core().compile_model(str(ROOT / "models" / "openvino" / "speciesnet.xml"), "CPU")
        runners["openvino"] = lambda value: compiled(value)[0]
    if "ncnn" in args.backends:
        import ncnn

        network = ncnn.Net()
        network.load_param(str(ROOT / "models" / "ncnn" / "speciesnet.ncnn.param"))
        network.load_model(str(ROOT / "models" / "ncnn" / "speciesnet.ncnn.bin"))

        def run_ncnn(value):
            with network.create_extractor() as extractor:
                extractor.input(network.input_names()[0], ncnn.Mat(value[0]).clone())
                status, result = extractor.extract("out0")
                if status:
                    raise RuntimeError(status)
                return np.asarray(result)[None]

        runners["ncnn"] = run_ncnn
    if "coreml" in args.backends:
        import coremltools as ct

        package = ROOT / "models" / "coreml" / "speciesnet.mlpackage"
        coreml = ct.models.MLModel(str(package))
        key = coreml.get_spec().description.input[0].name

        def run_coreml(value):
            pixels = np.rint(value[0].transpose(1, 2, 0) * 255).astype(np.uint8)
            return next(iter(coreml.predict({key: Image.fromarray(pixels)}).values()))

        runners["coreml"] = run_coreml
    report = {"passed": True, "labels": len(labels), "backends": {}}
    for backend, runner in runners.items():
        errors = []
        agreements = []
        for sample in samples:
            with torch.inference_mode():
                expected = model(torch.from_numpy(sample)).numpy()
            actual = runner(sample)
            if actual.shape != (1, len(labels)) or not np.isfinite(actual).all():
                raise ValueError(f"{backend}: invalid output {actual.shape}")
            errors.append(float(np.max(np.abs(expected - actual))))
            agreements.append(int(expected.argmax()) == int(actual.argmax()))
        report["backends"][backend] = {"max_logit_error": max(errors), "top1_agreement": all(agreements)}
        if not all(agreements):
            raise ValueError(f"{backend}: top-1 prediction changed")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
