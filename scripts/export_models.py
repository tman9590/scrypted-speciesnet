from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from speciesnet_adapter import IMAGE_SIZE, load_source


def export_onnx(model: torch.nn.Module) -> Path:
    output = ROOT / "models" / "onnx" / "speciesnet.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model,
        sample,
        output,
        input_names=["images"],
        output_names=["out0"],
        opset_version=17,
        dynamo=False,
    )
    return output


def export_openvino(onnx_path: Path) -> None:
    import openvino as ov

    output = ROOT / "models" / "openvino"
    output.mkdir(parents=True, exist_ok=True)
    converted = ov.convert_model(onnx_path)
    ov.save_model(converted, output / "speciesnet.xml", compress_to_fp16=True)


def export_ncnn(onnx_path: Path, pnnx: Path) -> None:
    output = ROOT / "models" / "ncnn"
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / "work").mkdir(parents=True, exist_ok=True)
    prefix = output / "speciesnet.ncnn"
    subprocess.run(
        [
            str(pnnx),
            str(onnx_path),
            f"inputshape=[1,3,{IMAGE_SIZE},{IMAGE_SIZE}]",
            "fp16=1",
            f"ncnnparam={prefix}.param",
            f"ncnnbin={prefix}.bin",
            f"pnnxparam={ROOT / 'work' / 'speciesnet.pnnx.param'}",
            f"pnnxbin={ROOT / 'work' / 'speciesnet.pnnx.bin'}",
            f"pnnxpy={ROOT / 'work' / 'speciesnet_pnnx.py'}",
            f"onnx={ROOT / 'work' / 'speciesnet.pnnx.onnx'}",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", choices=["onnx", "openvino", "ncnn"], default=["onnx", "openvino", "ncnn"])
    parser.add_argument("--pnnx", type=Path)
    args = parser.parse_args()
    model, _, _ = load_source()
    onnx_path = ROOT / "models" / "onnx" / "speciesnet.onnx"
    if "onnx" in args.backends or ({"openvino", "ncnn"} & set(args.backends) and not onnx_path.exists()):
        onnx_path = export_onnx(model)
    if "openvino" in args.backends:
        export_openvino(onnx_path)
    if "ncnn" in args.backends:
        pnnx = args.pnnx or Path(shutil.which("pnnx") or "")
        if not pnnx.is_file():
            raise FileNotFoundError("Pass --pnnx with the pnnx executable path")
        export_ncnn(onnx_path, pnnx)


if __name__ == "__main__":
    main()
