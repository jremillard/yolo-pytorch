import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "yolov9-pytorch"))

from export import export_onnx


def test_export_creates_file(tmp_path):
    out = tmp_path / "model.onnx"
    export_onnx(out)
    assert out.exists()
    model = onnx.load(str(out))
    assert model.ir_version >= 3
