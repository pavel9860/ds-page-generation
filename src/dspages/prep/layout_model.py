"""Picture detector for the layout: YOLO11n trained on DocLayNet, exported to ONNX with dynamic input size.

python -m dspages.prep.layout_model      (needs: pip install ultralytics)
Downloads PictureCfg.weights_url (AGPL-3.0 weights, kept out of this repository) and writes PictureCfg.model.
"""
import tempfile
import urllib.request
from pathlib import Path

from ..config import PictureCfg


def main():
    from ultralytics import YOLO
    c = PictureCfg()
    out = Path(c.model).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        pt = Path(d) / "yolov11n-doclaynet.pt"
        urllib.request.urlretrieve(c.weights_url, pt)
        Path(YOLO(pt).export(format="onnx", imgsz=c.imgsz, opset=17, dynamic=True, simplify=True)).replace(out)
    print(out)


if __name__ == "__main__":
    main()
