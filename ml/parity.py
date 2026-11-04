"""Browser/Python parity check for preprocessing + inference.

Copies a few held-out test photos into web/public/parity/ and writes the
probabilities Python's pipeline gives for them (expected.json). Load the same
files in the app and compare: the predicted grade should match and the
probabilities should agree within a few points (canvas resizing is not
bit-identical to PIL).

Usage: uv run python parity.py --model dinov3 [--n 8]
       uv run python parity.py --clean
"""

import argparse
import json
import shutil

import onnxruntime as ort
import pandas as pd
from PIL import Image

from common import ARTIFACTS, RAW, ROOT, SPLITS, crop_retina, to_tensor_array

OUT = ROOT.parent / "web" / "public" / "parity"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    if args.clean:
        shutil.rmtree(OUT, ignore_errors=True)
        return

    test = pd.read_csv(SPLITS).query("split == 'test'")
    # One or two photos per grade so every class is exercised.
    picks = test.groupby("diagnosis").head(max(1, args.n // 5)).head(args.n)
    sess = ort.InferenceSession(str(ARTIFACTS / args.model / "netra.f16w.onnx"), providers=["CPUExecutionProvider"])
    OUT.mkdir(parents=True, exist_ok=True)
    expected = []
    for row in picks.itertuples():
        src = RAW / "train_images" / f"{row.id_code}.png"
        shutil.copy(src, OUT / src.name)
        probs = sess.run(None, {"image": to_tensor_array(crop_retina(Image.open(src)))[None]})[0][0]
        expected.append({"file": src.name, "label": int(row.diagnosis), "probs": [round(float(p), 4) for p in probs]})
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1))
    print(f"wrote {len(expected)} images + expected.json to {OUT}")


if __name__ == "__main__":
    main()
