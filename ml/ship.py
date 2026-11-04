"""Copy an evaluated model into the web app and bump the offline cache version.

Usage: uv run python ship.py --model dinov3 [--variant f16w]
"""

import argparse
import re
import shutil

from common import ARTIFACTS, ROOT

WEB = ROOT.parent / "web"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", choices=["fp32", "f16w"], default="f16w")
    args = ap.parse_args()

    src = ARTIFACTS / args.model
    onnx_file = src / f"netra.{args.variant}.onnx".replace(".fp32", "")
    report = src / args.variant
    dest = WEB / "public" / "model"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_file, dest / "netra.onnx")
    for name in ("model-meta.json", "metrics.json"):
        shutil.copy(report / name, dest / name)

    # A new cache name makes installed apps fetch the new model once online.
    sw = WEB / "public" / "sw.js"
    text = sw.read_text()
    n = int(re.search(r'VERSION = "netra-v(\d+)"', text).group(1))
    sw.write_text(text.replace(f'"netra-v{n}"', f'"netra-v{n + 1}"'))
    print(f"shipped {onnx_file.name} ({onnx_file.stat().st_size / 1e6:.0f} MB) -> {dest}, cache netra-v{n + 1}")


if __name__ == "__main__":
    main()
