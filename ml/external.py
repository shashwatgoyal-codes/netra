"""External validation on DDR (Li et al., 2019): photos from 147 hospitals in
China, a different population and different cameras from APTOS.

The model never trains on DDR, and the referral threshold stays the one set
on the APTOS validation split. Nothing is tuned here: this measures what
happens when the app meets a hospital it has never seen. DDR grade 5
(ungradable) is excluded, as in the DDR paper's grading benchmark.

Adds an "external" block to artifacts/<model>/<variant>/metrics.json, which
ship.py copies into the app's model page.

Usage: uv run python external.py --model dinov3 [--variant f16w]
"""

import argparse
import json

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from common import ARTIFACTS, DATA, REFERABLE_FROM
from evaluate import refer_decision, referral_stats, run_paths

DDR = DATA / "ddr" / "DDR-dataset" / "DR_grading"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", choices=["fp32", "f16w"], default="f16w")
    args = ap.parse_args()
    report = ARTIFACTS / args.model / args.variant
    metrics = json.loads((report / "metrics.json").read_text())

    labels = pd.read_csv(DDR / "test.txt", sep=" ", header=None, names=["file", "grade"])
    labels = labels[labels.grade <= 4]
    onnx_file = ARTIFACTS / args.model / f"netra.{args.variant}.onnx".replace(".fp32", "")
    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    probs, _ = run_paths(sess, [DDR / "test" / f for f in labels.file])

    grades = labels.grade.to_numpy()
    is_ref = grades >= REFERABLE_FROM
    pred = refer_decision(probs, metrics["threshold"])
    external = {
        "dataset": "DDR test split (147 hospitals, China), excluding ungradable",
        "n": int(len(labels)),
        "n_referable": int(is_ref.sum()),
        "referral": {"auc": float(roc_auc_score(is_ref, probs[:, REFERABLE_FROM:].sum(1))), **referral_stats(pred, is_ref)},
        "grading": {
            "qwk": float(cohen_kappa_score(grades, probs.argmax(1), weights="quadratic")),
            "accuracy": float((probs.argmax(1) == grades).mean()),
        },
    }
    metrics["external"] = external
    (report / "metrics.json").write_text(json.dumps(metrics, indent=1))
    r = external["referral"]
    print(f"DDR external ({external['n']} photos, {external['n_referable']} referable): "
          f"AUC {r['auc']:.3f}  sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}  "
          f"QWK {external['grading']['qwk']:.3f}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
